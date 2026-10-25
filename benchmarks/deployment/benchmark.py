"""Benchmark the shared RGB-D-to-PointRAFT pipeline across hardware."""

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import open3d as o3d
import psutil
import torch
import torch_geometric

from .pipeline import PIPELINE_SCOPE, PROJECT_ROOT, InferencePipeline, resolve_device

STAGE_FIELDS = (
    "file_loading_ms",
    "rgbd_to_pointcloud_ms",
    "dbscan_ms",
    "preprocessing_ms",
    "network_inference_ms",
    "end_to_end_ms",
)
SAMPLE_FIELDS = (
    "sample_index",
    "rgb_file",
    "tuber_id",
    "points_unfiltered",
    "points_filtered",
    "points_model_input",
    "height_m",
    "prediction_g",
) + STAGE_FIELDS


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer command-line argument."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def run_command(command: Sequence[str]) -> Dict[str, Any]:
    """Run a read-only metadata command and retain all output.

    Parameters
    ----------
    command
        Command and arguments to execute without a shell.

    Returns
    -------
    dict
        Command, return code, standard output, and standard error.
    """
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, check=False
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except OSError as error:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }


def read_text_file(path: Path) -> str:
    """Read a text or device-tree file without trailing null bytes.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    str
        Decoded and stripped content, or an empty string when unavailable.
    """
    try:
        return path.read_bytes().replace(b"\x00", b"\n").decode().strip()
    except OSError:
        return ""


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_power_mode(expected_watts: Optional[int]) -> Optional[Dict[str, Any]]:
    """Read the Jetson nvpmodel mode when available.

    Parameters
    ----------
    expected_watts
        Required 10 W, 15 W, or 25 W mode. ``None`` disables the guard.

    Returns
    -------
    dict or None
        Parsed mode metadata on Jetson, otherwise ``None``.

    Raises
    ------
    RuntimeError
        If an expected mode is requested off Jetson or does not match.
    """
    if shutil.which("nvpmodel") is None:
        if expected_watts is not None:
            raise RuntimeError("--expected-power-watts requires Jetson nvpmodel")
        return None

    result = run_command(["nvpmodel", "-q"])
    name_match = re.search(r"NV Power Mode:\s*([^\r\n]+)", result["stdout"])
    watts_match = re.search(
        r"NV Power Mode:\s*(?:MODE[_ -]?)?(10|15|25)\s*W",
        result["stdout"],
    )
    id_match = re.search(r"(?m)^\s*(\d+)\s*$", result["stdout"])
    if name_match is None or id_match is None:
        raise RuntimeError(
            "Could not parse nvpmodel output:\n"
            f"{result['stdout']}\n{result['stderr']}"
        )
    watts = int(watts_match.group(1)) if watts_match else None
    if expected_watts is not None and watts != expected_watts:
        raise RuntimeError(
            f"Expected {expected_watts}W but nvpmodel reports "
            f"{name_match.group(1).strip()}"
        )
    return {
        "id": int(id_match.group(1)),
        "name": name_match.group(1).strip(),
        "configured_watts": watts,
        "command": result,
    }


def slugify(value: str) -> str:
    """Convert hardware text into a stable path component.

    Parameters
    ----------
    value
        Human-readable hardware description.

    Returns
    -------
    str
        Lowercase underscore-delimited slug.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown_hardware"


def device_description(
    device: torch.device, power_mode: Optional[Dict[str, Any]]
) -> str:
    """Return the measured CPU or GPU description.

    Parameters
    ----------
    device
        Model execution device.
    power_mode
        Optional Jetson power metadata.

    Returns
    -------
    str
        Hardware label suitable for reports.
    """
    model = read_text_file(Path("/sys/firmware/devicetree/base/model"))
    if model:
        suffix = (
            f" {power_mode['configured_watts']}W"
            if power_mode and power_mode["configured_watts"]
            else ""
        )
        return f"{model}{suffix}"
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    cpu = platform.processor() or platform.machine()
    return f"CPU {cpu} on {platform.node()}"


def create_run_directory(output_root: Path, description: str) -> Path:
    """Create a unique result directory for one benchmark run.

    Parameters
    ----------
    output_root
        Parent directory for inference results.
    description
        Measured hardware description.

    Returns
    -------
    pathlib.Path
        Newly created run directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{slugify(description)}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / base_name
    suffix = 2
    while run_dir.exists():
        run_dir = output_root / f"{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def configure_logging(run_dir: Path) -> logging.Logger:
    """Configure matching console and file benchmark logs.

    Parameters
    ----------
    run_dir
        Directory receiving ``run.log``.

    Returns
    -------
    logging.Logger
        Configured benchmark logger.
    """
    logger = logging.getLogger("deployment_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def benchmark_sample(
    sample_index: int,
    rgb_file: Path,
    pipeline: InferencePipeline,
) -> Dict[str, Any]:
    """Measure one complete RGB-D-to-regression sample.

    Parameters
    ----------
    sample_index
        One-based position in the measured sequence.
    rgb_file
        Masked RGB image path.
    pipeline
        Shared inference pipeline.

    Returns
    -------
    dict
        Sample identity, point counts, prediction, and stage timings.
    """
    total_started = time.perf_counter_ns()

    started = time.perf_counter_ns()
    rgba, depth = pipeline.load_rgbd(rgb_file)
    file_loading_ms = (time.perf_counter_ns() - started) / 1_000_000

    started = time.perf_counter_ns()
    unfiltered, tuber_id = pipeline.create_point_cloud(rgba, depth, rgb_file)
    rgbd_to_pointcloud_ms = (time.perf_counter_ns() - started) / 1_000_000

    started = time.perf_counter_ns()
    filtered = pipeline.filter_point_cloud(unfiltered)
    dbscan_ms = (time.perf_counter_ns() - started) / 1_000_000

    started = time.perf_counter_ns()
    data, object_height, filtered_points = pipeline.prepare_model_input(
        filtered, tuber_id
    )
    preprocessing_ms = (time.perf_counter_ns() - started) / 1_000_000

    started = time.perf_counter_ns()
    prediction = pipeline.predict(data)
    network_inference_ms = (time.perf_counter_ns() - started) / 1_000_000
    end_to_end_ms = (time.perf_counter_ns() - total_started) / 1_000_000

    return {
        "sample_index": sample_index,
        "rgb_file": str(rgb_file.relative_to(PROJECT_ROOT)),
        "tuber_id": tuber_id,
        "points_unfiltered": len(unfiltered.points),
        "points_filtered": filtered_points,
        "points_model_input": data.pos.size(0),
        "height_m": object_height,
        "prediction_g": prediction.item(),
        "file_loading_ms": file_loading_ms,
        "rgbd_to_pointcloud_ms": rgbd_to_pointcloud_ms,
        "dbscan_ms": dbscan_ms,
        "preprocessing_ms": preprocessing_ms,
        "network_inference_ms": network_inference_ms,
        "end_to_end_ms": end_to_end_ms,
    }


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    """Summarize one latency distribution.

    Parameters
    ----------
    values
        Measured values in milliseconds.

    Returns
    -------
    dict
        Mean, median, P95, population standard deviation, minimum, and maximum.
    """
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95)),
        "std_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def total_memory_bytes() -> int:
    """Return the host's total physical memory in bytes.

    Returns
    -------
    int
        Total installed physical memory reported by the operating system.
    """
    return int(psutil.virtual_memory().total)


def collect_system_metadata(
    pipeline: InferencePipeline,
    description: str,
    power_mode: Optional[Dict[str, Any]],
    sample_count: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Collect reproducibility metadata without changing system state.

    Parameters
    ----------
    pipeline
        Initialized shared pipeline.
    description
        Measured hardware description.
    power_mode
        Optional Jetson nvpmodel metadata.
    sample_count
        Number of measured RGB-D images.
    args
        Parsed benchmark arguments.

    Returns
    -------
    dict
        Hardware, software, source, dataset, and invocation metadata.
    """
    uv_binary = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    memory_bytes = total_memory_bytes()
    cuda_device = (
        torch.cuda.get_device_name(pipeline.device)
        if pipeline.device.type == "cuda"
        else None
    )
    cuda_capability = (
        list(torch.cuda.get_device_capability(pipeline.device))
        if pipeline.device.type == "cuda"
        else None
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_scope": PIPELINE_SCOPE,
        "invocation": sys.argv,
        "benchmark": {
            "samples": sample_count,
            "limit_argument": args.limit,
            "warmup_runs": args.warmup,
            "requested_device": args.device,
            "execution_device": pipeline.device.type,
            "expected_power_watts": args.expected_power_watts,
        },
        "power_mode": power_mode,
        "hardware": {
            "description": description,
            "hostname": platform.node(),
            "device_model": read_text_file(
                Path("/sys/firmware/devicetree/base/model")
            ),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": memory_bytes,
            "cuda_device": cuda_device,
            "cuda_compute_capability": cuda_capability,
            "lscpu": run_command(["lscpu"]),
        },
        "software": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "uv": run_command([uv_binary, "--version"]),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_geometric": torch_geometric.__version__,
            "open3d": o3d.__version__,
            "open3d_build_config": o3d._build_config,
            "opencv": cv2.__version__,
            "jetpack_packages": (
                run_command(
                    ["dpkg-query", "-W", "nvidia-jetpack", "nvidia-l4t-core"]
                )
                if power_mode is not None
                else None
            ),
        },
        "provenance": {
            "git_revision": run_command(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]
            ),
            "git_status": run_command(
                ["git", "-C", str(PROJECT_ROOT), "status", "--short"]
            ),
            "benchmark_script_sha256": sha256_file(Path(__file__)),
            "pipeline_script_sha256": sha256_file(Path(__file__).with_name("pipeline.py")),
            "checkpoint": str(pipeline.checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": pipeline.checkpoint_sha256,
            "splits_sha256": sha256_file(
                pipeline.dataset_root / "splits.csv"
            ),
            "ground_truth_sha256": sha256_file(
                pipeline.dataset_root / "ground_truth.csv"
            ),
        },
    }


def write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    """Write a human-readable benchmark summary.

    Parameters
    ----------
    path
        Destination Markdown path.
    summary
        Completed benchmark summary.
    """
    power = summary["power_mode_watts"]
    lines = [
        "# Inference Benchmark",
        "",
        f"- Hardware: {summary['hardware']}",
        f"- Execution device: {summary['execution_device']}",
        f"- Jetson power mode: {f'{power} W' if power else 'not applicable'}",
        f"- Samples: {summary['sample_count']}",
        f"- Throughput: {summary['throughput_fps']:.3f} samples/s",
        f"- Scope: {summary['pipeline_scope']}",
        "",
        "| Stage | Mean (ms) | Median (ms) | P95 (ms) | Std. (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage in STAGE_FIELDS:
        metric = summary["stages"][stage]
        lines.append(
            f"| {stage} | {metric['mean_ms']:.3f} | "
            f"{metric['median_ms']:.3f} | {metric['p95_ms']:.3f} | "
            f"{metric['std_ms']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add deployment benchmark arguments to a parser."""
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--expected-power-watts",
        type=non_negative_int,
        choices=(10, 15, 25),
        help="On Jetson, abort unless nvpmodel reports this configured mode.",
    )
    parser.add_argument(
        "--limit",
        type=non_negative_int,
        default=0,
        help="Maximum test samples; zero processes the complete test set.",
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=20,
        help="Unmeasured model warm-up runs.",
    )
    parser.add_argument(
        "--progress-every",
        type=non_negative_int,
        default=100,
        help="Log progress after this many samples; zero disables progress logs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "deployment",
        help="Parent directory for timestamped result directories.",
    )


def parse_args() -> argparse.Namespace:
    """Parse standalone deployment benchmark arguments.

    Returns
    -------
    argparse.Namespace
        Validated command-line values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """Run the cross-platform end-to-end deployment benchmark."""
    device = resolve_device(args.device)
    power_mode = query_power_mode(args.expected_power_watts)
    description = device_description(device, power_mode)
    run_dir = create_run_directory(args.output_root.resolve(), description)
    logger = configure_logging(run_dir)
    logger.info("Result directory: %s", run_dir)
    logger.info("Hardware: %s", description)
    logger.info("Execution device: %s", device.type)
    if power_mode:
        logger.info(
            "Jetson power mode: %s (ID %d)",
            power_mode["name"],
            power_mode["id"],
        )
    logger.info("Pipeline scope: %s", PIPELINE_SCOPE)

    setup_started = time.perf_counter_ns()
    pipeline = InferencePipeline(device)
    setup_ms = (time.perf_counter_ns() - setup_started) / 1_000_000
    samples = pipeline.test_samples(args.limit)

    rgba, depth = pipeline.load_rgbd(samples[0])
    unfiltered, tuber_id = pipeline.create_point_cloud(rgba, depth, samples[0])
    filtered = pipeline.filter_point_cloud(unfiltered)
    warmup_data, _, _ = pipeline.prepare_model_input(filtered, tuber_id)
    warmup_started = time.perf_counter_ns()
    pipeline.warmup(warmup_data, args.warmup)
    warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
    logger.info(
        "Prepared %d samples; setup %.3f ms; %d warm-ups %.3f ms",
        len(samples),
        setup_ms,
        args.warmup,
        warmup_ms,
    )

    system_metadata = collect_system_metadata(
        pipeline, description, power_mode, len(samples), args
    )
    (run_dir / "system.json").write_text(
        json.dumps(system_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stage_values: Dict[str, List[float]] = {
        stage: [] for stage in STAGE_FIELDS
    }
    measured_started = time.perf_counter_ns()
    with (run_dir / "samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for index, rgb_file in enumerate(samples, start=1):
            result = benchmark_sample(index, rgb_file, pipeline)
            writer.writerow(result)
            stream.flush()
            for stage in STAGE_FIELDS:
                stage_values[stage].append(float(result[stage]))
            if args.progress_every and (
                index % args.progress_every == 0 or index == len(samples)
            ):
                logger.info("Completed %d/%d samples", index, len(samples))
    measured_wall_ms = (time.perf_counter_ns() - measured_started) / 1_000_000

    stage_summary = {
        stage: summarize_metric(values)
        for stage, values in stage_values.items()
    }
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": description,
        "execution_device": device.type,
        "power_mode_watts": (
            power_mode["configured_watts"] if power_mode else None
        ),
        "sample_count": len(samples),
        "warmup_runs": args.warmup,
        "setup_ms_excluded": setup_ms,
        "warmup_ms_excluded": warmup_ms,
        "measured_wall_ms": measured_wall_ms,
        "pipeline_scope": PIPELINE_SCOPE,
        "throughput_fps": 1000 / stage_summary["end_to_end_ms"]["mean_ms"],
        "stages": stage_summary,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(run_dir / "summary.md", summary)
    logger.info("Benchmark complete: %d samples", len(samples))
    for stage in STAGE_FIELDS:
        metric = stage_summary[stage]
        logger.info(
            "%s mean=%.3f ms median=%.3f ms p95=%.3f ms std=%.3f ms",
            stage,
            metric["mean_ms"],
            metric["median_ms"],
            metric["p95_ms"],
            metric["std_ms"],
        )
    logger.info("Throughput: %.3f samples/s", summary["throughput_fps"])


def main() -> None:
    """Parse arguments and run the standalone deployment benchmark."""
    run(parse_args())


if __name__ == "__main__":
    main()
