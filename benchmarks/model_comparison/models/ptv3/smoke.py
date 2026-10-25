"""Download a pinned PTv3 snapshot and run a bounded minimal forward pass."""

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "3229e9b7de1770c8ad17c316f8e349982de509f8"
UPSTREAM_FILES = {
    "LICENSE": "8e293fb7c93586fb19294bcae2096755b467e465b010b1547f5658a3df1682ab",
    "model.py": "695372fe66630656b158bd2697e9874a5919c255cc8d90a0b9bd60e3fb833461",
    "serialization/__init__.py": "1e3aa8d47116a3d6cc86e816420483c496f4caac03efe8395f90b0aac5e97597",
    "serialization/default.py": "3d20c6cd07bcdf770740f2188a790d23707ab55d67d9b69d5ac763921d3f1084",
    "serialization/hilbert.py": "7166e14531407d93e151749f3b2cade69690516cea0d6a7a5968a3273770cc36",
    "serialization/z_order.py": "e94d7aebff6b1fbf92eb9d0ac8cdb7c62fe4d13f17c7255bbc5be96661190fac",
}


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


def prepare_upstream(project_dir: Path) -> Path:
    """Download and verify the pinned detached PTv3 source files.

    Parameters
    ----------
    project_dir
        PTv3 smoke-project directory.

    Returns
    -------
    Path
        Parent directory containing the generated ``pointtransformerv3``
        package.

    Raises
    ------
    RuntimeError
        If a downloaded file does not match its recorded digest.
    """
    package_dir = project_dir / ".upstream" / "pointtransformerv3"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").touch()

    for relative_path, expected_digest in UPSTREAM_FILES.items():
        destination = package_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256_file(destination) != expected_digest:
            url = (
                "https://raw.githubusercontent.com/Pointcept/"
                f"PointTransformerV3/{UPSTREAM_COMMIT}/{relative_path}"
            )
            temporary_path = destination.with_suffix(destination.suffix + ".tmp")
            print(f"Downloading {relative_path} from {UPSTREAM_COMMIT}", flush=True)
            urllib.request.urlretrieve(url, temporary_path)
            temporary_path.replace(destination)

        actual_digest = sha256_file(destination)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative_path}: "
                f"expected {expected_digest}, received {actual_digest}"
            )

    return package_dir.parent


def run_worker(
    project_dir: Path,
    device: str,
    points: int,
    model_variant: str,
    *,
    import_only: bool = False,
) -> None:
    """Import PTv3 and execute the reduced forward configuration.

    Parameters
    ----------
    project_dir
        PTv3 smoke-project directory.
    device
        Torch execution device.
    points
        Number of synthetic input points.
    model_variant
        Official encoder depth to exercise.
    import_only
        Stop after importing and reporting dependency versions.
    """
    sys.path.insert(0, str(project_dir / ".upstream"))

    import flash_attn
    import spconv
    import torch
    import torch_scatter

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    print(
        "Environment: "
        f"python={sys.version.split()[0]}, torch={torch.__version__}, "
        f"cuda={torch.version.cuda}, spconv={spconv.__version__}, "
        f"flash_attn={flash_attn.__version__}, "
        f"torch_scatter={torch_scatter.__version__}",
        flush=True,
    )
    if import_only:
        print("PTv3 import smoke passed", flush=True)
        return

    from adapter import OfficialPTv3ThreeStageRegressor, PTv3Regressor

    model_class = {
        "compact": PTv3Regressor,
        "official_3stage": OfficialPTv3ThreeStageRegressor,
    }[model_variant]
    model = model_class().to(device).eval()

    coordinates = torch.rand(points, 3, device=device) * 0.08
    data = {
        "feat": coordinates.clone(),
        "coord": coordinates,
        "grid_size": 0.002,
        "batch": torch.zeros(points, dtype=torch.long, device=device),
    }
    with torch.inference_mode():
        output = model(data["coord"], data["batch"], data["grid_size"])
    if device == "cuda":
        torch.cuda.synchronize()

    if output.shape != (1,) or not torch.isfinite(output).all():
        raise RuntimeError(f"Unexpected PTv3 regression output: {output}")
    print(
        f"PTv3 smoke passed: variant={model_variant}, input={points}, "
        f"output={tuple(output.shape)}, device={output.device}",
        flush=True,
    )


def main() -> None:
    """Prepare the pinned source and run the smoke in a timed subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument(
        "--model-variant",
        choices=("compact", "official_3stage"),
        default="official_3stage",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.points < 32:
        parser.error("--points must be at least 32")
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    project_dir = Path(__file__).resolve().parent
    prepare_upstream(project_dir)
    if args.worker:
        run_worker(
            project_dir,
            args.device,
            args.points,
            args.model_variant,
            import_only=args.import_only,
        )
        return

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--device",
        args.device,
        "--points",
        str(args.points),
        "--model-variant",
        args.model_variant,
    ]
    if args.import_only:
        command.append("--import-only")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=project_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        if error.stdout:
            print(error.stdout, end="")
        if error.stderr:
            print(error.stderr, end="", file=sys.stderr)
        raise SystemExit(
            f"PTv3 smoke timed out after {args.timeout} seconds"
        ) from error

    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
