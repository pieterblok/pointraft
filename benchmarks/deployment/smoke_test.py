"""Verify one real RGB-D sample with the shared inference pipeline."""

import argparse

import torch

from .pipeline import InferencePipeline, resolve_device


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add deployment smoke-test arguments to a parser."""
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def run(args: argparse.Namespace) -> None:
    """Run one complete RGB-D-to-regression deployment check."""
    device = resolve_device(args.device)
    pipeline = InferencePipeline(device)
    result = pipeline.run_sample(pipeline.test_samples(limit=1)[0])

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    print(f"device={device_name}")
    print(f"rgb_file={result['rgb_file']}")
    print(
        "points="
        f"{result['points_unfiltered']}->{result['points_filtered']}->"
        f"{result['points_model_input']}"
    )
    print(f"height_m={result['height_m']:.6f}")
    print(f"prediction_g={result['prediction_g']:.6f}")


def main() -> None:
    """Parse arguments and run the standalone deployment smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
