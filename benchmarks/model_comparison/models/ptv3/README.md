# PTv3 Isolated Regression Benchmark

This standalone uv project checks whether the official detached Point
Transformer V3 source can import and execute a reduced CUDA forward pass on the
current machine. The same environment now contains a compact three-stage PTv3
scalar-regression adapter and bounded paper-data training/evaluation workers.

The smoke uses the official upstream commit
`3229e9b7de1770c8ad17c316f8e349982de509f8`, verifies every downloaded source
file with SHA-256, disables FlashAttention, and terminates after a bounded
timeout.

From the PointRAFT repository root, run:

```bash
uv run --project benchmarks/model_comparison/models/ptv3 --locked \
  python benchmarks/model_comparison/models/ptv3/smoke.py
```

The first invocation creates
`benchmarks/model_comparison/models/ptv3/.venv/` and downloads the
pinned upstream files into the ignored `.upstream/` directory. Neither generated
directory is committed.

This environment intentionally follows the official PTv3-era CUDA stack rather
than PointRAFT's Python 3.12 environment:

- Python 3.11
- PyTorch 2.1.0 with CUDA 11.8
- spconv-cu118 2.3.6
- torch-scatter 2.1.2 for PyTorch 2.1/CUDA 11.8
- FlashAttention disabled

## Verified Smoke

Verified on 2026-08-01 with an NVIDIA GeForce RTX 3090 and driver 610.43.03.
Both bounded CUDA forwards completed successfully:

```text
Environment: python=3.11.12, torch=2.1.0+cu118, cuda=11.8,
spconv=2.3.6, torch_scatter=2.1.2+pt21cu118
PTv3 smoke passed: input=32, output=(32, 32), device=cuda:0
PTv3 smoke passed: input=128, output=(128, 32), device=cuda:0
```

The `uv.lock` SHA-256 is
`22d4858c357f46ea65a800c949b875a5a0118e9d3d55592df618c90a3d0d4136`.
## Bounded Paper Regression

Run a small end-to-end paper-data check from the repository root:

```bash
uv run python -m benchmarks model-comparison train --models ptv3 \
  --epochs 1 --batch-size 2 --max-train-batches 1 --max-val-batches 1 \
  --max-export-files-per-split 8
uv run python -m benchmarks model-comparison test --models ptv3 \
  --max-test-files 8 --max-export-files-per-split 8
```

The main Python 3.12 environment centers and farthest-point-samples the paper
PLY files, then exports non-pickle NumPy arrays under ignored `.data/`. The
Python 3.11/cu118 environment verifies the export hashes before training or
testing. Omitting `--max-export-files-per-split` replaces any bounded cache with
a complete export before a full run, so a prerun cannot be mistaken for the
complete dataset.

Bounded training targets the canonical `weights/ptv3.pth`; use it only when no
accepted checkpoint would be replaced.

The adapter is explicitly identified as `ptv3-compact-3stage` in provenance.
It uses XYZ features, serialized Z order, three encoder stages, global mean
pooling, and a two-layer scalar head. FlashAttention remains disabled.

The verified bounded integration run used 64 files per split, batch size 4,
three epochs, five training batches per epoch, and two validation batches per
epoch. It completed training, checkpoint loading, and 64-file evaluation. Its
metrics are engineering evidence only and are not reportable model-quality
results because the model received only 15 optimizer steps on a bounded export.
