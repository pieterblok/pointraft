# PTv3 Regression Benchmarks

This isolated uv project maintains two PTv3 scalar-regression models for the
PointRAFT comparison:

| CLI name | Model identity | Purpose |
| --- | --- | --- |
| `ptv3` | `ptv3-official-3stage-flash-regression` | Default architecture-faithful transfer baseline |
| `ptv3_compact` | `ptv3-compact-3stage` | Potato-scale optimized follow-up model |

> **CLI identity:** the unified command uses `--models` (plural).
> `--models ptv3` always runs PTv3-S. To run the task-optimized model, use
> `--models ptv3_compact`.

The runner resolves the public names explicitly:

```text
ptv3                     -> --model-variant official_3stage
ptv3_compact             -> --model-variant compact
```

Both models use the pinned detached PointTransformerV3 source at commit
`3229e9b7de1770c8ad17c316f8e349982de509f8`, centered XYZ input, 1,024 points,
a 2 mm grid, three encoder stages, per-cloud mean pooling, and scalar output.

The compact model uses fixed Z-order, standard attention with patch size 64,
no drop path, and a small `128 -> 64 -> 1` head. The official-style model
retains the four official serialization orders, shuffled orders,
FlashAttention with patch size 1,024, drop path 0.3, and a
`128 -> 256 -> 128 -> 1` BN/ReLU/dropout head.

Detailed motivation, stage-depth experiments, failed historical variants,
metrics, limitations, and artifact hashes are recorded in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Environment

The locked environment follows the PTv3-era CUDA stack:

- Python 3.11
- PyTorch 2.1.0 with CUDA 11.8
- spconv-cu118 2.3.6
- torch-scatter 2.1.2 for PyTorch 2.1/CUDA 11.8
- FlashAttention 2.5.5

The `uv.lock` SHA-256 is
`96083c3accdfbe0fbac35ce71c6231d009b30b8cdd66c3f3e0a5f56a22bdd51e`.

## Smoke Checks

Run the default official-style three-stage CUDA smoke from the repository root:

```bash
uv run --project benchmarks/model_comparison/models/ptv3 --locked \
  python benchmarks/model_comparison/models/ptv3/smoke.py --points 1024
```

The smoke default matches the unified CLI mapping: omitting `--model-variant`
in `smoke.py` checks `official_3stage`, and `--models ptv3` also means
`official_3stage`.

Run the compact smoke explicitly:

```bash
uv run --project benchmarks/model_comparison/models/ptv3 --locked \
  python benchmarks/model_comparison/models/ptv3/smoke.py \
  --model-variant compact --points 1024
```

## Default PTv3-S (`--models ptv3`)

Bounded engineering check:

```bash
uv run python -m benchmarks model-comparison train --models ptv3 \
  --epochs 1 --batch-size 2 --max-train-batches 1 --max-val-batches 1 \
  --max-export-files-per-split 8 --overwrite
uv run python -m benchmarks model-comparison test --models ptv3 \
  --max-test-files 8 --max-export-files-per-split 8
```

Complete run:

```bash
uv run python -m benchmarks model-comparison train --models ptv3 \
  --epochs 50 --batch-size 32 --overwrite
uv run python -m benchmarks model-comparison test --models ptv3
```

The 50-epoch rerun selected epoch 41 and achieved 24.842 g MAE, 33.201 g RMSE,
and R-squared 0.9066 on 5,254 test clouds. This intentionally remains the
default PTv3 result: it measures how the large-scene architecture transfers to
small isolated tubers before scenario-specific simplification.

The run started with benchmark source commit `dc31990` clean. Its final
repository-wide provenance reports `code_dirty=true` because unrelated
`tools/` files changed during training; `benchmarks/model_comparison/` and the
related tests remained identical to the recorded commit.

## PTv3-Compact (`--models ptv3_compact`)

Bounded engineering check:

```bash
uv run python -m benchmarks model-comparison train \
  --models ptv3_compact --epochs 1 --batch-size 2 \
  --max-train-batches 1 --max-val-batches 1 \
  --max-export-files-per-split 8 --overwrite
uv run python -m benchmarks model-comparison test \
  --models ptv3_compact --max-test-files 8 \
  --max-export-files-per-split 8
```

Complete run:

```bash
uv run python -m benchmarks model-comparison train \
  --models ptv3_compact --epochs 50 --batch-size 32 --overwrite
uv run python -m benchmarks model-comparison test \
  --models ptv3_compact
```

The retained clean compact result selected epoch 43 and achieved 14.944 g MAE,
20.746 g RMSE, and R-squared 0.9635. It is reported as a potato-scale
optimization rather than the default PTv3 transfer baseline.

## Artifact Contract

Each CLI name writes to independent flat paths:

```text
weights/<model>.pth
results/model_comparison/predictions/<model>.csv
results/model_comparison/logs/<model>.*
results/model_comparison/provenance/<model>.*.json
```

Consequently, `weights/ptv3.pth` is the PTv3-S checkpoint and
`weights/ptv3_compact.pth` is the task-optimized checkpoint.

Both canonical paths were promoted from retained runs. Their provenance JSON
keeps the original run name and command. The promotion record is
`results/model_comparison/provenance/ptv3.artifact_map.json`.

Training refuses to replace an existing checkpoint unless `--overwrite` is
passed. Bounded exports are verified before use and must not be reported as
complete model-quality results.
