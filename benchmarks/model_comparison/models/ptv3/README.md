# PTv3 Regression Benchmarks

This isolated uv project maintains two PTv3 scalar-regression models for the
PointRAFT comparison:

| CLI name | Model identity | Purpose |
| --- | --- | --- |
| `ptv3` | `ptv3-compact-3stage` | Strong task-optimized baseline |
| `ptv3_official_3stage` | `ptv3-official-3stage-flash-regression` | Official-style object-scale sensitivity baseline |

> **CLI identity:** the unified command uses `--models` (plural).
> `--models ptv3` always runs the compact model. To run the official-style
> model, use `--models ptv3_official_3stage`. There is no automatic selection
> of an "official" model behind the shorter `ptv3` name.

The runner resolves the public names explicitly:

```text
ptv3                     -> --model-variant compact
ptv3_official_3stage     -> --model-variant official_3stage
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

The smoke default is independent of the unified CLI mapping above: omitting
`--model-variant` in `smoke.py` checks `official_3stage`, while
`--models ptv3` in the unified benchmark still means compact.

Run the compact smoke explicitly:

```bash
uv run --project benchmarks/model_comparison/models/ptv3 --locked \
  python benchmarks/model_comparison/models/ptv3/smoke.py \
  --model-variant compact --points 1024
```

## Compact PTv3 (`--models ptv3`)

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

The retained clean result selected epoch 43 and achieved 14.944 g MAE,
20.746 g RMSE, and R-squared 0.9635 on 5,254 test clouds.

## Official-Style PTv3-S (`--models ptv3_official_3stage`)

Bounded engineering check:

```bash
uv run python -m benchmarks model-comparison train \
  --models ptv3_official_3stage --epochs 1 --batch-size 2 \
  --max-train-batches 1 --max-val-batches 1 \
  --max-export-files-per-split 8 --overwrite
uv run python -m benchmarks model-comparison test \
  --models ptv3_official_3stage --max-test-files 8 \
  --max-export-files-per-split 8
```

Complete run:

```bash
uv run python -m benchmarks model-comparison train \
  --models ptv3_official_3stage --epochs 50 --batch-size 32 --overwrite
uv run python -m benchmarks model-comparison test \
  --models ptv3_official_3stage
```

The completed diagnostic selected epoch 40 and achieved 20.593 g MAE,
27.915 g RMSE, and R-squared 0.9340. Its provenance reports
`code_dirty=true`; reproduce it from committed source before treating it as an
accepted manuscript artifact.

## Artifact Contract

Each CLI name writes to independent flat paths:

```text
weights/<model>.pth
results/model_comparison/predictions/<model>.csv
results/model_comparison/logs/<model>.*
results/model_comparison/provenance/<model>.*.json
```

Consequently, `weights/ptv3.pth` is the compact checkpoint and
`weights/ptv3_official_3stage.pth` is the official-style checkpoint.

Training refuses to replace an existing checkpoint unless `--overwrite` is
passed. Bounded exports are verified before use and must not be reported as
complete model-quality results.
