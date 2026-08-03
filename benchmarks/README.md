# PointRAFT Benchmarks

The benchmark package keeps reviewer model comparison separate from end-to-end
deployment measurement. Root `train.py` and `test.py` remain the direct
PointRAFT entry points.

## Layout

```text
benchmarks/
├── cli.py
├── model_comparison/
│   ├── runner.py
│   ├── train.py
│   ├── test.py
│   └── models/
│       └── ptv3/
└── deployment/
    ├── pipeline.py
    ├── pointcloud.py
    ├── smoke_test.py
    └── benchmark.py
```

The installed `data/3DPotatoTwin/` determines whether commands use the released
subset or a private paper dataset. Code and relative artifact paths are the same
for both, but datasets, checkpoints, and results remain repository-local.

## Model Comparison

PointNet and PointNet++ are selected by default:

```bash
uv run python -m benchmarks model-comparison train
uv run python -m benchmarks model-comparison test
```

Select one or more models explicitly when needed:

```bash
uv run python -m benchmarks model-comparison train \
  --models pointnet pointnet2 dgcnn pointmlp ptv3 --overwrite
uv run python -m benchmarks model-comparison test \
  --models pointnet pointnet2 dgcnn pointmlp ptv3
```

Available names are `pointnet`, `pointnet2`, `dgcnn`, `pointmlp`,
`pointtransformer`, `randlanet`, `ptv3`, and `ptv3_official_3stage`. Each model has one checkpoint at
`weights/<model>.pth`; training refuses to replace it unless `--overwrite` is
given.

Results use one common layout:

```text
results/model_comparison/
├── predictions/<model>.csv
├── logs/<model>.*
└── provenance/<model>.*.json
```

PTv3 data export runs in the main environment and the model runs automatically
in the locked project under `models/ptv3/`. See
[`model_comparison/models/ptv3/README.md`](model_comparison/models/ptv3/README.md)
for direct debugging commands.

## Dry Runs

Dry runs print commands and artifact destinations without exporting data,
launching model subprocesses, or replacing files:

```bash
uv run python -m benchmarks model-comparison train \
  --models pointnet pointmlp ptv3 --dry-run
uv run python -m benchmarks model-comparison test \
  --models pointnet pointmlp ptv3 --dry-run
uv run python -m benchmarks deployment run --dry-run
```

Bounded controls such as `--epochs`, `--max-train-batches`,
`--max-val-batches`, `--max-test-files`, and
`--max-export-files-per-split` are for integration checks only. They still target
canonical artifacts, so do not combine bounded training with `--overwrite` when
an accepted checkpoint exists.

## Deployment

The deployment workflow measures RGB-D point-cloud creation, DBSCAN filtering,
PointRAFT preprocessing, regression, total latency, and throughput. Instance
segmentation and one-time model loading are excluded from per-sample timing.

```bash
uv run python -m benchmarks deployment smoke --device cuda
uv run python -m benchmarks deployment run --device cuda --limit 10 --warmup 3
uv run python -m benchmarks deployment run --device cuda
uv run python -m benchmarks deployment run --device cpu
```

Every run reads `weights/pointraft.pth`, computes its actual SHA-256, and records
that digest in `system.json`. No repository-global checkpoint hash is enforced.
Timestamped artifacts are written under `results/deployment/`.

The Jetson uses the isolated Python 3.8/JetPack environment in `deployment/`.
See [`deployment/AGENTS.md`](deployment/AGENTS.md) for the no-sudo installation
and power-mode procedure.

## Third-Party Code

Model origins, scalar-regression adaptations, and included license texts are
documented in
[`model_comparison/README.md`](model_comparison/README.md).
