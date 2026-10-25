# PointRAFT Ablation Study

Each ablation retrains PointRAFT with the protocol from the root
[`train.py`](../../train.py) and changes exactly one component of the default
PointRAFT configuration, which uses a `[1, 8, 8]` height-embedding MLP. Each
ablation is an `AblationConfig` in its own module under `ablations/`; the shared
`train.py` and `test.py` in this folder read it. The ablation baseline in
`ablations/config.py` matches the PointRAFT default.

| #  | Name              | Change from the ablation baseline                        |
| -- | ----------------- | -------------------------------------------------------- |
| 1  | `mse`             | `nn.MSELoss()` instead of `nn.SmoothL1Loss(beta=20)`     |
| 2  | `l1`              | `nn.L1Loss()` instead of `nn.SmoothL1Loss(beta=20)`      |
| 3  | `height_emb_2`    | Height-embedding MLP `[1, 2, 2]` instead of `[1, 8, 8]`   |
| 4  | `height_emb_4`    | Height-embedding MLP `[1, 4, 4]` instead of `[1, 8, 8]`   |
| 5  | `height_emb_16`   | Height-embedding MLP `[1, 16, 16]` instead of `[1, 8, 8]` |
| 6  | `height_emb_32`   | Height-embedding MLP `[1, 32, 32]` instead of `[1, 8, 8]` |
| 7  | `height_emb_48`   | Height-embedding MLP `[1, 48, 48]` instead of `[1, 8, 8]` |
| 8  | `height_emb_64`   | Height-embedding MLP `[1, 64, 64]` instead of `[1, 8, 8]` |
| 9  | `no_augmentation` | No jitter, rotation, flip, or shear augmentation *       |
| 10 | `points_512`      | 512 input points instead of 1024 *                       |
| 11 | `points_2048`     | 2048 input points instead of 1024, batch size 16 † *     |

† 2048 points at batch size 32 does not fit in 16 GB of GPU memory.

## Running

From the repository root:

```bash
uv run python -m benchmarks ablation_study train
uv run python -m benchmarks ablation_study test
```

Without `--ablation`, all eleven run in the order above. To run selected
ablations or replace existing ablation weights:

```bash
uv run python -m benchmarks ablation_study train --ablation mse l1
uv run python -m benchmarks ablation_study test --ablation mse l1
uv run python -m benchmarks ablation_study train --ablation mse --overwrite
```

The runner accepts the same bounded controls as the model comparison
(`--epochs`, `--learning-rate` with default `0.0005`, `--batch-size`,
`--max-train-batches`, `--max-val-batches`, `--max-test-files`, `--dry-run`,
and `--continue-on-error`). When using the manually created environment from
[INSTALL.md](../../INSTALL.md), activate it and replace `uv run python` with
`python`.

## Cached data and run order

The order in `ablations/__init__.py` is the run order, and the runner keeps it
for any `--ablation` selection. Ablations marked `*` change the cached training
data and set `rebuild_processed=True`: before training they delete
`data/3DPotatoTwin/1_rgbd/2_pcd/processed/`, let `PointCloudDataset` rebuild it,
and write `processed/ablation_source.txt`. They therefore always run after
ablations 1-8, which use the existing default `processed/` folder. Ablations 1-8
refuse to train while that marker exists, so restore the default `processed/`
folder before retraining them.

## Results

Training saves one checkpoint per ablation, and every test run refreshes a
summary with the metrics of all tested ablations:

```text
weights/ablation_study/<name>.pth
results/ablation_study/
├── summary.csv            # refreshed after every test run
└── <name>/
    ├── train_metrics.csv
    ├── train.provenance.json
    ├── predictions.csv
    ├── test_metrics.txt
    ├── test.provenance.json
    ├── {train,test}.runner.log
    └── {train,test}.failure.json   # only after a failed run
```

## Adding an ablation

Create a module in `ablations/` that defines `ABLATION = AblationConfig(...)`
and add it to `ABLATION_REGISTRY` in `ablations/__init__.py`.
