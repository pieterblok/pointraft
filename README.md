# PointRAFT: Regression of 3D shape properties from partial point clouds

## Summary
PointRAFT is a point cloud regression network that directly predicts continuous 3D shape properties from partial point clouds. Instead of reconstructing complete 3D geometry, PointRAFT infers target values directly from raw 3D data.

PointRAFT is designed for high-throughput inference: on a laptop equipped with an NVIDIA GeForce RTX 4090 Laptop GPU, the average processing time was 6.3 ms per point cloud, enabling throughput of up to 150 point clouds per second. This efficiency and accuracy are achieved through a [PyTorch-based point cloud downsampling strategy](https://github.com/leonardodalinky/pytorch_fpsample) and an object-height embedding, as illustrated in the architecture below:<br/>
![PointRAFT](./assets/PointRAFT_network.png?raw=true)
<br/>

## Installation

### Recommended: uv

For a reproducible local environment, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --all-groups
```

This creates a local `.venv` with Python 3.12.3 and the CUDA 12.8 dependency versions specified in `uv.lock`. Commands can be run directly with `uv run`, without activating the environment.

### Alternative: pip

See [INSTALL.md](INSTALL.md) for the manual virtual environment and pip installation procedure.
<br/><br/>

## Dataset

A subset of 3DPotatoTwin is publicly available on
[Hugging Face](https://huggingface.co/datasets/UTokyo-FieldPhenomics-Lab/3DPotatoTwin).
Download and extract it into `data/3DPotatoTwin/`:

```text
./data/
|-- 3DPotatoTwin/
|   |-- 1_rgbd/
|   |   |-- 2_pcd/
|   |   |   |-- 2R1-1/
|   |   |   |   |-- 2R1-1_pcd_100.ply
|   |   |   |   |-- ...
|   |   |   |-- 2R1-2/
|   |   |   |   |-- 2R1-2_pcd_087.ply
|   |   |   |   |-- ...
|   |   |   |-- ...
|   |-- ground_truth.csv
|   |-- splits.csv

./weights/
|-- pointraft.pth
|-- pointnet.pth
|-- pointnet2.pth

./results/
|-- pointraft/
|   |-- predictions.csv
|-- model_comparison/
|   |-- predictions/
|-- deployment/
```

The released test split contains 2,053 point clouds from 73 tubers. The paper
uses a larger, currently non-public dataset containing 26,688 point clouds from
859 tubers, with 5,254 test point clouds from 172 tubers. Results from the two
test sets are not directly comparable because they use different data
distributions.

## Training and testing

Run all commands from the repository root. The recommended workflow starts by
training the model instead of downloading a checkpoint.

### 1. Train PointRAFT

```bash
uv run python train.py
```

This trains PointRAFT on the public dataset and saves the best validation
checkpoint as `weights/pointraft.pth`.

Training refuses to replace an existing checkpoint. To intentionally retrain
and replace it, use:

```bash
uv run python train.py --overwrite
```

### 2. Evaluate PointRAFT

```bash
uv run python test.py
```

This loads `weights/pointraft.pth`, evaluates the public test split, prints MAE,
RMSE, R-squared, and subgroup metrics, and writes per-point-cloud predictions to
`results/pointraft/predictions.csv`.

### 3. Train and evaluate the verified baselines

The benchmark runner defaults to the verified PointNet and PointNet++
baselines. Reviewer models such as PointMLP are available only when selected
explicitly with `--models`. PointRAFT remains under the dedicated `train.py`
and `test.py` workflow above.

```bash
uv run python -m benchmarks model-comparison train
uv run python -m benchmarks model-comparison test
```

The training command creates `weights/pointnet.pth` and
`weights/pointnet2.pth`. The test command writes predictions under
`results/model_comparison/predictions/`.

To run only one baseline or intentionally replace existing baseline weights:

```bash
uv run python -m benchmarks model-comparison train --models pointnet
uv run python -m benchmarks model-comparison test --models pointnet
uv run python -m benchmarks model-comparison train --overwrite
```

The same runner accepts bounded experiment controls such as `--epochs`,
`--learning-rate`, `--batch-size`, `--max-train-batches`,
`--max-val-batches`, and `--max-test-files`. It streams output while retaining
runner logs and explicit failure evidence under
`results/model_comparison/logs/` and `results/model_comparison/provenance/`.

See [benchmarks/README.md](benchmarks/README.md) for model comparison,
deployment, dry runs, PTv3's isolated environment, and artifact checks. Model
adaptations and third-party licenses are documented in
[benchmarks/model_comparison/README.md](benchmarks/model_comparison/README.md).

When using the manually created environment from [INSTALL.md](INSTALL.md),
activate it and replace `uv run python` with `python` in the commands above.

## Optional pretrained weight

Training locally is the primary documented workflow. To skip training, download
the author-provided
[pointraft.pth](https://drive.google.com/file/d/1EfBDY5WP037dblrIWtWqiO1EHMVs5EwP/view?usp=sharing)
and place it at `weights/pointraft.pth`, then run:

```bash
uv run python test.py
```

This checkpoint was trained on the released public dataset. It is not the model
used to report the manuscript results on the larger, non-public dataset, so its
evaluation should not be expected to reproduce the paper metrics.

## Results

Generated metrics depend on the installed dataset and checkpoint. Public-subset
and private paper-dataset results must be stored and reported separately; see
[`results/README.md`](results/README.md) for the local artifact schema.

![2024-107](./assets/2024-107.png?raw=true)

![R2-3](./assets/R2-3.png?raw=true)<br/><br/>

![2025-051](./assets/2025-051.png?raw=true)<br/><br/>

![2025-115](./assets/2025-115.png?raw=true)
<br/><br/>

## Citation
Refer to our research article:
```
@misc{blok2025pointraft3ddeeplearning,
      title={PointRAFT: 3D deep learning for high-throughput prediction of potato tuber weight from partial point clouds}, 
      author={Pieter M. Blok and Haozhou Wang and Hyun Kwon Suh and Peicheng Wang and James Burridge and Wei Guo},
      year={2025},
      eprint={2512.24193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.24193}, 
}
```
