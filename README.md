# PointRAFT: Regression of 3D shape properties from partial point clouds

## Summary
PointRAFT is a point cloud regression network that directly predicts continuous 3D shape properties from partial point clouds. Instead of reconstructing complete 3D geometry, PointRAFT infers target values directly from raw 3D data.

PointRAFT is designed for high-throughput inference: on a laptop equipped with an NVIDIA GeForce RTX 4090 Laptop GPU, point cloud preprocessing and network inference took on average 6.3 ms per point cloud, corresponding to approximately 160 point clouds per second. This efficiency and accuracy are achieved through a [PyTorch-based point cloud downsampling strategy](https://github.com/leonardodalinky/pytorch_fpsample) and an object-height embedding, as illustrated in the architecture below:<br/>
![PointRAFT](./assets/PointRAFT_network.png?raw=true)
<br/><br/>

## Installation

#### Recommended: uv

For a reproducible local environment, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --all-groups
```

This creates a local `.venv` with Python 3.12.3 and the CUDA 12.8 dependency versions specified in `uv.lock`. Commands can be run directly with `uv run`, without activating the environment.

#### Alternative: pip

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
<br/><br/>

## Training and testing

Run all commands from the repository root. The recommended workflow starts by
training the model instead of downloading a checkpoint.

#### 1. Train PointRAFT

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

#### 2. Evaluate PointRAFT

```bash
uv run python test.py
```

This loads `weights/pointraft.pth`, evaluates the public test split, prints MAE,
RMSE, R-squared, and subgroup metrics, and writes per-point-cloud predictions to
`results/pointraft/predictions.csv`.

#### 3. Train and evaluate the verified baselines

The six comparison models from the paper (PointNet, PointNet++, DGCNN, PointMLP,
RandLA-Net, and PTv3-Compact) are trained and evaluated with:

```bash
uv run python -m benchmarks model-comparison train
uv run python -m benchmarks model-comparison test
```

See [benchmarks/model_comparison/README.md](benchmarks/model_comparison/README.md)
for model selection, options, outputs, and the model adaptations, and
[benchmarks/README.md](benchmarks/README.md) for the deployment benchmark and the
ablation study.

When using the manually created environment from [INSTALL.md](INSTALL.md),
activate it and replace `uv run python` with `python` in the commands above.
<br/><br/>

## Optional pretrained weight

Download the author-provided
[pointraft.pth](https://drive.google.com/file/d/1J0V2mqxob999znSpQvMQfe1ghsdjBD_k/view?usp=sharing)
and place it at `weights/pointraft.pth`.
<br/><br/>

## Results

On a test set of 5,254 point clouds representing 172 unique tubers, PointRAFT achieved a mean absolute error (MAE) of 11.7 g and a root mean squared error (RMSE) of 17.1 g, outperforming the six mentioned point cloud regression networks.

![Predicted weights of four example tubers by PointRAFT and the six comparison networks](./assets/tuber_mosaic.png?raw=true)
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
