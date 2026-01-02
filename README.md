# PointRAFT: Regression of 3D shape properties from partial point clouds

## Summary
PointRAFT is a point cloud regression network that directly predicts continuous 3D shape properties from partial point clouds. Instead of reconstructing complete 3D geometry, PointRAFT infers target values directly from raw 3D data.

PointRAFT is designed for high-throughput inference: on a laptop equipped with an NVIDIA GeForce RTX 4090 Laptop GPU, the average processing time was 6.3 ms per point cloud, enabling throughput of up to 150 point clouds per second. This efficiency and accuracy are achieved through a [PyTorch-based point cloud downsampling strategy](https://github.com/leonardodalinky/pytorch_fpsample) and an object-height embedding, as illustrated in the architecture below:<br/>
![PointRAFT](./misc/PointRAFT_network.png?raw=true)
<br/>

## Installation
[INSTALL.md](INSTALL.md)
<br/><br/>

## Dataset
A subset of our dataset is publicly available at [3DPotatoTwin](https://huggingface.co/datasets/UTokyo-FieldPhenomics-Lab/3DPotatoTwin). Download and unzip the dataset locally to reproduce our training and testing procedures. The partial point clouds (in .ply format) are organized in the following folder structure:

    3DPotatoTwin/
    |-- 1_rgbd/
    |   |-- 2_pcd/
    |   |   |-- 2R1-1/
    |   |   |   |-- 2R1-1_pcd_100.ply
    |   |   |   |-- ...
    |   |   |-- 2R1-2/
    |   |   |   |-- 2R1-2_pcd_087.ply
    |   |   |   |-- ...
    |   |   |-- ...
<br/>

## Pretrained weights
[pointraft_potato.pth](https://drive.google.com/file/d/1EfBDY5WP037dblrIWtWqiO1EHMVs5EwP/view?usp=sharing)
<br/><br/>

## Instructions
1. Download the [3DPotatoTwin dataset](https://huggingface.co/datasets/UTokyo-FieldPhenomics-Lab/3DPotatoTwin).
2. Place the ZIP file in the data/ directory and unzip its contents.
3. Activate your virtual environment and navigate to the local repository:
    ```command
    cd pointraft
    source ~/prft/bin/activate
    ```
4. Update the file paths and experiment-specific parameters in **train.py** to match your local setup.
5. Train PointRAFT
    ```python
    python train.py
    ```
6. Evaulate PointRAFT
    ```python
    python test.py
    ```
<br/>

## Results
On a test set of 5,254 point clouds from 172 unique potato tubers, PointRAFT achieved a mean absolute error (MAE) of 12.0 g and a root mean squared error (RMSE) of 17.2 g, substantially outperforming a linear regression baseline, which achieved an MAE of 23.0 g and an RMSE of 31.8 g.
![2024-107](./misc/2024-107.png?raw=true)

![R2-3](./misc/R2-3.png?raw=true)<br/><br/>

![2025-051](./misc/2025-051.png?raw=true)<br/><br/>

![2025-115](./misc/2025-115.png?raw=true)
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
