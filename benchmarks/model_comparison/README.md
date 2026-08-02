# Model Comparison Suite

These implementations are isolated from the PointRAFT model under root
`models/`. They share PointRAFT's dataset split, preprocessing conventions,
training budget, flat checkpoint contract, and result schema.

## Adapted Implementations

| Model | Upstream | Local adaptation | License |
| --- | --- | --- | --- |
| DGCNN | [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py) | [`models/dgcnn.py`](models/dgcnn.py), adapted to scalar regression | MIT, [included text](licenses/DGCNN-MIT.txt) |
| PointMLP | [ma-xu/pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ModelNet40/models/pointmlp.py) | [`models/pointmlp.py`](models/pointmlp.py), adapted to scalar regression | Apache-2.0, [included text](licenses/POINTMLP-APACHE-2.0.txt) |

The remaining comparison implementations are repository-specific regression
baselines. PTv3 downloads a pinned subset of the upstream Pointcept source at
runtime and verifies each downloaded file; its details are documented in
[`models/ptv3/README.md`](models/ptv3/README.md).
