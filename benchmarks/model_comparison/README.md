# Model Comparison Suite

These implementations are isolated from the PointRAFT model under root
`models/`. They share PointRAFT's dataset split, preprocessing conventions,
training budget, flat checkpoint contract, and result schema.

## Adapted Implementations

| Model | Upstream | Local adaptation | License |
| --- | --- | --- | --- |
| DGCNN | [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py) | [`models/dgcnn.py`](models/dgcnn.py), adapted to scalar regression | MIT, [included text](licenses/DGCNN-MIT.txt) |
| PointMLP | [ma-xu/pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ModelNet40/models/pointmlp.py) | [`models/pointmlp.py`](models/pointmlp.py), adapted to scalar regression | Apache-2.0, [included text](licenses/POINTMLP-APACHE-2.0.txt) |
| RandLA-Net | [RandLA-Net paper and official SemanticKITTI configuration](https://github.com/QingyongHu/RandLA-Net) | [`models/randlanet.py`](models/randlanet.py), independently written PyTorch/PyG four-stage encoder adapted to scalar regression | Local implementation follows the project license; official reference code is CC BY-NC-SA 4.0 and was not copied |
| PTv3-Compact (`ptv3`) | [PointTransformerV3 detached source](https://github.com/Pointcept/PointTransformerV3) | [`models/ptv3/adapter.py`](models/ptv3/adapter.py), task-optimized three-stage Z-order model with patch size 64 | MIT; pinned upstream files are hash-verified at runtime |
| PTv3-S (`ptv3_official_3stage`) | [PointTransformerV3 detached source](https://github.com/Pointcept/PointTransformerV3) | [`models/ptv3/adapter.py`](models/ptv3/adapter.py), official-style object-scale three-stage model with four serialization orders and FlashAttention | MIT; pinned upstream files are hash-verified at runtime |

The remaining comparison implementations are repository-specific regression
baselines. PTv3 downloads a pinned subset of the upstream Pointcept source at
runtime and verifies each downloaded file; its details are documented in
[`models/ptv3/README.md`](models/ptv3/README.md).

## PTv3 CLI Mapping

The unified benchmark CLI uses `--models` (plural). The two PTv3 names are not
aliases:

```text
--models ptv3
  -> run name: ptv3
  -> model variant: compact
  -> checkpoint: weights/ptv3.pth

--models ptv3_official_3stage
  -> run name: ptv3_official_3stage
  -> model variant: official_3stage
  -> checkpoint: weights/ptv3_official_3stage.pth
```

Therefore `--models ptv3` always selects PTv3-Compact, never the official-style
variant. Historical four/five-stage and multiscale experiments are no longer
CLI models; their evidence is retained in
[`models/ptv3/EXPERIMENTS.md`](models/ptv3/EXPERIMENTS.md).

The canonical `randlanet` model retains four encoder levels with SemanticKITTI widths
`d_out=(16, 64, 128, 256)`, 16-neighbor grouping, 4x seeded random
subsampling, two attentive pools per local feature aggregation block, and
neighborhood max pooling. The segmentation decoder is omitted because the
benchmark target is object-level; deepest features are globally max-pooled and
passed through a `512 -> 64 -> 32 -> 1` scalar head. It is a PyTorch
implementation of the official configuration, not the official implementation.
It replaces the earlier two-stage PyG classification adaptation, which remains
documented only in historical benchmark provenance.

The accepted four-stage artifacts were originally produced under the
`randlanet_official` run name and promoted byte-for-byte to the canonical
`randlanet` paths. Their provenance intentionally retains the original command
and artifact name; the superseded two-stage artifacts are archived locally with
the `randlanet_legacy` prefix.
