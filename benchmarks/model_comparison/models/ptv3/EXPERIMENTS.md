# PTv3 Object-Scale Adaptation Experiments

This report records the PTv3 adaptation and ablation work for the PointRAFT
model comparison. It preserves the evidence for historical variants that are no
longer executable from the benchmark CLI. The maintained implementations are:

- `ptv3`: the official-style object-scale three-stage transfer baseline;
- `ptv3_compact`: the task-optimized compact three-stage model.

The experiments use 1,024-point XYZ tuber clouds, a 2 mm grid, the private
paper split, Smooth L1 loss, Adam, batch size 32, and seed 42 unless explicitly
noted otherwise.

## Why the Official Defaults Require Adaptation

The detached PTv3 source is a general point encoder whose published Pointcept
configuration targets large indoor/outdoor scenes and point-wise prediction.
The benchmark instead predicts one scalar from a small isolated object. The
following changes are required for a fair PointRAFT comparison rather than
optional accuracy tuning.

| Official setting | PointRAFT comparison setting | Reason |
| --- | --- | --- |
| Six input feature channels | Centered XYZ only | Every comparison model must receive the same input information |
| Grid supplied by a scene pipeline, commonly centimeter scale | 2 mm grid | Preserve geometry on centimeter-scale tubers |
| Point-wise output and four-stage decoder | Encoder-only object representation | Scalar regression does not require per-point resolution restoration |
| No detached scalar head | Per-cloud pooling plus scalar MLP | Convert point features into one weight prediction |
| Pointcept task schedule | Shared 50-epoch benchmark schedule | Compare backbones rather than dataset-specific recipes |
| Downstream-dependent initialization | Training from scratch | Avoid a pretraining advantage over PointRAFT and other baselines |

These task adaptations do not alter the defining serialized-attention blocks.
Stage depth is treated separately because the default scene hierarchy is not
scale-compatible with a 1,024-point object.

## Why Five Stages Become Three

PTv3 pooling merges occupied `2 x 2 x 2` voxel parents at each stride-2 level;
it does not retain a fixed point ratio. On the complete 5,254-cloud test split,
the measured token hierarchy was:

| Encoder stage | Mean tokens | Minimum | Maximum | Effective grid |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,024.00 | 1,024 | 1,024 | 2 mm |
| 2 | 425.49 | 136 | 903 | 4 mm |
| 3 | 135.34 | 35 | 350 | 8 mm |
| 4 | 40.75 | 13 | 113 | 16 mm |
| 5 | 11.25 | 3 | 33 | 32 mm |

The official hierarchy is designed for scenes containing tens of thousands of
voxels. On a tuber only a few centimeters across, stages four and five reduce
the object to roughly 41 and 11 spatial tokens. The three-stage adaptation
retains about 135 tokens while preserving the first three official depths,
channels, heads, pooling strides, and serialized-attention blocks.

The depth diagnostics support this scale argument:

| Variant | Budget | Best epoch | MAE (g) | RMSE (g) | R-squared | Prediction SD (g) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Official 5-stage, Flash off | 50 | 21 | 110.164 | 131.509 | -0.4649 | 65.961 | Full run; failed |
| Official 5-stage, Flash on | 10 | 2 | 124.372 | 162.145 | -1.2269 | 10.824 | Low-biased collapse |
| Official 4-stage, Flash on | 10 | 2 | 107.835 | 146.691 | -0.8226 | 0.337 | Constant collapse |
| Official 3-stage, Flash on, historical | 50 | 40 | 20.593 | 27.915 | 0.9340 | 105.779 | Learned valid mapping |
| PTv3-S rerun, canonical default | 50 | 41 | 24.842 | 33.201 | 0.9066 | 95.690 | Valid but weak large-scene transfer |

The unequal budgets mean the table is an engineering ablation, not a formal
same-budget leaderboard. The controlled 10-epoch validation runs nevertheless
showed the same transition: the five- and four-stage variants remained narrow
or unstable, whereas the three-stage variant recovered a target-scale
prediction spread. Architecture selection must be justified from validation
behavior, not from test-set ranking.

## Why Compact Improves on Official-Style Three-Stage

Both maintained variants have the same stage count and backbone widths:

```text
depths   = (2, 2, 2)
channels = (32, 64, 128)
heads    = (2, 4, 8)
stride   = (2, 2)
```

The compact model removes mechanisms inherited from large-scene PTv3 and uses
a smaller scalar head:

| Setting | Official-style three-stage | Compact three-stage |
| --- | --- | --- |
| Serialization | Four orders, shuffled | Fixed Z-order |
| Attention | FlashAttention FP16 | Standard attention with FP32 upcast |
| Patch size | 1,024 | 64 |
| Drop path | 0.3 | 0 |
| Scalar head | `128 -> 256 -> 128 -> 1`, two BN layers, dropout 0.5 | `128 -> 64 -> 1`, no BN, dropout 0.2 |
| Parameters | 1,817,633 | 1,759,137 |

With 1,024/425/135 tokens, patch size 1,024 makes every official-style stage
effectively whole-object attention. Patch size 64 retains approximately
16/7/3 local serialized windows and therefore a stronger local-shape inductive
bias. The compact head also avoids BatchNorm statistics learned under the
class-balanced sampler and removes stacked drop-path/dropout/order-shuffle
regularization.

The complete 50-epoch results were:

| Model | Best epoch | MAE (g) | RMSE (g) | R-squared | Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PTv3-S, default transfer baseline | 41 | 24.842 | 33.201 | 0.9066 | 13.131 |
| PTv3-Compact, potato-scale optimization | 43 | **14.944** | **20.746** | **0.9635** | **11.960** |

The compact model reduced MAE by 9.898 g (39.8% relative to PTv3-S) and RMSE by
12.454 g (37.5%). This is a combined scenario-adaptation effect; the current
evidence does not attribute the gain to one setting in isolation.
Head-only, patch-only, drop-path, order-shuffle, and precision ablations would
be required for a causal decomposition.

## Multiscale Readout Check

A five-stage variant pooled stages 3, 4, and 5 independently and concatenated
their 128/256/512-channel features before the scalar head. It retained the
complete official encoder but produced MAE 119.707 g, RMSE 158.650 g, and
R-squared -1.1319 after 10 epochs. Direct access to stage-three features was
therefore insufficient while the deeper hierarchy remained jointly optimized.

## Reproducibility and Reporting Boundaries

- The accepted compact result was produced from clean commit
  `37f5839966cff2852a54b79f4a56ccc9ec47c8cc`.
- The canonical PTv3-S result completed 50 epochs from benchmark source commit
  `dc3199004a4aea2b485c7aaba9293dfd741f63e6`. Its final provenance reports
  `code_dirty=true` because unrelated `tools/` files changed during training;
  the benchmark implementation and tests remained identical to that commit.
- An earlier official-style three-stage run achieved 20.593 g MAE but also
  reports `code_dirty=true` at commit
  `133379ec7681167582c37c083313d8f593202c5b`.
- The split SHA-256 is
  `bf2471121a816829e33fa207225cb5c419996b0b6dd19da3bb835c3d4602d9c1`.
- The ground-truth SHA-256 is
  `6cc17dc5b4240c1650cd39daf5849065cb7eaf80f8bf2b3d95248f3925278f10`.
- Historical results are evidence for architecture decisions, not accepted
  manuscript rows unless reproduced from committed source under one export.

Primary artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| Canonical PTv3-S checkpoint | `adae094b127414169c3894a43b10e97d4be46421516476b24e944bcc22d089a4` |
| Canonical PTv3-S predictions | `fb879fb1145bd2928884c369bdedb101a3d5f318c9ca155353b50e56f7d86a61` |
| PTv3-Compact checkpoint | `a4c34c75cbed4dec7c05fe84f49deee9018e1182691f7aa71a39300d249ee8ce` |
| PTv3-Compact predictions | `8206b6eef92b771a8f78f25143a6ef9d6c1647365a1a0191bbc4b9fbda799c24` |
| Historical official-style 3-stage checkpoint | `d56a0fad94067f111c1eec5a5b3cdbc7b715f847df2ad79e4ff682828b604148` |
| Historical official-style 3-stage predictions | `44cba8e168670cdc05557cf49f4029973b181eb1cad8cea124a3841357693c6e` |
| Official 5-stage Flash-off checkpoint | `187e8b68ebd7105be9d4072a96de615e68e68edcb8dfa0949b56883fc0fef886` |
| Official 5-stage Flash-on checkpoint | `7dc367aad6131deb2cd10fb0e7cb3dd9249bb078d27d7582736007e8cf4c8d7a` |
| Official 4-stage checkpoint | `d87ec43772194e84a0e04a90cb87bf2de39b52b8b8134155b27507d6d6dbc3de` |
| Official 5-stage multiscale checkpoint | `0395c7cc4faf2a50c51e88a2ad9fbb48efed12ed64795a50c27c2521e63482cd` |

The historical weights, predictions, logs, and provenance remain under
`weights/` and `results/model_comparison/`; removing their executable registry
entries does not delete those records.

## Recommended Manuscript Presentation

- Report `PTv3-S` (`ptv3`) as the default transfer baseline. Its weak result
  demonstrates that large-scene PTv3 settings do not transfer directly to
  small isolated tubers.
- Report `PTv3-Compact` (`ptv3_compact`) as a subsequent scenario optimization,
  preferably in the discussion rather than as the default PTv3 row.
- Describe XYZ input, 2 mm grid, encoder-only operation, scalar pooling/head,
  and from-scratch shared training as task-required adaptations.
- Describe three stages as a validation-motivated scale adaptation.
- Keep the failed four/five-stage and multiscale variants in supplementary
  material rather than the primary benchmark table.
