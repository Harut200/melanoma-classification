# Final model results

## Per fold

| experiment | fold | ROC-AUC | PR-AUC | Sens@95Spec | best epoch | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `final_tf_efficientnet_b4_300` | 0 | 0.8863 | 0.2253 | 0.5726 | 10 | 139 |
| `final_tf_efficientnet_b4_300` | 1 | 0.9012 | 0.1689 | 0.4655 | 8 | 140 |
| `final_tf_efficientnet_b4_300` | 2 | 0.9182 | 0.2901 | 0.6207 | 11 | 139 |
| `final_tf_efficientnet_b4_300` | 3 | 0.9119 | 0.2763 | 0.5299 | 10 | 139 |
| `final_tf_efficientnet_b4_300` | 4 | 0.9255 | 0.3044 | 0.6102 | 11 | 139 |

## Averaged over folds

| experiment | folds | ROC-AUC | PR-AUC | Sens@95Spec |
| --- | ---: | ---: | ---: | ---: |
| `final_tf_efficientnet_b4_300` | 5 | 0.9086 ± 0.0153 | 0.2530 ± 0.0557 | 0.5598 |
| _resnet34 baseline_ | 5 | 0.8873 | 0.2285 | 0.5189 |

## Out of fold

Every competition photo scored once, by the fold model that did not
train on it. This is the number to quote.

| experiment | photos | ROC-AUC | PR-AUC | Sens@95Spec | threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final_tf_efficientnet_b4_300` | 33126 | 0.9076 | 0.2507 | 0.5479 | 0.6245 |

## Versus the baseline

| | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| resnet34 @224 baseline | 0.8873 | 0.2285 |
| `final_tf_efficientnet_b4_300` out of fold | 0.9076 | 0.2507 |
| **change** | **+0.0203** | **+0.0222** |

