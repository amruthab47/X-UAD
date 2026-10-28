# Revision-3 results

## 1. Does the current code reproduce Table I? (X-UAD, P1, 10 epochs, seed 0)

| Metric | Paper (freeze code) | Current code | Diff |
|---|---|---|---|
| ACER | 1.46 | 3.33 | +1.88 |
| APCER | 2.08 | 5.67 | +3.58 |
| BPCER | 0.83 | 1.00 | +0.17 |
| AUC | 99.89 | 99.86 | -0.03 |
| EER | 0.76 | 1.33 | +0.58 |
| ACC | 94.53 | 92.81 | -1.72 |
| macroF1 | 94.53 | 92.79 | -1.74 |
| ECE | 1.94 | 1.27 | -0.67 |
| Region IoU | 77.77 | 77.66 | -0.10 |

## 2. 12-epoch recipe, all models (mean ± sample std over seeds)

| Model | ACER | APCER | BPCER | AUC | ACC | macroF1 | ECE | Region IoU |
|---|---|---|---|---|---|---|---|---|
| X-UAD | 0.58 (1 seed) | 0.67 (1 seed) | 0.50 (1 seed) | 99.98 (1 seed) | 94.58 (1 seed) | 94.59 (1 seed) | 3.81 (1 seed) | 80.73 (1 seed) |
| CDCN++ | 0.46 (1 seed) | 0.42 (1 seed) | 0.50 (1 seed) | 99.99 (1 seed) | 95.19 (1 seed) | 95.17 (1 seed) | 5.07 (1 seed) | 72.71 (1 seed) |
| ResNet-18 | 1.88 (1 seed) | 3.08 (1 seed) | 0.67 (1 seed) | 99.95 (1 seed) | 96.93 (1 seed) | 96.92 (1 seed) | 2.35 (1 seed) | — |

McNemar exact test, seed 0, same 9,600 test samples (b = X-UAD right & other wrong, c = the reverse):

| vs | 8-way b / c | p | binary b / c | p |
|---|---|---|---|---|
| CDCN++ | 225 / 283 | 0.0114 | 10 / 28 | 0.0051 |
| ResNet-18 | 153 / 378 | 5.43e-23 | 67 / 21 | 9.2e-07 |

## 3. Leave-one-attack-out at the full P1 budget (10 epochs, 19,200-sample split)


**ACER (%) ↓**

| Model | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| X-UAD | — | — | — | — | — |
| CDCN++ | — | — | — | — | — |
| ResNet-18 | — | — | — | — | — |

**8-way accuracy (%) ↑**

| Model | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| X-UAD | — | — | — | — | — |
| CDCN++ | — | — | — | — | — |
| ResNet-18 | — | — | — | — | — |

**Unseen-family detection rate (%) ↑**

| Model | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| X-UAD | — | — | — | — | — |
| CDCN++ | — | — | — | — | — |
| ResNet-18 | — | — | — | — | — |

**Novelty AUROC, model's own uncertainty (vacuity for X-UAD, 1-max softmax for baselines) ↑**

| Model | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| X-UAD | — | — | — | — | — |
| CDCN++ | — | — | — | — | — |
| ResNet-18 | — | — | — | — | — |

**Novelty AUROC, 1-max softmax (all models) ↑**

| Model | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| X-UAD | — | — | — | — | — |
| CDCN++ | — | — | — | — | — |
| ResNet-18 | — | — | — | — | — |

**Novelty AUROC, feature distance (Mahalanobis, fitted on val) ↑**

| Model | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| X-UAD | — | — | — | — | — |
| CDCN++ | — | — | — | — | — |
| ResNet-18 | — | — | — | — | — |

**Novelty AUROC, uncertainty + feature distance (fixed equal weights, z-scored on val) ↑**

| Model | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| X-UAD | — | — | — | — | — |
| CDCN++ | — | — | — | — | — |
| ResNet-18 | — | — | — | — | — |

## 4. Referral rule u ≥ 0.55 (X-UAD), % of each group referred to a human

| Fold | unseen attack | seen attacks | genuine |
|---|---|---|---|
