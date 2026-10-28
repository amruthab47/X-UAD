# X-UAD — Protocol P1, full run (`xuad_P1_full_retry`)

Trained locally on the verified 33,600-sample, 8-class LFW-based benchmark. This run superseded an earlier attempt (`results_full/`) that was externally interrupted near the end of its final epoch with no Python error; that attempt's partial history (epochs 0–4) is preserved untouched at `results_full/xuad_P1_full_hist.json` and was not overwritten.

## Exact command used

```bash
cd /Users/swethaapremnath/Desktop/X-UAD-final
mkdir -p results_full_retry
caffeinate -i python3 scripts/run_experiment.py --data data --model xuad --protocol P1 \
    --epochs 6 --tag xuad_P1_full_retry --out results_full_retry
```
`caffeinate -i` was used to prevent the macOS sleep-related interruption that killed the first attempt. All other flags left at their defaults (`--bs 48 --lr 2.5e-3 --img 112 --threads 2 --workers 1 --seed 0`).

## Dataset

| | |
|---|---|
| Total benchmark size | 33,600 samples (8 classes) |
| Domain A (LFW-Sim, 7 synthetic classes) | 29,400 samples |
| Adversarial class (PGD transfer attack) | 4,200 samples |
| Train split | 19,200 |
| Validation split | 4,800 |
| Test split | 9,600 |
| Classes | bonafide, print, replay, mask3d, faceswap, reenact, synth, adversarial |
| Source corpus | LFW (13,233 images, 5,749 identities), identity-disjoint train/val/test split |

## Model configuration (XUAD defaults, unmodified)

| Param | Value |
|---|---|
| dim | 128 |
| depth | 4 (AQA blocks) |
| heads | 4 |
| circuit_depth | 4 (Givens layers) |
| n_basis | 8 (quantum evidence basis states) |
| mlp_ratio | 2.0 |
| central-difference theta | 0.7 |
| dropout / drop_path | 0.05 / 0.05 |
| image size | 112×112 (random crop from 128px store; centre crop at test) |
| gate_bias | -2.0 |
| Trainable parameters | 1.294 M |
| Optimizer | AdamW, lr=2.5e-3, weight_decay=0.02 |
| LR schedule | OneCycleLR (pct_start=0.25) |
| Gradient clipping | max-norm 3.0 |
| Loss weights | cls=1.0, loc=0.6, basis=0.25, sup=0.3, pues=0.15, coh=0.02, conf=0.4 |

## Training duration

- `fit()` phase (6 epochs, train+per-epoch val): **1,466.5 s ≈ 24.4 min**
- Total wall-clock (launch → all output files written, incl. final val/test eval + save): **≈ 26 min**
- No sleep-related interruptions this run (`caffeinate -i` held throughout)

## Per-epoch training and validation

| Epoch | Epoch time | Train loss (total) | Val AUC | Val ACER | Val ACC | Val ECE |
|---|---|---|---|---|---|---|
| 0 | 258s | 3.530 | 83.41 | 44.17 | 56.17% | 2.37 |
| 1 | 237s | 2.516 | 92.92 | 20.42 | 69.02% | 4.02 |
| 2 | 234s | 2.042 | 98.06 | 10.42 | 75.52% | 2.87 |
| 3 | 234s | 1.667 | 99.72 | 6.42 | 85.27% | 2.13 |
| 4 | 232s | 1.344 | 99.86 | 2.00 | 89.06% | 3.30 |
| 5 | 261s | 1.193 | 99.92 | 2.50 | 89.96% | 2.77 |

Training loss fell monotonically every epoch across all 7 loss components (cls, loc, basis, sup, pues, coh, conf) with no divergence or instability.

## Final test-set metrics (9,600 held-out samples)

| Metric | Value |
|---|---|
| AUC (binary attack detection) | 99.88 |
| ACER | 2.417 (APCER 3.083 / BPCER 1.75) |
| EER | 1.494 |
| 8-way accuracy | 89.49% |
| macro F1 | 89.48% |
| ECE | 3.356 |
| Brier score | 0.153 |
| mean vacuity | 0.185 |
| Localisation IoU / Dice / pixel-AUC | 0.841 / 0.903 / 0.932 |
| Localisation mean (region-level classes only) | 0.726 |

## Class-wise evaluation (test set, n=1,200 per class)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bonafide | 0.9569 | 0.9625 | 0.9597 | 1,200 |
| print | 0.9168 | 0.8908 | 0.9036 | 1,200 |
| replay | 0.9158 | 0.9158 | 0.9158 | 1,200 |
| mask3d | 0.8942 | 0.7608 | 0.8222 | 1,200 |
| faceswap | 0.8246 | 0.8425 | 0.8335 | 1,200 |
| reenact | 0.8857 | 0.8975 | 0.8916 | 1,200 |
| synth | 0.7857 | 0.8892 | 0.8342 | 1,200 |
| **adversarial** | **0.9950** | **1.0000** | **0.9975** | 1,200 |
| **macro avg** | **0.8968** | **0.8949** | **0.8948** | 9,600 |

### Adversarial class, reported separately

The adversarial class is the strongest-performing class by a wide margin: **precision 0.9950, recall 1.0000, F1 0.9975** — every one of the 1,200 adversarial test samples was correctly detected (APCER = 0.0%), with only 6 non-adversarial samples anywhere misclassified as adversarial. Its mean Dirichlet vacuity (0.085) is well below the overall mean (0.185) and its mean predicted-class confidence is 0.990 — the model is highly confident and correct on this class. Its mean risk score (1.067) is also the highest of any class, consistent with its severity weight (0.95, the highest in the taxonomy). This strong result is expected for protocol P1 specifically: the model *saw* adversarial examples during training here (closed-set), unlike the project's leave-one-attack-out (P2) protocol, where prior results (`RESULTS.md`) show the adversarial class is considerably harder when genuinely unseen.

### Disproportionate confusions

One confusion pair stands out well above the rest:

- **mask3d → synth: 264/1,200 (22.0%)** of true mask3d samples were misclassified as synth — by far the largest single confusion in the matrix, and the main driver of mask3d's comparatively low recall (0.7608).
- synth → mask3d: 103/1,200 (8.6%) — a smaller reverse leakage, so the confusion is asymmetric but present in both directions.
- faceswap ↔ reenact: a more symmetric pair (faceswap→reenact 8.2%, reenact→faceswap 8.9%) — plausible, since both are digital manipulations concentrated on similar facial regions.
- print → replay: 5.1% (comparatively minor).

All other off-diagonal confusions are below 5% of their true class's support. The mask3d/synth pair is the one worth further attention if pursuing model improvements — both involve broad, texture-level changes (silicone-mask smoothing vs. generator-fingerprint textures) that may share overlapping low-level statistics the current feature space doesn't fully separate.

## Generated artifacts

| File | Contents |
|---|---|
| `xuad_P1_full_retry.json` | Full metrics dict (test set) |
| `xuad_P1_full_retry.pt` | Trained model checkpoint |
| `xuad_P1_full_retry_hist.json` | Per-epoch train loss + val metrics |
| `xuad_P1_full_retry_pred.npz` | Raw test predictions (`prob, y, vacuity, risk`) |
| `figures/confusion_matrix.png` | Row-normalised 8×8 confusion matrix |
| `figures/roc_curve.png` | Binary attack-detection ROC (AUC 99.88) |
| `figures/precision_recall_curve.png` | Binary attack-detection PR curve (AUC 99.98) |
| `figures/calibration.png` | Reliability diagram + vacuity distribution |
| `figures/localization_examples.png` | Predicted vs. ground-truth evidence maps, one example per class |
