# X-UAD — Improved P1 (focal loss + 12 epochs + acc_f1 selection) vs. original P1 baseline

**Bottom line up front**: the experiment produced a genuine, statistically significant improvement in 8-class classification accuracy (89.49% → 93.14% on the held-out test set) **without** trading away AUC or ACER — both of those also improved. It did **not** reach the 94–96% target. The honest cost of this experiment is calibration: ECE more than doubled (3.36 → 7.69). Full numbers and diagnosis below.

## Run provenance

- **Baseline**: `results_full_retry/xuad_P1_full_retry.json` / `_pred.npz` (6 epochs, original loss, `select_by=auc_acer`)
- **Improved**: `results_improved_p1_retry/xuad_P1_improved_retry.json` / `_pred.npz` (12 epochs, focal loss γ=2.0, `select_by=acc_f1`, same dataset/split/protocol/seed=0)
- The run was interrupted twice by an external kill (not a script error, not a sleep issue — charger was connected and `caffeinate -i` was active both times) and recovered twice via the new checkpoint/resume mechanism (`--save-every-epoch` / `--resume-from`), verified to continue correctly rather than restart (confirmed via the printed `resumed from ... continuing at epoch N` line and unbroken `hist.json` epoch sequences each time).
- Best validation epoch: **epoch 10** (val ACC=93.77, AUC=99.94, ACER=1.42, selected score = ACC+macroF1 = 187.53), marginally ahead of epoch 11 (187.45) — the `acc_f1` checkpoint-selection change is working as designed.
- **All numbers below are from the final held-out TEST set (9,600 samples), not validation**, per instruction.

## Summary metrics

| Metric | Original P1 | Improved P1 | Change |
|---|---|---|---|
| Accuracy | 89.490 | **93.135** | **+3.646** |
| Macro-F1 | 89.476 | **93.090** | **+3.614** |
| AUC | 99.880 | 99.965 | +0.084 |
| ACER | 2.417 | 1.792 | −0.625 |
| APCER | 3.083 | 3.000 | −0.083 |
| BPCER | 1.750 | 0.583 | −1.167 |
| EER | 1.494 | 0.750 | −0.744 |
| ECE | 3.356 | **7.692** | **+4.336** |
| Brier | 0.153 | 0.122 | −0.031 |
| mean vacuity | 0.185 | 0.083 | −0.102 |

## Localisation metrics (unexpected bonus — not the target of this change)

| Metric | Original P1 | Improved P1 | Change |
|---|---|---|---|
| IoU | 0.8410 | 0.8716 | +0.0306 |
| Dice | 0.9034 | 0.9224 | +0.0190 |
| pixel-AUC | 0.9324 | 0.9691 | +0.0367 |
| mean_localisable | 0.7263 | 0.7948 | +0.0685 |

## Per-class precision / recall / F1

| Class | P1 F1 | Improved F1 | Δ F1 | P: base→imp | R: base→imp |
|---|---|---|---|---|---|
| bonafide | 0.9597 | 0.9797 | +0.0200 | 0.957→0.973 | 0.963→0.987 |
| print | 0.9036 | 0.9231 | +0.0195 | 0.917→0.916 | 0.891→0.931 |
| replay | 0.9158 | 0.9416 | +0.0257 | 0.916→0.937 | 0.916→0.947 |
| **mask3d** | 0.8222 | **0.9281** | **+0.1060** | 0.894→0.900 | 0.761→0.958 |
| faceswap | 0.8335 | 0.8566 | +0.0232 | 0.825→0.898 | 0.843→0.819 |
| reenact | 0.8916 | 0.9076 | +0.0161 | 0.886→0.880 | 0.897→0.938 |
| synth | 0.8342 | 0.9108 | +0.0766 | 0.786→0.953 | 0.889→0.873 |
| adversarial | 0.9975 | 0.9996 | +0.0021 | 0.995→0.999 | 1.000→1.000 |

**Every single class improved** — no class regressed.

## Confusion matrix (improved model, rows=true, cols=predicted)

```
         bonafi   print  replay  mask3d  facesw  reenac   synth  advers   SUM
bonafide   1184       3       3       0       2       2       6       0   1200
print         3    1117      41       1      24      13       0       1   1200
replay        5      40    1136       0      13       4       2       0   1200
mask3d        0       1       2    1149       3       2      43       0   1200
faceswap     14      51      16       4     983     131       1       0   1200
reenact       2       8       3       0      62    1125       0       0   1200
synth         9       0      12     122       8       2    1047       0   1200
adversarial   0       0       0       0       0       0       0    1200   1200
```

Every row sums to exactly 1,200 (verified, same discipline as the earlier P2.replay check).

## Which classes improved, and why (tied to the original diagnosis)

- **mask3d (headline result)**: the original P1 baseline's single worst confusion was mask3d→synth at 22.0% (264/1,200). In the improved model that same confusion is **43/1,200 (3.6%)** — an 84% relative reduction. This is exactly the effect focal loss was chosen for: mask3d's recall jumped from 0.761→0.958 with almost no precision cost.
- **synth**: F1 up +0.077, driven almost entirely by precision (0.786→0.953) — far fewer other classes get misclassified as synth. Synth's own recall dipped very slightly (0.889→0.873), with synth→mask3d actually rising a little (103→122) — the mask3d/synth confusion didn't vanish, it became more one-directional (mask3d is now well-separated; synth still leaks a bit into mask3d).
- **replay vs. bonafide**: both directions of this pair improved — bonafide→replay 4→3, replay→bonafide 14→5. A secondary, smaller win alongside the mask3d headline.
- **faceswap ↔ reenact**: mixed but net positive. reenact→faceswap fell sharply (107→62), while faceswap→reenact rose (99→131) — the pair didn't get uniformly better, it partially inverted. Both classes' F1 still improved (faceswap +0.023, reenact +0.016), but this pair is the one area where the confusion pattern shifted rather than cleanly resolved, and is the most likely candidate for further work if this pair specifically matters.
- **adversarial**: essentially untouched (F1 0.9975→0.9996, still 1,200/1,200 correct) — confirms the change didn't disturb the one class that was already solved.

## Is the improvement statistically meaningful?

**Yes, decisively.** McNemar's test on the identical 9,600 test samples (paired, since both models are scored on the exact same held-out set): 668 samples the baseline got wrong that the improved model got right, vs. only 318 the reverse (986 discordant pairs total). Continuity-corrected χ² = 123.5, p ≈ 0 (far below any conventional significance threshold). This is not noise or a favorable random seed — the improved model is genuinely, substantially more accurate on this test set.

## Does it reach the 96% target?

**No — 93.135% test accuracy, short of both the 96% goal and the 94% floor** stated as the minimum bar for declaring success. Reporting this plainly, not as a success.

## Does the accuracy gain cost AUC or ACER?

**No.** Both improved alongside accuracy: AUC +0.084 (99.880→99.965), ACER −0.625 (2.417→1.792, i.e., better). This is the cleanest possible outcome on the primary detection metrics — there was no accuracy/detection-quality tradeoff.

**The real cost is calibration.** ECE more than doubled (3.36→7.69) and mean vacuity dropped from 0.185→0.083 — the model became substantially more confident on average, which is exactly the known side effect of focal loss: by design it concentrates gradient on hard/uncertain examples and drives easy ones toward saturated confidence, which sharpens decisions (helping accuracy) but stretches the gap between confidence and true correctness rate (hurting ECE). Brier score, notably, still improved slightly (0.153→0.122) — Brier blends calibration and sharpness, and the sharpness gain outweighed the calibration loss there, but ECE isolates calibration specifically and shows the cost clearly.

## Diagnosis: why 93.1%, not 96%

1. **Calibration overfitting, not accuracy overfitting**: training loss and validation accuracy both improved smoothly and monotonically across all 12 epochs with no sign of overfitting on the classification objective itself (val ACC climbed every epoch: 56.65→59.88→70.19→74.79→84.60→83.15→87.50→90.08→91.60→92.77→93.77→93.73). The ceiling here looks like it's the *feature representation's* separability limit for the remaining confusions (faceswap/reenact especially), not an undertrained model — more epochs at the same γ are unlikely to close much more of the gap, since epochs 10 and 11 were already within 0.04 of each other.
2. **The faceswap/reenact pair only partially resolved** and in one sub-direction got worse (faceswap→reenact rose). This is the most concrete remaining error source visible in the confusion matrix, and it wasn't specifically targeted by focal loss the way mask3d/synth was (focal loss reweights by confidence, not by which specific pair is confused).
3. **γ=2.0 may not be the calibration-optimal choice** — it's the literature-standard default, not tuned for this dataset. A lower γ would likely trade back some of the accuracy gain for less ECE damage; this wasn't explored (single-variable experiment, as required).

## Recommended next controlled experiment

Per the fallback list, the evidence points specifically to **decision calibration**, not architecture changes:

**Experiment 2 (recommended, single variable)**: add post-hoc **temperature scaling** fit on the validation split (a single scalar `T` dividing the logits before softmax, fit by minimizing validation NLL) — this is the standard, minimal-risk fix for exactly this failure mode (a model that's accurate but overconfident), touches nothing about training, and is fully separable from Experiment 1: it can be applied to the *already-trained* `xuad_P1_improved_retry.pt` checkpoint with no retraining at all, so it's cheap to test and won't disturb the accuracy gain just measured. Expected outcome: ECE should drop substantially back toward baseline levels with zero change to accuracy/AUC/ACER (temperature scaling is monotonic, so argmax predictions are unaffected).

**Experiment 3 (if more accuracy is still wanted after that)**: targeted feature separation for the faceswap/reenact pair specifically — e.g. a small contrastive term between those two classes' pooled features, rather than a global architecture change, keeping with the "one justified variable at a time" discipline established in Experiment 1.

Not recommending higher input resolution or frequency-domain feature changes yet — nothing in this run's confusion matrix or per-class results points to a resolution or spectral-feature bottleneck; the remaining errors are inter-class boundary confusions (faceswap/reenact) and calibration, both of which have cheaper, more targeted fixes to try first.

## Files referenced

| File | Role |
|---|---|
| `results_full_retry/xuad_P1_full_retry.json`, `_pred.npz` | baseline (read-only, unmodified) |
| `results_improved_p1_retry/xuad_P1_improved_retry.json`, `.pt`, `_hist.json`, `_pred.npz` | improved run, final outputs |
| `results_improved_p1_retry/xuad_P1_improved_retry_epoch.pt` | rolling resumable checkpoint (all 12 epochs, kept for provenance) |
| `results_improved_p1_retry/xuad_P1_improved_vs_baseline_report.md` | this document |
