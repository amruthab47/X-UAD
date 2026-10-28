# X-UAD binary-preserving refinement — SUCCESS, all three criteria met

**Bottom line**: this experiment achieved what the hierarchical ablation could not. Accuracy improved (93.14%→94.29%), and — unlike the hierarchical attempt — ACER *also* improved (1.79%→0.75%), bonafide recall improved rather than regressed (0.987→0.991), and the two dominant confusion pairs shrank overall (though not uniformly). Statistically the strongest result of the three follow-up experiments (McNemar p=1.35e-05).

## Provenance

- **Protected baseline**: `results_improved_p1_retry/` (93.14% accuracy, 1.79% ACER) — confirmed byte-identical before and after this run.
- **Also preserved, untouched**: `results_hierarchical_p1/` (the ablation this experiment was designed to fix).
- **This run**: `results_binary_preserving_refinement/` — identical dataset, split, seed=0, backbone (`hierarchical=False`, same `EvidentialHead` as the baseline), optimizer, focal_gamma=2.0, epochs=12. The only additions: `bce` (binary-reinforcement, weight 0.5) and `pcp` (targeted faceswap/reenact + mask3d/synth penalty, weight 0.15) loss terms, both computed directly on the existing classifier's output probability (no new parameters, no second decision surface); and `select_by=acc_f1_acer_guard` for checkpoint selection.
- No errors or warnings in the training log. Training completed in one uninterrupted pass under `caffeinate -i` on AC power (no kill this time).

## Success criteria — all three met

| Criterion | Requirement | Result | Met? |
|---|---|---|---|
| ACER | not worse than 1.79% | **0.750%** | ✅ (better, not just non-worse) |
| Accuracy | improve beyond 93.14% | **94.292%** | ✅ (+1.156 pp) |
| Bonafide recall | must not regress | 0.9867 → **0.9908** | ✅ (improved) |

## Summary metrics: protected baseline vs. refined

| Metric | Baseline (93.14%) | Refined | Change |
|---|---|---|---|
| **Accuracy** | 93.135 | **94.292** | **+1.156 pp** |
| **Macro-F1** | 93.090 | **94.293** | **+1.203** |
| AUC | 99.965 | 99.987 | +0.022 |
| **ACER** | 1.792 | **0.750** | **−1.042 (better)** |
| APCER | 3.000 | 1.000 | −2.000 (better) |
| BPCER | 0.583 | 0.500 | −0.083 (better) |
| EER | 0.750 | 0.500 | −0.250 (better) |
| ECE | 7.692 | 4.635 | −3.057 (better, unrequested bonus) |
| Brier | 0.122 | 0.092 | −0.029 (better) |
| mean vacuity | 0.083 | 0.064 | −0.019 |

**Every single metric improved** — no tradeoffs anywhere in the summary table, unlike either prior experiment.

## Localisation

| Metric | Baseline | Refined | Change |
|---|---|---|---|
| IoU | 0.8716 | 0.8840 | +0.0124 |
| Dice | 0.9224 | 0.9313 | +0.0089 |
| pixel-AUC | 0.9691 | 0.9675 | −0.0016 (negligible) |

## Per-class precision / recall / F1

| Class | Baseline P/R/F1 | Refined P/R/F1 | Δ F1 |
|---|---|---|---|
| **bonafide** | 0.973 / 0.987 / 0.9797 | **0.993 / 0.991 / 0.9917** | **+0.0119** |
| print | 0.916 / 0.931 / 0.9231 | 0.910 / 0.938 / 0.9236 | +0.0005 |
| replay | 0.937 / 0.947 / 0.9416 | 0.948 / 0.930 / 0.9390 | −0.0026 (only regression, negligible) |
| mask3d | 0.900 / 0.958 / 0.9281 | 0.928 / 0.968 / 0.9474 | +0.0193 |
| faceswap | 0.898 / 0.819 / 0.8566 | 0.877 / 0.876 / 0.8762 | +0.0196 |
| reenact | 0.880 / 0.938 / 0.9076 | 0.924 / 0.924 / 0.9242 | +0.0165 |
| synth | 0.953 / 0.873 / 0.9108 | 0.967 / 0.918 / 0.9414 | +0.0306 |
| adversarial | 0.999 / 1.000 / 0.9996 | 1.000 / 1.000 / 1.0000 | +0.0004 |

**7 of 8 classes improved**; the one regression (replay, −0.0026) is negligible.

## Confusion matrix (refined model)

```
         bonafi   print  replay  mask3d  facesw  reenac   synth  advers   SUM
bonafide   1189       0       2       1       2       2       4       0   1200
print         0    1125      29       0      37       9       0       0   1200
replay        2      53    1116       0      24       4       1       0   1200
mask3d        0       0       2    1161       4       2      31       0   1200
faceswap       4      50      15       5    1051      73       2       0   1200
reenact         2       8       3       0      78    1109       0       0   1200
synth           1       0      10      84       3       1    1101       0   1200
adversarial     0       0       0       0       0       0       0    1200   1200
```

Every row sums to exactly 1,200 (verified). Total errors: 548 (vs. baseline's 659).

## Did the targeted pair penalty work?

| Confusion | Baseline | Refined | Change |
|---|---|---|---|
| faceswap → reenact | 131 | **73** | **−58 (much better)** |
| reenact → faceswap | 62 | 78 | +16 (worse) |
| mask3d → synth | 43 | **31** | **−12 (better)** |
| synth → mask3d | 122 | **84** | **−38 (much better)** |
| **Total (4 cells)** | **358** | **266** | **−92 net reduction** |

**Mostly yes, with one exception.** 3 of the 4 targeted confusion directions improved substantially; only reenact→faceswap moved the wrong way (+16), likely because pulling faceswap's own probability mass away from reenact (the pcp penalty is symmetric, targeting both directions equally) freed up some ambiguity that reenact samples now fall into. Net effect across both pairs is a clear win (92 fewer errors in exactly the cells the penalty targeted), but it's not a uniform, clean fix — the same kind of nuance the earlier error analysis would predict for a "small, targeted" intervention rather than a perfect one.

## Statistical significance

McNemar's test on the identical 9,600 test samples vs. the protected baseline: **375 baseline-wrong→refined-right vs. 264 baseline-right→refined-wrong** (639 discordant pairs). χ²=18.936, **p=1.35×10⁻⁵**.

This is a much stronger result than the hierarchical experiment's p=0.018 — decisively significant, not borderline. The improvement is real, not noise.

## Contact sheet

`figures/error_contact_sheet.png` — remaining misclassified examples for the refined model, same pairs as before (faceswap↔reenact, synth↔mask3d, print↔replay), with true label, predicted label, confidence, and vacuity captioned.

## Why this worked where the hierarchical experiment didn't

The hierarchical ablation introduced a **separate, independently-parameterized binary sub-head** whose own decision boundary could diverge from — and in that case did worse than — the original flat softmax's boundary. This experiment instead added loss pressure **directly on the same probability distribution** the original classifier already used to achieve 1.79% ACER (`out["prob"][:,0]`), with no new head and therefore no second boundary that could diverge. The binary-reinforcement term (`bce`, weight 0.5, deliberately dominant over the 0.15-weight targeted penalty) sharpened that existing boundary further rather than replacing it — which is the direct, causal explanation for why ACER *improved* here instead of regressing. The `acc_f1_acer_guard` checkpoint-selection rule (disqualifying any epoch whose validation ACER drifted more than 0.5pp from its best-seen value) also structurally prevented the training run itself from drifting toward a high-accuracy/high-ACER epoch, even if one had occurred.

## Files

| File | Role |
|---|---|
| `results_improved_p1_retry/` | protected baseline (unmodified, checksummed) |
| `results_hierarchical_p1/` | protected ablation (unmodified, checksummed) |
| `results_binary_preserving_refinement/xuad_P1_refined.json/.pt/_hist.json/_pred.npz` | this run's final outputs |
| `results_binary_preserving_refinement/xuad_P1_refined_epoch.pt` | rolling resumable checkpoint (all 12 epochs) |
| `results_binary_preserving_refinement/figures/error_contact_sheet.png` | misclassified examples |
| `results_binary_preserving_refinement/xuad_P1_refined_report.md` | this document |
