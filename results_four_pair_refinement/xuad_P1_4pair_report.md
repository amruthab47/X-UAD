# X-UAD four-pair refinement — UNSUCCESSFUL, reported as such

**Result: this experiment does not improve on `results_binary_preserving_refinement/` and should not replace it as the primary model.** Accuracy decreased, ACER got slightly worse, bonafide recall regressed slightly, and the difference from the reference model is not statistically significant. Per your instructions, this is reported as a failed/negative experiment and **no further experiment was launched automatically.**

## Provenance

- **Reference (protected, untouched)**: `results_binary_preserving_refinement/xuad_P1_refined.json` / `_pred.npz` — checksummed before and after this run, confirmed byte-identical.
- **This run**: `results_four_pair_refinement/` — identical dataset, split, seed=0, backbone, optimizer, `focal_gamma=2.0`, `epochs=12`, `select_by=acc_f1_acer_guard`. The only change: the `pcp` targeted-pair penalty was extended from 2 confusion clusters (`faceswap↔reenact`, `mask3d↔synth`) to 4 (`+ replay↔print, faceswap↔print`), using the **same weights** (`bce=0.5`, `pcp=0.15`) as the reference — as specified before training.
- Killed once during training (epoch 9→10, not a sleep issue — AC power was connected throughout), recovered correctly via the checkpoint/resume mechanism (verified: "resumed... completed epochs 0-9, continuing at epoch 10"). No errors in either log segment.

## Independent verification (as required, since this experiment approached but did not exceed the 98% threshold — verification run anyway for rigor)

Accuracy and macro-F1 independently recomputed from `xuad_P1_4pair_pred.npz` (`argmax(prob)` vs. `y`) **exactly match** the saved JSON (94.01042% both ways). Confirmed the test set is byte-identical to the reference's (`np.array_equal` on `y`, passed).

## Summary metrics: reference vs. four-pair

| Metric | Reference (`binary_preserving_refinement`) | Four-pair | Change |
|---|---|---|---|
| **Accuracy** | 94.292 | **94.010** | **−0.281 pp (worse)** |
| **Macro-F1** | 94.293 | **94.001** | **−0.293 (worse)** |
| AUC | 99.987 | 99.987 | ~0 |
| **ACER** | 0.750 | **0.792** | **+0.042 (worse)** |
| APCER | 1.000 | 1.083 | +0.083 (worse) |
| BPCER | 0.500 | 0.500 | unchanged |
| EER | 0.500 | 0.500 | unchanged |
| ECE | 4.635 | 4.423 | −0.212 (better) |
| Brier | 0.092 | 0.096 | +0.004 (slightly worse) |
| mean vacuity | 0.064 | 0.072 | +0.008 |

**By your stated success criteria** ("meaningful improvement while preserving or improving ACER and bonafide performance"): **not met.** Accuracy went down, not up; ACER went up, not down or flat.

## Bonafide precision/recall/F1

| | Reference | Four-pair |
|---|---|---|
| Precision | 0.9925 | 0.9933 |
| Recall | 0.9908 | **0.9842 (regressed)** |
| F1 | 0.9917 | 0.9887 |

Bonafide recall regressed — a small but real move in the wrong direction relative to the "must not regress" bar applied to the previous experiment.

## Total errors

**Reference: 548. Four-pair: 575 (+27 more errors, not fewer.)**

## Full confusion matrix (four-pair model)

```
         bonafi   print  replay  mask3d  facesw  reenac   synth  advers   SUM
bonafide   1181       3       8       3       2       1       2       0   1200
print         0    1115      48       0      27      10       0       0   1200
replay        1      26    1152       2      14       3       2       0   1200
mask3d        1       0       2    1165       3       2      27       0   1200
faceswap       2      32      33       4    1033      95       1       0   1200
reenact         3      14       9       0      81    1093       0       0   1200
synth           1       1      11      93       5       3    1086       0   1200
adversarial     0       0       0       0       0       0       0    1200   1200
```

## The four target confusion pairs: reference vs. four-pair

| Pair | Reference | Four-pair | Change |
|---|---|---|---|
| faceswap → reenact | 73 | 95 | **+22 (worse)** |
| reenact → faceswap | 78 | 81 | +3 (worse) |
| synth → mask3d | 84 | 93 | **+9 (worse)** |
| mask3d → synth | 31 | 27 | −4 (better) |
| replay → print | 53 | 26 | **−27 (much better)** |
| print → replay | 29 | 48 | **+19 (worse)** |
| faceswap → print | 50 | 32 | **−18 (better)** |
| print → faceswap | 37 | 27 | −10 (better) |

**The key finding**: the two *new* pairs added this experiment (`replay↔print`, `faceswap↔print`) mostly improved — but the two *original* pairs the reference model was exclusively tuned for (`faceswap↔reenact`, `synth↔mask3d`) got measurably **worse**, most notably `faceswap→reenact` (+22 errors). The net effect across all four pairs is roughly a wash, and the model's *overall* error count still went up because of collateral effects outside the four targeted pairs too.

## Statistical comparison

McNemar's test on the identical 9,600-sample test set: **265 reference-wrong→4pair-right vs. 292 reference-right→4pair-wrong** (557 discordant). χ²=1.214, **p=0.2706** — far from significant. The difference between the two models is statistically indistinguishable from noise; there is no basis to claim the four-pair model is different from the reference in either direction with any confidence.

## Interpretation (not independently verifiable, reasoned from the numbers above)

The `pcp` loss term is a single aggregate weight (0.15) that **averages** over however many (sample, partner) terms are active in a batch. Extending from 2 pairs to 4 didn't increase total penalty strength (by design, to stay "conservative" — confirmed correct, since ACER didn't blow up), but it did **dilute how much of that fixed budget lands on any one pair**. The two original pairs, which previously received the full undivided benefit of the `pcp` term, now share it with two more pairs — and their confusion counts moved backward as a result. This is a plausible, evidence-consistent explanation, not a proven mechanism.

## Conclusion and next step

**This experiment failed to improve on the protected reference and is not adopted.** `results_binary_preserving_refinement/xuad_P1_refined.pt` remains the primary model. Per your instructions, **no further experiment has been launched automatically** — this stops here, awaiting your direction. If pursued further, the diagnosis above suggests the fixed-weight-averaging dilution effect (not the choice of which pairs to target) is the more likely culprit, which would point toward either per-pair weighting (giving each cluster its own small weight rather than sharing one aggregate term) or tackling the pairs one at a time in sequence rather than all four simultaneously — but this is a suggestion for a *possible* future experiment, not something started.

## Files

| File | Role |
|---|---|
| `results_binary_preserving_refinement/` | reference model (unmodified, checksummed) |
| `results_four_pair_refinement/xuad_P1_4pair.json/.pt/_hist.json/_pred.npz` | this (unsuccessful) run's outputs |
| `results_four_pair_refinement/xuad_P1_4pair_epoch.pt` | rolling checkpoint (all 12 epochs) |
| `results_four_pair_refinement/xuad_P1_4pair_report.md` | this document |
