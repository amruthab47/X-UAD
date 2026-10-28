# X-UAD per-pair-weighted refinement — UNSUCCESSFUL by your stated criterion, reported as such

**Result: this experiment does not meet your success bar and should not replace `results_binary_preserving_refinement/` as the primary model.** Accuracy and Macro-F1 both decreased versus the reference (not statistically significant, but a decrease is not an improvement). Binary-detection quality (ACER/APCER/EER) improved substantially and bonafide precision/F1 improved, but your criterion required accuracy to *meaningfully improve* while these were *preserved or improved* — accuracy failing that bar means the overall criterion is not met. Per your instructions, **no further experiment has been launched automatically.**

## Provenance

- **Reference (protected, untouched)**: `results_binary_preserving_refinement/xuad_P1_refined.json` / `_pred.npz` — checksummed before and after this run, confirmed byte-identical (`9c1896a8...` / `73cd5e43...`).
- **This run**: `results_pair_weighted_refinement/` — identical dataset, split, seed=0, backbone, optimizer, `focal_gamma=2.0`, `epochs=12` (ran to completion, no kill/resume needed this time), `select_by=acc_f1_acer_guard`. The only change: replaced the shared-mean `pcp` term with the independent per-pair-weighted `weighted_pair_confusion_penalty` (`--refine-weighted`), keeping `bce=0.5` unchanged and assigning the original two pairs (faceswap↔reenact, mask3d↔synth) the same 0.15 weight as the successful reference, plus two new pairs (replay↔print, faceswap↔print) at 0.05 each — as shown and approved before training.

## Independent verification

Recomputed `argmax(prob)` vs. `y` directly from `xuad_P1_pairweighted_pred.npz` for both accuracy and macro-F1: **exact match** to the saved JSON (94.0625% / 94.05595% both ways). Confirmed the test set is byte-identical to the reference's (`np.array_equal` on `y` passed, n=9600).

## Summary metrics: reference vs. pair-weighted

| Metric | Reference | Pair-weighted | Change |
|---|---|---|---|
| **Accuracy** | 94.292 | **94.063** | **−0.229 pp (worse)** |
| **Macro-F1** | 94.293 | **94.056** | **−0.237 (worse)** |
| AUC | 99.987 | 99.983 | ~0 (marginally worse) |
| **ACER** | 0.750 | **0.417** | **−0.333 (better)** |
| **APCER** | 1.000 | **0.333** | **−0.667 (much better)** |
| BPCER | 0.500 | 0.500 | unchanged |
| **EER** | 0.500 | **0.417** | **−0.083 (better)** |
| ECE | 4.635 | 5.173 | +0.538 (worse) |
| Brier | 0.092 | 0.096 | +0.004 (worse) |

**By your stated success criteria** ("meaningfully improves the reference model while preserving or improving Accuracy, Macro-F1, ACER, APCER/BPCER, EER, Bonafide recall"): **not met.** Accuracy and Macro-F1 both went down, not up — the primary criterion fails even though the binary-detection metrics (ACER/APCER/EER) improved clearly.

## Bonafide precision/recall/F1

| | Reference | Pair-weighted |
|---|---|---|
| Precision | 0.9925 | **0.9975 (better)** |
| Recall | 0.9908 | 0.9908 (unchanged, exactly preserved) |
| F1 | 0.9917 | **0.9941 (better)** |

Bonafide performance is preserved/improved — this part of the criterion is satisfied on its own.

## Total errors

**Reference: 548. Pair-weighted: 570 (+22 more errors, not fewer.)**

## Full confusion matrix (pair-weighted model, row=true, col=pred; order: bonafide, print, replay, mask3d, faceswap, reenact, synth, adversarial)

```
             bonafi   print  replay  mask3d  facesw  reenac   synth  advers   SUM
bonafide       1189       2       2       1       2       0       3       1   1200
print             1    1156       8       3      23       7       0       2   1200
replay             1      40    1107       8      27       5       7       5   1200
mask3d             0       1       4    1131       8       1      55       0   1200
faceswap           0      48      16      11    1052      67       6       0   1200
reenact            1      16       5       0      65    1113       0       0   1200
synth              0       1      13      90       6       4    1083       3   1200
adversarial        0       0       0       0       0       0       0    1199   1200 (1 misclassified as bonafide, see row 1)
```

## The four target confusion pairs: reference vs. pair-weighted

| Pair | Reference | Pair-weighted | Change |
|---|---|---|---|
| faceswap → reenact | 73 | 67 | **−6 (better)** |
| reenact → faceswap | 78 | 65 | **−13 (better)** |
| mask3d → synth | 31 | **55** | **+24 (much worse)** |
| synth → mask3d | 84 | 90 | +6 (worse) |
| replay → print | 53 | 40 | **−13 (better)** |
| print → replay | 29 | 8 | **−21 (much better)** |
| faceswap → print | 50 | 48 | −2 (~same) |
| print → faceswap | 37 | 23 | **−14 (better)** |

**Key finding**: the independent-weighting fix worked exactly as intended for the **faceswap↔reenact** pair (both directions improved, unlike the four-pair experiment where this pair got worse from dilution) and for both **new** pairs (replay↔print, faceswap↔print — all four directions improved or held). But **mask3d→synth got substantially worse** (31→55, +24 errors) despite receiving the identical 0.15 weight, undiluted, that produced good results in the reference run. This means the earlier "dilution" diagnosis was only part of the story for this pair — giving mask3d↔synth its own independent 0.15 weight did *not* reproduce the reference's mask3d↔synth behavior, which points to some interaction between this specific pair's gradient and the *other* three pairs' terms now being trained simultaneously (optimization interference, not simple dilution), rather than a weight-sharing artifact. The net effect: three of the four pairs improved, but the one that regressed did so by more than the other three combined, so total error count rose.

## Statistical comparison

McNemar's test on the identical 9,600-sample test set: **307 reference-right→pair-weighted-wrong vs. 285 reference-wrong→pair-weighted-right** (592 discordant). χ²=0.7449, **p=0.3881** — not significant. The accuracy/macro-F1 decrease is not statistically distinguishable from noise, but per your explicit criterion the model must *improve*, not merely "not be significantly worse" — so this still counts as not meeting the bar.

## Conclusion and next step

**This experiment does not meet your stated success criterion and is not adopted.** `results_binary_preserving_refinement/xuad_P1_refined.pt` remains the primary model. Per your instructions, **no further experiment has been launched automatically** — this stops here, awaiting your direction.

If pursued further, the diagnosis above suggests the mask3d↔synth pair may need a smaller weight than 0.15 once other pair-terms are co-active (since it regressed even without dilution), or the pairs may need to be introduced sequentially rather than jointly — but these are only suggestions for a *possible* future experiment, not anything started.

## Files

| File | Role |
|---|---|
| `results_binary_preserving_refinement/` | reference model (unmodified, checksummed) |
| `results_pair_weighted_refinement/xuad_P1_pairweighted.json/.pt/_hist.json/_pred.npz` | this (unsuccessful, by your accuracy criterion) run's outputs |
| `results_pair_weighted_refinement/xuad_P1_pairweighted_epoch.pt` | rolling checkpoint (all 12 epochs, no kill/resume needed) |
| `results_pair_weighted_refinement/xuad_P1_pairweighted_report.md` | this document |
