# X-UAD hierarchical P1 experiment — results, honestly reported

**Bottom line**: the hierarchical classifier produced a small, marginally statistically-significant accuracy gain (93.14%→93.77%, p=0.018) — but at a real cost to ACER (1.79%→3.42%), and it did **not** achieve its primary intended mechanism: cross-superclass error leakage did not decrease, and by one measure ticked up slightly (56.1%→57.7% of errors). This is not a clean win. Reporting it as it is.

## Provenance

- **Protected baseline**: `results_improved_p1_retry/` (93.14% test accuracy, focal loss + 12 epochs + acc_f1 selection) — confirmed untouched throughout (checksummed before and after this run).
- **Hierarchical**: `results_hierarchical_p1/` — identical dataset, split, seed=0, backbone, optimizer, focal_gamma=2.0, select_by=acc_f1, epochs=12; the only change is `HierarchicalEvidentialHead` replacing `EvidentialHead`, plus the `hbin`/`hsup` auxiliary losses replacing the old `sup` term.
- No errors or warnings anywhere in the training log. No lingering processes. Checkpoint/resume mechanism was in place (`--save-every-epoch`) but was not needed this run — training completed in one uninterrupted pass (with one long idle gap mid-epoch-1, same power-management pattern as previous runs, but the process was never killed this time).

## Summary metrics: protected baseline vs. hierarchical

| Metric | Baseline (93.14%) | Hierarchical | Change |
|---|---|---|---|
| **Accuracy** | 93.135 | **93.771** | **+0.635 pp** |
| **Macro-F1** | 93.090 | **93.761** | **+0.671** |
| AUC | 99.965 | 99.915 | −0.050 |
| **ACER** | 1.792 | **3.417** | **+1.625 (worse)** |
| APCER | 3.000 | 5.583 | +2.583 (worse) |
| BPCER | 0.583 | 1.250 | +0.667 (worse) |
| EER | 0.750 | 1.488 | +0.738 (worse) |
| ECE | 7.692 | 7.256 | −0.436 (better) |
| Brier | 0.122 | 0.110 | −0.012 (better) |
| mean vacuity | 0.083 | 0.085 | +0.003 |

## Localisation

| Metric | Baseline | Hierarchical | Change |
|---|---|---|---|
| IoU | 0.8716 | 0.8842 | +0.0126 |
| Dice | 0.9224 | 0.9307 | +0.0083 |
| pixel-AUC | 0.9691 | 0.9653 | −0.0038 |

## Per-class F1

| Class | Baseline F1 | Hierarchical F1 | Δ | P: base→hier | R: base→hier |
|---|---|---|---|---|---|
| bonafide | 0.9797 | **0.9693** | **−0.0104** | 0.973→0.964 | 0.987→0.975 |
| print | 0.9231 | 0.9326 | +0.0095 | 0.916→0.925 | 0.931→0.940 |
| replay | 0.9416 | 0.9463 | +0.0047 | 0.937→0.953 | 0.947→0.939 |
| mask3d | 0.9281 | 0.9377 | +0.0096 | 0.900→0.923 | 0.958→0.953 |
| faceswap | 0.8566 | 0.8718 | +0.0152 | 0.898→0.882 | 0.819→0.862 |
| reenact | 0.9076 | 0.9127 | +0.0051 | 0.880→0.906 | 0.938→0.919 |
| synth | 0.9108 | 0.9304 | +0.0196 | 0.953→0.948 | 0.873→0.913 |
| adversarial | 0.9996 | 1.0000 | +0.0004 | 0.999→1.000 | 1.000→1.000 |

**bonafide is the only class that regressed** — its F1 fell because more bonafide samples are now being misclassified as attacks (recall dropped 0.987→0.975), which directly explains the BPCER/APCER/ACER regression above: the hierarchical binary gate is a *slightly worse* bonafide/attack discriminator than the flat model's implicit boundary, even though the fine-grained attack-family classification improved.

## Confusion matrix (hierarchical model)

```
         bonafi   print  replay  mask3d  facesw  reenac   synth  advers   SUM
bonafide   1170       5       5       2       4       8       6       0   1200
print         3    1128      29       1      29      10       0       0   1200
replay        6      40    1127       3      17       4       3       0   1200
mask3d        1       0       2    1144       3       0      50       0   1200
faceswap      27      35       9       4    1034      90       1       0   1200
reenact        3      10       3       0      81    1103       0       0   1200
synth          4       1       7      86       4       2    1096       0   1200
adversarial     0       0       0       0       0       0       0    1200   1200
```

Every row sums to exactly 1,200 (verified).

## Cross-superclass leakage — did the hierarchical design work as intended?

| | Baseline | Hierarchical |
|---|---|---|
| Total errors | 659 | 598 |
| Cross-superclass errors | 370 (56.1%) | 345 (57.7%) |
| 3-way superclass-level accuracy | 96.146% | 96.406% |

**No — not as hypothesized.** The absolute number of cross-superclass errors fell (370→345, tracking the overall error reduction), but the *proportion* of remaining errors that cross the physical/digital/adversarial boundary did not improve — it ticked up slightly. Superclass-level accuracy improved only marginally (+0.26 pp). The hierarchical factorization did not deliver the structural separation the error analysis predicted; the accuracy gain that did occur appears to come mostly from the fine-grained within-superclass classification improving (see per-class F1 above: print/replay/mask3d/faceswap/reenact/synth all improved modestly), not from the cross-family leakage the design specifically targeted.

## Statistical significance

McNemar's test on the identical 9,600 test samples: **352 baseline-wrong→hierarchical-right vs. 291 baseline-right→hierarchical-wrong** (643 discordant pairs). χ²=5.599, **p=0.0180**.

**This is statistically significant (p<0.05), but much more marginal than the previous experiment's result** (which had p≈0, essentially certain). A p-value of 0.018 means there's roughly a 1.8% chance this net improvement is due to chance alone — meaningful, but this is a real, modest effect, not an overwhelming one. Given the accompanying ACER regression, I would characterize this as "a genuine but small and mixed effect," not a strong win.

## Contact sheet

`figures/error_contact_sheet.png` — misclassified examples from the hierarchical model's confusion pairs, same captioning convention as the baseline analysis (true label, predicted label, confidence, vacuity).

## Honest conclusion

1. **Accuracy target (99%)**: not remotely approached, and no claim of that is made. 93.77% test accuracy.
2. **Did it improve on the protected baseline?** Marginally, and unevenly: yes on accuracy/macro-F1/localisation/calibration, no on ACER/APCER/BPCER/EER/AUC. Whether this is a net improvement depends entirely on which metric matters more for the deployment — for a security-relevant detector, the ACER regression is arguably the more consequential number, since it means more bonafide users get incorrectly flagged and more attacks get missed at the operating threshold, even while the *specific attack family* label is more often correct when the system does get the case right.
3. **Did the hierarchical structure fix the diagnosed problem (cross-superclass leakage)?** No — this is the most important honest finding. The mechanism did not work as the error analysis predicted. The gain that did materialize looks like it came from elsewhere (probably the auxiliary hbin/hsup supervision acting as a mild regularizer / extra training signal, rather than the structural gating preventing cross-family errors).
4. **Recommendation for any follow-up** (not executed, per your instruction to run only one experiment): if hierarchical classification is worth pursuing further, the next step would need to diagnose *why* the structural gating didn't reduce cross-superclass leakage as designed — plausibly the multiplicative probability factorization still allows gradient to route around the intended constraint, or the `hbin`/`hsup` auxiliary loss weights (0.3 each, matched to the old `sup` term's scale) weren't strong enough to actually enforce hard separation. That diagnosis, not a bigger architecture change, would be the evidence-justified next single variable.

## Files

| File | Role |
|---|---|
| `results_improved_p1_retry/` | protected baseline (read-only, unmodified, checksummed) |
| `results_hierarchical_p1/xuad_P1_hierarchical.json/.pt/_hist.json/_pred.npz` | this run's final outputs |
| `results_hierarchical_p1/xuad_P1_hierarchical_epoch.pt` | rolling resumable checkpoint (all 12 epochs) |
| `results_hierarchical_p1/figures/error_contact_sheet.png` | misclassified examples |
| `results_hierarchical_p1/xuad_P1_hierarchical_report.md` | this document |
