# X-UAD region-pooled refinement, stop-gradient fix — MIXED RESULT, does not unambiguously meet the success bar

**The stop-gradient fix worked mechanically exactly as diagnosed**: localisation quality now *improves* over the reference (loc_IoU 0.884→0.889, loc_Dice 0.931→0.935) instead of degrading as it did in the non-detached run (0.884→0.868), confirming the root-cause finding in `results_region_pooled_refinement/xuad_P1_regionpool_report.md` was correct. Accuracy, Macro-F1, ACER, APCER, EER, and localisation quality all improved numerically over the reference. **But**: the accuracy gain is not statistically significant (McNemar p=0.24 — indistinguishable from noise), and **bonafide recall regressed** (0.9908→0.9875, 11→15 bonafide samples now misclassified as attack) — one of your explicit "must preserve or improve" criteria. Per the strict letter of your success bar, this is **not a clean success**, and per your instruction ("if the experiment fails, STOP"), I am **not** launching anything further and **not** proposing this as the new primary model. This report presents the full, honest picture for your judgment call.

## Provenance

- **Reference (protected, untouched)**: `results_binary_preserving_refinement/xuad_P1_refined.json` / `_pred.npz`.
- **Prior failed attempt (protected, untouched)**: `results_region_pooled_refinement/` — non-detached region-pool, real/significant regression (McNemar p<0.0001).
- **This run**: `results_region_pooled_detached/` — identical dataset, split, seed=0, backbone, optimizer, `focal_gamma=2.0`, `bce=0.5`, `pcp=0.15`, `epochs=12` (ran to completion, no kill/resume needed), `select_by=acc_f1_acer_guard`. The only change from the failed run: `loc.detach()` before it enters `RegionPool`, exactly as reviewed and approved — verified by direct gradient tests before training that classification loss cannot reach `self.loc`'s parameters while the localisation loss still can.

## Independent verification

`argmax(prob)` vs. `y` recomputed directly from `xuad_P1_regionpool_detached_pred.npz`: **exact match** to the saved JSON (94.58333% accuracy, 94.59246 macro-F1, both ways). Test set confirmed byte-identical to the reference's (n=9600).

## Summary metrics: reference vs. this run vs. the prior failed (non-detached) run

| Metric | Reference | Detached (this run) | Change vs. ref | Failed non-detached run |
|---|---|---|---|---|
| **Accuracy** | 94.292 | **94.583** | **+0.291 pp (better, not significant)** | 93.042 (significant regression) |
| **Macro-F1** | 94.293 | **94.592** | +0.299 (better) | 93.061 |
| AUC | 99.987 | 99.983 | ~0 | 99.984 |
| **ACER** | 0.750 | **0.583** | −0.167 (better) | 0.708 |
| **APCER** | 1.000 | **0.667** | −0.333 (better) | 1.000 |
| BPCER | 0.500 | 0.500 | unchanged | 0.417 |
| **EER** | 0.500 | **0.417** | −0.083 (better) | 0.429 |
| ECE | 4.635 | 3.809 | better | 4.852 |
| Brier | 0.092 | 0.086 | better | 0.109 |
| **loc_IoU** | 0.884 | **0.889** | **better (confirms the fix worked)** | 0.868 (degraded) |
| **loc_Dice** | 0.931 | **0.935** | **better** | 0.920 (degraded) |
| loc_pixel_AUC | 0.967 | 0.969 | better | 0.945 (degraded) |

## Bonafide precision/recall/F1 — the one metric that regressed

| | Reference | Detached (this run) |
|---|---|---|
| Precision | 0.9925 | 0.9900 (slightly worse) |
| **Recall** | 0.9908 | **0.9875 (worse — 4 more bonafide samples misclassified as attack, 11→15)** |
| F1 | 0.9917 | 0.9887 (worse) |

This is a small but real regression against your explicit "preserve or improve bonafide recall" requirement.

## Total errors

Reference: 548. This run: **520 (34 fewer)**. Failed non-detached run: 668 (120 more).

## Full confusion matrix (this run, row=true, col=pred; order: bonafide, print, replay, mask3d, faceswap, reenact, synth, adversarial)

```
             bonafi   print  replay  mask3d  facesw  reenac   synth  advers   SUM
bonafide       1185       1       4       2       6       0       1       1   1200
print              2    1136      18       0      32      10       0       2   1200
replay              3      51    1109       3      27       5       2       0   1200
mask3d              1       0       1    1154       5       2      37       0   1200
faceswap            1      35      15       3    1078      66       2       0   1200
reenact             4      11       5       0      83    1097       0       0   1200
synth               1       0       8      64       2       3    1122       0   1200
adversarial          0       1       0       0       0       0       0    1199   1200
```

## The four target confusion pairs: mixed, not uniformly improved

| Pair | Reference | This run | Change |
|---|---|---|---|
| faceswap → reenact | 73 | **66** | **−7 (better)** |
| reenact → faceswap | 78 | 83 | +5 (worse) |
| mask3d → synth | 31 | 37 | +6 (worse) |
| synth → mask3d | 84 | **64** | **−20 (much better)** |
| replay → print | 53 | 51 | −2 (~same) |
| print → replay | 29 | 18 | **−11 (better)** |
| faceswap → print | 50 | 35 | **−15 (better)** |
| print → faceswap | 37 | 32 | −5 (better) |

Net across the two originally-targeted pairs (faceswap↔reenact, mask3d↔synth): −16 errors, driven almost entirely by `synth→mask3d` (−20). The direction the region-pool mechanism was specifically designed to fix — `faceswap↔reenact` — is itself a wash (−7 one way, +5 the other, net −2), not the clear win the architecture investigation predicted. Most of this run's improvement over the reference comes from `faceswap↔print`, `print↔replay`, and `synth→mask3d` rather than from `faceswap↔reenact` specifically.

## Statistical comparison

McNemar's test on the identical 9,600-sample test set: **251 reference-right→this-run-wrong vs. 279 reference-wrong→this-run-right** (530 discordant). χ²=1.3755, **p=0.2409** — not significant. The accuracy/macro-F1 improvement, while real in this one run, is **not distinguishable from noise** at conventional significance levels. This is the same statistical situation as the two loss-based experiments (four-pair, per-pair-weighted) that were reported as not meeting the bar for the same reason.

## Verdict against your stated success criteria

| Criterion | Required | Result |
|---|---|---|
| Meaningful improvement in accuracy | Yes | Numerically yes (+0.29pp), but **not statistically significant** |
| Macro-F1 preserved/improved | Yes | ✅ improved |
| ACER preserved/improved | Yes | ✅ improved |
| APCER preserved/improved | Yes | ✅ improved |
| BPCER preserved/improved | Yes | ✅ unchanged |
| EER preserved/improved | Yes | ✅ improved |
| **Bonafide recall preserved/improved** | Yes | ❌ **regressed** (0.9908→0.9875) |
| loc IoU/Dice preserved/improved | Yes | ✅ improved |

**Seven of eight criteria are satisfied; one (bonafide recall) is not, and the headline accuracy gain lacks statistical significance.** This is not the clean, unambiguous success your protocol requires to adopt a new primary model, but it is also not the clear failure the non-detached run was. I am reporting it exactly as this — a mixed result — rather than rounding it to either "success" or "failure."

## Conclusion and next step

**Per your instruction, I am not launching anything further automatically**, and I am **not** proposing this as a replacement for `results_binary_preserving_refinement/`, which remains the protected primary model. The stop-gradient fix is mechanically validated (localisation quality now improves rather than degrades, confirming last run's root-cause diagnosis), but the net classification result is a genuine mixed bag — some real gains (loc quality, ACER, EER, several confusion pairs), one real regression (bonafide recall), and a headline accuracy number that cannot yet be called more than noise. This is a judgment call for you: whether the bonafide-recall trade-off and the lack of statistical significance are acceptable given the other improvements, or whether this should be treated as a fourth non-adopted experiment like the three before it.

## Files

| File | Role |
|---|---|
| `results_binary_preserving_refinement/` | reference model (unmodified, checksummed) |
| `results_region_pooled_refinement/` | prior failed (non-detached) attempt (unmodified, checksummed) |
| `results_region_pooled_detached/xuad_P1_regionpool_detached.json/.pt/_hist.json/_pred.npz` | this run's outputs |
| `results_region_pooled_detached/xuad_P1_regionpool_detached_epoch.pt` | rolling checkpoint (all 12 epochs, no kill/resume needed) |
| `results_region_pooled_detached/xuad_P1_regionpool_detached_report.md` | this document |
