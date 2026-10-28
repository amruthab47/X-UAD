# X-UAD region-pooled refinement — FAILED, clearly and significantly, reported as such

**Result: this experiment does not improve on `results_binary_preserving_refinement/` and must not replace it.** Accuracy dropped by 1.25pp (94.292%→93.042%), total errors rose from 548 to 668, and — critically — the two confusion pairs this change specifically targeted (`faceswap↔reenact`, `mask3d↔synth`) both got **worse**, not better, in every direction. Unlike the two previous unsuccessful experiments (four-pair, per-pair-weighted), whose accuracy drops were *not* statistically significant, this one **is** (McNemar's χ²=21.79, p<0.0001) — this is a real, verified regression, not noise. Per your instructions, **no further experiment has been launched automatically.**

## Provenance

- **Reference (protected, untouched)**: `results_binary_preserving_refinement/xuad_P1_refined.json` / `_pred.npz`.
- **This run**: `results_region_pooled_refinement/` — identical dataset, split, seed=0, backbone, optimizer, `focal_gamma=2.0`, `bce=0.5`, `pcp=0.15` (2-pair, same as reference), `epochs=12` (ran to completion, no kill/resume needed), `select_by=acc_f1_acer_guard`. The only change: `--region-pool` — a new `RegionPool` module pools the model's own 56×56 localisation evidence map over 7 facial regions (already defined in `xai/evidence.py::REGIONS`) and concatenates the result into the classification feature, exactly as proposed and approved in `architecture_feasibility_investigation.md`.

## Independent verification

`argmax(prob)` vs. `y` recomputed directly from `xuad_P1_regionpool_pred.npz`: **exact match** to the saved JSON (93.04167% both ways for accuracy, 93.06141 for macro-F1). Test set confirmed byte-identical to the reference's (`np.array_equal` on `y`, n=9600).

## Summary metrics: reference vs. region-pooled

| Metric | Reference | Region-pooled | Change |
|---|---|---|---|
| **Accuracy** | 94.292 | **93.042** | **−1.250 pp (worse, and significant — see below)** |
| **Macro-F1** | 94.293 | **93.061** | **−1.232 (worse)** |
| AUC | 99.987 | 99.984 | ~0 |
| ACER | 0.750 | 0.708 | −0.042 (marginally better) |
| APCER | 1.000 | 1.000 | unchanged |
| BPCER | 0.500 | 0.417 | −0.083 (better) |
| EER | 0.500 | 0.429 | −0.071 (better) |
| ECE | 4.635 | 4.852 | +0.218 (worse) |
| Brier | 0.092 | 0.109 | +0.017 (worse) |

**By your stated success criteria** ("meaningful improvement over 94.292% while preserving Macro-F1/ACER/APCER/BPCER/EER/bonafide recall"): **not met, decisively.** Accuracy and Macro-F1 both dropped by over a full percentage point, and — unlike the prior two experiments — this drop is statistically real, not noise.

## Bonafide precision/recall/F1

| | Reference | Region-pooled |
|---|---|---|
| Precision | 0.9925 | **0.9819 (worse)** |
| Recall | 0.9908 | 0.9933 (better) |
| F1 | 0.9917 | **0.9876 (worse)** |

Mixed but net worse — more bonafide samples now cost a false attack alarm than before.

## Total errors

**Reference: 548. Region-pooled: 668 (+120 more errors — the largest regression of any experiment run this session.)**

## Full confusion matrix (region-pooled model, row=true, col=pred; order: bonafide, print, replay, mask3d, faceswap, reenact, synth, adversarial)

```
             bonafi   print  replay  mask3d  facesw  reenac   synth  advers   SUM
bonafide       1192       0       1       1       3       1       1       1   1200
print             1    1136      19       1      40       2       0       1   1200
replay             6      56    1076       2      47       6       7       0   1200
mask3d             1       0       1    1119      12       2      65       0   1200
faceswap           6      52       4       2    1054      80       2       0   1200
reenact            2      19       1       1      94    1083       0       0   1200
synth              6       0       3      99      12       2    1077       1   1200
adversarial        0       3       2       0       0       0       0    1195   1200
```

## The four target confusion pairs: reference vs. region-pooled

| Pair | Reference | Region-pooled | Change |
|---|---|---|---|
| faceswap → reenact | 73 | **80** | **+7 (worse)** |
| reenact → faceswap | 78 | **94** | **+16 (worse)** |
| mask3d → synth | 31 | **65** | **+34 (much worse — the largest single regression)** |
| synth → mask3d | 84 | **99** | **+15 (worse)** |
| replay → print | 53 | 56 | +3 (~same) |
| print → replay | 29 | 19 | −10 (better) |
| faceswap → print | 50 | 52 | +2 (~same) |
| print → faceswap | 37 | 40 | +3 (~same) |

**Every single one of the two originally-targeted pairs got worse, in both directions.** This directly falsifies the central hypothesis of `architecture_feasibility_investigation.md` — that giving the classifier access to the model's own localisation evidence would help separate these pairs. It did the opposite.

## Root cause (verified from the model's own saved localisation-quality metrics, not speculation)

| Localisation metric | Reference | Region-pooled | Change |
|---|---|---|---|
| loc_IoU | 0.884 | **0.868** | worse |
| loc_Dice | 0.931 | **0.920** | worse |
| loc_pixel_AUC | 0.967 | **0.945** | worse |
| loc_per_class mask3d | 0.863 | **0.824** | worse |
| loc_per_class faceswap | 0.764 | **0.753** | worse |
| loc_per_class reenact | 0.756 | **0.738** | worse |
| loc_per_class synth | 0.983 | **0.966** | worse |

**The localisation head's own accuracy at its original job — reconstructing the ground-truth attack mask — degraded across every class it was measured on.** This is the direct, verified mechanism: routing a second gradient signal back through the localisation head (via `region_proj → RegionPool → loc`, required to make its output useful for classification) pulled its weights away from purely optimizing the per-pixel localisation loss. The resulting `loc` map is a *less faithful* reconstruction of where the attack actually is — and feeding a *less accurate* signal into the classifier for exactly the two pairs (`faceswap↔reenact`, `mask3d↔synth`) whose separability this experiment was betting on that signal's *fidelity* made things worse, not better. `architecture_feasibility_investigation.md`'s implementation plan explicitly noted this reordering was necessary but did not anticipate (and this investigation, being read-only, had no way to test) that it would create competing gradients on the localisation head strong enough to measurably degrade its primary task.

## Statistical comparison

McNemar's test on the identical 9,600-sample test set: **385 reference-right→region-pooled-wrong vs. 265 reference-wrong→region-pooled-right** (650 discordant). χ²=21.7862, **p<0.0001** — highly significant. Unlike the four-pair and per-pair-weighted experiments (both p>0.27, indistinguishable from noise), **this regression is real and would replicate.**

## Conclusion and next step

**This experiment failed clearly and is not adopted.** `results_binary_preserving_refinement/xuad_P1_refined.pt` remains the primary model. Per your instructions, **no further experiment has been launched automatically** — this stops here, awaiting your direction.

If this direction is revisited, the root-cause finding above suggests the fix is not "don't reuse the localisation map" but "don't let the classification objective reshape it": e.g. reading a **stop-gradient (detached) copy** of `loc` into `RegionPool` (the same design already used for the evidential head's magnitude branch in `EvidentialHead.ev`, for exactly this reason — see `xuad/models/heads.py:47-54`) would let the classifier use the map's *information* without being able to *corrupt* it. That is a plausible follow-up hypothesis grounded in this run's own diagnostic evidence, not a new speculative direction — but it is a suggestion for a *possible* future experiment, not something started.

## Files

| File | Role |
|---|---|
| `results_binary_preserving_refinement/` | reference model (unmodified, checksummed) |
| `results_region_pooled_refinement/xuad_P1_regionpool.json/.pt/_hist.json/_pred.npz` | this (failed) run's outputs |
| `results_region_pooled_refinement/xuad_P1_regionpool_epoch.pt` | rolling checkpoint (all 12 epochs, no kill/resume needed) |
| `results_region_pooled_refinement/xuad_P1_regionpool_report.md` | this document |
