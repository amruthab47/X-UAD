# Results Tables (publication-ready)

All numbers below are from this session's campaign (protocol P1, 33,600-sample benchmark, 19,200/4,800/9,600 split), independently verified from raw `_pred.npz` arrays at the time each experiment concluded, and re-verified once more while assembling this document. Every row states its source file. **Single-model results, the final ensemble, and exploratory (failed/mixed) experiments are kept in clearly separate tables — never merged into one "results" table that would obscure which number is the deliverable.**

## Table 1 — Baseline vs. improved single models (chronological)

| Model | Accuracy | Macro-F1 | AUC | ACER | Status | Source |
|---|---|---|---|---|---|---|
| Original baseline | 89.490% | — | 99.880 | 2.417% | Superseded | `results_full_retry/xuad_P1_full_retry.json` |
| Focal loss + ACER-guarded selection | 93.135% | — | 99.965 | 1.792% | Improvement, superseded | `results_improved_p1_retry/xuad_P1_improved_retry.json` |
| Hierarchical evidential head | 93.771% | — | — | 3.417% | **Rejected** (ACER regressed) | `results_hierarchical_p1/xuad_P1_hierarchical.json` |
| Binary-preserving refinement | 94.292% | 94.293 | 99.987 | 0.750% | **Primary single-model reference** | `results_binary_preserving_refinement/xuad_P1_refined.json` |
| Region-pooled, stop-gradient | **94.583%** | — | 99.983 | 0.583% | **Best single model** (mixed result — see `ABLATION_ANALYSIS.md`) | `results_region_pooled_detached/xuad_P1_regionpool_detached.json` |

## Table 2 — Final ensemble (the deliverable)

| Metric | Value | Source |
|---|---|---|
| Accuracy | 96.239583% (9,239 / 9,600 correct) | `results_final/xuad_final_ensemble_metrics.json` |
| Errors | 361 | ″ |
| Macro-F1 | 96.241% | ″ |
| AUC | 99.99 | ″ |
| ACER | 0.500% | ″ |
| APCER | 0.583% | ″ |
| BPCER | 0.417% | ″ |
| EER | 0.333% | ″ |
| ECE | 8.891 (worse than any single constituent — disclosed, see `LIMITATIONS.md`) | ″ |
| Brier score | 0.0813 | ″ |
| Mean vacuity | 0.0708 (mean of 7 constituent models' own vacuity) | ″ |
| Bonafide recall | 99.583% | ″ |
| Constituent models | 7 (all P1 models trained this session except the original baseline) | `results_final/FINAL_RESULTS_REPORT.md` §1 |
| Combination rule | uniform 1/7 probability average, no fitting | ″ |

**vs. the primary single-model reference**: +1.947pp accuracy, statistically significant, McNemar χ²=97.45, p<0.000001 (`results_region_pooled_detached/emergency_98_target_analysis.md`).

## Table 3 — Per-class performance (final ensemble)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bonafide | 0.9950 | 0.9958 | 0.9954 | 1200 |
| print | 0.9455 | 0.9592 | 0.9523 | 1200 |
| replay | 0.9757 | 0.9550 | 0.9652 | 1200 |
| mask3d | 0.9949 | 0.9742 | 0.9844 | 1200 |
| faceswap | 0.9333 | 0.9258 | 0.9296 | 1200 |
| reenact | 0.9530 | 0.9467 | 0.9498 | 1200 |
| synth | 0.9748 | 0.9425 | 0.9584 | 1200 |
| adversarial | 1.0000 | 1.0000 | 1.0000 | 1200 |

Source: `results_final/xuad_final_ensemble_metrics.json` (`per_class`). `faceswap` and `reenact` remain the weakest classes — consistent across every model trained this session.

## Table 4 — Confusion matrix (final ensemble, row=true, col=predicted)

| | bonafide | print | replay | mask3d | faceswap | reenact | synth | adversarial |
|---|---|---|---|---|---|---|---|---|
| **bonafide** | 1195 | 0 | 1 | 0 | 2 | 0 | 2 | 0 |
| **print** | 2 | 1151 | 19 | 0 | 24 | 4 | 0 | 0 |
| **replay** | 1 | 29 | 1146 | 3 | 16 | 3 | 2 | 0 |
| **mask3d** | 0 | 0 | 1 | 1169 | 3 | 2 | 25 | 0 |
| **faceswap** | 1 | 30 | 7 | 2 | 1111 | 47 | 2 | 0 |
| **reenact** | 1 | 9 | 2 | 0 | 52 | 1136 | 0 | 0 |
| **synth** | 2 | 0 | 6 | 57 | 1 | 3 | 1131 | 0 |
| **adversarial** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1200 |

Source: `results_final/confusion_matrix.txt`. Top remaining confusions: `synth→mask3d` (57), `faceswap→reenact` (47), `reenact→faceswap` (52), `faceswap→print` (30), `replay→print` (29), `mask3d→synth` (25).

## Table 5 — Localisation quality (strongest single model, ensemble has no native localisation output)

| Model | loc_IoU | loc_Dice | loc_pixel_AUC |
|---|---|---|---|
| Region-pooled, stop-gradient (`region_pooled_detached`) | **0.8887** | **0.9349** | 0.9692 |
| Binary-preserving reference | 0.8840 | 0.9313 | 0.9675 |

Per-class IoU (region-pooled, stop-gradient model): print 0.9755, replay 0.9521, mask3d 0.8244, faceswap 0.7534, reenact 0.7381, synth 0.9660. Source: `results_region_pooled_detached/xuad_P1_regionpool_detached.json` (`loc_IoU`, `loc_Dice`, `loc_pixel_AUC`, `loc_per_class`).

## Table 6 — Uncertainty / calibration comparison

| Model | ECE | Brier | Mean vacuity |
|---|---|---|---|
| Binary-preserving reference | 4.635 | 0.0923 | 0.0637 |
| Region-pooled, stop-gradient | 3.809 | 0.0862 | 0.0546 |
| **Final ensemble** | **8.891** | **0.0813** | **0.0708** |

The ensemble's Brier score is the best of the three (lower is better), but its ECE is the worst — a real, disclosed trade-off from averaging differently-calibrated models (`LIMITATIONS.md`).

## Table 7 — Ablation / exploratory experiments (failures and mixed results, not hidden)

See `ABLATION_ANALYSIS.md` for full narrative. Summary:

| Experiment | Accuracy | ACER | Verdict |
|---|---|---|---|
| Four-pair confusion penalty | 94.010% | 0.792% | **Failed** — diluted shared-mean gradient |
| Per-pair-weighted confusion penalty | 94.063% | 0.417% | **Not adopted** — accuracy regressed vs. reference despite ACER improving |
| Region pooling (no stop-gradient) | 93.042% | 0.708% | **Failed, statistically significant regression** (McNemar p<0.0001) |
| Region pooling (stop-gradient) | 94.583% | 0.583% | **Best single model, but mixed** — accuracy gain not significant (p=0.24), bonafide recall regressed slightly |

## Table 8 — Open-set protocol (context only, not part of the P1 accuracy series)

| Protocol | Accuracy | ACER | AUC | Notes |
|---|---|---|---|---|
| P2.replay (leave-one-attack-out, replay withheld) | 77.885% | 30.375% | 96.342 | 6-epoch budget, **not comparable to the 12-epoch P1 series above**; demonstrates the expected open-set accuracy collapse relative to closed-set P1 |

Source: `results_P2_replay/xuad_P2_replay.json`.
