# Paper Figures — What Exists, What's Missing

## Figures that already exist and are directly usable

| Figure | Location | Produced this session? | Notes |
|---|---|---|---|
| ROC curve (final ensemble) | `results_final/figures/roc_curve.png` | ✅ yes | Binary attack-vs-bonafide, AUC=99.99. |
| Precision-Recall curve (final ensemble) | `results_final/figures/pr_curve.png` | ✅ yes | Attack detection, bonafide vs. attack. |
| Calibration / reliability diagram (final ensemble) | `results_final/figures/calibration.png` | ✅ yes | Shows the ECE=8.89 trade-off visually. |
| Class-wise Precision/Recall/F1 bar chart (final ensemble) | `results_final/figures/class_wise_performance.png` | ✅ yes | Per `RESULTS_TABLES.md` Table 3. |
| Confusion matrix (numeric, final ensemble) | `results_final/confusion_matrix.txt` | ✅ yes | Raw counts + row-normalized; **not yet rendered as a heatmap image** — see "missing" below. |

## Figures that exist from the earlier campaign — usable only with the campaign clearly labeled

| Figure | Location | Belongs to |
|---|---|---|
| Architecture diagram | `paper/figures/architecture.png` / `.pdf` | Earlier campaign. **Verify before reuse**: this session added the `RegionPool` path (`xuad/models/xuad.py`, `xuad/models/heads.py::RegionPool`) — check whether this figure already depicts it or needs updating. |
| Dataset taxonomy figure | `paper/figures/taxonomy.png` | Earlier campaign. Likely still valid (the 8-class taxonomy itself is unchanged), but verify the sample counts shown match this session's 33,600/19,200/4,800/9,600 figures, not the earlier campaign's 29,400+4,900 composition. |
| ROC/DET curves | `paper/figures/roc_det.png` / `.pdf`, `curves.png` / `.pdf` | Earlier campaign, different dataset build — **do not caption as this session's result.** |
| Confusion matrix (heatmap) | `paper/figures/confusion.png` / `.pdf` | Earlier campaign — numbers will not match `results_final/confusion_matrix.txt`. If a matching heatmap is wanted for this session's ensemble, it must be freshly rendered from `results_final/confusion_matrix.txt` (recommended figure, listed below). |
| Calibration curve | `paper/figures/calibration.png` / `.pdf` | Earlier campaign — do not conflate with `results_final/figures/calibration.png` (different filename, different underlying model). |
| Qualitative examples | `paper/figures/qualitative.png` / `.pdf` | Earlier campaign; unknown whether the specific samples shown are still representative of this session's error patterns — verify before reuse or regenerate. |
| Leave-one-attack-out figure | `paper/figures/loao.png` / `.pdf` | Earlier campaign's P2 series across all 4 held-out families; this session ran only one P2.replay fold — do not present the earlier figure as this session's P2 result. |

## Recommended figures — full list, with existence status

1. **Overall X-UAD architecture diagram** — exists (earlier campaign, `[VERIFY]` for the `RegionPool` addition; regenerate if it's missing that path).
2. **Dataset taxonomy / sample examples** (one crop per class) — exists as a diagram (`paper/figures/taxonomy.png`, `[VERIFY]` counts); **actual example image crops are not yet assembled into a figure** — `[TO GENERATE]` from `data/A_x.npy` (memmapped array) + `data/A_manifest.csv` for correct labeling, remembering the stored array is BGR (convert before rendering, per this session's earlier BGR/RGB finding).
3. **Training/evaluation pipeline diagram** — no dedicated figure found; `[TO GENERATE]`, content specified in `METHODOLOGY.md` §"Pipeline overview" ASCII diagram.
4. **Confusion matrix (final ensemble, heatmap)** — numeric version exists (`results_final/confusion_matrix.txt`); **heatmap image `[TO GENERATE]`** (straightforward — the numbers already exist, just needs a `matplotlib.imshow` render, no new computation).
5. **ROC curve** — exists, this session, `results_final/figures/roc_curve.png`.
6. **PR curve** — exists, this session, `results_final/figures/pr_curve.png`.
7. **Calibration curve** — exists, this session, `results_final/figures/calibration.png`.
8. **Class-wise performance** — exists, this session, `results_final/figures/class_wise_performance.png`.
9. **Localisation examples** (predicted map vs. ground-truth mask, side by side) — **`[TO GENERATE]`**. Would require loading `results_region_pooled_detached/xuad_P1_regionpool_detached.pt`, running inference on a few test samples per class (especially `faceswap`/`reenact`, the hardest-to-localise per `RESULTS_TABLES.md` Table 5), and rendering predicted `loc` maps next to the ground-truth masks from `data/A_m.npy`. Not fabricated — must be generated from the actual checkpoint and actual stored masks.
10. **Uncertainty examples** (a few samples with their vacuity/confidence and the resulting accept/reject/refer decision) — **`[TO GENERATE]`**, from `xuad/xai/reasoning.py::build_record` run on real test samples plus the saved `vacuity` array in `results_final/xuad_final_ensemble_pred.npz`.
11. **Error analysis examples** (misclassified samples with true/predicted labels) — **`[TO GENERATE]`**, sample indices already identified in `results_binary_preserving_refinement/error_and_data_quality_diagnosis.md` §2 ("top-20 highest-confidence errors") and reproducible via `results_final/xuad_final_ensemble_pred.npz`'s `pred`/`y` arrays cross-referenced with `data/A_manifest.csv`.
12. **Progression bar chart** (baseline 89.49% → final ensemble 96.24%) — **`[TO GENERATE]`**, trivial from `RESULTS_TABLES.md` Table 1 + 2, no new computation needed.

## Explicit statement on fabrication

No plot, image, or metric in this document or in any file within `results_final/` was invented. Every figure marked "exists" was verified present on disk during this task; every figure marked `[TO GENERATE]` is flagged because generating it (correctly, from real checkpoints/arrays) was outside the scope of this documentation task, not because a fabricated placeholder was produced instead.
