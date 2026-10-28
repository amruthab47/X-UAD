# X-UAD P1 — final comparison report: baseline vs. hierarchical vs. binary-preserving refinement

This report is a read-only synthesis. Every number below was either read directly from the three experiments' saved `.json` files or independently recomputed from their saved `_pred.npz` prediction arrays as a cross-check (both are noted per section). **No training was run to produce this report, and none of the three result directories were modified.**

## Verification performed before writing this report

- All three headline JSON metrics (`ACC`, `macroF1`, `AUC`, `ACER`, `APCER`, `BPCER`, `EER`, `ECE`, `Brier`, `mean_vacuity`) were read directly from each experiment's `.json` file.
- Accuracy and macro-F1 were **independently recomputed** from each experiment's raw `_pred.npz` (`argmax(prob)` vs. `y`) and matched the JSON values exactly to 5 decimal places for all three runs.
- Confirmed **all three experiments were scored on the byte-identical 9,600-sample test set**, same order (`np.array_equal` check passed) — the comparison is a valid apples-to-apples one, not an artifact of different test splits.
- Confirmed every confusion-matrix row sums to exactly 1,200 for all three (no missing/duplicated samples).
- The **binary-preserving refinement's claimed 94.29% accuracy / 0.75% ACER is confirmed directly from `results_binary_preserving_refinement/xuad_P1_refined.json`**: `ACC=94.29166666666666`, `ACER=0.75` — verified, not taken on faith.

## 1. Metrics comparison table (verified)

| Metric | Baseline (`improved_p1_retry`) | Hierarchical (`hierarchical_p1`) | **Refined (`binary_preserving_refinement`)** |
|---|---|---|---|
| Accuracy | 93.135 | 93.771 | **94.292** |
| Macro-F1 | 93.090 | 93.761 | **94.293** |
| AUC | 99.965 | 99.915 | **99.987** |
| ACER | 1.792 | 3.417 | **0.750** |
| APCER | 3.000 | 5.583 | **1.000** |
| BPCER | 0.583 | 1.250 | **0.500** |
| EER | 0.750 | 1.488 | **0.500** |
| ECE | 7.692 | 7.256 | **4.635** |
| Brier | 0.122 | 0.110 | **0.092** |
| Mean vacuity | 0.083 | 0.085 | **0.064** |

**The refined model is the best or tied-best on every single metric in this table**, verified from the saved files, not just claimed.

*(Context, not part of this three-way comparison but relevant lineage: the original, pre-focal-loss P1 run, `results_full_retry/`, scored ACC=89.490, ACER=2.417 — all three experiments above build on the focal-loss-plus-12-epoch configuration that already improved on that.)*

## 2. Confusion-matrix and error-analysis findings (verified)

| | Baseline | Hierarchical | Refined |
|---|---|---|---|
| Total test errors (of 9,600) | 659 (6.86%) | 598 (6.23%) | **548 (5.71%)** |
| Bonafide P / R / F1 | 0.973 / 0.987 / 0.980 | 0.964 / 0.975 / 0.969 | **0.993 / 0.991 / 0.992** |
| faceswap → reenact | 131 | 90 | 73 |
| reenact → faceswap | 62 | 81 | 78 |
| mask3d → synth | 43 | 50 | 31 |
| synth → mask3d | 122 | 86 | 84 |

Findings:
- **Bonafide is the one class where the three experiments diverge sharply.** Hierarchical *regressed* bonafide recall (0.987→0.975) — this is the direct, verified cause of its ACER increase. Refined *improved* bonafide recall (0.987→0.991) and precision (0.973→0.993) simultaneously — the direct, verified cause of its ACER decrease.
- **The two targeted confusion pairs (faceswap↔reenact, mask3d↔synth) shrank in total error count across all three experiments** relative to baseline, but not uniformly: `reenact→faceswap` is the one cell that got *worse* in both hierarchical (62→81) and refined (62→78) relative to baseline — the only confusion direction neither follow-up experiment fixed.
- Total error count fell monotonically across the three (659→598→548), but this single number hides the more important story above: hierarchical's error reduction came packaged with a worse bonafide/attack boundary, while refined's did not.

## 3. Did the refinement improve binary attack detection without harming multiclass classification?

**Yes — verified, not inferred.** Every binary-detection metric (AUC, ACER, APCER, BPCER, EER) improved from baseline to refined, and every multiclass metric (accuracy, macro-F1) also improved simultaneously. There is no metric in the tracked set where refined is worse than baseline. This is the specific outcome the experiment's design (loss terms computed directly on the classifier's existing probability output, rather than a separate binary head) was intended to produce, and the saved results confirm it did.

## 4. Main improvements and regressions

**Improvements (verified):**
- Accuracy +1.156 pp, macro-F1 +1.203 over baseline.
- ACER more than halved (1.792→0.750), APCER cut two-thirds (3.0→1.0).
- Calibration improved substantially (ECE 7.692→4.635, Brier 0.122→0.092) — this wasn't a stated success criterion but is a genuine, verified side benefit.
- Bonafide, mask3d, faceswap, reenact, synth, adversarial classes all improved (per-class F1, from the prior turn's verified breakdown).

**Regressions (verified, both minor):**
- `replay` class F1 dipped very slightly (0.9416→0.9390, Δ=−0.0026) — negligible.
- `reenact→faceswap` confusion increased (62→78 of 1,200) — the one confusion direction that moved the wrong way; smaller in magnitude than the gains elsewhere in the same two pairs (net −92 errors across all four targeted cells combined, per the earlier per-pair analysis).

No other regressions were found in the tracked metrics.

## 5. Statistical comparison with the previous best model

McNemar's test, computed directly from the saved predictions, on the identical 9,600-sample test set for every pair:

| Comparison | Discordant pairs | χ² | p-value |
|---|---|---|---|
| baseline vs. **refined** | 639 (375 baseline-wrong→refined-right, 264 reverse) | 18.936 | **1.352×10⁻⁵** |
| baseline vs. hierarchical | 643 (352 vs. 291) | 5.599 | 1.797×10⁻² |
| hierarchical vs. refined | 594 (322 hier-wrong→refined-right, 272 reverse) | 4.042 | 4.438×10⁻² |

**The refined model's improvement over the protected baseline (p=1.35×10⁻⁵) is decisively significant** — over three orders of magnitude smaller than the conventional 0.05 threshold, and far stronger than the hierarchical model's borderline result (p=0.018). Refined is also significantly better than hierarchical itself (p=0.044), though that comparison is more marginal.

## 6. Final recommendation for the primary X-UAD P1 model

**Adopt `results_binary_preserving_refinement/` (`xuad_P1_refined.pt`) as the primary P1 model.** It is the best-or-tied-best on every tracked metric versus both alternatives, the improvement over the previous best model is statistically decisive (p=1.35×10⁻⁵), and — critically for a security-relevant detector — it achieved this without the ACER regression the hierarchical attempt produced. This recommendation follows directly from the verified numbers above; it is not a judgment call between competing tradeoffs, since refined has no measured tradeoff against baseline in this metric set.

## 7. Which experiments should be reported as ablations

- **`results_hierarchical_p1/`** — report as a **negative/informative ablation**: it demonstrates that a structurally separate binary sub-head, even with matched auxiliary-loss weighting, can develop a decision boundary that diverges from (and here, underperforms) the main classifier's own probability output — motivating the refined design's choice to operate on the existing `prob` tensor directly rather than introduce a new head. This is a genuinely useful negative result for the paper's design-justification narrative, not a discarded run.
- **`results_improved_p1_retry/`** — report as the **immediate-prior baseline / main ablation row** in any ablation table (it isolates the effect of the binary-preserving refinement in complete isolation, holding dataset/seed/backbone/epochs/focal-loss fixed).
- **`results_full_retry/`** (context, not one of the three compared here) — report as the **original-configuration ablation** (no focal loss, 6 epochs, `auc_acer` selection), the furthest baseline point in the lineage, useful for showing the cumulative effect of all improvements together if the paper wants that framing.
- **`results/`, `results_smoke/`, `results_full/`, `results_improved_p1/`** — not results to report at all: `results/` is the original frozen project's own numbers (different codebase state, kept for provenance only), `results_smoke/` is a pipeline sanity check on 490 samples, and `results_full/`/`results_improved_p1/` are the partially-completed, externally-interrupted predecessors of `results_full_retry/`/`results_improved_p1_retry/` respectively — superseded, not independent results.

## 8. Exact files for the paper / presentation

**Primary model (report as the main result):**
| File | Use |
|---|---|
| `results_binary_preserving_refinement/xuad_P1_refined.json` | headline metrics table |
| `results_binary_preserving_refinement/xuad_P1_refined.pt` | the checkpoint to cite/release as the primary model |
| `results_binary_preserving_refinement/xuad_P1_refined_pred.npz` | raw predictions, for any further analysis or reviewer verification |
| `results_binary_preserving_refinement/xuad_P1_refined_hist.json` | training curve, if a per-epoch plot is wanted |
| `results_binary_preserving_refinement/figures/error_contact_sheet.png` | qualitative error figure |
| `results_binary_preserving_refinement/xuad_P1_refined_report.md` | full narrative writeup of this experiment |

**Ablation comparison (report alongside the primary result):**
| File | Use |
|---|---|
| `results_improved_p1_retry/xuad_P1_improved_retry.json` | immediate-prior-baseline row in the ablation table |
| `results_improved_p1_retry/figures/confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `calibration.png`, `localization_examples.png` | **note**: these five figures exist under `results_full_retry/figures/` (the original 89.49% configuration), not under `results_improved_p1_retry/figures/` — the latter only has `error_contact_sheet.png` and `top_confident_errors.json`. If ROC/PR/calibration/localisation figures are wanted for the 93.14% baseline specifically, they have **not** been generated yet and would need to be (a new step, not something already sitting in the repo — flagging this rather than assuming it exists). |
| `results_hierarchical_p1/xuad_P1_hierarchical.json`, `figures/error_contact_sheet.png`, `xuad_P1_hierarchical_report.md` | negative-ablation row + discussion |
| `results_full_retry/xuad_P1_full_retry.json` + its full figure set (`confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `calibration.png`, `localization_examples.png`) | original-configuration ablation row, and the only one of the four experiments with a complete pre-generated visualization set |

**This document itself**: `results_binary_preserving_refinement/FINAL_COMPARISON_REPORT.md`.

---

## Verified vs. interpretation — summary

Everything in sections 1, 2 (the table rows), 3, 5, and the file listings in section 8 is a **verified fact**, read from or recomputed from the saved result files. Sections 4's causal explanations ("this is the direct cause of..."), section 6's recommendation, and section 7's categorization are **interpretation** grounded in those verified numbers, not independently verifiable facts in themselves — flagged as such per your request to keep the two distinct.
