# X-UAD primary model (`xuad_P1_refined.pt`) — detailed error and data-quality diagnosis

Read-only analysis of `results_binary_preserving_refinement/xuad_P1_refined_pred.npz` (9,600 test samples) plus direct inspection of the dataset splits and training history. **No training, dataset modification, or file overwrite was performed.** All figures below are computed directly from saved files; interpretive claims are labeled as such.

Current state: **94.292% test accuracy (548 errors), 0.750% ACER.** Target: ~98–99%.

## 1. Exact error counts and percentages, every confusion pair

| True → Predicted | Count | % of all 548 errors | % of that true class (1,200) |
|---|---|---|---|
| synth → mask3d | 84 | 15.33% | 7.00% |
| reenact → faceswap | 78 | 14.23% | 6.50% |
| faceswap → reenact | 73 | 13.32% | 6.08% |
| replay → print | 53 | 9.67% | 4.42% |
| **faceswap → print** | **50** | **9.12%** | **4.17%** |
| print → faceswap | 37 | 6.75% | 3.08% |
| mask3d → synth | 31 | 5.66% | 2.58% |
| print → replay | 29 | 5.29% | 2.42% |
| replay → faceswap | 24 | 4.38% | 2.00% |
| faceswap → replay | 15 | 2.74% | 1.25% |
| all remaining 21 pairs | 74 | 13.5% | — |

**New finding not previously flagged**: `faceswap ↔ print` (50+37=87 errors, 15.9% of all errors) is now a cluster comparable in size to `mask3d↔synth`, and larger than several previously-tracked pairs. This wasn't part of the original two "dominant pairs" and should be included in any further targeted work.

### Bonafide-specific breakdown (items 3–4 of your request)

| | Count | Rate | % of all errors |
|---|---|---|---|
| Bonafide misclassified as attack | 11 / 1,200 | 0.92% | 2.01% |
| Attack misclassified as bonafide | 9 / 8,400 | 0.107% | 1.64% |

Both are now small in absolute terms (20 of 548 errors combined, 3.65%) — the refinement's binary-preserving design achieved its goal; **the bonafide/attack boundary is not where the remaining accuracy gap lives**. The remaining 96.35% of errors are entirely inter-attack-family confusions.

## 2. Example sample IDs (item 2)

Full top-20 highest-confidence errors saved to `figures/top20_high_confidence_errors.json`. Selected examples:

| Row | Identity | True | Predicted | Confidence | Vacuity |
|---|---|---|---|---|---|
| 7412 | Shafal_Mosed | synth | mask3d | 0.991 | 0.052 |
| 2036 | Demi_Moore | synth | mask3d | 0.988 | 0.048 |
| 4407 | John_Negroponte | faceswap | reenact | 0.985 | 0.037 |
| 592 | Bernice_Wong | faceswap | reenact | 0.972 | 0.063 |
| 1758 | Colin_Powell | print | reenact | 0.929 | 0.038 |
| 6120 | Nick_Nolte | replay | print | 0.926 | 0.044 |
| 7572 | Steven_Spielberg | reenact | faceswap | 0.921 | 0.098 |

(Rows are indices into the test split's iteration order, reproducible via `XUADDataset(root="data", domain="A", split="test", ...)` — the same construction used throughout this project. "Identity" is the LFW source-person string, not a file path, since the benchmark stores samples as memory-mapped arrays rather than individual image files.)

## 3. Confidence and vacuity for incorrect predictions (item 3)

| Error group | n | mean confidence | mean vacuity | conf > 0.9 | conf < 0.6 |
|---|---|---|---|---|---|
| faceswap ↔ reenact | 151 | 0.654 | 0.147 | 6.6% | 38.4% |
| synth ↔ mask3d | 115 | 0.692 | 0.122 | 10.4% | 28.7% |
| bonafide → attack | 11 | 0.574 | 0.172 | 0.0% | 63.6% |
| attack → bonafide | 9 | 0.652 | 0.142 | 0.0% | 44.4% |
| **all 548 errors** | 548 | 0.602 | 0.161 | 4.6% | 55.3% |
| *(for reference) correct predictions* | 9,052 | 0.915 | 0.058 | — | — |

**The uncertainty channel is behaving sensibly**: correct predictions are confident (0.915) and low-vacuity (0.058); errors are markedly less confident (0.602) and higher-vacuity (0.161), and the two bonafide-boundary error groups have zero high-confidence mistakes at all. Only 4.6% of all errors (25 samples) are high-confidence (>90%) failures — these are the ones most worth treating as genuine model/data weaknesses rather than defensible uncertainty; the synth↔mask3d pair contributes disproportionately to this small high-confidence-error set (10.4% vs. 6.6% for faceswap/reenact), suggesting synth/mask3d confusion is the more "confidently wrong" (harder to fix via better calibration alone) of the two dominant pairs.

## 4. Cause diagnosis

| Candidate cause | Verdict | Evidence |
|---|---|---|
| **Weak visual differences** | **Supported** | Confidence stratification above (mean conf on errors 0.602 vs. 0.915 on correct) and the previously-generated contact sheets (`results_improved_p1_retry/figures/error_contact_sheet.png`, `results_hierarchical_p1/figures/error_contact_sheet.png`) show faceswap/reenact and synth/mask3d pairs sharing visually similar artifacts at this crop/resolution. |
| **Synthetic generation artifacts** | **Supported** | Prior contact-sheet inspection found mask3d and synth sharing a similar coarse bluish color-cast — a synthesis-pipeline-level correlate, not a class-defining feature. This is a data-generation characteristic, not a model failure. |
| **Class imbalance** | **Ruled out** | Directly re-verified: all 8 classes have exactly 1,200 test samples. No imbalance exists to explain any error pattern. |
| **Label noise** | **Ruled out by construction** | Labels are assigned deterministically by which synthesizer function generated each sample (`xuad/data/attacks.py`) — there is no human annotation step and therefore no traditional mislabeling risk. This is a structural guarantee, not a statistical inference. |
| **Data leakage / split issues** | **Ruled out** | Directly re-verified this session: zero identity overlap between train/val/test in every pairwise direction, confirmed on the live dataset (not from memory of an earlier check). |
| **Preprocessing problems** | **Unlikely primary cause** | Test-time preprocessing uses a deterministic **center** crop (not the random crop used only during training augmentation), so crop-position noise cannot explain test-set errors. A minor, unconfirmed possibility: the fixed 128→112 center crop could still clip peripheral evidence for boundary-localized attacks (reenact's mouth/eye region, faceswap's inner-face boundary) for faces that aren't well-centered in their original crop — flagged as a hypothesis, not confirmed. |
| **Insufficient image resolution** | **Plausible partial contributor**, pair-dependent | Plausible for faceswap/reenact (fine blending-boundary detail may not survive 112px). Less plausible for synth/mask3d, whose shared confusion is a coarse, whole-image color/texture correlate that resolution is less likely to disambiguate. |
| **Model capacity limitations** | **Inconclusive, leaning "not primary"** | Training-loss trajectory (`xuad_P1_refined_hist.json`) shows train classification loss still falling at epoch 11 (0.2169→0.2033) while validation accuracy has essentially plateaued (94.85%→94.98%, epochs 10→11). This pattern is more consistent with an emerging generalization/separability ceiling than with the 1.29M-parameter model being too small to fit the training distribution — but this is an interpretation, not a proof; a capacity ablation (a wider/deeper variant) is the only way to confirm it directly, and none has been run. |

**Additional finding, not one of the listed candidates but directly relevant**: per-identity error clustering initially looked alarming (109 of 679 test identities account for 64.2% of all errors) — but a direct correlation check found this is **not** evidence of "hard identities": error rate per sample is statistically flat regardless of how many source photos an identity contributes to the test set (r=−0.003; identities with 1 photo have a 5.68% error rate, identities with ≥5 photos have 5.95% — effectively identical). The apparent clustering is simply because LFW identities have wildly uneven photo counts (up to 87 photos for one person), so a person with many photos mechanically accumulates more error *instances* at the same underlying *rate* — not a sign of systematically harder faces.

## 5. Is 98–99% realistically achievable on this dataset/setup?

**Conditionally yes, but only with a coordinated multi-pronged effort — not from any single change, and 99% specifically is a stretch even then.**

Ceiling analysis, computed directly from the current error distribution (cumulative, in order of cluster size):

| Scenario | Errors resolved | Resulting accuracy |
|---|---|---|
| Current | 0 | 94.292% |
| + faceswap↔reenact fully resolved | 151 | 95.865% |
| + synth↔mask3d fully resolved | 266 | 97.062% |
| + replay↔print fully resolved | 348 | 97.917% |
| **+ faceswap↔print fully resolved (all 4 dominant clusters)** | **435** | **98.823%** |
| All remaining errors (long tail, 113 errors across 21+ smaller pairs) also resolved | 548 | 100.000% |

**Resolving all four dominant confusion clusters essentially completely reaches 98.8%** — inside your target range. But this requires *near-perfect* resolution of four separate, only-partially-related confusion patterns simultaneously, which is a substantially higher bar than the partial improvement the pair-penalty already achieved on two of them (it reduced but did not eliminate faceswap↔reenact and synth↔mask3d). Reaching 99% specifically requires also cleaning up the long tail of 21+ smaller pairs (113 more errors, an 82.5% cut in total errors from where the model stands now) — a materially higher bar than 98%.

## 6. Three highest-impact changes (ranked)

1. **Extend the existing targeted pair-penalty to all four dominant clusters**, not just the two used in the refinement experiment (`faceswap↔reenact`, `synth↔mask3d`). Adding `replay↔print` and `faceswap↔print` to the `PAIR_PENALTY_MAP` is a direct, low-risk, low-cost extension of a mechanism already proven to work (net −92 errors across its two original targets) — highest expected-benefit-per-unit-cost of any option, since it reuses working machinery rather than introducing something new.
2. **Higher input resolution** (112→~160px, using more of the already-stored 128px source, or regenerating at higher native resolution). Directly targets the plausible fine-detail bottleneck for faceswap/reenact specifically; moderate cost (more compute per step, some risk of needing to retune batch size/LR), uncertain benefit for the synth/mask3d pair specifically since that confusion looks coarse-artifact-driven rather than fine-detail-driven.
3. **Supervised contrastive / metric-learning loss** between the four confusable-pair classes, pulling same-class features together and pushing confusable-pair features apart in representation space — a deeper intervention than a probability-level penalty. Higher cost (a genuinely new loss component requiring its own tuning) and higher risk (more can go wrong, as the hierarchical-head experiment demonstrated), but addresses the *representation*, not just the *output*, which the current pair-penalty doesn't do.

*(Not recommending frequency-domain feature changes as a top-3 pick: the model already has a dedicated spectral branch, and the dominant synth/mask3d confusion looks more like a shared spatial color-cast than a spectral-signature overlap — no direct evidence points there yet.)*

## 7. Estimated upper bound

**~97–98% is a realistic, evidence-grounded ceiling for a well-executed combination of the above changes; 99% is achievable only if essentially every confusion pair in the long tail is also resolved, which is a much higher bar than anything demonstrated so far.** This estimate is interpretation grounded in the verified cluster-resolution table above, not a guarantee — resolving a confusion cluster "on paper" (subtracting its error count) assumes the intervention achieves *complete* separation, which no experiment run so far has done for any pair (the pair-penalty reduced but never eliminated its two targets).

## 8. Concrete experiment plan, ranked by expected benefit vs. computational cost

| Rank | Experiment | Expected benefit | Cost | Risk |
|---|---|---|---|---|
| 1 | Extend `pcp` pair-penalty to all 4 dominant clusters (same mechanism as the successful refinement, just a larger `PAIR_PENALTY_MAP`) | Medium-high (plausibly +1–2 pp, based on the ~92-error reduction the 2-pair version already achieved, scaled to twice the pairs) | Low (one dict edit, same 12-epoch/~50min run, no new failure mode) | Low — same design that already worked cleanly |
| 2 | Higher input resolution (start with 128px, the native store size, before considering true upscaling) | Medium, pair-dependent (likely helps faceswap/reenact more than synth/mask3d) | Medium (more compute/epoch, possible batch-size/LR retune) | Low-medium |
| 3 | Supervised contrastive loss between the 4 confusable classes | Medium-high if it works, but uncertain | High (new component, needs its own tuning pass, likely needs its own smoke test + possibly its own controlled ablation before a full run) | Medium — representation-level changes are less predictable, as the hierarchical-head experiment showed |
| 4 (only if 1–3 plateau) | Capacity ablation (modestly wider/deeper backbone) to directly test the "model capacity" hypothesis this diagnosis left inconclusive | Low-medium, and mainly *diagnostic* (tells you whether capacity is the bottleneck) rather than a guaranteed accuracy win | Medium (larger model, longer training) | Low |

Recommendation: run rank 1 next as a single, isolated, low-risk continuation of the currently-working design; only escalate to ranks 2–3 if it plateaus below the 97–98% ceiling estimated above.

## Verified facts vs. interpretation — summary

**Verified** (read directly from saved files or freshly recomputed): all counts/percentages in sections 1–3, the class-balance check, the split-disjointness check, the training-loss trajectory numbers, and the per-identity correlation check.

**Interpretation** (reasoned from the verified facts, not independently provable): the cause attributions in section 4 (except imbalance/label-noise/leakage, which are hard rule-outs), the achievability judgment in section 5, the three ranked changes in section 6, the upper-bound estimate in section 7, and the risk/benefit ratings in section 8.

## Files

| File | Role |
|---|---|
| `results_binary_preserving_refinement/xuad_P1_refined.pt` | model inspected (inference/read-only) |
| `results_binary_preserving_refinement/xuad_P1_refined_pred.npz` | source of all counts in this report |
| `results_binary_preserving_refinement/xuad_P1_refined_hist.json` | source of the training-trajectory analysis |
| `results_binary_preserving_refinement/figures/top20_high_confidence_errors.json` | full sample-ID list for the highest-confidence errors |
| `results_binary_preserving_refinement/error_and_data_quality_diagnosis.md` | this document |
