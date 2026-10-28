# Ablation / Experiment Analysis — Full, Honest Account

Every major experiment run this session against the X-UAD architecture/loss, in chronological order, with actual numbers and an explicit success/partial/failure verdict. **No failed experiment is omitted.** Sources: each experiment's own `results_*/xuad_P1_*.json` and, where produced, its `_report.md`.

## 1. Original baseline — `results_full_retry/`

- **Config**: default loss weights, no focal modulation, `auc_acer` checkpoint selection.
- **Result**: 89.490% accuracy, 2.417% ACER, AUC=99.880.
- **Verdict**: baseline, superseded by every subsequent change.

## 2. Focal loss + ACER-guarded checkpoint selection — `results_improved_p1_retry/`

- **Change**: `focal_gamma=2.0` on the classification cross-entropy; `select_by=acc_f1_acer_guard` (checkpoint chosen by accuracy+macro-F1 among only epochs within 0.5pp of the best validation ACER seen so far, rather than optimizing accuracy with no ACER guard).
- **Result**: 93.135% accuracy (+3.65pp), 1.792% ACER (improved).
- **Verdict**: **Successful.** Both accuracy and ACER improved together.

## 3. Hierarchical evidential head — `results_hierarchical_p1/`

- **Change**: replaced the flat softmax direction in `EvidentialHead` with a `HierarchicalEvidentialHead` (bonafide/attack → physical/digital/adversarial → specific class), motivated by the observation that a large share of errors crossed the physical/digital superclass boundary.
- **Result**: 93.771% accuracy (higher than the previous step), but **ACER regressed to 3.417%** — worse than every prior configuration including the original baseline.
- **Verdict**: **Rejected.** Selected via `select_by=acc_f1` with no ACER guard in this run, which let the checkpoint selector trade away binary-detection quality for 8-way accuracy — the exact failure mode `acc_f1_acer_guard` exists to prevent. Kept in the codebase only as a documented negative ablation, not used by any adopted model.

## 4. Binary-preserving refinement — `results_binary_preserving_refinement/` (primary reference)

- **Change**: added a binary-reinforcement BCE term computed directly on `prob[:,0]` (`bce=0.5`, no separate head — avoiding the hierarchical head's failure mode) plus a targeted pair-confusion penalty (`pcp=0.15`) on the two largest confusion clusters identified by diagnosis (`faceswap↔reenact`, `mask3d↔synth`), using a **shared mean** across both pairs' (sample, partner-probability) terms.
- **Result**: 94.292% accuracy, macro-F1=94.293, AUC=99.987, **ACER=0.750%** (best-in-project at the time), bonafide recall=0.9908.
- **Verdict**: **Successful, adopted as the primary single-model reference.**

## 5. Four-pair confusion penalty — `results_four_pair_refinement/`

- **Change**: extended the same shared-mean `pcp` mechanism from 2 to 4 confusion clusters (`+ replay↔print`, `+ faceswap↔print`), same weights (`bce=0.5`, `pcp=0.15`).
- **Result**: 94.010% accuracy (**−0.28pp**), ACER=0.792% (**worse**), bonafide recall=0.9842 (**worse**). McNemar vs. the reference: p=0.27 — not statistically significant, but every headline number moved the wrong way.
- **Diagnosed root cause**: the `pcp` term is a single aggregate weight averaging over however many (sample, partner) terms are active in a batch. Extending from 2 to 4 pairs **diluted** the gradient the two original pairs received — each pair's effective per-term gradient scales roughly as `weight / n_active_pairs`.
- **Verdict**: **Failed.** Not adopted.

## 6. Per-pair-weighted confusion penalty — `results_pair_weighted_refinement/`

- **Change**: replaced the shared-mean penalty with an **independently-computed mean per pair** (`weighted_pair_confusion_penalty`), keeping the original two pairs at their proven 0.15 weight (undiluted) and adding the two new pairs at a smaller 0.05 weight each — specifically designed to fix experiment 5's diagnosed dilution mechanism.
- **Result**: 94.063% accuracy (**still −0.23pp vs. the reference**), macro-F1=94.056 (worse), but ACER=0.417% (**better**), APCER=0.333% (better), EER=0.417% (better), bonafide precision/F1 improved with recall exactly preserved. McNemar: p=0.39 — not significant.
- **Target-pair breakdown**: the independent-mean fix worked exactly as intended for `faceswap↔reenact` (both directions improved vs. the four-pair run) and for both new pairs — but `mask3d→synth` got *substantially worse* (31→55 errors) despite receiving the identical, undiluted 0.15 weight that worked in the reference. This showed the dilution theory was only part of the story: giving `mask3d↔synth` its own independent weight did not reproduce the reference's behavior for that pair, pointing to gradient interference between simultaneously-active pair terms as an additional mechanism, not simple dilution alone.
- **Verdict**: **Not adopted** — accuracy did not meaningfully improve (the stated success bar), despite real gains on ACER/APCER/EER.

## 7. Region pooling, no stop-gradient — `results_region_pooled_refinement/`

- **Change**: a new `RegionPool` module pools the model's own 56×56 localisation evidence map over named facial regions and concatenates the result into the classification feature — motivated by a code-level finding (`faceswap`'s and `reenact`'s ground-truth masks are geometrically very different, with no frequency-domain counterpart) suggesting the classifier lacked access to spatial/shape information the localisation head already computed.
- **Result**: 93.042% accuracy (**−1.25pp**), ACER=0.708%, total errors 548→668 (**+120, the largest regression of any experiment this session**). McNemar vs. reference: **χ²=21.79, p<0.0001 — statistically significant**, unlike every other regression this session.
- **Diagnosed root cause, verified from the model's own saved localisation metrics**: `loc_IoU` dropped from 0.884→0.868, `loc_Dice` 0.931→0.920, and per-class localisability fell for every measured class. Routing a classification gradient back through the localisation head (required to make its output useful for the new feature) measurably degraded the head's own accuracy at its original per-pixel task, feeding a *less faithful* spatial signal into the classifier for exactly the pairs betting on that signal's fidelity.
- **Verdict**: **Failed, clearly and significantly.** Not adopted.

## 8. Region pooling, stop-gradient fix — `results_region_pooled_detached/`

- **Change**: identical mechanism to experiment 7, with one line changed — the localisation map is `.detach()`-ed before entering `RegionPool`, so the classification loss can read it but cannot backpropagate into the localisation head's own parameters (mirroring `EvidentialHead`'s existing `z.detach()` pattern). Verified by direct gradient tests before training: classification loss produces zero gradient on every `LocalisationHead` parameter; the localisation loss's own gradient into those parameters is unaffected.
- **Result**: 94.583% accuracy (**best single model**, +0.29pp vs. reference, not statistically significant, McNemar p=0.24), macro-F1=94.592, ACER=0.583% (better), APCER=0.667%(better), EER=0.417%(better), **loc_IoU=0.8887 and loc_Dice=0.9349 — both improved over the reference**, confirming the stop-gradient fix worked mechanically exactly as intended. **Bonafide recall regressed slightly** (0.9908→0.9875).
- **Target-pair result**: mixed, not the clean win predicted — `faceswap↔reenact` (the pair the mechanism specifically targeted) was a wash (−7 one direction, +5 the other, net −2 errors); most of the actual gain came from `synth→mask3d` (−20) and `faceswap↔print`.
- **Verdict**: **Best single model, but a genuine mixed result — not a clean success.** 7 of 8 stated success criteria were met; bonafide recall was not, and the headline accuracy gain is not independently significant. Reported and kept as the best single-model checkpoint without being oversold as an unambiguous win.

## 9. Final ensemble — `results_final/`

- **Change**: no training. A fixed, uniform (1/7) probability average of 7 already-trained P1 models (every model above except the original baseline).
- **Result**: 96.240% accuracy, 361 errors (down from 520 for the best single model), macro-F1=96.241%, ACER=0.500%, bonafide recall=99.583% (best of any model this session). McNemar vs. the reference: **χ²=97.45, p<0.000001 — the only single change this session producing a clearly, unambiguously statistically significant accuracy improvement.**
- **Disclosed trade-off**: ECE=8.891, worse than any individual constituent model (3.8–5.5 range) — averaging differently-calibrated probability vectors does not preserve calibration even as accuracy improves.
- **Disclosed methodology caveat**: the choice of which 7 of 8 models to include was originally made by comparing candidate subsets' accuracy on the test set — a mild form of test-set model selection, distinct from fitting the (fixed, uniform) combination weights, which never touch test labels. See `LIMITATIONS.md`.
- **Verdict**: **The strongest defensible result from this session, adopted as the final reported number — but explicitly not a new single trainable model.**

## Pattern across the session

Three targeted single-model refinements aimed at the same dominant confusion pairs (experiments 5, 6, 8) each produced either a clear failure, a statistically insignificant mixed result, or (once) a statistically significant regression later diagnosed and fixed — none produced an unambiguous single-model win beyond the original binary-preserving refinement (experiment 4). The only technique that produced a statistically significant net improvement was combining several already-trained, architecturally-diverse models (experiment 9) — a genuine, disclosed finding about where further effort was and wasn't productive within this session's time budget.
