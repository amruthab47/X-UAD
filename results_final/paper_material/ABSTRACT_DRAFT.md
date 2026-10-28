# Abstract Draft

Uses only supported results (`results_final/`, `results_region_pooled_detached/`, `results_binary_preserving_refinement/`). No SOTA claim, no "first unified detector" claim, no "first ViT for FAS" claim, no unqualified "quantum" claim.

---

## Draft

Face verification systems face two structurally different threats: physical presentation attacks (print, replay, 3-D mask) and digital manipulations (face-swap, reenactment, fully synthetic faces, adversarial perturbations). Detectors specialised to one family typically degrade on the other, and most unified detectors that do address both report a single uncalibrated score, with no account of *how much* the model does not know, no spatial evidence for *where* an attack was found, and no explanation that can be checked against the decision that was actually made. We present X-UAD, a single model that performs fine-grained (8-way) unified classification across both attack families, estimates Dirichlet-evidential uncertainty on the same decision, localises the manipulated region for both physical and digital attacks from a shared supervision signal, and produces a natural-language explanation generated exclusively from a typed evidence record extracted from the model's own internal state — by construction, no sentence in the explanation can be produced without a corresponding measured quantity. X-UAD combines a central-difference convolutional stem, a whole-image spectral branch, and a transformer stack whose attention adaptively mixes a standard dot-product channel with a classical, simulated Born-rule-inspired scoring channel. On a controlled, identity-disjoint, LFW-derived benchmark of eight attack families (33,600 samples; 19,200/4,800/9,600 train/validation/test), a single X-UAD model reaches 94.58% 8-class test accuracy with 0.583% average classification error rate (ACER) and 0.889 localisation IoU. We report a full, honest ablation series: two targeted loss-level refinements to the model's largest remaining confusion pairs failed or produced statistically insignificant results, and an architectural change reusing the model's own localisation evidence for classification initially caused a significant accuracy regression, traced to an unintended gradient path into the localisation head; a stop-gradient correction fixed the mechanism (confirmed by improved localisation quality) but yielded only a marginal, non-significant single-model gain. Combining seven independently-trained X-UAD variants with a fixed, unweighted probability average — requiring no additional training — reached 96.24% accuracy (361/9,600 errors), a statistically significant improvement over the strongest single model (McNemar p<0.0001), with improved bonafide recall (99.58%) but a disclosed calibration cost (ensemble ECE higher than any individual constituent). We report this 96.24% figure as our final result; we do not claim state-of-the-art accuracy, external generalisation, or novelty of unified physical-digital detection as a direction, and we discuss the specific, code-level reasons — including a documented artifact overlap between two attack classes' synthesis pipelines — that a substantially higher accuracy on this benchmark appears out of reach without changing the benchmark itself.

---

## Checklist against the "do not overclaim" instructions

- ❌ No claim of novelty for "unified physical+digital detection" as a research direction.
- ❌ No SOTA claim (external comparison not performed this session).
- ❌ No "first-ever" claim of any kind.
- ✅ "classical, simulated Born-rule-inspired" — explicitly qualifies the AQA mechanism as non-quantum-hardware.
- ✅ States the honest ablation failures and the stop-gradient fix's mixed result, not just the final number.
- ✅ States the ensemble's calibration trade-off in the abstract itself, not hidden in a limitations section only.
- ✅ 96.24% stated as "our final result," not as a comparison to any external benchmark.
