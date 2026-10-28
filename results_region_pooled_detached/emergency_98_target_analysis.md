# Emergency 98–99% feasibility analysis (read-only) — X-UAD, `results_region_pooled_detached/`

**No training, no dataset modification, no label modification, and no new experiment launch were performed for this analysis.** Every number below is either read directly from saved `.json`/`_pred.npz` files or independently recomputed from the raw prediction arrays (`argmax(prob)` vs. `y`). The strongest single existing model, `results_region_pooled_detached/xuad_P1_regionpool_detached_pred.npz` (94.583% accuracy, 520 errors, the base model this analysis was asked to inspect), is used for items 1–8; item 9 additionally uses `results_binary_preserving_refinement/` (94.292%, the protected reference); item 10 uses both plus, for the ensemble finding, all 8 P1 models ever trained this session.

**Headline honest answer, stated up front**: **98–99% is not realistically achievable by tomorrow without changing the benchmark itself, and the evidence below explains concretely why.** The best immediately-available, zero-training result (a probability-averaging ensemble of existing models, computed below) reaches **96.24%**, comfortably clears 95–96% and is statistically significant — but falls 169 errors short of 98% and 265 short of 99%. Details and the ranked plan follow.

---

## 1. Full 8×8 confusion matrix (`results_region_pooled_detached`, row=true, col=pred)

```
             bonafide   print  replay  mask3d  faceswap  reenact   synth  adversarial
bonafide         1185       1       4       2         6        0       1            1
print               2    1136      18       0        32       10       0            2
replay              3      51    1109       3        27        5       2            0
mask3d               1       0       1    1154         5        2      37            0
faceswap             1      35      15       3      1078       66       2            0
reenact              4      11       5       0        83     1097       0            0
synth                1       0       8      64         2        3    1122            0
adversarial          0       1       0       0         0        0       0         1199
```

## 2. Class-wise error counts (of 1,200 each)

| Class | Errors | % of class |
|---|---|---|
| faceswap | 122 | 10.17% |
| reenact | 103 | 8.58% |
| replay | 91 | 7.58% |
| synth | 78 | 6.50% |
| print | 64 | 5.33% |
| mask3d | 46 | 3.83% |
| bonafide | 15 | 1.25% |
| adversarial | 1 | 0.08% |

## 3–4. Every confusion pair, sorted; top 10 contribution

| Rank | Pair | Count | % of 520 | Cumulative |
|---|---|---|---|---|
| 1 | reenact → faceswap | 83 | 15.96% | 16.0% |
| 2 | faceswap → reenact | 66 | 12.69% | 28.7% |
| 3 | synth → mask3d | 64 | 12.31% | 41.0% |
| 4 | replay → print | 51 | 9.81% | 50.8% |
| 5 | mask3d → synth | 37 | 7.12% | 57.9% |
| 6 | faceswap → print | 35 | 6.73% | 64.6% |
| 7 | print → faceswap | 32 | 6.15% | 70.8% |
| 8 | replay → faceswap | 27 | 5.19% | 76.0% |
| 9 | print → replay | 18 | 3.46% | 79.4% |
| 10 | faceswap → replay | 15 | 2.88% | 82.3% |
| 11–38 | (28 smaller pairs) | 92 | 17.7% | 100% |

**Top 10 pairs account for 428/520 = 82.31% of all errors.** The remaining 28 nonzero pairs are small (1–11 errors each) and mostly involve bonafide or adversarial at the margins.

## 5. Errors that must be corrected for each accuracy target (from this model's 520)

| Target | Allowed errors (of 9,600) | Must fix at least |
|---|---|---|
| 95% | 480 | 40 |
| 96% | 384 | 136 |
| 97% | 288 | 232 |
| 98% | 192 | **328** |
| 99% | 96 | **424** |

## 6. Minimum confusion-pair set that could mathematically reach 98%

Taking pairs in descending size until the cumulative count clears 328:

| Pair | Count | Running total |
|---|---|---|
| reenact → faceswap | 83 | 83 |
| faceswap → reenact | 66 | 149 |
| synth → mask3d | 64 | 213 |
| replay → print | 51 | 264 |
| mask3d → synth | 37 | 301 |
| faceswap → print | 35 | **336** ✅ (≥328) |

**Reaching 98% requires *fully eliminating* all 6 of these pairs simultaneously** (not "substantially reducing" — full elimination; a 50% reduction across these 6 pairs only recovers ~168 errors, worth ~1.75pp, landing at ~96.3%, not 98%). For 99%, the top 10 pairs (428 cumulative) must be almost entirely eliminated — an even higher, less realistic bar (81.5% of all current errors gone).

**This is exactly the set of pairs three separate targeted experiments this session already tried to fix** (four-pair penalty, per-pair-weighted penalty, region-pooled localisation reuse) — two failed outright, one produced a statistically non-significant, direction-mixed result (see item 9).

## 7. High-confidence errors

| | Count | % of 520 errors |
|---|---|---|
| Mean confidence on errors | 0.611 | — |
| Confidence > 0.90 | 31 | 6.0% |
| Confidence > 0.95 | 16 | 3.1% |
| Confidence > 0.99 | 0 | 0.0% |
| Confidence < 0.60 | 281 | **54.0%** |

**94% of remaining errors are not high-confidence mistakes.** Over half (54%) are genuinely low-confidence — the model is already "unsure" on these, which is the honest, well-calibrated behavior a confidently-wrong bug would not show. Of the 31 high-confidence errors, 20 (65%) are `synth↔mask3d` alone (13 + 7) — this is the one pair where the model is most often confidently wrong, consistent with the code-level finding in the dataset-generation inspection below.

## 8. Error concentration by attack type

No per-sample generator-method or strength metadata exists in `data/A_manifest.csv` (columns: `idx, identity, split, label, label_name` only) — each class is produced by exactly one synthesizer function with internally randomized parameters, so class-level granularity (item 2's table) is the finest breakdown available; there is no finer "generator variant" to check. At that granularity, errors are heavily concentrated in `faceswap`, `reenact`, `replay`, and `synth` (394 of 520 errors, 75.8%), while `mask3d`, `print`, `bonafide`, and `adversarial` are comparatively well-separated.

## 9. Reference (94.292%) vs. detached region-pool (94.583%) — sample-level transitions

Overall: 279 samples flipped wrong→correct, 251 flipped correct→wrong (net +28, giving the +0.29pp headline, not statistically significant per the previous turn's McNemar test, p=0.24).

| True class | Improved (ref-wrong→det-right) | Regressed (ref-right→det-wrong) | Net |
|---|---|---|---|
| faceswap | 90 | 63 | **+27** |
| synth | 58 | 37 | **+21** |
| print | 38 | 27 | **+11** |
| replay | 29 | 36 | −7 |
| mask3d | 22 | 29 | −7 |
| reenact | 40 | 52 | **−12** |
| bonafide | 2 | 6 | −4 |
| adversarial | 0 | 1 | −1 |

The detached region-pool model is a real trade: it meaningfully helps `faceswap` and `synth` classification, but **regresses `reenact` specifically** (net −12) — the class the region-pooling mechanism was designed to help most. This matches the earlier finding that `faceswap↔reenact` was a wash net (−7/+5) rather than the clean win predicted.

## 10. Samples both models consistently misclassify

**269 of the detached model's 520 errors (51.7%) are also wrong in the reference model** — a large, persistent core of hard samples, not model-specific noise. Of these 269 jointly-wrong samples, **both models predict the identical wrong class in 237 cases (88.1%)** — i.e., these aren't two models failing in different, idiosyncratic ways; they are converging on the *same* wrong answer, strong evidence of a genuinely hard/ambiguous signal rather than two independent sets of model-specific mistakes.

| True class | Jointly-wrong count |
|---|---|
| faceswap | 59 |
| reenact | 51 |
| replay | 55 |
| synth | 41 |
| print | 37 |
| mask3d | 17 |
| bonafide | 9 |
| adversarial | 0 |

Joint-failure pairs (both models agree on the wrong answer, sorted): `reenact→faceswap` (41), `replay→print` (36), `synth→mask3d` (30), `faceswap→reenact` (26), `faceswap→print` (25), `print→faceswap` (22) — the same top pairs as item 3, now confirmed to be **model-invariant**, not an artifact of one particular training run.

### Extended check: an 8-model oracle ceiling

Every P1 model trained this session (`full_retry`, `improved_retry`, `hierarchical`, `refined` [reference], `4pair`, `pairweighted`, `regionpool_nondet`, `regionpool_det`) shares the identical 9,600-sample test set. Checking whether **any** of the 8 gets each sample right (an oracle upper bound, not an achievable classifier):

**99.198% oracle ceiling — but only 77 of 9,600 samples are wrong in every single one of the 8 models tried.** This says two things: (a) most of the "hard" samples are only hard for *some* architectures/losses, so there is real headroom in combining what already exists; (b) a genuine, small, irreducible-so-far core of 77 samples (0.8%) has resisted every variant tried this session.

**A real, implementable-tonight ensemble** (unweighted probability averaging, top-7 models by individual accuracy — everything except the weakest `full_retry` baseline) reaches:

**96.240% accuracy, 361 errors** — computed directly from the already-saved `.npz` files, **zero additional training required**. Independently verified: McNemar's test vs. the reference model gives χ²=97.45, **p<0.000001** — this is a real, highly significant improvement, not noise (unlike every single-model experiment tried this session). Bonafide recall for this ensemble is **0.9958**, better than both the reference (0.9908) and the detached model alone (0.9875) — no bonafide trade-off.

| Target | Errors needed from the ensemble's 361 |
|---|---|
| 95% | already cleared (−119 margin) |
| 96% | already cleared (−23 margin) |
| 97% | 73 more |
| 98% | **169 more** |
| 99% | **265 more** |

This ensemble is the single strongest, most defensible number available tonight — but it still falls well short of 98%.

## Dataset-generation code inspection — dominant error classes only

Per your instruction, I inspected `xuad/data/attacks.py` for the classes responsible for the largest error clusters. **No dataset or label was modified; this is code reading only.**

**`mask3d` ↔ `synth` — a genuine, code-verified overlapping artifact**, and the pair with the most high-confidence (>0.9) errors (20/31, 65%): both synthesizers apply `cv2.bilateralFilter` (an edge-preserving smoothing filter) as their *primary* artifact step, with overlapping parameter ranges:
- `synth_mask3d` (`attacks.py:216-217`): `cv2.bilateralFilter(img, 9, lerp(35,95), lerp(6,14))`
- `synth_generated` (`attacks.py:333-334`): `cv2.bilateralFilter(img, 7, lerp(25,70), lerp(5,11))`

Sigma-color ranges overlap at 35–70, sigma-space ranges overlap at 6–11 — for a meaningful fraction of randomized samples, the two classes' *dominant* low-level texture signature is produced by numerically similar filter calls. This is not a hypothesis; it is what the code does.

**`faceswap` ↔ `reenact`** — already established in prior diagnostics this session: neither has a designed frequency-domain fingerprint (both are purely spatial/geometric edits), so any separating signal must be local/shape-based, which is exactly the mechanism the region-pool experiment (partially, non-significantly) targeted.

**`faceswap` ↔ `print`** — a secondary, weaker overlap: both apply a `cv2.GaussianBlur` as part of their pipeline with overlapping sigma (`synth_faceswap`: `lerp(0.25,0.8)`, `attacks.py:282`; `synth_print`: `lerp(0.45,1.05)`, `attacks.py:149`), overlapping at 0.45–0.8.

**Can this be corrected without touching the benchmark?** **No, not fully.** These overlaps are *designed into* the synthesis functions' parameter ranges, not a bug in how the model reads the data. A model-side fix (better architecture, better loss) can only ever partially compensate for genuinely overlapping ground-truth artifact statistics — it cannot manufacture a separating signal that the generator didn't put there. Any fix that touches the actual separation (narrowing the parameter ranges, adding a distinguishing artifact to one class) would be a change to the dataset-generation code, which you have explicitly excluded from tonight's scope. This is the single most important honesty caveat for the target: **part of the remaining error mass is attributable to intentional design overlap between attack classes, not solely to model limitations**, and three separate model-side experiments this session (four-pair, per-pair-weighted, region-pooled) have already spent their best shots at compensating for exactly this.

---

## Ranked emergency plan

### A. Highest-probability path (tonight)
**Ship the 7-model probability-averaging ensemble.** It already exists in the sense that all 7 underlying models are trained and their predictions are saved — this requires no training, only inference-time combination and evaluation, which can be finished and verified within an hour. Result: **96.24% accuracy, statistically significant vs. the reference (p<0.000001), bonafide recall improved to 0.9958.** This is the single most defensible number to present tomorrow if the deadline cannot move. It does **not** reach 98%.

### B. Second-best path (higher risk, lower probability, only if there's training time left)
A knowledge-distillation-style short run: train one more model using the 7-model ensemble's soft probabilities as an auxiliary target alongside the existing labels (still the same dataset, same split, no label changes) — a legitimate way to try to fold the ensemble's disagreement-recovery into a single deployable model. This is **untested and unapproved** — it is a plan, not a result — and given that three prior targeted experiments this session (each aimed at exactly the dominant confusion pairs) produced two failures and one statistically-insignificant mixed result, there is no strong evidence basis to expect this closes the ~169-to-265-error remaining gap to 98–99%. If pursued, it should be treated as experiment #4 under the exact same rigor as the prior three (one controlled change, show design, verify independently, report honestly) — not launched tonight without your separate explicit approval.

### C. What can realistically be completed tonight
- Finalizing and re-verifying the 7-model ensemble above (already done in this analysis; a small evaluation script would formalize it).
- This full error-budget/diagnostic writeup (done).
- At most **one** additional short training run (~50–90 minutes based on this session's observed per-epoch timings), if you explicitly authorize a specific, reviewed design — not a second concurrent experiment.

### D. What cannot realistically be completed tonight
- **Reaching 98–99% test accuracy.** The math in item 6 requires *fully* eliminating 6–10 confusion pairs that three targeted experiments already failed or only marginally moved, and item 10's dataset-code finding shows part of that gap is a designed-in overlap between attack classes' core artifacts — not purely a fixable model deficiency.
- Any dataset-generation change to reduce the `mask3d`/`synth` bilateral-filter overlap or the `faceswap`/`print` blur overlap (explicitly out of scope, and would also require regenerating and re-verifying the entire 33,600-sample benchmark — a multi-hour, multi-step process this session's own history shows takes far longer than one night when done with the required rigor).
- Landmark-based face alignment, a genuinely higher-input-resolution retraining pipeline, or an architecture redesign beyond what's already been tried — each would need its own design-review-test-train-verify cycle, and none has evidence behind it yet at the level the three completed experiments do.
- Closing the 77-sample, 8-model-unanimous "hard core" (item 10) — by definition, every technique tried so far already failed on these.

---

## Honest bottom line

**98–99% is not achievable by tomorrow with the existing benchmark and existing models.** The best real, verifiable number available tonight is **96.24%** (the 7-model ensemble), which is a genuine, statistically significant improvement worth presenting — but it is 1.76pp short of 98%, and the gap is concentrated in confusion pairs that (a) three separate targeted experiments this session already tried to fix without a clean success, and (b) are partly explained by artifact overlaps designed into the dataset-generation code itself, which cannot be touched under tonight's constraints. Presenting the 96.24% ensemble result with this analysis attached is the most defensible position available; presenting a false claim of 98–99% is not.
