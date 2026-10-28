# X-UAD improved P1 — detailed error analysis and 99%-accuracy diagnosis

Source: `results_improved_p1_retry/xuad_P1_improved_retry_pred.npz` (9,600 test samples). Baseline and improved results are read-only inputs; nothing in either protected directory was modified.

## 1. Confusion matrix — raw counts and row percentages

```
raw:               bonafi   print  replay  mask3d  facesw  reenac   synth  advers
bonafide             1184       3       3       0       2       2       6       0
print                   3    1117      41       1      24      13       0       1
replay                  5      40    1136       0      13       4       2       0
mask3d                   0       1       2    1149       3       2      43       0
faceswap                14      51      16       4     983     131       1       0
reenact                  2       8       3       0      62    1125       0       0
synth                    9       0      12     122       8       2    1047       0
adversarial               0       0       0       0       0       0       0    1200

% of true class:  bonafi   print  replay  mask3d  facesw  reenac   synth  advers
bonafide            98.7%    0.2%    0.2%    0.0%    0.2%    0.2%    0.5%    0.0%
print                0.2%   93.1%    3.4%    0.1%    2.0%    1.1%    0.0%    0.1%
replay                0.4%    3.3%   94.7%    0.0%    1.1%    0.3%    0.2%    0.0%
mask3d                0.0%    0.1%    0.2%   95.8%    0.2%    0.2%    3.6%    0.0%
faceswap              1.2%    4.2%    1.3%    0.3%   81.9%   10.9%    0.1%    0.0%
reenact               0.2%    0.7%    0.2%    0.0%    5.2%   93.8%    0.0%    0.0%
synth                 0.8%    0.0%    1.0%   10.2%    0.7%    0.2%   87.2%    0.0%
adversarial            0.0%    0.0%    0.0%    0.0%    0.0%    0.0%    0.0%  100.0%
```

Total errors: **659 / 9,600 (6.86%)** → accuracy 93.135% (matches the reported test result exactly).

## 2. Per-class precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bonafide | 0.9729 | 0.9867 | 0.9797 | 1,200 |
| print | 0.9156 | 0.9308 | 0.9231 | 1,200 |
| replay | 0.9365 | 0.9467 | 0.9416 | 1,200 |
| mask3d | 0.9005 | 0.9575 | 0.9281 | 1,200 |
| faceswap | 0.8977 | 0.8192 | 0.8566 | 1,200 |
| reenact | 0.8796 | 0.9375 | 0.9076 | 1,200 |
| synth | 0.9527 | 0.8725 | 0.9108 | 1,200 |
| adversarial | 0.9992 | 1.0000 | 0.9996 | 1,200 |

## 3. Most frequently misclassified pairs (ranked by count)

| True → Predicted | Count | % of all 659 errors |
|---|---|---|
| faceswap → reenact | 131 | 19.9% |
| synth → mask3d | 122 | 18.5% |
| reenact → faceswap | 62 | 9.4% |
| faceswap → print | 51 | 7.7% |
| mask3d → synth | 43 | 6.5% |
| print → replay | 41 | 6.2% |
| replay → print | 40 | 6.1% |
| print → faceswap | 24 | 3.6% |
| faceswap → replay | 16 | 2.4% |
| faceswap → bonafide | 14 | 2.1% |
| replay → faceswap | 13 | 2.0% |
| print → reenact | 13 | 2.0% |
| synth → replay | 12 | 1.8% |
| synth → bonafide | 9 | 1.4% |
| synth → faceswap | 8 | 1.2% |

Two pairs alone — **faceswap↔reenact (193, 29.3%)** and **synth↔mask3d (165, 25.0%)** — account for **54.3%** of every error the model still makes.

## 4. Exact samples responsible for the largest (highest-confidence) errors

Full list of the 16 highest-confidence misclassifications saved to `figures/top_confident_errors.json`. The top few:

| Row | Identity | True | Predicted | Confidence | Vacuity |
|---|---|---|---|---|---|
| 2633 | Elvis_Presley | print | bonafide | 0.992 | 0.006 |
| 1885 | Cristina_Torrens_Valero | replay | print | 0.984 | 0.037 |
| 1609 | Colin_Powell | synth | mask3d | 0.944 | 0.069 |
| 5074 | Larry_Greene | synth | mask3d | 0.942 | 0.074 |
| 1889 | Cristina_Torrens_Valero | synth | bonafide | 0.941 | 0.091 |
| 6576 | Prince_Felipe | mask3d | synth | 0.939 | 0.051 |

**11 of the 16** highest-confidence errors are the synth↔mask3d pair — this is the single hardest residual confusion even after the focal-loss improvement; these are the specific cases most worth inspecting individually.

## 5. Contact sheet

`figures/error_contact_sheet.png` — 8 rows (one per major confusion direction: faceswap↔reenact, synth↔mask3d, print↔replay, replay↔bonafide), 4 highest-confidence examples each, captioned with true label, predicted label, model confidence, and Dirichlet vacuity.

## 6. Concentration in the flagged pairs

| Pair | Errors (both directions) | % of all errors |
|---|---|---|
| mask3d ↔ synth | 165 | 25.0% |
| faceswap ↔ reenact | 193 | 29.3% |
| replay ↔ bonafide | 8 | 1.2% |
| adversarial ↔ any other class | 1 | 0.2% |
| **Sum of the 3 flagged pairs** | **366** | **55.5%** |

**Cross-superclass leakage** (physical ↔ digital, per the taxonomy's own superclass grouping): **370 errors (56.1%)** cross the physical/digital boundary; 289 (43.9%) stay within it. Adversarial and replay/bonafide are both non-issues at this point — the model has essentially solved them. The remaining problem is concentrated almost entirely in the digital-manipulation cluster (faceswap/reenact) and the physical-vs-digital-texture cluster (mask3d/synth).

## 7. Label ambiguity / visual indistinguishability

Two independent signals, both pointing the same way:

- **Confidence/vacuity split of the errors**: 57.2% of all errors (377/659) were made with <60% confidence — the model was appropriately unsure, not confidently wrong. Only **2.4% (16/659)** were high-confidence (>90%) mistakes — genuine failures rather than flagged uncertainty. Mean confidence on correct predictions (0.874) is well above mean confidence on incorrect ones (0.592), and mean vacuity is the mirror image (0.075 vs. 0.191) — the uncertainty channel is behaving sensibly even where the hard decision is wrong.
- **Visual inspection of the contact sheet**: several synth↔mask3d and print↔replay examples share a similar bluish color-cast that makes them genuinely hard to distinguish even by eye at this resolution, and multiple faceswap↔reenact examples show only subtle facial-region edits. This is consistent with the residual errors being largely a genuine class-overlap problem rather than a model-capacity problem the current architecture is failing to exploit.

## 8. Theoretical ceiling if the dominant pairs were perfectly separated

| Scenario | Errors fixed | Resulting accuracy |
|---|---|---|
| faceswap↔reenact fully resolved | 193 | 95.146% |
| mask3d↔synth fully resolved | 165 | 94.854% |
| print↔replay fully resolved | 81 | 93.979% |
| **All three of the above fully resolved** | **439** | **97.708%** |
| Every remaining confusion pair fully resolved (absolute upper bound) | 659 | 100.000% |

**Even a perfect fix of every flagged pair together only reaches 97.71%** — not 99%.

---

## Direct answers

### Is 99% eight-class accuracy realistically achievable on this dataset?

**No — not realistically, and not from any single targeted intervention.** To reach 99% you need **563 more correct predictions**, i.e. errors must fall from 659 to 96 — an **85.4% reduction across every remaining confusion**, not just the headline pairs. Fully solving the three flagged pairs (55.5% of all errors) only gets to 97.71%; fully eliminating *all* cross-superclass leakage (a perfect hierarchical gate) only reaches ~96.99% (calculated as (9600-289)/9600). To close the rest of the gap you would need to also resolve the smaller residual confusions (print↔faceswap, faceswap↔bonafide, synth↔replay, etc.) almost completely — a far higher bar than fixing the two dominant pairs. There's also a structural reason to expect this: the benchmark's own attack-synthesis code (`xuad/data/attacks.py`) documents that its difficulty `strength` parameter was deliberately tuned "so that a plain ResNet-18 reference detector lands in a target error band... preventing the trivially-separable regime that inflates simulated benchmarks." A 99% ceiling would imply the benchmark has become close to trivially separable — which contradicts its own design intent. 99% is a reasonable aspiration only if paired with a change to the benchmark's difficulty calibration, which you've told me not to touch.

### How many additional correct predictions are required to reach 99%?

**563** (from 8,941 correct today to 9,504 needed out of 9,600).

### Which class pairs account for most of the missing accuracy?

**faceswap↔reenact (29.3% of all errors)** and **synth↔mask3d (25.0%)** together account for **54.3%**. Everything else is comparatively minor (print↔replay is the only other pair above 10%, at 12.3%).

### Would higher resolution likely help?

**Plausibly, for one pair, not the other.** faceswap/reenact confusion is about warping/blending boundary artifacts that may genuinely have more discriminating detail at higher resolution than the current 112px crop preserves. mask3d/synth confusion is different in character — both classes' errors correlate with a shared coarse color-cast/texture-smoothing signature (visible directly in the contact sheet) that is a low-frequency, whole-image effect, not a fine-detail one; higher resolution is unlikely to help there and might not move the needle much on the single largest contributor to remaining error.

### Would a hierarchical classifier help?

**Yes, meaningfully, but not enough alone.** 56.1% of current errors cross the physical/digital superclass boundary — a hierarchical gate that nails the physical-vs-digital-vs-adversarial split first would plausibly prevent a large share of these (e.g. `faceswap→print`, `faceswap→bonafide`, `synth→bonafide`, `synth→replay` — all cross-family errors visible in the confusion matrix). But even a perfect superclass gate caps accuracy at ~96.99%, because the two dominant pairs (faceswap↔reenact, synth↔mask3d) are each **within** the same superclass (both digital) — a coarse hierarchical split doesn't resolve them; you'd still need a second-level classifier for exactly this sub-discrimination, which is really the hierarchical design already proposed (bonafide/attack → physical/digital/adversarial → specific class).

### Would a binary attack detector plus attack-family classifier be more appropriate?

**This is effectively the same idea as the hierarchical classifier**, and the diagnosis above already shows binary attack/bonafide detection is *not* the bottleneck — AUC is already 99.965%. The real difficulty is entirely in the second stage (which specific family), so decomposing the problem into (a) attack/bonafide, (b) family-within-attack doesn't create new information beyond what a well-designed hierarchical classifier already captures — it's the same recommendation under a different name.

### Are there any data-generation artifacts limiting the ceiling?

**Yes, plausibly one.** The contact sheet shows several mask3d, synth, and even print/replay examples sharing a similar bluish color-cast at this attack strength setting — a shared low-level visual signature across attack types that isn't discriminative and may be acting as a nuisance correlate the model has to work around rather than a genuinely separating feature. This is consistent with (not proof of) the synthesizers' own documented design: mask3d's "waxy albedo tint" and synth's spectral/checkerboard fingerprinting are different mechanisms but can produce visually similar coarse color shifts at strength=0.62. I want to be careful here: this is an observation from visual inspection of a sample of errors, not a rigorous audit of the synthesis code — it's suggestive, not confirmed.

## Recommended single next experiment

**Hierarchical classification**, exactly as you framed it: `bonafide/attack → physical/digital/adversarial → specific attack class`.

Justification, tied directly to the evidence above:
- It's the only proposed technique that directly targets the *largest single measured effect* in this analysis — the 56.1% cross-superclass error rate — rather than a hypothesis (resolution, frequency features) that the evidence only partially supports.
- It's naturally compatible with what's already working: AUC (top-level bonafide/attack split) is already 99.965%, and the model's existing superclass auxiliary loss (`sup` term in `XUADLoss`) already supervises a physical/digital/none split as a *soft signal* — a true hierarchical architecture would make that a hard gate instead, which is a small, well-scoped structural change rather than a redesign.
- It won't fix everything (ceiling ~97%, not 99%, per the ceiling analysis above) — but it's the change most likely to produce the next meaningful jump, and it's compatible with layering a second, more targeted change (e.g. contrastive separation specifically between faceswap and reenact) afterward, following the same single-variable discipline used for the focal-loss experiment.

**Not recommending** higher-resolution input or frequency-domain features as the *first* move — the evidence for them is partial (plausible for faceswap/reenact, not for mask3d/synth) rather than the clear, dominant, measured effect that the cross-superclass finding represents. Keeping this as a possible Experiment 3 rather than the primary one, per the "one variable at a time" instruction.

**Baseline status**: `results_improved_p1_retry/` (93.14% test accuracy) remains the protected baseline for the next comparison. Nothing in this analysis modified it.
