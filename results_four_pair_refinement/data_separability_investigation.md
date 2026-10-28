# Data-generation and class-separability investigation

Read-only investigation of `data/A_x.npy` (raw generated pixels) and the reference model's predictions (`results_binary_preserving_refinement/xuad_P1_refined_pred.npz`), focused on the four dominant confusion pairs. **No dataset changes, no result-directory modifications, no training.**

## Correction to an earlier claim, found during this investigation

A prior turn's error analysis described mask3d/synth sharing "a similar bluish color-cast" as a possible data-generation artifact. **That claim was wrong, and the cause was found and fixed during this investigation**: the dataset stores images in OpenCV's native BGR channel order (confirmed directly — `cv2.imwrite` of the raw stored array produces a normal, correctly-colored photo; swapping the channels produces the blue-tinted version). Every contact sheet generated in earlier turns fed BGR arrays directly to `matplotlib.imshow`, which assumes RGB — producing a blue tint on **every class uniformly, including bonafide**. Since the tint appears everywhere, it cannot be a class-discriminating signature, and the earlier interpretation is retracted. This does **not** affect any of the quantitative statistics in this report or prior reports (brightness/contrast/sharpness were computed via `cv2.cvtColor(..., BGR2GRAY)`, which correctly accounts for channel order; the colorfulness metric explicitly unpacked `b,g,r = img[...,0],img[...,1],img[...,2]`, the correct BGR assignment) — only the *visual display* was wrong. Corrected visuals: `figures/pair_visual_comparison_corrected_rgb.png`.

## 1. Resolution (verified, not a factor)

All classes are stored identically: `(128, 128, 3) uint8`, confirmed by direct inspection. Resolution is a controlled variable across every class by construction — it cannot explain any of the confusion patterns.

## 2. Per-class image statistics (n=300/class, test split, computed directly from raw stored pixels)

| Class | Brightness | Contrast | Sharpness (Laplacian var) | Blockiness | Colorfulness |
|---|---|---|---|---|---|
| bonafide | 112.11 | 52.59 | 191.5 | 21.00 | 44.84 |
| print | **102.80** | 48.39 | **96.9** | 19.29 | 42.82 |
| replay | 113.30 | 49.43 | 187.8 | 20.10 | **39.38** |
| mask3d | 111.16 | 52.38 | 191.2 | 20.74 | 44.99 |
| faceswap | 111.80 | 51.76 | 162.8 | 21.13 | 43.67 |
| reenact | 111.94 | 52.26 | 160.5 | 20.82 | 44.46 |
| synth | 111.65 | 53.00 | **298.5** | 21.14 | 44.37 |

**print** and **synth** are the two most statistically distinctive classes in this table — print is notably darker and much blurrier (its synthesis applies Gaussian blur + paper-texture smoothing), synth is much sharper than everything else (its synthesis explicitly adds a high-pass "over-sharpening" boost on top of a bilateral-smoothed base, by design in `xuad/data/attacks.py`). This makes the confusion patterns involving them (`replay↔print`, `faceswap↔print`, `synth↔mask3d`) *not* a case of "these classes look statistically identical" — at the population level they're clearly separable. The question is why specific *samples* still get confused (next section).

## 3. Are misclassified samples statistical outliers within their own class? (the key finding)

For each pair, compared correctly-classified samples of the true class, misclassified samples, and correctly-classified samples of the class they were confused with:

| Pair | Metric | True-class mean | **Misclassified mean** | Confused-into-class mean | Misclassified samples drift toward |
|---|---|---|---|---|---|
| faceswap→reenact | sharpness | 165.0 | **130.7** | 156.9 | reenact |
| faceswap→reenact | brightness | 112.64 | 115.90 | 113.15 | reenact |
| synth→mask3d | brightness | 112.82 | 111.27 | 112.01 | mask3d |
| synth→mask3d | colorfulness | 44.85 | 45.29 | 45.54 | mask3d |
| replay→print | brightness | 114.64 | **106.51** | 103.39 | print |
| replay→print | sharpness | 186.4 | **108.1** | 92.6 | print |
| faceswap→print | sharpness | 165.0 | **73.0** | 92.6 | print |
| faceswap→print | colorfulness | 44.27 | 42.07 | 43.28 | print |

**This is the central, verified finding of this investigation**: in every one of the four pairs, on at least one and usually two of the measured statistics, **the misclassified samples are quantitatively closer to the class the model confused them with than to their own true class's typical population** — most dramatically for `replay→print` (sharpness: misclassified samples average 108.1, almost exactly between replay's typical 186.4 and print's typical 92.6, but much closer to print) and `faceswap→print` (sharpness: 73.0, actually *below* print's own typical value). These are not random model failures on typical, well-formed samples — they are genuine boundary cases where the specific generated image's statistical properties happen to overlap with the confused-into class, largely because of natural per-sample variation in the randomized synthesis parameters (blur amount, JPEG quality, perspective jitter, etc. — all sampled per-sample in `xuad/data/attacks.py`).

## 4. Frequency/spectral characteristics (radial FFT log-magnitude profile, 8 bins low→high frequency)

| Class | bin0 | bin1 | bin2 | bin3 | bin4 | bin5 | bin6 | bin7 |
|---|---|---|---|---|---|---|---|---|
| bonafide | 4.59 | 3.26 | 2.43 | 1.81 | 1.33 | 0.98 | 0.73 | 0.42 |
| print | 4.51 | 3.11 | 2.19 | 1.51 | 1.05 | 0.79 | 0.63 | 0.40 |
| replay | 4.52 | 3.20 | 2.37 | 1.76 | 1.30 | 0.97 | 0.77 | 0.49 |
| mask3d | 4.58 | 3.21 | 2.37 | 1.77 | 1.32 | 0.99 | 0.78 | 0.49 |
| **faceswap** | 4.59 | 3.23 | 2.36 | 1.73 | 1.26 | 0.94 | 0.74 | 0.46 |
| **reenact** | 4.58 | 3.22 | 2.35 | 1.72 | 1.26 | 0.93 | 0.73 | 0.46 |
| synth | 4.59 | 3.28 | 2.49 | 1.89 | 1.46 | 1.14 | 0.88 | 0.56 |

**The single most important quantitative finding in this investigation**: **faceswap and reenact's spectral profiles are essentially identical** — maximum bin-wise difference of 0.01 across all 8 frequency bands (see the exact deltas below). This means the model's dedicated spectral branch (`xuad/models/stem.py`'s `SpectralBranch`) has **almost no aggregate spectral signal to discriminate these two classes with**. By contrast, `synth↔mask3d` has a real, measurable spectral gap (synth has 10-14% more high-frequency energy, consistent with its designed sharpening step), and `replay↔print` / `faceswap↔print` have larger, clearly usable spectral gaps (print has substantially less mid/high-frequency energy than either, consistent with its blur+paper-texture smoothing).

| Pair | High-freq (bins 5-7) delta |
|---|---|
| **faceswap − reenact** | **0.006 (negligible — essentially indistinguishable)** |
| synth − mask3d | 0.103 (small but real) |
| replay − print | 0.138 (clearly usable) |
| faceswap − print | 0.108 (clearly usable) |

## 5. Compression artifacts, blur/sharpness

Blockiness (8×8 JPEG-boundary discontinuity proxy) is similar across all classes (19.3–21.1), unsurprising since every class's synthesis pipeline ends with its own JPEG re-encoding step at broadly similar quality ranges (`xuad/data/attacks.py`'s `_jpeg()` calls, quality 64-96 depending on class) — this dimension does not obviously separate the four target pairs from each other. Blur/sharpness (Laplacian variance) is the more informative axis and is covered in sections 2-3 above.

## 6. Face-region characteristics

Not separately measured beyond what's captured by the whole-crop statistics above — a dedicated face-landmark-level analysis (e.g., comparing only the inner-face region for faceswap/reenact, since both attacks concentrate their edits there) was not performed in this pass; flagged as a gap, not a finding.

## 7. Obvious synthetic-generation signatures — by design, per the source code

Each class's *intended* discriminating signature, as implemented in `xuad/data/attacks.py` (verified from source, not inferred):

| Class | Designed signature |
|---|---|
| print | halftone dot screen + paper-substrate texture + blur + perspective warp + re-capture sensor noise |
| replay | sub-pixel RGB stripe lattice (moiré), rolling-shutter banding, specular glare |
| mask3d | localized skin-region bilateral smoothing + waxy tint + rim discontinuity at mask edge |
| faceswap | blended inner-face region with a resolution-mismatch (donor upsampled from 48-96px) |
| reenact | localized non-rigid warp confined to mouth/eye ellipse regions |
| synth | bilateral-smoothed base + high-pass sharpening boost + checkerboard + azimuthal spectral fingerprint |

**Why the four confusion pairs are inherently harder than the other 12 class pairs**: `faceswap` and `reenact` are the *only two classes whose designed signature is purely spatial/geometric* (a blended region, a local warp) rather than a distinct texture/frequency artifact — neither has a designed frequency fingerprint, which directly explains the near-zero spectral gap measured in section 4. `print`, `replay`, and `synth` each have an explicit, designed frequency-domain signature (halftone, moiré, sharpening/checkerboard respectively) that should in principle be learnable — and section 4 confirms `synth↔mask3d` and `replay/faceswap↔print` do have real, if sometimes modest, spectral gaps. The confusions in those three pairs are therefore more plausibly attributable to per-sample randomization pushing specific instances close to the boundary (section 3) than to a fundamental absence of signal.

## 8. Comparison against bonafide

| Pair | bonafide vs. member 1 | bonafide vs. member 2 |
|---|---|---|
| synth vs. mask3d | bonafide has **less** high-freq energy than synth (delta −0.146) — synth is clearly separable from bonafide | bonafide ≈ mask3d (both near baseline) |
| replay vs. print | bonafide ≈ replay (both near baseline) | bonafide has **more** high-freq energy than print (delta +0.106) — print is clearly separable from bonafide |

Both classes in each of the confused pairs remain well-separated from bonafide even where they're not well-separated from *each other* — consistent with the earlier finding that bonafide/attack confusion is now a minor contributor to total error (see the prior error-diagnosis report), and that the remaining accuracy gap is specifically an inter-attack-family problem.

## 9. Do the classes have distinctive cues the model can realistically learn?

**Yes for 3 of 4 pairs, with caveats; largely no (at the aggregate/global level) for faceswap↔reenact.**
- `synth↔mask3d`, `replay↔print`, `faceswap↔print`: real, measurable statistical and spectral gaps exist at the population level (section 2, 4), and specific errors are traceable to individual samples drifting toward the boundary (section 3) rather than the classes being fundamentally indistinguishable — a model with better sensitivity to these existing cues could plausibly do better here.
- `faceswap↔reenact`: the aggregate spectral signal is essentially absent. Any separability here likely has to come from *localized, region-specific* cues (the model's evidence-localisation and quantum-basis machinery already attempt this) rather than global image statistics — this is the pair with the weakest evidence for "the classes are separable at all" using the whole-image statistics measured here.

## 10. Primary cause: A, B, C, D, or E?

**E — a combination, but weighted heavily toward A (data-generation/separability), with a secondary contribution from C (loss/objective).**

- **(A) Data-generation/separability — primary, strongly supported**: sections 3 and 4 together are direct, quantitative evidence that a meaningful share of the remaining errors are genuine near-boundary samples (statistically closer to the wrong class than their own) and that one pair (faceswap/reenact) has almost no aggregate discriminating signal at all in the frequency domain, which the model's own architecture is specifically built to exploit for other classes.
- **(C) Loss/objective — secondary, supported by the four-pair experiment result**: the four-pair experiment (this session's prior turn) showed that redistributing the *same* fixed-weight `pcp` penalty across more pairs measurably traded off performance between them (two pairs improved, two got worse) rather than improving all four — that's a loss-design limitation (a single shared-weight averaging term can't independently tune per-pair pressure), not a data-separability limitation, and it's a concrete, already-observed data point, not speculation.
- **(B) Model capacity — weakly supported, not ruled in or out**: the earlier diagnosis found training loss still falling at the final epoch while validation had plateaued — ambiguous, and no capacity ablation has been run.
- **(D) Preprocessing — not supported**: test-time preprocessing is a deterministic center crop; nothing in this investigation points to a preprocessing-driven artifact.

## Top 3 highest-impact changes, ranked by expected benefit vs. risk

1. **Per-pair-weighted loss instead of one shared `pcp` weight** (directly targets finding #10/C). Give each of the four confusion pairs its own small weight rather than one aggregate term that dilutes when the pair set grows — directly addresses the specific failure mode observed in the four-pair experiment. **Low-medium cost** (a small, well-scoped code change reusing the existing mechanism), **medium-high expected benefit** (it's fixing an identified, not hypothesized, limitation), **low risk** (same conservative design family as the successful binary-preserving refinement).
2. **Region-localized features for faceswap vs. reenact specifically** (directly targets finding #9's weakest pair). Since the aggregate spectral/statistical signal is nearly absent for this pair, a global feature emphasis won't help much further — a targeted mechanism that compares the model's own localisation-head output (which already marks *where* the evidence is) between these two classes, or a small crop/attention emphasis on the inner-face vs. mouth-eye regions specifically, is more likely to find the separating signal than any global-statistics-based change. **Medium cost** (needs new, class-pair-specific machinery, though it can reuse the existing localisation head's output rather than building a new one), **medium expected benefit** (this is the hardest pair, so a real fix here has high value, but the "aggregate signal is nearly absent" finding also means it's the least certain to work), **medium risk**.
3. **Verify and, if needed, correct any RGB/BGR handling anywhere color actually matters to the model** (a direct follow-up to the correction made in section 0 of this report). The model itself is unaffected by channel order (nothing in `xuad/models/` treats channels asymmetrically by name), so this is **not** expected to change accuracy — but it's worth a quick, cheap audit given a real bug was just found in this session's own visualization code, purely as due diligence before investing in the other two changes. **Very low cost, low expected benefit** (most likely a non-issue for the model itself, confirmed diagnostic value only), **no risk** (read-only check).

*(Deliberately not ranking "more epochs" or "higher resolution" in the top 3 — the training-trajectory evidence from the prior diagnosis was ambiguous on capacity, and this investigation's resolution check found no evidence any of the four pairs is resolution-limited rather than statistics/spectrum-limited.)*

## Files

| File | Role |
|---|---|
| `figures/pair_visual_comparison_corrected_rgb.png` | corrected-colour visual comparison, all 4 pairs |
| `figures/pair_visual_comparison.png` | the original (BGR-displayed-as-RGB) version, kept for transparency about the correction, not for use |
| `data_separability_investigation.md` | this document |

## Verified facts vs. interpretation

**Verified**: all statistics in sections 1–8 (computed directly from `data/A_x.npy` and the reference model's saved predictions), the BGR/RGB display-bug diagnosis in section 0, and the source-code-derived synthesis mechanisms in section 7 (read directly from `xuad/data/attacks.py`).

**Interpretation**: the "why" narratives connecting statistics to confusion patterns (sections 3, 7, 9), the A–E cause attribution in section 10, and the ranked recommendations — all reasoned from the verified facts above, not independently provable claims.
