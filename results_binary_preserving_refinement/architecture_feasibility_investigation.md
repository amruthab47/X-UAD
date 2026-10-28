# X-UAD architecture feasibility investigation (read-only) — a credible path toward 98–99%?

**No training, dataset changes, model-code changes, checkpoints, or background processes were run for this investigation.** Everything below comes from reading `xuad/models/{xuad,stem,aqa,heads,baselines}.py`, `xuad/data/{attacks,faces,dataset}.py`, `xuad/xai/evidence.py`, and the existing saved evidence in `results_binary_preserving_refinement/{xuad_P1_refined_pred.npz, error_and_data_quality_diagnosis.md}` and `results_four_pair_refinement/data_separability_investigation.md`. The protected reference model (`xuad_P1_refined.pt`, 94.292% accuracy, 0.750% ACER) was not touched — verified by checksum at the end of this report.

## 1. Current architecture — full data flow

```
input 112x112x3 BGR crop
  (Haar-cascade face detect, fixed 12% margin, square-crop + resize;
   NOT landmark-aligned -- xuad/data/faces.py::crop_face_box)
        |
   CDCStem (xuad/models/stem.py)
   s0: CDConv2d stride2 -> 56x56 (24ch)
   s1: CDCBlock  stride2 -> 28x28 (48ch)   [returned as "f32"]
   s2: CDCBlock  stride2 -> 14x14 (96ch)   [returned as "f16"]
   s3: CDCBlock  stride2 ->  7x7 (128ch=dim) [returned as "f8"]
        |                                  \
        |                                   +--> SpectralBranch: whole-image
        |                                        luminance -> log|FFT| -> polar
        |                                        (16 radius x 12 angle) histogram
        |                                        -> tiny MLP -> 4 spectral tokens
        |                                        (position-less, one histogram
        |                                        per whole image, not per patch)
        v
  flatten 7x7 CNN map -> 49 spatial tokens (+ learned position embedding)
  concat: [CLS token, 49 spatial tokens, 4 spectral tokens] = 54 tokens, dim=128
        |
  4x AQABlock (xuad/models/aqa.py): classical dot-product attention channel
  mixed with a Born-rule ("quantum") channel via a per-head input-conditioned
  gate; operates on the same 54 tokens throughout -- no further spatial
  resolution change happens here.
        |
  QuantumEvidenceBasis: measures the 49 spatial tokens (post-transformer)
  against 8 learned orthonormal basis states -> Born probabilities (N,49,8),
  mean-pooled over the 49 tokens -> born_pool (N,8)
        |
  ================== FEATURE FUSION (the fork in the road) ==================
  feat = concat[ cls_tok,            (128, attention-pooled -- the ONLY
                                       location-sensitive piece)
                 spat.mean(1),       (128, UNWEIGHTED mean over 49 tokens)
                 f8.mean((2,3)),     (128, UNWEIGHTED mean over the 7x7 CNN
                                       map, bypassing the transformer)
                 born_pool*8 ]       (8,  UNWEIGHTED mean over 49 tokens)
  -> EvidentialHead -> 8-class probability, vacuity, evidence
  ==============================================================================
        |
  (separately, in parallel, NOT feeding classification):
  LocalisationHead: FPN decode of [7x7 token map, 14x14 f16, 28x28 f32]
  -> 56x56 evidence map, sigmoid'd, supervised per-pixel against the real
  attack mask -> collapsed to a single scalar "area" (mean activation) that
  feeds only RiskCalibrator (severity-weighted risk score), never `feat`.
```

Input resolution used by every run in this project: **112px** (`scripts/run_experiment.py:23`, `img=112` default, never overridden in any experiment run so far). `self.grid = img_size // 16 = 7` (`xuad/models/xuad.py:41-42`) is exact.

## 2. Current bottleneck

The classification feature `feat` (`xuad/models/xuad.py:115-116`) is built from **three unweighted global averages and one attention-pooled token**, all operating on a coarse 7×7 grid (49 tokens for the whole face). Two of the four terms (`spat.mean(1)`, `born_pool`) are literal `.mean()` over all spatial tokens; a third (`f8.mean((2,3))`) is a raw CNN global average that bypasses the transformer entirely. Only `cls_tok` can, in principle, weight different face positions unequally — and it has to compress that into a single 128-dim vector via attention over the same 7×7 grid.

Meanwhile the model **already computes and separately supervises** much finer spatial maps: the CDC stem produces 28×28 and 14×14 intermediate maps, and `LocalisationHead` decodes all three scales into a 56×56 map trained directly against the ground-truth per-pixel attack mask. That map — which is exactly "where did this attack manipulate the face" — is collapsed to **one scalar** and fed only to the risk calibrator, never to the classifier.

**This is the bottleneck**: the network has already learned the fine-grained *where* signal needed to separate several of the hardest confusion pairs, and then discards essentially all of it — down to a single number that doesn't even reach the classification head — before the 8-way decision is made.

## 3. `faceswap ↔ reenact` — why this pair is hard and what the architecture preserves

Reading the actual synthesis code (`xuad/data/attacks.py`) shows the two attacks' designed ground-truth regions are geometrically distinct, and neither has a designed frequency fingerprint:

- **`synth_faceswap`** (`attacks.py:253-293`): manipulates **one large feathered ellipse**, `scale ∈ [0.52, 0.66]` of the face box — forehead, both cheeks, nose and mouth together. Evidence: a blending boundary at the ellipse edge plus a resolution mismatch from a 48–96px upsampled donor.
- **`synth_reenact`** (`attacks.py:299-323`): manipulates **three small, disjoint ellipses** — one mouth ellipse (`0.28·bw × 0.17·bh`) and two eye ellipses (`0.18·bw × 0.11·bh`) — everything else in the frame is untouched.

This matches the already-saved finding in `results_four_pair_refinement/data_separability_investigation.md` (§4/§9): **the aggregate spectral/statistical signal for this pair is essentially absent**, because neither attack has a designed frequency signature (unlike print/replay/synth). It also matches the confidence evidence already in `error_and_data_quality_diagnosis.md` (§3): `faceswap ↔ reenact` errors have the lowest mean confidence of any error cluster (0.654, vs. 0.915 on correct predictions) and the highest share of low-confidence (<0.6) mistakes (38.4%) — consistent with the model having only weak, non-geometric cues to go on for this pair, exactly as predicted by the fact that its two classes' *only* real distinguishing signal is spatial extent/shape, which `feat` cannot represent.

**Does the current architecture preserve the information needed to distinguish them?** It computes it (in the localisation decoder) but does **not** preserve it through to classification — the answer is no, not because the information doesn't exist inside the network, but because of the fusion bottleneck in §2.

## 4. The other three pairs — same lens, different verdict

| Pair | Ground-truth mask shape (from `attacks.py`) | Does region/shape info plausibly help? |
|---|---|---|
| `synth ↔ mask3d` | `synth`: `np.ones(h,w)` — **entire frame**. `mask3d`: large face-shaped region (`scale∈[0.80,0.92]`) **excluding** eye/mouth holes, plus a thin boundary "ring" — bounded to the face oval. | **Weak-to-moderate.** Both are large-area masks, so shape alone is a much weaker differentiator than for faceswap/reenact — but `mask3d`'s mask is *zero outside the face oval* while `synth`'s is 1 *everywhere including background*, so a region descriptor that includes the periphery/corners of the crop could still pick up a signal. Consistent with `error_and_data_quality_diagnosis.md`'s independent finding that this pair looks like a shared **coarse colour-cast/texture** correlate rather than a spatial-extent one — texture/spectral fixes are more directly targeted here than region-shape fixes. |
| `replay ↔ print` | `print`: `np.ones(h,w)` — entire frame (halftone screen + paper texture, uniform). `replay`: `np.clip(0.55 + 0.45·glare_ellipse, 0,1)` — also **high almost everywhere** (baseline 0.55), with an added local glare boost. | **Weak.** Both masks are close to uniform/whole-frame; there is little shape signal for a region-pooling scheme to exploit. Both attacks *do* have designed, distinct spatial-frequency lattices (halftone dot screen vs. RGB sub-pixel/moiré stripe, different periods and orientations) — this is a **texture/frequency** problem, which is exactly what the existing `SpectralBranch` is built for, not a locality problem. |
| `faceswap ↔ print` | `faceswap`: bounded face-oval ellipse. `print`: entire frame. | **Moderate.** Unlike replay/print, here the two masks *do* differ sharply in spatial extent (bounded oval vs. everywhere) — a region descriptor that includes the crop's outer/background area could plausibly separate "attack confined to the face" from "attack covers the whole frame," which is a real, if untested, distinguishing signal. |

This directly implies: any single architectural change is **not equally beneficial across all four pairs**. A region/locality-based fix (§5-7) is best targeted at `faceswap ↔ reenact` (strong geometric evidence) and, more speculatively, `faceswap ↔ print`; it is not well-motivated for `replay ↔ print` (a texture/frequency problem) or, to a lesser extent, `synth ↔ mask3d` (a coarse colour/texture problem per the existing diagnosis). No single architectural change proposed here is expected to move all four pairs at once — this differs from the pair-penalty loss experiments, which applied a uniform mechanism to all pairs and, per your own diagnosis, produced mixed/negative net effects for exactly this reason (diluting a shared mechanism across mechanistically different problems).

## 5. Candidate architectural solutions — A–F evaluated

**F** below (region-pooled reuse of the existing localisation map) is not an arbitrary sixth option — it is the direct, code-grounded elaboration of the #2-ranked idea already on record in `results_four_pair_refinement/data_separability_investigation.md` ("a targeted mechanism that compares the model's own localisation-head output... reuse the existing localisation head's output"), now backed by the exact mask-geometry evidence in §3–4 rather than a hypothesis.

| Option | 1. Info added/preserved | 2. Pair(s) helped | 3. Why current arch is insufficient | 4. Accuracy effect | 5. ACER/APCER/BPCER effect | 6. CPU cost | 7. Complexity | 8. Overfitting risk | 9. Paper-defensible | 10. No dataset change needed |
|---|---|---|---|---|---|---|---|---|---|---|
| **A. New lightweight local/multi-scale CNN branch** | A fresh small conv head reading the 28×28/14×14 stem maps directly, pooled into a handful of local descriptors (edge density, local variance per sub-region) | faceswap↔reenact (moderate), faceswap↔print (weak-moderate) | `feat` never sees anything but the coarsest 7×7 scale (§2) | Moderate | Low risk if gated/small, but untested new parameters could add noise near the bonafide boundary | Low-moderate (a few small conv layers, but on top of an already-computed forward pass) | Moderate — new module, needs its own init/regularisation to avoid destabilising early training (cf. AQABlock's LayerScale design, which exists for exactly this reason) | Moderate (new capacity, no reuse of existing supervision) | Reasonable | Yes |
| **B. Reusing localisation features generically** (e.g., global-pool the 48-channel pre-final-conv FPN feature, no named regions) | The loc decoder's internal feature map, not just its final 1-channel output | faceswap↔reenact (moderate-high) | Same as F, weaker version | Moderate-good | Low (reuses an already-supervised tensor) | Very low | Low | Low | Good, but discards the region *semantics* the codebase already defines | Yes |
| **C. Multi-scale CNN features (`f16`/`f32`) fused as extra transformer tokens** | Raw, un-decoded intermediate CNN maps, added to the token sequence itself | faceswap↔reenact (moderate), general (untargeted) | Transformer only ever sees the 7×7 scale | Moderate | Moderate risk — changes token count/attention shape model-wide, touches every downstream consumer of the token sequence (AQA gates, attention rollout in `xai/`) | Moderate-high (more tokens -> attention cost grows ~quadratically in token count) | Moderate-high (new position-embedding scheme for the added tokens, changes shapes checked in several places) | Moderate | Reasonable, but heavier-handed than the problem requires | Yes |
| **D. Higher-resolution input (112→160/224)** | More raw pixels everywhere, uniformly | Unclear which pair, if any, specifically | Resolution *might* not even be the binding constraint — see §2: the problem is that positional info is pooled away, not that there isn't enough of it at 112px | Uncertain, plausibly small for this specific bottleneck | Low (no logic change) — but a bigger, slower-to-converge model could plausibly *drift* on ACER within the same 12-epoch budget | **High** — CDC stem compute scales ~with H×W (roughly 2-4x for 160px), and the 7x7->~10x10 or ~14x14 token grid grows attention cost by roughly (T_new/54)² (~2-14x depending on target resolution), on a CPU budget already at ~47 min/12 epochs | Low (one config value) | Low | Weak standalone justification given §2's finding that pooling, not resolution, is the demonstrated cause | Yes (same stored 128px source can be cropped larger) |
| **E. Region/crop-based facial-feature branch** (separate small CNN(s) run on named crops of the *input image*, not reusing the loc head) | Raw pixels from fixed sub-regions (eyes, mouth, cheeks) run through their own (small, possibly shared-weight) CNN | faceswap↔reenact (high, same geometric argument as F), faceswap↔print (moderate) | Same as F/A | Moderate-good | Low-moderate | Low-moderate (extra small CNN passes over 6 crops, but each crop is small) | **Higher than F** — needs crop extraction logic, a shared or per-region CNN, and a fusion scheme; more new, untested parameters than F | Moderate (more new capacity than F) | Good | Yes — crops taken from the existing 112px array at inference time, no re-generation |
| **F. Region-pooled reuse of the existing localisation map** (recommended) | The already-computed, already-supervised 56×56 evidence map, pooled over the 6 named regions already defined in `xuad/xai/evidence.py::REGIONS` (forehead, eye region, nose/midface, mouth+chin, left/right cheek, face-boundary ring) | **faceswap↔reenact (high — direct match to §3's mask geometry)**, faceswap↔print (moderate, via the face-boundary/background region specifically), synth↔mask3d (weak, per §4) | Same diagnosis as §2, but this is the option that fixes it with the least new machinery | **Best-supported of the six** | **Very low risk** — reuses a tensor already computed and already trained; adds no new source of evidence, only a new *view* of an existing one | **Lowest of all six** — a handful of masked-mean-pool ops (precomputable as a static sparse pooling matrix, since region boundaries are fixed normalised coordinates) + one small `Linear` layer, on a tensor (`loc`) already materialised in the existing forward pass | **Lowest of all six** — `REGIONS` and the loc map both already exist; only a differentiable (torch) version of the existing numpy `region_profile()` needs writing | **Lowest of all six** — reuses supervision that already exists, adds only a few thousand parameters (a `Linear(n_regions, region_dim)`) on top of a 1.29M-parameter model | **Strong** — "the model already learns *where*; let the classifier use *where* to help decide *what*" is a clean, falsifiable architectural story, and it is explicitly grounded in the attack-synthesis code, not a hunch | Yes |

## 6. Best single architectural modification

**Option F**: pool the existing `loc` (localisation) evidence map into region-aware descriptors using the `REGIONS` coordinates already defined in `xuad/xai/evidence.py`, and concatenate the result into `feat` before the classification head.

Rationale for choosing F over the other five: it is the only option that (a) is backed by exact, code-verified evidence that the ground-truth signal it exploits differs geometrically between the target pair's two classes (§3), (b) adds new information to the classifier without adding a new *source* of evidence (lowest overfitting/ACER-regression risk of the six), and (c) costs the least on the CPU-only training budget this project runs under. It is a smaller, cheaper, and more targeted change than A/C/D/E, while directly attacking the mechanism identified in §2.

Two honest caveats carried over from §4, not glossed over: it is **not expected to help `replay ↔ print`** at all (both masks are near-uniform, a texture/frequency problem, not a locality one), and its benefit for `synth ↔ mask3d` is **speculative and likely small** (that pair's confusion looks colour/texture-driven per the existing diagnosis, not shape-driven). This is a targeted fix for the single largest confusion pair (faceswap↔reenact, 151 errors, 27.6% of all 548), not a fix for the whole 94.29%→98% gap in one move.

## 7. Exact implementation plan (not executed)

| Element | Detail |
|---|---|
| New function | A differentiable torch re-implementation of `xai/evidence.py::region_profile()` — e.g. `region_pool(loc, regions=REGIONS)` — masked mean-pool of the `(N, 56, 56)` `loc` tensor over each of the 6-7 region boxes (excluding the special-cased "face boundary" ring, or included as a 7th descriptor), returning `(N, n_regions)`. Boundaries are static normalised coordinates, so the per-region pixel masks can be precomputed once (as boolean or float masks) and reused every forward pass — no per-sample recomputation cost. |
| File/location for the new function | New small function in `xuad/models/heads.py` (co-located with `LocalisationHead`, which produces its input) or a new `xuad/models/regions.py` if kept separate from the head definitions — either is fine; the function itself is a handful of lines. |
| `xuad/models/xuad.py::XUAD.__init__` | Add `self.region_proj = nn.Sequential(nn.LayerNorm(n_regions), nn.Linear(n_regions, region_dim), nn.GELU())` (e.g. `region_dim=16`, a small addition relative to `dim=128`). Change `feat = dim*3 + n_basis` to `feat = dim*3 + n_basis + region_dim`. |
| `xuad/models/xuad.py::XUAD.forward` | After `loc = torch.sigmoid(loc_logit)` (already computed, line ~130, *before* it is collapsed to `area`), compute `region_feat = self.region_proj(region_pool(loc))` and concatenate it into the `feat` tensor at lines 115-116, before `self.head(feat)` is called. |
| Unchanged | `xuad/losses.py`, `xuad/engine/train.py`, all of `xuad/data/*`, the evidential head's direction/magnitude decoupling, the AQA stack, the spectral branch, the risk calibrator, and the localisation *loss* itself — this change only adds one new input path into `feat`; nothing else in the model or training loop needs to change. |

## 8. Training plan (proposed, **not executed**)

One controlled experiment, holding every other component of the current reference config fixed:

- **Dataset/split/seed**: identical to `results_binary_preserving_refinement` (same `data/`, same identity-disjoint split, `seed=0`).
- **Optimizer**: identical AdamW + OneCycleLR config (`lr`, `wd`, `pct_start`, etc., unchanged).
- **Loss config**: identical `focal_gamma=2.0`, `bce=0.5` binary-preserving reinforcement term, same `select_by=acc_f1_acer_guard`. No change to `pcp`/`pair_weights` — this experiment isolates the architectural change; do not combine it with a new loss change in the same run, per your one-controlled-experiment-at-a-time methodology.
- **Epochs**: 12 (matches both prior refinement experiments; no evidence in this investigation suggests this new, small addition needs a different schedule).
- **Evaluation**: identical held-out 9,600-sample test set, identical metric suite (`summarise()` from `xuad/engine/metrics.py`), independent recomputation from the raw `_pred.npz` before accepting any result, McNemar's test against `results_binary_preserving_refinement`'s saved predictions.
- **Command shape** (illustrative — would need a new `--region-pool` flag wired through `run_experiment.py` analogous to the existing `--refine`/`--refine-weighted` flags before this could actually run):
  ```
  caffeinate -i python3 scripts/run_experiment.py \
    --data data --model xuad --protocol P1 --epochs 12 \
    --focal-gamma 2.0 --select-by acc_f1_acer_guard \
    --refine --region-pool --save-every-epoch \
    --tag xuad_P1_regionpool --out results_region_pooled_refinement
  ```
- **New output directory** (would be, not created): `results_region_pooled_refinement/`.

**This plan is not being executed.** No CLI flag, model code, or training run has been created or started for it.

## 9. Success criteria

Primary goal: **meaningful improvement over 94.292% accuracy / 94.293 macro-F1**, specifically driven by a measurable reduction in the `faceswap→reenact` and `reenact→faceswap` confusion counts (73 and 78 in the reference model) — since that is the mechanism this change is designed to move, a result that improves overall accuracy without moving those two counts would need separate explanation before being trusted.

While **preserving or improving**, verified independently from the raw `_pred.npz` exactly as in the last two experiments:
- Macro-F1 ≥ 94.293
- ACER ≤ 0.750
- APCER ≤ 1.000, BPCER ≤ 0.500
- EER ≤ 0.500
- Bonafide recall ≥ 0.9908

A result that improves accuracy but regresses any of the above — as the four-pair and per-pair-weighted experiments both did to overall accuracy while sometimes improving ACER — does **not** meet the bar and must be reported as such, not adopted.

## Final recommendation

**Yes, another architecture experiment is worth running — but it should be this one, and only this one, next.** It is the first of the investigated options that adds new *information* to the classifier rather than reshaping how existing probabilities are penalised (which is what both prior, unsuccessful loss-based experiments did), and it is grounded in exact, verified evidence (§3) about what distinguishes the single largest remaining confusion pair, not a general hope that "more architecture" helps.

**Which one**: Option F — region-pooled reuse of the existing localisation evidence map, fed into `feat` (§6-7).

**Why it has a realistic chance of moving toward 98–99%**: `faceswap↔reenact` is the reference model's largest single confusion pair by error count (151/548, 27.6%) and, per `error_and_data_quality_diagnosis.md`'s own ceiling table, resolving it alone would move accuracy from 94.292%→95.865%. Option F is the only investigated change specifically and mechanistically targeted at *why* this pair is hard (a purely geometric/spatial signal with no frequency counterpart) using information the network already computes and already supervises, at the lowest cost and risk of the six options considered.

**What result would make us stop further experimentation on this path**: if a correctly-executed run of this experiment (a) does not move the `faceswap→reenact`/`reenact→faceswap` counts materially, or (b) improves them but the improvement doesn't survive McNemar's test against the reference, or (c) regresses ACER/bonafide recall — any of these would indicate that either (i) the loc map's *predicted* shape doesn't actually track the *ground-truth* mask shape closely enough at inference time to be useful (a genuinely open question this investigation could not resolve without training), or (ii) the remaining confusion is dominated by cases where the true region overlaps ambiguously between the two attacks' plausible shapes, rather than being resolved by the region prior alone. In either case, per your standing instruction, the result should be reported honestly and no further experiment launched automatically — the next step would then more likely be the texture/frequency-oriented ideas already on record for `replay↔print` (§4) rather than a second, larger architectural change to this same mechanism.

**Do not start training under any circumstances. This report and its plan are pending your review and explicit approval.**
