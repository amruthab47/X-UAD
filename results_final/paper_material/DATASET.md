# Dataset — Provenance and Construction

**Explicit distinction, stated up front**: **Original LFW is an untouched, real, publicly-sourced face dataset. X-UAD's benchmark is a constructed derivative that uses LFW images as raw face material and synthesises all attack classes programmatically.** X-UAD's benchmark must never be described as an official or native face-anti-spoofing dataset, and no claim should imply LFW itself contains attack labels — it does not.

## 1. Original LFW

- **Labeled Faces in the Wild** (`huang2007lfw`, UMass Amherst technical report 07-49): 13,233 images, 5,749 identities, unconstrained "in the wild" face photographs. This is the standard, published face-verification dataset — used here purely as a source of real face images, not as a spoofing/forgery benchmark (LFW carries no attack labels of its own).
- Acquisition for this project: downloaded and checksum-verified (per this project's own records; exact download source/checksum should be pulled from the acquisition log/plan if a citation-grade provenance statement is needed — `[VERIFY FROM PROJECT LOGS]`, since the acquisition transcript is not one of the files re-inspected for this specific document).
- **Original LFW is not modified anywhere in this pipeline.** Every downstream step reads LFW images and writes new, derived files; the source LFW archive itself is untouched.

## 2. X-UAD-derived benchmark — what is constructed, not native

- **Face extraction**: Haar-cascade frontal-face detection (`xuad/data/faces.py::crop_face_box`) on each LFW image — the largest detected face, square-cropped with a 12% margin, resized to a fixed size. If no face is detected, a centered fallback crop is used and flagged as `detected=False`. **Not landmark-based alignment** — no eye/mouth registration is performed.
- **Attack synthesis** (`xuad/data/attacks.py`): six attack classes are generated programmatically from the cropped bonafide face, each by an explicit, inspectable function:
  - `print`: tone-curve compression, ordered-dither halftone screen, paper-substrate texture noise, blur, perspective warp, sensor noise, JPEG re-compression.
  - `replay`: color-channel shift, RGB sub-pixel stripe lattice (moiré), low-frequency beat pattern, rolling-shutter banding, a specular glare ellipse, sensor noise, JPEG re-compression.
  - `mask3d`: bilateral-filter skin smoothing confined to a face-shaped region (excluding eye/mouth holes), a waxy tint + specular sheen, a discontinuity "ring" at the simulated mask edge, sensor noise, JPEG re-compression.
  - `faceswap`: a donor face (by default, the same image horizontally flipped) geometrically aligned onto the target's face box, alpha-blended into a large feathered inner-face ellipse, generator-style low-resolution-then-upsample step, Lab-space color transfer (Reinhard), JPEG re-compression.
  - `reenact`: a local non-rigid warp confined to three small disjoint ellipses (mouth + two eyes) driven by smooth value-noise displacement fields, localized soft-blur, JPEG re-compression.
  - `synth` ("fully synthetic"): bilateral-filter smoothing + unsharp-mask over-sharpening (a generator "over-smooth/over-sharp" signature), a transposed-convolution-stride checkerboard pattern, an azimuthally-anisotropic boost in the Fourier magnitude at a randomized radius (a designed generator spectral fingerprint), JPEG re-compression.
  - A **per-pixel ground-truth evidence mask** is emitted alongside every attack sample by the same function, used to supervise the localisation head. `print` and `synth`'s ground-truth masks are the entire frame (whole-image edits); `replay`'s is near-uniform with a localized glare boost; `mask3d` and `faceswap`'s are face-shaped regions; `reenact`'s is three small disjoint regions. This geometric distinction is directly relevant to the error analysis in `ABLATION_ANALYSIS.md`/`emergency_98_target_analysis.md`.
- **Adversarial class**: crafted via projected gradient descent (`madry2018pgd`) against a trained surrogate model (`xuad/data/adversarial.py`, `data/surrogate.pt`), a digitally-perturbed variant distinct from the six synthesiser-based classes.

## 3. Eight-class taxonomy

| ID | Class | Category |
|---|---|---|
| 0 | bonafide | genuine |
| 1 | print | physical presentation attack |
| 2 | replay | physical presentation attack |
| 3 | mask3d | physical presentation attack |
| 4 | faceswap | digital manipulation |
| 5 | reenact | digital manipulation |
| 6 | synth | digital manipulation (fully generated) |
| 7 | adversarial | adversarial perturbation |

## 4. Sample counts (this session's benchmark build)

| Component | Count |
|---|---|
| Domain-A (bonafide + 6 synthetic attack classes, class-balanced) | 29,400 |
| Adversarial (PGD-crafted) | 4,200 |
| **Total** | **33,600** |
| Train | 19,200 |
| Validation | 4,800 |
| Test | 9,600 |
| Per-class balance (overall) | 4,200 samples/class |
| Per-class balance (test split) | 1,200 samples/class |

**Note on the earlier project campaign**: `PROJECT_STATE.md` (dated 2026-08-28) describes an earlier benchmark build of 29,400 Domain-A + 4,900 **Domain-B** samples (a "Wild-Sim" domain for cross-domain-transfer protocol P3), which does not match this session's Domain-A + adversarial composition. **This session's 33,600-sample, adversarial-inclusive build is the one used for every accuracy number in this paper-material package**; the earlier Domain-B/P3 material is a separate artifact from a prior campaign and should not be merged into this session's counts without separately re-verifying it.

## 5. Split protocol

- **Identity-disjoint**: no source LFW identity appears in more than one of train/val/test. Verified directly this session (zero identity overlap, every pairwise direction) rather than assumed.
- **Protocol P1** (used for every experiment in this deliverable): closed-set — all 8 classes present in both training and test.
- A separate **P2.replay** run (leave-one-attack-out, `replay` withheld from training) exists as a single exploratory protocol demonstration (`results_P2_replay/`), not part of the main P1 accuracy-improvement series, and run at a different (6-epoch) budget.

## 6. What is original vs. constructed — summary

| Element | Original LFW | X-UAD-constructed |
|---|---|---|
| Raw face pixels (bonafide crops) | ✅ sourced from LFW | — |
| Face detection/crop | — | ✅ (Haar-cascade, this pipeline) |
| Attack images (print/replay/mask3d/faceswap/reenact/synth) | — | ✅ (synthesised, `xuad/data/attacks.py`) |
| Adversarial images | — | ✅ (PGD, `xuad/data/adversarial.py`) |
| Attack ground-truth masks | — | ✅ (emitted by the same synthesiser functions) |
| Identity labels | ✅ from LFW filename/identity metadata | — |
| Attack class labels | — | ✅ (deterministic — assigned by which synthesiser generated the sample, no human annotation) |
| Train/val/test split | — | ✅ (identity-disjoint, constructed this pipeline) |

No dataset provenance beyond what is stated above is invented. If the paper requires the exact LFW download source/checksum, retrieve it from this project's acquisition records before submission — not reconstructed here from memory.
