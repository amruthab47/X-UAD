# X-UAD Methodology — Complete Pipeline

All details below are read directly from the code files named; nothing is inferred beyond what the code states. Any detail that could not be directly confirmed is marked `[VERIFY FROM CODE]`.

## Pipeline overview

```
input (112x112x3, BGR) → CDC stem → {spatial tokens} ⊕ {spectral tokens} ⊕ {CLS}
   → 4x Adaptive Quantum Attention transformer blocks
   → quantum evidence basis measurement
   → feature fusion (+ optional region-pooled localisation feature)
   → evidential Dirichlet classifier head
        (in parallel: localisation head; risk calibrator; rule-based reasoning)
```

## 1. Input / preprocessing

- Face crop: Haar-cascade frontal-face detector, largest detected face, square crop with a 12% margin around the detected box, resized to a fixed size (`xuad/data/faces.py::crop_face_box`). **Not landmark-aligned** — no eye/mouth registration, only a Haar-cascade bounding box.
- Stored resolution: 128×128 (per `xuad/data/dataset.py`); model input: 112×112, obtained by a **random crop during training augmentation** and a **deterministic center crop at test time** (`xuad/data/dataset.py::__getitem__`).
- Channel order: BGR (OpenCV convention) throughout, including in the stored `.npy` arrays.
- Training-time augmentation: random horizontal flip (p=0.5) and photometric jitter (gain 0.90–1.10, bias ±0.06) only — no geometric warps, by design, to avoid destroying the periodic forensic signatures the model is meant to detect (`xuad/data/dataset.py`, inline comment).

## 2. CNN stem (`xuad/models/stem.py::CDCStem`)

- 4 stages, each stride-2: input → 56×56 (24ch) → 28×28 (48ch) → 14×14 (96ch) → 7×7 (128ch, `dim`), for the 112px input size used throughout this project.
- Each stage after the first is a `CDCBlock`: two `CDConv2d` layers + BatchNorm + GELU + a residual skip.
- `CDConv2d` implements **Central Difference Convolution**: `y = conv(x) - theta * conv_1x1_sum(x)`, a computationally cheap way (one extra 1×1 conv) to make the layer sensitive to local gradient/texture detail rather than only raw intensity. `theta=0.7` by default.
- Motivation stated in-code: this texture sensitivity targets print halftone, silicone/mask smoothing, and generator over-smoothing — all textural, not purely intensity-based, artifacts.

## 3. Spectral branch (`xuad/models/stem.py::SpectralBranch`)

- Computes a 2D FFT of the input's luminance channel, takes `log1p(|FFT|)`, pools it onto a **polar (radius × angle) grid** (16 radius bins × 12 angle bins = 192-dim histogram) — translation-invariant, rotation-equivariant by construction.
- A small MLP encodes this histogram into 4 "spectral tokens" of the same width (`dim=128`) as the spatial tokens, concatenated into the token sequence with their own learned positional bias (not tied to spatial position, since the spectral signal is whole-image, not per-location).
- **Important limitation, verified this session** (`results_binary_preserving_refinement/architecture_feasibility_investigation.md`): this branch computes one histogram per whole image — it cannot supply a spatially-localized signal, and it provides essentially no separating signal for the `faceswap`/`reenact` pair specifically, since neither attack has a designed frequency-domain fingerprint.

## 4. Token assembly and transformer (`xuad/models/xuad.py::XUAD.forward`, `xuad/models/aqa.py`)

- Token sequence: 1 learned CLS token + 49 spatial tokens (flattened 7×7 stem output, each with a learned position embedding) + 4 spectral tokens = 54 tokens total, width 128.
- 4 `AQABlock`s (depth=4, heads=4), each: LayerNorm → `AdaptiveQuantumAttention` → LayerScale-gated residual → LayerNorm → MLP (GELU, ratio 2.0) → LayerScale-gated residual.
- LayerScale initialised near zero (`1e-2`), so the network begins training as close to "CDC stem + classifier" and the attention stack is recruited only as its residual earns gradient — stated design rationale in-code, not just an implementation detail.

### Adaptive Quantum Attention (`xuad/models/aqa.py::AdaptiveQuantumAttention`)

- **Classical channel**: standard scaled dot-product attention logits, `(q @ k^T) * scale`.
- **Born channel**: queries and keys are L2-normalised ("amplitude-encoded"), each passed through a per-head `GivensCircuit` (a depth-4 cascade of paired 2D rotations, orthogonal by construction — `O(d·L)` parameters instead of `O(d²)`), then scored by squared inner product (`|⟨ψ_q|ψ_k⟩|²`), scaled by a learned temperature `beta`.
- **Adaptive gate**: a per-head, input-conditioned sigmoid gate (`sigmoid(Linear([mean(x); std(x)]))`), initialised near zero (`gate_bias=-2.0`) so the block starts almost purely classical and learns how much of the Born channel to mix in.
- **This is a classical/simulated numerical procedure** — real-valued vectors, real orthogonal rotation matrices, computed on ordinary hardware. It is inspired by the mathematical form of quantum-measurement (Born rule) scoring, not an implementation on quantum hardware. See `CLAIMS_AUDIT.md` for exact wording guidance.

### Quantum evidence basis (`xuad/models/aqa.py::QuantumEvidenceBasis`)

- A learnable set of `n_basis=8` orthonormal "basis states," produced by applying a (separate, depth-6) `GivensCircuit` to the computational-basis vectors — orthonormality holds exactly by construction, no Gram-Schmidt or penalty needed.
- Every post-transformer spatial token is measured against this basis (Born probabilities, `(ψ · φ)²`), mean-pooled into an 8-dim vector (`born_pool`) that is (a) concatenated into the classification feature and (b) exported unchanged as part of the intrinsic explanation (`xuad/xai/evidence.py::name_basis_states`) — the same numbers used for the decision are what get explained, not a separate post-hoc computation.

## 5. Feature fusion (`xuad/models/xuad.py::XUAD.forward`)

`feat = concat[cls_token, mean(spatial_tokens), mean(stem_output_map), born_pool·n_basis]`, dimension `3·dim + n_basis = 392`. **Architecture-investigation finding, this session** (`results_binary_preserving_refinement/architecture_feasibility_investigation.md`): two of these four components are unweighted global means — positional/shape information about *where* on the face something changed is destroyed before classification, even though the model's own localisation head computes exactly that signal.

**This session's addition**: an optional `RegionPool` module (`xuad/models/heads.py`) pools the localisation map over 6 named facial regions plus a face-boundary ring — the same normalised-coordinate regions already used for post-hoc explanation (`xuad/xai/evidence.py::REGIONS`) — projects them through a small linear layer, and appends the result to `feat` (dimension becomes 408 when enabled). **Critically, the localisation map is `.detach()`-ed before this pooling** (`xuad/models/xuad.py`, in the region-pooled path) so the classification objective can *read* the localisation head's output without being able to *reshape* it — mirroring the existing stop-gradient design already used in `EvidentialHead`'s magnitude branch (`z.detach()`, `xuad/models/heads.py`). This was added and verified after an earlier, non-detached version of the same mechanism was found to measurably degrade the localisation head's own accuracy (see `ABLATION_ANALYSIS.md`).

## 6. Evidential classification head (`xuad/models/heads.py::EvidentialHead`)

- Decoupled direction/magnitude Dirichlet parameterisation: `pi = softmax(cls(trunk(feat)))` (direction), `E = exp(ev(feat.detach()))` (magnitude, reading a **stop-gradient copy** of the feature), `alpha = 1 + E · pi.detach()`, `S = K + E`, `vacuity = K/S`.
- Stated rationale (in-code docstring): predicting direction and magnitude jointly through one parameterisation routes every gradient through `alpha`, whose fit term saturates and whose regulariser drives the pre-activation negative — an "evidence starvation" effect measured (per the docstring) to cost significant accuracy; decoupling avoids this while keeping the output a valid Dirichlet posterior (subjective logic, `josang2016subjective`).
- A separate `HierarchicalEvidentialHead` (bonafide/attack → physical/digital/adversarial → specific class) exists and was tested this session as an alternative; it is **not** the head used by any adopted model (see `ABLATION_ANALYSIS.md`).

## 7. Localisation head (`xuad/models/heads.py::LocalisationHead`)

- A light FPN-style decoder: the 7×7 post-transformer token map plus the stem's 14×14 and 28×28 intermediate maps are progressively fused and upsampled to a 56×56 single-channel map, sigmoid-activated.
- Supervised per-pixel against the ground-truth attack mask produced by the same synthesiser function that generated the sample (`xuad/data/attacks.py`), except for print/replay/synth, whose mask is the whole frame (`[VERIFY FROM CODE]`: confirm no attack-type-specific exception exists elsewhere that alters this for the loss computation).

## 8. Risk calibrator (`xuad/models/heads.py::RiskCalibrator`)

- `risk = (1 − vacuity) · Σ(prob · severity) + λ_u · vacuity · u_severity + λ_a · localisation_area`, clamped to `[0, 1.5]`.
- `λ_u`, `λ_a` are `nn.Parameter`s but are **explicitly documented as deployment policy, not learned from a training signal** — `risk` is never consumed by `XUADLoss` (verified this session by grep), so these coefficients receive no gradient in the current training pipeline unless a separate risk-labeled fine-tuning stage is added elsewhere `[VERIFY FROM CODE — confirm no other script trains them]`.

## 9. Rule-based reasoning (`xuad/xai/reasoning.py`)

- `build_record()` assembles a typed `EvidenceRecord` dataclass from a single forward pass: predicted class, confidence, vacuity, attack probability, superclass, risk score/label, top-3 activated facial regions (via `xai/evidence.py::region_profile`), spatial statistics of the localisation map, spectral descriptors, and the top-3 quantum evidence-basis modes.
- `GroundedReasoner` is a **deterministic text realiser**: every sentence is emitted by a rule guarded by a numeric predicate over the record — the hallucination rate is zero by construction because no sentence can be generated without a corresponding measured quantity.
- `MLLMReasoner` is an **optional** hosted-LLM back-end that receives the evidence record as the only admissible source of facts and is instructed to paraphrase, not add; `verify()` re-scans the produced text for numeric claims not traceable to the record.

## Training objective

`xuad/losses.py::XUADLoss` combines (weights in the primary reference configuration in parentheses where known): evidential classification loss with optional focal modulation (`cls`, focal_gamma=2.0 in the reference/best configs), localisation BCE (`loc=0.6` default), quantum-basis auxiliary CE (`basis=0.25` default), pseudo-unseen-evidence-suppression (`pues=0.15` default), attention-coherence regulariser (`coh=0.02` default), confidence-calibration term (`conf=0.4` default), a binary-preserving BCE reinforcement term computed directly on `prob[:,0]` (`bce=0.5`, introduced this session), and a targeted pair-confusion penalty (`pcp=0.15` or the per-pair-weighted variant, introduced this session — see `ABLATION_ANALYSIS.md`). Exact default weight values are read from `xuad/losses.py::XUADLoss.__init__`; per-experiment overrides are recorded in each `results_*/` directory's training command.

## Full list of files inspected for this document

`xuad/models/xuad.py`, `xuad/models/stem.py`, `xuad/models/aqa.py`, `xuad/models/heads.py`, `xuad/models/baselines.py`, `xuad/losses.py`, `xuad/data/faces.py`, `xuad/data/dataset.py`, `xuad/xai/evidence.py`, `xuad/xai/reasoning.py` — all read in full this session.
