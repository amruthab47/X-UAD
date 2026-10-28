# Novelty and Research Gap

## What is already established in the literature (per `paper/refs.bib`, real, pre-existing citations)

- **Unified physical + digital face attack detection is an active, published line of work**, not an open problem: `yu2024unified` (UniAttackData, IJCAI 2024), `yuan2024challenge` (challenge report, CVPRW 2024), `ma2025uniattackplus` (UniAttackData+, ICCVW 2025), `chen2025uniattack` (IJCV 2025), `zhang2024unidefense` (IJCV 2024). These establish joint corpora and joint detectors already.
- **Classical single-domain FAS** (texture/depth-driven): `liu2018auxiliary`, `yu2020cdcn`, `yu2021revisiting`, `wang2022ssan`, plus domain-generalization work (`shao2019mada`, `jia2020ssdg`, `liu2024openset`).
- **Forgery forensics** (frequency/blending-driven): `rossler2019ff`, `li2020celebdf`, `frank2020frequency`, `wang2020cnnspot`.
- **VLM/MLLM-based explainable forgery detection**: `huang2024ffaa` (FFAA), `guo2025m2f2det` (M2F2-Det) — both produce natural-language explanations from a multimodal LLM.
- **Evidential/Dirichlet uncertainty estimation**: `sensoy2018edl`, `josang2016subjective` (subjective logic), `malinin2018prior`, plus the OOD/uncertainty baselines `hendrycks2017msp`, `hendrycks2019oe`.
- **Quantum-inspired attention in vision transformers** (a real, small, recent sub-literature, not unique to this project): `cherrat2024qvt`, `comajoan2024qgluon`, `evans2025on`.

**Named by the user but not present in `refs.bib` or otherwise verified in this project**: "MAYA", "One Model, All Attacks", "I-FAS", and unspecified "recent VLM-based unified FAS work" beyond FFAA/M2F2-Det. **No citation for these should be fabricated.** If they are to be discussed, their actual bibliographic details must be located and added to `refs.bib` before submission; until then, `PAPER_OUTLINE.md` §2 and `CLAIMS_AUDIT.md` flag this explicitly rather than inventing entries.

## What X-UAD does differently (supported by code inspected this session)

Per the user's own framing — and matching what the code actually implements, not a redescription of the two already-established directions ("ViT for FAS" or "first unified physical+digital detector") — the defensible novelty is the **specific combination**, in one architecture, of:

1. **Fine-grained (8-way) unified attack classification** in a single closed-set head, rather than binary presentation-attack detection or a separately-trained digital-forgery classifier (`xuad/models/xuad.py`, `xuad/data/attacks.py` 8-class taxonomy).
2. **Evidential (Dirichlet) uncertainty on the same decision**, decoupling direction from magnitude (`xuad/models/heads.py::EvidentialHead`) — a design choice with a stated, code-documented rationale (avoiding "evidence starvation" from a joint parameterisation), not just "we used evidential deep learning" (`sensoy2018edl`) applied unchanged.
3. **Pixel-level spatial attack localisation supervised for both physical *and* digital attacks** (`xuad/models/heads.py::LocalisationHead`, ground truth from `xuad/data/attacks.py`'s synthesis masks) — most cited FAS localisation work (`liu2018auxiliary`, `yu2021revisiting`) supervises a depth map, which is a liveness proxy meaningless for digital attacks; X-UAD's mask is the actual manipulated region for every attack type.
4. **A rule-based, zero-hallucination-by-construction explanation layer that reads only from a typed evidence record** (`xuad/xai/reasoning.py::GroundedReasoner`), with an optional constrained-MLLM back-end whose output is verified against the same record (`MLLMReasoner.verify()`) — distinct from the VLM-forensics line (`huang2024ffaa`, `guo2025m2f2det`), which narrates from the image directly and has no mechanism to check the narrative against the model's actual internal decision.
5. **An adaptive, per-head, input-conditioned gate mixing a classical dot-product channel with a Born-rule (squared-overlap) channel** (`xuad/models/aqa.py::AdaptiveQuantumAttention`) — a specific mechanism within the small "quantum-inspired attention" sub-literature (`cherrat2024qvt`, `comajoan2024qgluon`, `evans2025on`), not a novel field by itself.

The genuine, defensible novelty claim is the **integration** of (1)–(5) into one forward pass whose classification, uncertainty, localisation, and explanation all derive from the same computation — not any single one of these five components in isolation.

## What claims are safe

- "X-UAD integrates fine-grained 8-way unified attack classification with evidential uncertainty, pixel-level localisation for both physical and digital attacks, and a zero-hallucination-by-construction explanation layer within a single forward pass" — **safe**, directly supported by the code.
- "On our constructed, LFW-derived 8-class benchmark, a single X-UAD model (region-pooled, stop-gradient variant) reaches 94.58% 8-class test accuracy, and a 7-model ensemble of already-trained X-UAD variants reaches 96.24%" — **safe**, independently re-verified this session (`results_final/`).
- "The Adaptive Quantum Attention mechanism is a classical, simulated numerical procedure (real-valued amplitude encoding, real orthogonal Givens rotations) inspired by the mathematical form of quantum measurement, not an implementation on quantum hardware" — **safe, and necessary** (see `CLAIMS_AUDIT.md`).
- "The localisation ground truth is the mask our own attack synthesiser injected, not an independently annotated forensic ground truth" — **safe, and necessary** for the localisation claims to be honest.

## What claims must NOT be made

- **"First unified physical + digital face attack detector."** False — `yu2024unified` and the UniAttackData line predate and already establish this.
- **"State-of-the-art accuracy."** Not established. This session's numbers are self-contained (compared only across X-UAD variants trained by this project); no head-to-head against a current external SOTA system on a shared benchmark was performed this session. The earlier campaign's baseline comparison (`COMPARISON.md`) used a *different* dataset build than this session's, and even there X-UAD trailed the best convolutional baseline (CDCN++) on raw ACER/accuracy.
- **"First ViT for FAS."** False — `dosovitskiy2021vit` and numerous FAS-ViT works predate this; the transformer-for-vision idea itself is well established.
- **"First explainable FAS system."** False — VLM-based forensic explanation already exists (`huang2024ffaa`, `guo2025m2f2det`); the defensible claim is a *different mechanism* (grounded, record-verified, not post-hoc narration), not temporal priority.
- **Any claim implying quantum hardware, quantum speedup, or quantum advantage.** The mechanism is classical/simulated.
- **Any claim of generalization beyond this project's own constructed benchmark.** No external, real-world capture-based corpus was evaluated this session (`PROJECT_STATE.md` explicitly states this was true even before this session, and nothing in this session's work changes it).

## Strongest defensible novelty statement

> X-UAD integrates fine-grained unified (physical + digital, 8-class) attack classification, evidential Dirichlet uncertainty estimation, pixel-level spatial evidence localisation for both physical and digital attack families, and a rule-based, record-grounded natural-language explanation layer within a single model and a single forward pass, evaluated on a controlled, identity-disjoint, LFW-derived benchmark; a classical (non-quantum-hardware) Born-rule-inspired attention channel is one architectural component within this integration, adaptively gated against standard dot-product attention.

## 3–5 concise research contributions

1. A single-model architecture (X-UAD) that jointly performs 8-way unified physical/digital attack classification, Dirichlet-evidential uncertainty estimation, and pixel-level attack localisation from one forward pass, reaching 94.58% 8-class accuracy as a single trained model on a controlled benchmark.
2. A rule-based explanation layer (`GroundedReasoner`) that generates natural-language forensic explanations exclusively from a typed, numeric evidence record extracted from the model's own internal state, with zero hallucination by construction, plus an optional constrained-MLLM back-end whose output is programmatically auditable against that same record.
3. An honestly-reported experiment series documenting which refinements to a unified detector's loss and architecture succeed and which fail, including two negative loss-based results (with a diagnosed dilution mechanism) and one negative-then-fixed architectural result (with a diagnosed gradient-contamination mechanism and its stop-gradient correction) — a methodological contribution in its own right for a field where negative results are rarely published.
4. An evidence-grounded finding that a zero-additional-training probability-averaging ensemble of independently-trained model variants (96.24% accuracy, statistically significant over the best single model) outperforms every attempted single-model architectural or loss-level refinement, informing where marginal engineering effort is best spent for this class of detector.
5. (Only if included, and only with the honest caveat in `LIMITATIONS.md`) An adaptive per-head gate mixing classical and Born-rule attention channels, evaluated as one of several architectural components within the above system rather than as a standalone contribution.
