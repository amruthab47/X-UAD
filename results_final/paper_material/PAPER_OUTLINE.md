# X-UAD Paper Outline

**Grounding note, read first**: this project contains **two distinct experimental campaigns**. (1) An earlier, "frozen" campaign (`PROJECT_STATE.md`, dated 2026-08-28; `README.md`, `COMPARISON.md`, `RESULTS.md`; `paper/xuad.tex` + `paper/refs.bib`) with its own dataset composition (29,400 Domain-A + 4,900 Domain-B, 10-epoch budget), baselines (ResNet-18, ViT-Tiny, CDCN++), and protocols P1/P2/P3 + 7 ablations + localisation + faithfulness + cost tables, reporting X-UAD at ACC=94.53%, ACER=1.46%. (2) **This session's campaign** (2026-09-04 to 2026-09-14), which rebuilt/used a dataset of 29,400 Domain-A + 4,200 adversarial = 33,600 samples (train 19,200 / val 4,800 / test 9,600), ran only protocol P1 (plus one P2.replay run) at a 12-epoch budget, and iteratively improved 8-class accuracy from 89.49% to a final 96.24% ensemble. **These two campaigns' absolute numbers are not directly comparable** (different dataset composition, different epoch budget) and must not be merged into one results table. This outline uses **this session's verified numbers as the primary results** and clearly marks anywhere the earlier campaign's material (architecture prose, `paper/xuad.tex`'s Related Work, `refs.bib` citations) is being reused for framing rather than as a reproduced number.

---

## Title
"Explainable Unified Face Attack Detection via Adaptive Vision Transformers and Multimodal Reasoning" (as specified). Note: the existing `paper/xuad.tex` uses a different working title ("...Adaptive Quantum Attention and Evidence-Grounded Reasoning..."). Reconcile before final submission — pick one title and keep terminology consistent with §NOVELTY on how "quantum" is described (the AQA mechanism is a simulated/classical numerical procedure, not physical quantum hardware; see `CLAIMS_AUDIT.md`).

## Abstract
Draft in `ABSTRACT_DRAFT.md`. Must report this session's final ensemble (96.24%) as the headline number, not the earlier campaign's 94.53%, since this session's campaign is what has been independently re-verified end-to-end in this deliverable.

## Keywords
face anti-spoofing, presentation attack detection, deepfake detection, unified attack detection, vision transformer, evidential deep learning, uncertainty estimation, spoof localisation, explainable AI, model ensembling. (Drop "quantum-inspired attention" from the keyword list unless the paper is careful to immediately qualify it as simulated/classical — see `CLAIMS_AUDIT.md`.)

## 1. Introduction
Motivate the physical/digital split in FAS and forgery-detection literature and the case for a unified detector (supported: `paper/xuad.tex` §Introduction, `refs.bib` citations to the unified-detection line: `yu2024unified`, `yuan2024challenge`, `ma2025uniattackplus`, `chen2025uniattack`, `zhang2024unidefense`). State the three capabilities X-UAD targets: uncertainty, localisation, grounded explanation (supported: `xuad/models/heads.py` EvidentialHead + LocalisationHead + RiskCalibrator; `xuad/xai/reasoning.py` GroundedReasoner). End with the this-session result as the concrete accuracy evidence: 89.49% → 96.24% across a documented, honestly-reported experiment series (`results_final/FINAL_RESULTS_REPORT.md`).

## 2. Related Work
Reuse `paper/xuad.tex`'s Related Work section and `refs.bib` as the literature base — these are pre-existing, real citations, not invented for this task. Structure per `NOVELTY_AND_RESEARCH_GAP.md`: (a) classical FAS (`liu2018auxiliary`, `yu2020cdcn`, `yu2021revisiting`, `wang2022ssan`), (b) forgery forensics (`rossler2019ff`, `li2020celebdf`, `frank2020frequency`, `wang2020cnnspot`), (c) unified physical+digital detection (`yu2024unified`, `yuan2024challenge`, `ma2025uniattackplus`, `chen2025uniattack`, `zhang2024unidefense`), (d) VLM/MLLM-based forensics (`huang2024ffaa`, `guo2025m2f2det`), (e) evidential/uncertainty learning (`sensoy2018edl`, `josang2016subjective`, `malinin2018prior`, `hendrycks2017msp`), (f) quantum-inspired attention (`cherrat2024qvt`, `comajoan2024qgluon`, `evans2025on`). **The user's message named additional works — "MAYA", "One Model, All Attacks", "I-FAS", recent VLM-based unified FAS work — that do not appear in `refs.bib` and have no verified bibliographic record in this project.** They must not be cited with invented author/venue/year details; either locate and verify their real citations before submission, or omit them. Flagged explicitly in `CLAIMS_AUDIT.md`.

## 3. Research Gap
Per `NOVELTY_AND_RESEARCH_GAP.md`: unified detection is solved in-distribution but not out-of-distribution (supported by the earlier campaign's P2 leave-one-attack-out collapse, `RESULTS.md` §P2, and this session's own `results_P2_replay/` single-fold result: ACC=77.89%, ACER=30.37%, well below the closed-set numbers); most unified detectors report one scalar with no uncertainty, no localisation for digital attacks, and no auditable explanation.

## 4. Contributions
Use `CONTRIBUTIONS_DRAFT.md` version B (strong-but-defensible) as the default; do not use version C (slide-shortened) in the paper body.

## 5. Dataset and Benchmark Construction
Full detail in `DATASET.md`. Key facts to carry into the paper: LFW as the sole real-face source (`huang2007lfw`), Haar-cascade detection + fixed-margin square crop (`xuad/data/faces.py::crop_face_box`, **not** landmark-aligned), 6 synthetic attack families generated by explicit, inspectable code (`xuad/data/attacks.py`) plus one adversarial (PGD, `madry2018pgd`) family, identity-disjoint split verified this session, 33,600 total samples (this session's build), 8-class taxonomy.

## 6. X-UAD Architecture
Full detail in `METHODOLOGY.md`. Stem → spectral branch → AQA transformer → evidential head → localisation head → risk calibrator, plus this session's addition (`RegionPool`, `xuad/models/heads.py`), which one of the two "best single model" checkpoints (`region_pooled_detached`) uses.

## 7. Training Objective
`xuad/losses.py::XUADLoss` — evidential loss with optional focal modulation, localisation loss, quantum-basis auxiliary loss, superclass auxiliary loss, pseudo-unseen-evidence-suppression, attention-coherence regulariser, the binary-preserving BCE reinforcement term and the (weighted) pair-confusion penalty introduced during this session's refinement experiments. Describe each term's role and cite which experiment (`ABLATION_ANALYSIS.md`) established or tested it.

## 8. Explainability and Uncertainty
`xuad/xai/evidence.py` (region profiles, spectral descriptors, basis naming, attention rollout, deletion/insertion faithfulness) and `xuad/xai/reasoning.py` (`EvidenceRecord`, `GroundedReasoner` — deterministic, zero-hallucination-by-construction realiser; `MLLMReasoner` — optional hosted-LLM back-end constrained to the record, with a `verify()` audit function). **Faithfulness numbers (deletion/insertion AUC) exist only in the earlier campaign's `RESULTS.md`** and were not reproduced this session — mark as `[EARLIER CAMPAIGN, NOT REPRODUCED THIS SESSION]` if included.

## 9. Experimental Setup
This session: AdamW + OneCycleLR, 112×112 input, CPU-only training, seed=0, up to 12 epochs, `caffeinate`-protected background runs with per-epoch checkpointing (an engineering detail worth a sentence — every long run in this project survived unexplained external kills via a resume mechanism, evidence of a controlled, reproducible protocol). Protocol P1 (closed-set, all 8 classes seen at train and test) is the primary protocol; one P2.replay (leave-one-attack-out) run exists (`results_P2_replay/`, 6-epoch budget, not comparable to the 12-epoch P1 series) demonstrating the open-set direction without claiming it as a main result.

## 10. Results
Primary table: `RESULTS_TABLES.md` §Final Ensemble and §Baseline vs Improved. State plainly: single best model = `region_pooled_detached` (94.583%), final reported result = the 7-model ensemble (96.240%), and that the ensemble is a **combination procedure over already-trained models**, not a new trainable architecture — important for correctly scoping the paper's contribution (the architecture + training recipe reach 94.58% as a single model; the ensembling step is presented as a supplementary, zero-additional-training result, not part of the core architectural contribution).

## 11. Ablation Study
`ABLATION_ANALYSIS.md` — every experiment this session, successes and failures alike, with root-cause analysis for the failures (dilution in the shared-mean pair penalty, gradient contamination in the non-detached region-pooling). This is a genuine strength of the paper: three honestly-reported negative/mixed results with mechanistic explanations are more defensible than a clean success list with no failure analysis.

## 12. Error Analysis
`results_region_pooled_detached/emergency_98_target_analysis.md` (already produced this session) is the primary source: full confusion matrix, per-pair error ranking, confidence distribution of errors, class-wise error concentration, cross-model agreement on hard samples, and the code-level finding that `mask3d`/`synth` share an overlapping `cv2.bilateralFilter` artifact by design (`xuad/data/attacks.py:216-217` vs `333-334`).

## 13. Localisation Analysis
Report `region_pooled_detached`'s loc_IoU=0.8887, loc_Dice=0.9349, loc_pixel_AUC=0.9692 (this session, verified). The earlier campaign's per-class localisation table (`RESULTS.md`, X-UAD vs CDCN++) is a different run and must be labeled `[EARLIER CAMPAIGN]` if referenced for context, not presented as this session's number.

## 14. Uncertainty/Calibration Analysis
Mean vacuity, ECE, Brier for the reference model, the best single model, and the final ensemble (`RESULTS_TABLES.md` §Uncertainty). Must include the honest finding that ensemble ECE (8.89) is *worse* than any individual constituent (3.8–5.5 range) — a genuine, disclosed limitation, not glossed over.

## 15. Discussion
Synthesize: architecture + refined loss gets a single model to 94.58% with strong localisation; naive extensions of the loss-based refinement (four-pair, per-pair-weighted) and a naive architectural reuse of the localisation map (non-detached region pooling) all failed or regressed, each for an identifiable mechanistic reason; the stop-gradient fix validated the mechanism but produced only a statistically non-significant single-model gain; ensembling existing models, with no additional training, was the only technique this session found that produced a statistically significant improvement (96.24%, McNemar p<0.000001) — a finding worth discussing as a methodological lesson (diminishing returns from further single-model refinement vs. straightforward gains from combining diverse existing models).

## 16. Limitations
`LIMITATIONS.md` in full — LFW-derived/synthetic benchmark, no external corpus validation this session, no SOTA claim, the ensemble's test-set-informed model-selection caveat, ECE=8.89, localisation borrowed from a single constituent rather than ensembled, and (from the earlier campcampaign, if referenced) the domain-shift collapse and the vacuity channel's inconsistent unseen-family behavior.

## 17. Conclusion
State the verified final number (96.24%) without overclaiming; point to `ABLATION_ANALYSIS.md`'s honest failure analysis as a contribution in its own right; state open directions (a genuinely single deployable model matching ensemble accuracy remains unsolved by this session's work).

---

**Files this outline depends on** (all inspected before writing): `README.md`, `COMPARISON.md`, `RESULTS.md`, `PROJECT_STATE.md`, `paper/xuad.tex` (first ~140 lines), `paper/refs.bib` (full), `xuad/xai/reasoning.py` (full), `xuad/xai/evidence.py` (full, read earlier this session), `xuad/models/*.py` (full, read earlier this session), `xuad/losses.py` (read earlier this session), `xuad/data/attacks.py` (full, read earlier this session), `xuad/data/faces.py` (read earlier this session), every `results_*/*.json` and `*_report.md` produced this session, `results_final/FINAL_RESULTS_REPORT.md`, `results_region_pooled_detached/emergency_98_target_analysis.md`, `results_P2_replay/xuad_P2_replay.json`.
