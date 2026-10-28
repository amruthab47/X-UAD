# Presentation Narrative and Slide Plan

## Story arc

**Problem → Existing gap → X-UAD idea → Dataset → Architecture → Experiments → Ablations → Final 96.24% result → Explainability → Uncertainty → Limitations → Contributions**

---

### Slide 1 — Title
Content: paper title (as specified in `PAPER_OUTLINE.md`), author, affiliation. Keep "quantum" terminology qualified if used ("Adaptive Quantum Attention" is fine as a component name; avoid unqualified "quantum" in the title alone without the paper immediately clarifying it's classical/simulated).

### Slide 2 — The problem
Content: two structurally different attack surfaces (physical presentation attacks vs. digital manipulation), each historically handled by separate detectors; a fielded system needs one decision, one threshold, one audit trail. Visual: a simple two-column diagram (physical attack examples vs. digital attack examples) — see `PAPER_FIGURES.md` for what exists/what needs generating.

### Slide 3 — Existing gap
Content: unified physical+digital detection already exists in the literature (`yu2024unified` and the UniAttackData line) and is largely solved *in-distribution*; it collapses out-of-distribution (cite the same paper's own reported numbers). Most unified detectors report one uncalibrated scalar with no localisation for digital attacks and no auditable explanation. Source: `NOVELTY_AND_RESEARCH_GAP.md`.

### Slide 4 — The X-UAD idea
Content: not "first unified detector," but integration of fine-grained unified classification + evidential uncertainty + spatial localisation (for both attack families) + grounded explanation, in one forward pass. Use the strongest defensible novelty statement verbatim from `NOVELTY_AND_RESEARCH_GAP.md`.

### Slide 5 — Dataset
Content: LFW as the real-face source (13,233 images / 5,749 identities), Haar-cascade crop (not landmark-aligned), 6 synthesised attack classes + 1 adversarial class, identity-disjoint split, 33,600 total samples / 19,200-4,800-9,600 split. Explicitly state: original LFW untouched, all attack content constructed. Source: `DATASET.md`.

### Slide 6 — Architecture overview
Content: the pipeline diagram from `METHODOLOGY.md` — CDC stem → spectral branch → AQA transformer → feature fusion → evidential head / localisation head / risk calibrator / reasoning. Visual: `paper/figures/architecture.png` already exists (from the earlier campaign) — verify it still matches the current code before reusing (`[VERIFY]`, since this session added the `RegionPool` path that the existing figure may predate).

### Slide 7 — Adaptive Quantum Attention
Content: classical + Born-rule channel, adaptive per-head gate, Givens-circuit parameterisation (`O(dL)` not `O(d²)`). State plainly, on this slide, that it is a simulated/classical mechanism — do this here, not only in a limitations footnote, since it's the component most likely to be misread as "quantum computing."

### Slide 8 — Evidential head + localisation + reasoning
Content: decoupled direction/magnitude Dirichlet head; localisation head supervised per-pixel for every attack type (not a depth proxy); `GroundedReasoner`'s zero-hallucination-by-construction design (every sentence traceable to a measured quantity in the evidence record).

### Slide 9 — Experimental setup
Content: protocol P1, 112×112 input, CPU-only, AdamW+OneCycleLR, 12-epoch budget, seed 0, per-epoch checkpointing for resilience against interrupted runs. One sentence acknowledging the P2.replay exploratory fold exists but is not part of the main series.

### Slide 10 — Progression: baseline to best single model
Content: Table 1 from `RESULTS_TABLES.md` as a simple bar/line chart — 89.49% → 93.14% → (93.77%, flagged as regressed ACER, not a clean step) → 94.29% → 94.58%. Make the hierarchical-head regression visible on the chart, not silently skipped, since the ablation slide (12) explains it.

### Slide 11 — Final result: the ensemble
Content: 96.24% accuracy headline, but immediately followed by "how" — 7 already-trained models, uniform probability average, zero additional training, statistically significant over the best single model (McNemar p<0.000001). This is the single most important slide to get exactly right: state clearly this is a combination procedure, not a new architecture, and that it is the final reported number, not a claim of 98–99%.

### Slide 12 — Ablations: what worked and what didn't
Content: Table 7 from `RESULTS_TABLES.md`, presented honestly — two failed loss experiments (four-pair, mechanistically diluted; region-pool-no-detach, mechanistically gradient-contaminated) and one non-significant mixed result (per-pair-weighted). Frame this as a strength: each failure has a diagnosed mechanism, not just a number.

### Slide 13 — Error analysis
Content: from `results_region_pooled_detached/emergency_98_target_analysis.md` — top confusion pairs, the 51.7% of errors that are jointly wrong across models (converging on the *same* wrong answer 88% of the time), the code-level `mask3d`/`synth` bilateral-filter overlap finding. This slide is where the "why can't we just push to 98%" question gets a real, evidence-based answer instead of a hand-wave.

### Slide 14 — Localisation results
Content: loc_IoU=0.8887 / loc_Dice=0.9349 (best single model), per-class IoU table, and — if a qualitative figure is generated — a side-by-side predicted-map vs. ground-truth-mask example for `faceswap` and `reenact` (the two hardest-to-localise classes per the per-class IoU numbers). See `PAPER_FIGURES.md` — this figure does not yet exist and would need to be generated from saved model checkpoints, not fabricated.

### Slide 15 — Uncertainty and calibration
Content: Table 6 from `RESULTS_TABLES.md`, with the reliability diagram (`results_final/figures/calibration.png`, already generated). State the ECE trade-off plainly (ensemble worse-calibrated despite higher accuracy) — this is a genuine, interesting finding worth a full slide, not a buried footnote.

### Slide 16 — Explainability example
Content: one worked example of `GroundedReasoner`'s output on a real test sample (headline + explanation text) alongside its evidence record fields. This requires running the reasoning module on a specific sample and capturing real output — do not write a fabricated example explanation text; generate it from the actual model (`[TO GENERATE]`, see `PAPER_FIGURES.md`).

### Slide 17 — Limitations
Content: the five or six most important items from `LIMITATIONS.md` — LFW-derived/synthetic benchmark, no external validation, no SOTA claim, ensemble's test-set-selection caveat, ECE trade-off, classical (not quantum-hardware) AQA.

### Slide 18 — Contributions
Content: `CONTRIBUTIONS_DRAFT.md` version C (short conference-slide version) verbatim.

### Slide 19 — Conclusion / future work
Content: the verified 96.24% headline restated without overclaiming; open direction — a genuinely single deployable model matching ensemble accuracy remains unsolved by this session's work; external/real-corpus validation as the clear next step.
