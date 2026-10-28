# Contributions — Three Versions

## A. Conservative academic version

1. We describe X-UAD, a single-model architecture that jointly performs 8-way unified physical/digital face-attack classification, Dirichlet-evidential uncertainty estimation, and pixel-level attack localisation supervised for both physical and digital attack families, and evaluate it on a controlled, identity-disjoint benchmark derived from LFW.
2. We describe a rule-based explanation mechanism that generates text exclusively from a typed evidence record computed during the same forward pass as the classification decision, together with an optional constrained large-language-model back-end whose output can be programmatically checked against that record.
3. We report a full ablation series, including experiments that did not improve on the baseline, together with a code-level diagnosis of why two of them failed (a diluted shared-gradient penalty; an unintended gradient path into the localisation head) and the specific architectural correction (a stop-gradient) that addressed the second.
4. We report that a simple, fixed, unweighted probability-average ensemble of independently-trained variants of this model reaches a statistically significant accuracy improvement over any single trained variant, without additional training, and we quantify the calibration cost of doing so.

## B. Strong but defensible version (recommended default for the paper body)

1. **Integrated architecture.** X-UAD combines fine-grained unified (physical + digital) attack classification, Dirichlet-evidential uncertainty, and pixel-level spatial localisation for *both* attack families — not a depth proxy meaningful only for physical attacks — within a single model and a single forward pass, reaching 94.58% 8-class accuracy as a single trained model on our benchmark.
2. **Auditable explanation by construction.** Rather than narrating from the image, X-UAD's `GroundedReasoner` generates every sentence of its explanation from a rule guarded by a numeric predicate over a typed evidence record extracted from the model's own state — the hallucination rate is zero by construction, not empirically low — and an optional LLM back-end's output is programmatically verified against the same record.
3. **A genuinely honest negative-result account.** We show, with root-cause diagnosis rather than a bare number, why two loss-level refinements targeting the model's dominant confusion pairs failed (gradient dilution in a shared-mean penalty; an unintended classification-to-localisation gradient path that measurably degraded the localisation head's own accuracy), and we show the specific one-line architectural fix (a stop-gradient, mirroring an existing pattern already used elsewhere in the model) that corrected the mechanism, confirmed by improved localisation IoU/Dice.
4. **Ensembling outperforms further single-model refinement.** Across every architectural and loss-level refinement attempted, only combining independently-trained model variants produced a statistically significant accuracy gain (96.24%, McNemar p<0.0001 vs. the best single model) — a finding with a direct engineering implication for where further effort on this class of detector is best spent, reported alongside its real calibration cost (ensemble ECE worse than any constituent).
5. **A classical, quantum-measurement-inspired attention channel**, adaptively and per-head gated against standard dot-product attention, is evaluated as one architectural component within the integrated system above (not claimed as a standalone contribution or as quantum computation).

## C. Short conference-slide version

- One model, physical + digital attacks, 8 classes, with uncertainty, localisation, and a grounded (not hallucinated) explanation.
- 94.58% single-model accuracy; 96.24% with a free (no extra training) 7-model ensemble.
- Three refinement attempts failed or fell short — we show why, and fixed one with a one-line change.
- Ensembling beat every architecture/loss tweak we tried.
- Honest about limits: constructed benchmark, no external validation, no SOTA claim.
