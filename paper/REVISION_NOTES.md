# Paper revision 2 (2026-09-25)

`xuad_v2.tex` is the current manuscript. It combines:

- the prose, title and new Section VIII ("Directions We Think Are Genuinely Open")
  of the revised draft `xuad_v2_draft.pdf`, and
- all 8 figures and the supplementary tables (per-family APCER, quantum modes,
  capability matrix) of the original `xuad.tex`, which the draft had dropped.

Every number is `\input` from `tables/*.tex` (generated from `results/*.json`), so
no figure in the text was retyped by hand. `xuad.tex` / `xuad.pdf` are kept unchanged
for reference.

## Numbers checked

Every table in the draft was compared against `results/*.json` and `tables/*.tex`:
P1, per-family APCER, P2 (all four folds and four metrics), P3, localisation IoU,
faithfulness, ablation, open-set ablation, gates, efficiency. **No numeric
discrepancies.** Fold means were recomputed by hand and match.

## Errors found and fixed

| # | Where | Error | Evidence | In old `xuad.tex` too? |
|---|---|---|---|---|
| 1 | Protocol P2, Sec. V-C, V-E, Table II note | Novelty AUROC described as held-out family **vs. genuine**. The code computes held-out family **vs. seen attack families**. | `scripts/openset.py:42-44`, `scripts/run_experiment.py:313-314` | Yes (P2 protocol text, calibration section) |
| 2 | Conclusion | Old paper claimed gains on "cross-domain transfer". X-UAD is at chance on P3 (AUC 50.90 vs CDCN++ 65.96). | `tables/crossdomain.tex` | Yes (draft had already fixed it) |
| 3 | Risk equation | Draft dropped the unknown-attack severity term `ς_unk`. The code uses it (`u_severity = 0.75`). | `xuad/models/heads.py:243-252` | No |
| 4 | Taxonomy caption | Implied region-level IoU averages 4 families incl. adversarial. It averages 3 (3-D mask, face-swap, reenact): (83.57+78.09+71.64)/3 = 77.77. | `results/xuad_P1.json` `loc_per_class` | Yes |
| 5 | Qualitative figure caption | Claimed the localiser "recovers the blending ellipse of a face-swap". In the figure, the face-swap prediction is an eye/mouth pattern like reenactment, not the ellipse; replay misses the glare spot. | `figures/qualitative.png` | Yes |
| 6 | Confusion figure caption | "Overwhelmingly within a super-class" overstated. Largest errors are face-swap↔reenact (within digital), but print→digital (6%) and 3-D mask→synthetic (4%) cross the boundary. | `figures/confusion.png` | Yes |
| 7 | Abstract / setup / conclusion | Single-seed limitation was only in the old paper's limitations; the draft dropped it. Restored as its own limitation and stated in abstract, setup and conclusion. | `results/*_P1.json` all `seed: 0` | Draft only |

## New analysis added (from existing result files, no new training)

The P2 JSONs already store per-group mean vacuity and a max-softmax novelty score
(written by `scripts/openset.py:47-53`). Two things follow:

| Fold | novelty AUROC, vacuity | same model, max-softmax | mean u: held-out / seen attacks / genuine |
|---|---|---|---|
| replay | 71.03 | 39.49 | 0.536 / 0.415 / 0.537 |
| mask3d | 50.41 | 56.80 | 0.309 / 0.322 / 0.372 |
| faceswap | 87.86 | 86.70 | 0.429 / 0.265 / 0.420 |
| adversarial | 43.04 | 40.74 | 0.394 / 0.430 / 0.582 |
| **mean** | **63.09** | **55.93** | |

8. **Added to Sec. V-C:** vacuity beats the same model's own softmax score by
   +7.2 novelty AUROC on average, mostly from replay. This is a cleaner test of
   the evidential head than the cross-model comparison. The per-group means also
   explain the two failing folds.
9. **Corrected Gap 5:** both earlier versions cited "the abstention behaviour of
   the u ≥ 0.55 rule on held-out families" as evidence. The data does not support
   it: genuine faces get at least as much vacuity as the held-out family in every
   fold. Gap 5 now says this, and it is a new limitation. Exact referral rates need
   per-sample P2 outputs, which were not saved.

10. **Training budget misstated.** Both versions said "every model in every table"
    was trained 10 epochs. True only for P1/P3. P2 and the open-set ablation were
    5 epochs on a 4,800-sample subset (`epochs: 5, n_train: 4382` in
    `results/*_P2.*.json`; command in `scripts/_session_runs/laneB5.sh`). Setup
    now states each budget. Consequence: the "ACER jumps from ~1% to 29-41% when a
    family is withheld" sentence was removed, because the full X-UAD already hits
    38% ACER at a similar small budget with **all** families seen (ablation
    table). Budget and held-out family are confounded; now a stated limitation.
    Also fixed in `README.md` (reproduce commands) and `RESULTS.md` (header).

## Figures restored

`figures/calibration.png` and `figures/roc_det.png` in the tree were **blank
axes**: they had been regenerated after the freeze without the prediction files.
The originals were extracted from the untouched `xuad.pdf` (PyMuPDF) and put back;
their legends match Table I (AUC 99.89 / 99.92 / 99.92 / 73.43). The blank
versions (and their `.pdf` twins) were moved to
`figures/_blank_regenerated_after_freeze/`. The restored PNGs are re-encoded, so
their hashes differ from the freeze manifest while the image content is the
original.

## Code no longer matches the freeze

Comparing the tree against `MANIFEST_2026-08-28_freeze.sha256` (the original
manifest, kept): 27 files changed after the freeze, including
`xuad/models/heads.py`, `xuad/losses.py`, `xuad/models/xuad.py`,
`xuad/engine/train.py`, `xuad/data/attacks.py` and `scripts/run_experiment.py`.
The changes come from the later 12-epoch campaign (new `HierarchicalEvidentialHead`,
`RegionPool`, focal/pair-confusion losses, checkpoint-selection options). They
appear to be off by default (new loss weights 0, new flags off, `--select-by`
default documented as the original), but without the original files this
**cannot be verified**. No copy of the freeze-time code exists on this machine
(`X-UAD-final.zip` and `x-uad_comp.zip` contain the same modified files). Before
claiming "code released", either recover the freeze-time code or re-run P1 for
X-UAD with the current code and confirm the Table I numbers.

## Later campaign kept out on purpose

`results_final/` and the `results_*refinement*` / `results_region_pooled_*` folders
hold a later set of X-UAD variants (best single model 0.583% ACER / 94.58% ACC; a
7-model ensemble 96.24% ACC / 0.50% ACER). They are **not** in revision 2 because:

- trained 12 epochs on a different data build, while the baselines were trained
  10 epochs and never retrained, so a comparison would not be controlled;
- `results_final/paper_material/CLAIMS_AUDIT.md` itself says not to merge the two
  campaigns;
- they give up X-UAD's calibration advantage (ECE 3.81 and 4.63 vs 1.94; the
  ensemble 8.89);
- the ensemble has no localisation or native uncertainty output, and its member
  set was chosen by test-set accuracy (`FINAL_RESULTS_REPORT.md` §4).

To use them, retrain CDCN++ and ResNet-18 under the same 12-epoch budget and data
build first.

Softened: "nobody had combined ..." / "never been connected" → "to our knowledge" /
"has not been connected" (novelty claims without a systematic literature search).

## Dropped on purpose

- The patent-style **Appendix A: Claim Set** of `xuad.tex` is not in revision 2
  (the draft also omits it). It is unusual in a peer-reviewed paper; it remains
  available in `xuad.tex` if needed.

## Not yet done

- **Not compiled.** No LaTeX toolchain on the machine used for this revision.
  Compile with `pdflatex xuad_v2 && bibtex xuad_v2 && pdflatex xuad_v2 && pdflatex xuad_v2`
  (or upload `paper/` to Overleaf). Static checks passed: all 40 citation keys exist
  in `refs.bib`, all `\ref`s have labels, all `\input` tables and figure files exist,
  environments and braces balance.
- **No new experiments.** Real-corpus validation and multi-seed runs need Python +
  PyTorch and the datasets, which were not available. See `PROJECT_STATE.md`.
