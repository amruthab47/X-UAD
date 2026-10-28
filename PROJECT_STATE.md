# X-UAD — project state at freeze

Frozen 2026-08-28. This tree is the authoritative final version.

## What is validated

All reported numbers come from **one simulated benchmark** derived from LFW
(identity-disjoint, 8 attack families, 29,400 Domain-A samples + 4,900 Domain-B).
Protocols P1 (seen), P2 (leave-one-attack-out), P3 (cross-domain) and the seven
ablations are complete and recorded in `results/*.json`.

## What is NOT validated

**No experiment has ever been run on a real corpus.** Adapters for nine public
datasets are implemented and regression-tested against synthetic mock fixtures
only. FaceForensics++ external validation is prepared but **not started** — it
requires the real c23 corpus, which is not present in this repository.

Nothing produced from `tests/make_mock_corpora.py` may be reported as a result.

## Reproducing

    pip install -r requirements.txt
    python -m xuad.data.build_benchmark --lfw <lfw-dir> --out data   # rebuilds the benchmark
    python scripts/run_experiment.py --data data --model xuad --protocol P1 --epochs 10

The benchmark rebuild is deterministic: sampled entries reproduce byte-for-byte.

## Integrity check (no training, no data required)

    python3 tests/test_real_adapters.py          # 118 checks, all pass

## Known limitations to disclose in any write-up

1. No external validation on any captured corpus.
2. X-UAD is **not** best on cross-domain transfer: CDCN++ 65.96 AUC vs X-UAD 50.90.
3. The vacuity channel fails on two of four unseen folds (adversarial 9.81,
   mask3d 32.76 unseen-AUROC); on the adversarial fold it is anti-correlated.
4. Localisation IoU binarises the sigmoid at a fixed 0.5 — a convention, never fitted.
5. Localisation is measured on the first 600 test samples; pixel-AUC on 200.
6. The FF++ ground-truth mask decode threshold (0.15) is unvalidated against real
   annotations. It affects no published number.
7. Where a corpus ships no mask, a blurred face-ellipse proxy is substituted and is
   not filtered out of IoU. Not triggered by any published result.

## Revision 2026-09-25 (paper only; no results changed)

- `paper/xuad_v2.tex` is now the current manuscript: revised prose + new
  "open directions" section + all figures restored. `paper/REVISION_NOTES.md`
  lists 7 corrections, e.g. novelty AUROC is held-out vs **seen attacks**
  (`scripts/openset.py:42-44`), not vs bonafide, and the region-level IoU
  averages 3 families, not 4.
- `paper/xuad_v2_draft.pdf` is the prose draft it was built from.
- The later 12-epoch campaign (`results_final/`, `results_*refinement*`,
  `results_region_pooled_*`) is deliberately not in the paper; its baselines were
  never retrained under the same budget. Reasons in `paper/REVISION_NOTES.md`.
- Not yet compiled to PDF (no LaTeX on that machine).
- **P2 budget:** P2 and the open-set ablation were 5 epochs on a 4,800-sample
  subset, not 10 epochs as previously written. P1 and P2 are not comparable.
- **Figures:** `calibration.png` and `roc_det.png` had been overwritten with
  blank plots after the freeze; originals restored from `paper/xuad.pdf`.
- **Code drift:** 27 files differ from the freeze manifest (now kept as
  `MANIFEST_2026-08-28_freeze.sha256`), including the model, losses and training
  code. The code that produced the reported numbers is not in this tree. See
  `paper/REVISION_NOTES.md`.
- Code note, no effect on any number: `RiskCalibrator.lam_u`
  (`xuad/models/heads.py:247`) is an `nn.Parameter`, but no loss uses the risk
  score, so it never receives a gradient. This matches the paper ("no training
  signal touches them"). Turning it into a buffer would be cleaner but changes
  checkpoint keys.

## Revision 3 experiments (started 2026-09-25 13:58, running)

- Environment: `.venv/` (Python 3.11, torch 2.14 CPU). On Windows use
  `--workers 0`: DataLoader workers cannot pickle the memory-mapped dataset.
- Queue: `scripts/_session_runs/v3_queue.sh` runs `v3_jobs.txt` (21 jobs, 3 in
  parallel) into `results_v3/`. Re-launching skips finished jobs.
  `keep_awake.ps1` stops Windows sleeping while `results_v3/QUEUE_RUNNING` exists.
- Jobs: Table I reproduction with current code; CDCN++ and ResNet-18 under the
  12-epoch recipe (seed 0); all three models under that recipe for seeds 1-2;
  leave-one-attack-out for all three models at the full P1 budget.
- `run_experiment.py` now also saves test/val features, val predictions and the
  threshold in `*_pred.npz` (additive; no metric changes).
- Analysis: `.venv/Scripts/python.exe scripts/analysis_v3.py` writes
  `results_v3/SUMMARY_v3.md`.
- Not possible here: cross-domain (P3) work, because Domain B (`B_*.npy`) and its
  700 source face images are not on this machine.

## Next experiments, in priority order (need Python + PyTorch)

0. **Check the current code still reproduces Table I**: re-run
   `run_experiment.py --model xuad --protocol P1 --epochs 10` and compare with
   `results/xuad_P1.json` (ACER 1.46, ACC 94.53, ECE 1.94). Everything below
   depends on this.
1. **One real corpus.** `bash scripts/run_real.sh <root> <corpus> 30 <tag>` on
   CASIA-FASD, OULU-NPU or FF++ c23. Removes limitation 1.
2. **Seeds 1 and 2** for P1 and the four P2 folds, all four models
   (`--seed` in `run_experiment.py`). Report mean ± std in Tables I and II.
   The P2 margins (e.g. novelty AUROC 63.09 vs 63.15) are within seed noise
   until this is done.
3. **Referral rule and vacuity failure.** Partly done from the saved means (see
   `paper/REVISION_NOTES.md`): on mask3d the held-out family gets the same vacuity
   as seen attacks, on adversarial less; genuine faces get at least as much
   vacuity as held-out attacks in every fold, so the u ≥ 0.55 referral rule is
   unvalidated. Next: re-run P2 saving per-sample `_pred.npz` and measure the
   referral rate per group.
4. **Insertion-faithfulness conjecture.** Re-run faithfulness on the
   region-level families only (3-D mask, face-swap, reenact). If insertion then
   beats the control, the whole-frame-family explanation holds.

## Must never be published

- Pre-fix P3 numbers (test set was 19.7% contaminated).
- Faithfulness/gates/basis values from the 1-epoch smoke-test overwrite.
- Any number produced from the mock FF++ fixture, including the four LOMO folds.
- `pilot_*` results (different budget, not comparable).
- Ablations (5 epochs) compared against P1 mains (10 epochs).
