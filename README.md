# X-UAD — Explainable Hybrid Vision Transformer with Adaptive Quantum Attention and Evidence-Grounded Reasoning for Unified Face Attack Detection

One detector for **physical presentation attacks** (print, replay, 3-D mask) and
**digital attacks** (face-swap, reenactment, fully-synthetic, adversarial), which
also (i) estimates *how much it does not know*, (ii) says *where* the evidence
is, and (iii) writes a *traceable* natural-language explanation.

```
Attack: Replay attack
Confidence: 97%
Risk: Medium
```

## Why this exists

| Capability | Unified FAS (UniAttackDetection, La-SoftMoE) | MLLM forensics (FFAA, M2F2-Det) | CDCN / pixel-wise FAS | **X-UAD** |
|---|---|---|---|---|
| Physical **and** digital attacks in one model | yes | partly | no | **yes** |
| Calibrated uncertainty on the decision | no | no | no | **yes (Dirichlet vacuity)** |
| Explicit *unknown-attack* channel | no | no | no | **yes** |
| Pixel-level evidence localisation | no | rarely | yes (depth proxy) | **yes (attack-mask supervised)** |
| Natural-language explanation | no | yes | no | **yes** |
| Explanation provably grounded in the decision | — | no (post-hoc narration) | — | **yes (evidence record)** |
| Attention adapted to *interference* evidence | no | no | no | **yes (Born-rule channel)** |

## Layout

```
xuad/
  data/     faces.py  attacks.py  adversarial.py  dataset.py  build_benchmark.py
  models/   stem.py (CDC + spectral)  aqa.py (Adaptive Quantum Attention)
            heads.py (evidential / localisation / risk)  xuad.py  baselines.py
  engine/   train.py  metrics.py (ISO/IEC 30107-3)
  xai/      evidence.py (basis naming, rollout, faithfulness)  reasoning.py
  losses.py
scripts/    run_experiment.py  make_figures.py  demo.py
configs/    reproduction configs, including for the public capture-based corpora
paper/      IEEEtran manuscript + figures
```

## Reproduce

```bash
# 1. benchmark
python -m xuad.data.build_benchmark --lfw <lfw_images> --wild <wild_faces> \
       --out data --strength 0.62
python -m xuad.data.adversarial            # crafts the transfer-adversarial class

# 2. train + evaluate
# budgets used for the reported numbers: P1 = 10 epochs, full train split;
# P2 = 5 epochs on a 4,800-sample subset (see scripts/_session_runs/laneB5.sh)
python scripts/run_experiment.py --model xuad     --protocol P1 --epochs 10
python scripts/run_experiment.py --model cdcn     --protocol P1 --epochs 10
python scripts/run_experiment.py --model xuad     --protocol P2.faceswap --epochs 5 --subsample 4800

# 3. figures, tables, demo
python scripts/make_figures.py
python scripts/demo.py --ckpt results/xuad_P1.pt
```

## Running on capture-based corpora

`xuad/data/real.py` implements the mapping in `configs/real_datasets.yaml` for
nine public corpora. Discovery is glob-based rather than path-hard-coded, every
run starts with a `--dry-run` inventory, splits are subject-disjoint (a
validation set is carved out of train for corpora that ship only train/test), and
FaceForensics++ mask videos are read as real localisation targets.

```bash
# 0. look before you leap -- one second, prints the full inventory
python scripts/prepare_real.py --root /data/CASIA-FASD --corpus auto --dry-run

# 1. everything, one command: cache -> P1 -> leave-one-attack-out -> figures/tables
bash scripts/run_real.sh /data/CASIA-FASD      casia-fasd    30 casia
bash scripts/run_real.sh /data/FaceForensics++ faceforensics 30 ffpp
bash scripts/run_real.sh /data/OULU-NPU        auto          30 oulu

# 2. cross-dataset: train on one corpus, test unchanged on another
python scripts/run_experiment.py --data cache_oulu --data-test cache_casia \
       --model xuad --protocol P3 --eval-checkpoint results_oulu/xuad_P1.pt \
       --tag xuad_P3 --out results_oulu
```

| Corpus | `--corpus` | Maps onto | Ships masks | Ships a protocol |
|---|---|---|---|---|
| CASIA-FASD | `casia-fasd` | bonafide, print, replay | no | train/test |
| Replay-Attack | `replay-attack` | bonafide, print, replay | no | train/devel/test |
| OULU-NPU | `oulu-npu` | bonafide, print, replay | no | train/dev/test + P1-P4 |
| SiW-Mv2 | `siw-mv2` | + mask3d (13 sub-types) | no | native leave-one-attack-out |
| CelebA-Spoof | `celeba-spoof` | + mask3d (pass `--label-json`) | no | train/test |
| FaceForensics++ | `faceforensics` | bonafide, faceswap, reenact | **yes** | no |
| Celeb-DF-v2 | `celeb-df` | bonafide, faceswap | no | test list |
| DFDC | `dfdc` | bonafide, faceswap | no | metadata.json |
| UniAttackData | `uniattackdata` | + physical & digital | no | train/dev/test |

Only the data layer changes: the model, losses, engine, metrics, explainability
and reasoning code are identical to the simulated-benchmark runs. Classes absent
from a corpus are dropped from the label space automatically, so a three-family
corpus is not scored against eight. Corpora with no annotated masks train with
`--no-loc`, which the driver sets for you.

### Verifying the adapters without the data

```bash
python tests/make_mock_corpora.py     # miniature corpora with the exact layouts
python tests/test_real_adapters.py    # 118 checks at the 2026-08-28 freeze, incl. training end-to-end
```

`tests/make_mock_corpora.py` writes real `.avi`/`.mp4`/`.png` files in each
corpus's own directory structure and filename convention. The regression test
asserts that every adapter recognises its own layout and no other's, that the
taxonomy mapping matches the published documentation, that splits are
subject-disjoint, that FF++ mask videos are targets rather than samples, and that
P1, leave-one-attack-out and the cross-dataset protocol all run to completion.

## Code map

| Path | What is in it |
|---|---|
| `xuad/models/aqa.py` | Adaptive Quantum Attention: Givens circuits, Born-rule channel, the adaptive gate, the orthonormal evidence basis |
| `xuad/models/stem.py` | Central-difference convolutions, the polar log-DFT spectral branch |
| `xuad/models/heads.py` | Decoupled direction/magnitude evidential head, localisation decoder, risk calibrator |
| `xuad/models/xuad.py` | The assembled detector |
| `xuad/models/baselines.py` | ResNet-18, ViT-Tiny, CDCN++ under the same interface |
| `xuad/losses.py` | Evidential loss, PUES, evidence-correctness calibration, coherence regulariser |
| `xuad/data/attacks.py` | The eight attack synthesisers and their evidence masks |
| `xuad/data/real.py` | Adapters for the nine public corpora |
| `xuad/xai/evidence.py` | Basis naming, attention rollout, deletion/insertion faithfulness |
| `xuad/xai/reasoning.py` | Evidence record, grounded realiser, constrained MLLM back-end |
| `xuad/engine/metrics.py` | ISO/IEC 30107-3 metrics, calibration, localisation, open-set |
| `scripts/` | Experiment runner, real-data preparation, analyses, figures, LaTeX tables |
| `paper/` | IEEEtran manuscript, bibliography, generated figures and tables |
| `paper/xuad_v2.tex` | Current manuscript (revision 2): revised prose, all figures, corrected metric definitions. See `paper/REVISION_NOTES.md` |
| `paper/xuad_v2_draft.pdf` | The revised-prose draft that revision 2 was built from (no figures; superseded by `xuad_v2.tex`) |
