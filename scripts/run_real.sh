#!/bin/bash
# One-command X-UAD run on a real capture-based corpus.
#
#   bash scripts/run_real.sh <dataset-root> [corpus] [epochs] [out-tag]
#
#   bash scripts/run_real.sh /data/CASIA-FASD casia-fasd 30 casia
#   bash scripts/run_real.sh /data/FaceForensics++ faceforensics 30 ffpp
#   bash scripts/run_real.sh /data/OULU-NPU auto 30 oulu
#
# Runs, in order: inventory -> cache build -> P1 (seen attacks) for X-UAD and the
# three baselines -> leave-one-attack-out for every attack family the corpus
# actually contains -> the post-hoc analyses, figures and LaTeX tables.
#
# Cross-dataset (train here, test there) is one extra command, printed at the end.
set -euo pipefail

ROOT=${1:?usage: run_real.sh <dataset-root> [corpus] [epochs] [tag]}
CORPUS=${2:-auto}
EPOCHS=${3:-30}
TAG=${4:-real}
FRAMES=${FRAMES:-8}
WORKERS=${WORKERS:-8}
THREADS=${THREADS:-8}
BS=${BS:-64}

cd "$(dirname "$0")/.."
CACHE="cache_${TAG}"
RES="results_${TAG}"
mkdir -p "$RES"

echo "=============================================================="
echo " 1/5  inventory"
echo "=============================================================="
python3 scripts/prepare_real.py --root "$ROOT" --corpus "$CORPUS" --dry-run \
        --frames-per-video "$FRAMES" | tee "$RES/inventory.txt"

echo
echo "=============================================================="
echo " 2/5  building cache -> $CACHE"
echo "=============================================================="
python3 scripts/prepare_real.py --root "$ROOT" --corpus "$CORPUS" --out "$CACHE" \
        --frames-per-video "$FRAMES" --workers "$WORKERS"

# corpora without annotated forgery masks must not fit the localisation head
NOLOC=""
if ! grep -q '"has_masks": true' "$CACHE/A_source.json"; then
  NOLOC="--no-loc"
  echo "note: no annotated masks in this corpus -> training with $NOLOC"
fi
CLASSES=$(python3 - "$CACHE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/A_source.json"))
print(",".join(c for c in m["classes_present"] if c != "bonafide"))
PY
)
echo "attack families present: $CLASSES"

echo
echo "=============================================================="
echo " 3/5  protocol P1 - seen attacks, all four detectors"
echo "=============================================================="
for M in xuad cdcn resnet18 vit_tiny; do
  echo "--- $M"
  python3 scripts/run_experiment.py --data "$CACHE" --model $M --protocol P1 \
      --epochs "$EPOCHS" --bs "$BS" --threads "$THREADS" --workers 2 $NOLOC \
      --tag "${M}_P1" --out "$RES" 2>&1 | tail -n 24
done

echo
echo "=============================================================="
echo " 4/5  protocol P2 - leave-one-attack-out"
echo "=============================================================="
IFS=',' read -ra FAMILIES <<< "$CLASSES"
for C in "${FAMILIES[@]}"; do
  [ -z "$C" ] && continue
  for M in xuad cdcn resnet18; do
    echo "--- hold out '$C', model $M"
    python3 scripts/run_experiment.py --data "$CACHE" --model $M --protocol "P2.$C" \
        --epochs "$EPOCHS" --bs "$BS" --threads "$THREADS" --workers 2 $NOLOC \
        --tag "${M}_P2.$C" --out "$RES" 2>&1 | tail -n 6
  done
done

echo
echo "=============================================================="
echo " 5/5  analyses, figures and tables"
echo "=============================================================="
python3 scripts/openset.py || true
python3 scripts/analysis.py --what all --ckpt "$RES/xuad_P1.pt" --out "$RES" \
        --threads "$THREADS" || true
python3 scripts/make_qualitative.py "$RES/xuad_P1.pt" "$RES" || true
python3 scripts/make_figures.py || true
python3 scripts/make_tables.py || true

echo
echo "done.  results in $RES/, figures in paper/figures/, tables in paper/tables/"
echo
echo "cross-dataset (train on this corpus, test on another):"
echo "  bash scripts/run_real.sh /data/OTHER-CORPUS auto $EPOCHS other   # build its cache"
echo "  python3 scripts/run_experiment.py --data $CACHE --data-test cache_other \\"
echo "      --model xuad --protocol P3 --eval-checkpoint $RES/xuad_P1.pt --tag xuad_P3 --out $RES"
