#!/bin/bash
# Leave-one-attack-out sweep (both lanes feed this once the ablations finish).
# $1 = lane name, remaining args = the fold list for this lane.
cd /root/xuad
R=/root/xuad/results
LANE=$1; shift
WAIT=$1; shift
while ! grep -q DONE $R/$WAIT 2>/dev/null; do sleep 30; done

SUB=6400
EP=6
for C in "$@"; do
  for M in xuad cdcn resnet18; do
    python3 scripts/run_experiment.py --model $M --protocol P2.$C --epochs $EP \
      --subsample $SUB --threads 1 --workers 1 --tag ${M}_P2.$C > $R/${M}_P2.$C.log 2>&1
  done
done
echo WAVE_E_${LANE}_DONE
