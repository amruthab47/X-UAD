#!/bin/bash
cd /root/xuad
R=/root/xuad/results
python3 scripts/run_experiment.py --model xuad --protocol P1 --epochs 10 --threads 1 --workers 1 > $R/xuad_P1.log 2>&1
echo XUAD_P1_DONE
S=4800; E=5
for C in faceswap adversarial; do
  for M in xuad cdcn resnet18; do
    python3 scripts/run_experiment.py --model $M --protocol P2.$C --epochs $E --subsample $S \
      --threads 1 --workers 1 --tag ${M}_P2.$C > $R/${M}_P2.$C.log 2>&1
  done
done
echo LANE_A3_DONE
