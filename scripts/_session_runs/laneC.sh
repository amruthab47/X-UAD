#!/bin/bash
# Open-set ablation: does PUES (and AQA) actually buy the novelty channel?
cd /root/xuad
R=/root/xuad/results
while ! grep -q LANE_B5_DONE $R/laneB5.log 2>/dev/null; do sleep 30; done
python3 scripts/run_experiment.py --model xuad --no-pues --protocol P2.faceswap --epochs 5 \
  --subsample 4800 --threads 1 --workers 1 --tag xuad_no_pues_P2.faceswap > $R/xuad_no_pues_P2.faceswap.log 2>&1
python3 scripts/run_experiment.py --model xuad_no_aqa --protocol P2.faceswap --epochs 5 \
  --subsample 4800 --threads 1 --workers 1 --tag xuad_no_aqa_P2.faceswap > $R/xuad_no_aqa_P2.faceswap.log 2>&1
echo LANE_C_DONE
