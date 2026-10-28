#!/bin/bash
# Baseline lane: every model gets the identical 10-epoch budget.
cd /root/xuad
R=/root/xuad/results
python3 scripts/run_experiment.py --model vit_tiny --protocol P1 --epochs 10 --threads 1 --workers 1 > $R/vit_tiny_P1.log 2>&1
python3 scripts/run_experiment.py --model cdcn     --protocol P1 --epochs 10 --threads 1 --workers 1 > $R/cdcn_P1.log 2>&1
python3 scripts/run_experiment.py --model resnet18 --protocol P1 --epochs 10 --threads 1 --workers 1 > $R/resnet18_P1.log 2>&1
echo WAVE_B_DONE
