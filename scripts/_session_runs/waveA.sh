#!/bin/bash
# Main lane: X-UAD under the identical 10-epoch budget, then the cross-domain
# evaluation of every P1 checkpoint (no extra training).
cd /root/xuad
R=/root/xuad/results
python3 scripts/run_experiment.py --model xuad --protocol P1 --epochs 10 --threads 1 --workers 1 > $R/xuad_P1.log 2>&1
echo WAVE_A_DONE
