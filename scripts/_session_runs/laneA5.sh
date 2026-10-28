#!/bin/bash
cd /root/xuad
R=/root/xuad/results
while ! grep -q LANE_A3_DONE $R/laneA3.log 2>/dev/null; do sleep 30; done
python3 scripts/run_experiment.py --model xuad --protocol P1 --epochs 10 --threads 1 --workers 1 > $R/xuad_P1.log 2>&1
echo LANE_A5_DONE
