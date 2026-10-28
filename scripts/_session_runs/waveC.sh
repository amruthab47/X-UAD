#!/bin/bash
# Lane A continuation: ablation sweep at a reduced, *shared* budget, then the
# cross-domain evaluation of the P1 checkpoints (evaluation only, no training).
cd /root/xuad
R=/root/xuad/results
while ! grep -q WAVE_A_DONE $R/waveA.log 2>/dev/null; do sleep 30; done

SUB=6400
EP=6
run () { python3 scripts/run_experiment.py --protocol P1 --epochs $EP --subsample $SUB \
         --threads 1 --workers 1 "$@" ; }

run --model xuad          --tag xuad_ab        > $R/xuad_ab.log 2>&1
run --model xuad_no_aqa   --tag xuad_no_aqa_ab > $R/xuad_no_aqa_ab.log 2>&1
run --model xuad_no_evid  --tag xuad_no_evid_ab> $R/xuad_no_evid_ab.log 2>&1
run --model xuad --no-pues --tag xuad_no_pues_ab > $R/xuad_no_pues_ab.log 2>&1
echo WAVE_C_DONE
