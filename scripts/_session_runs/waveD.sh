#!/bin/bash
# Lane B continuation: the remaining ablations, then the leave-one-attack-out
# sweep for X-UAD, CDCN++ and ResNet-18 at the shared reduced budget.
cd /root/xuad
R=/root/xuad/results
while ! grep -q WAVE_B_DONE $R/waveB.log 2>/dev/null; do sleep 30; done

SUB=6400
EP=6
run () { python3 scripts/run_experiment.py --epochs $EP --subsample $SUB \
         --threads 1 --workers 1 "$@" ; }

run --model xuad_no_spec --protocol P1 --tag xuad_no_spec_ab > $R/xuad_no_spec_ab.log 2>&1
run --model xuad_no_cdc  --protocol P1 --tag xuad_no_cdc_ab  > $R/xuad_no_cdc_ab.log 2>&1
run --model xuad --no-loc --protocol P1 --tag xuad_no_loc_ab > $R/xuad_no_loc_ab.log 2>&1
echo WAVE_D_DONE
