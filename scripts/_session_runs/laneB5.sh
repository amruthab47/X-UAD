#!/bin/bash
cd /root/xuad
R=/root/xuad/results
while ! grep -q LANE_B3_DONE $R/laneB3.log 2>/dev/null; do sleep 30; done
S=4000; E=5
ab () { python3 scripts/run_experiment.py --protocol P1 --epochs $E --subsample $S \
        --threads 1 --workers 1 "$@" ; }
ab --model xuad           --tag xuad_ab         > $R/xuad_ab.log 2>&1
ab --model xuad_no_aqa    --tag xuad_no_aqa_ab  > $R/xuad_no_aqa_ab.log 2>&1
ab --model xuad --no-pues --tag xuad_no_pues_ab > $R/xuad_no_pues_ab.log 2>&1
ab --model xuad_no_spec   --tag xuad_no_spec_ab > $R/xuad_no_spec_ab.log 2>&1
ab --model xuad_no_cdc    --tag xuad_no_cdc_ab  > $R/xuad_no_cdc_ab.log 2>&1
ab --model xuad --no-loc  --tag xuad_no_loc_ab  > $R/xuad_no_loc_ab.log 2>&1
echo ABLATIONS2_DONE
for C in faceswap replay; do
  python3 scripts/run_experiment.py --model xuad --protocol P2.$C --epochs 5 --subsample 4800 \
    --threads 1 --workers 1 --tag xuad_P2.$C > $R/xuad_P2.$C.log 2>&1
done
echo LANE_B5_DONE
