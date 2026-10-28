#!/bin/bash
cd /root/xuad
R=/root/xuad/results
python3 scripts/run_experiment.py --model vit_tiny --protocol P1 --epochs 10 --threads 1 --workers 1 > $R/vit_tiny_P1.log 2>&1
S=4000; E=5
ab () { python3 scripts/run_experiment.py --protocol P1 --epochs $E --subsample $S \
        --threads 1 --workers 1 "$@" ; }
ab --model xuad           --tag xuad_ab          > $R/xuad_ab.log 2>&1
ab --model xuad_no_aqa    --tag xuad_no_aqa_ab   > $R/xuad_no_aqa_ab.log 2>&1
ab --model xuad_no_evid   --tag xuad_no_evid_ab  > $R/xuad_no_evid_ab.log 2>&1
ab --model xuad --no-pues --tag xuad_no_pues_ab  > $R/xuad_no_pues_ab.log 2>&1
ab --model xuad_no_spec   --tag xuad_no_spec_ab  > $R/xuad_no_spec_ab.log 2>&1
ab --model xuad_no_cdc    --tag xuad_no_cdc_ab   > $R/xuad_no_cdc_ab.log 2>&1
ab --model xuad --no-loc  --tag xuad_no_loc_ab   > $R/xuad_no_loc_ab.log 2>&1
echo ABLATIONS_DONE
S=4800
for C in replay mask3d; do
  for M in xuad cdcn resnet18; do
    python3 scripts/run_experiment.py --model $M --protocol P2.$C --epochs $E --subsample $S \
      --threads 1 --workers 1 --tag ${M}_P2.$C > $R/${M}_P2.$C.log 2>&1
  done
done
echo LANE_B3_DONE
