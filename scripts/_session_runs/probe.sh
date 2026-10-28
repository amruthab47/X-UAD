#!/bin/bash
# Fast, budget-matched hyper-parameter probe: identical data (small subsample) and
# identical schedule length for every configuration, so the comparison is fair.
cd /root/xuad
R=/root/xuad/results/probe
mkdir -p $R
S=3200
E=3
p () { TAG=$1; shift; python3 scripts/run_experiment.py --protocol P1 --epochs $E \
       --subsample $S --threads 1 --workers 0 --tag $TAG --out $R "$@" > $R/$TAG.log 2>&1; }

p pr_base   --model xuad
p pr_lr12   --model xuad --lr 1.2e-3
p pr_lr40   --model xuad --lr 4.0e-3
p pr_auxoff --model xuad --aux off
p pr_ce     --model xuad_no_evid
p pr_cdcn   --model cdcn
echo PROBE_DONE
