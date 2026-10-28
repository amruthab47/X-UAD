#!/bin/bash
cd /root/xuad
R=/root/xuad/results
for M in xuad cdcn resnet18 vit_tiny; do
  python3 scripts/run_experiment.py --model $M --protocol P3 --tag ${M}_P3 \
    --eval-checkpoint $R/${M}_P1.pt --threads 1 --workers 1 > $R/${M}_P3.log 2>&1
done
echo P3_DONE
python3 scripts/analysis.py --what all --ckpt $R/xuad_P1.pt --threads 1 > $R/analysis.log 2>&1
echo ANALYSIS_DONE
