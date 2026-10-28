#!/bin/bash
# Cross-domain evaluation of the P1 checkpoints (no training), then all the
# post-hoc analyses, figures and LaTeX tables.
cd /root/xuad
R=/root/xuad/results
for M in xuad cdcn resnet18 vit_tiny; do
  if [ -f $R/${M}_P1.pt ]; then
    python3 scripts/run_experiment.py --model $M --protocol P3 --tag ${M}_P3 \
      --eval-checkpoint $R/${M}_P1.pt --threads 2 --workers 1 > $R/${M}_P3.log 2>&1
  fi
done
python3 scripts/analysis.py --what all --ckpt $R/xuad_P1.pt --threads 2 > $R/analysis.log 2>&1
python3 scripts/make_qualitative.py $R/xuad_P1.pt                        > $R/qualitative.log 2>&1
python3 scripts/make_figures.py                                          > $R/figures.log 2>&1
python3 scripts/make_tables.py                                           > $R/tables.log 2>&1
echo FINALIZE_DONE
