#!/bin/bash
cd /root/xuad
bash scripts/waveD.sh >> /root/xuad/results/waveD.log 2>&1
bash scripts/waveE.sh B waveD.log synth adversarial >> /root/xuad/results/waveE_B.log 2>&1
echo LANE_B_DONE
