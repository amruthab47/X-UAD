#!/bin/bash
cd /root/xuad
bash scripts/waveC.sh >> /root/xuad/results/waveC.log 2>&1
bash scripts/waveE.sh A waveC.log replay mask3d faceswap >> /root/xuad/results/waveE_A.log 2>&1
echo LANE_A_DONE
