#!/bin/bash
# Build the delivery bundle: source, configs, paper (tex + pdf + figures + tables),
# results JSONs and logs.  Model checkpoints and the 1.6 GB benchmark tensors are
# excluded; the benchmark is regenerated deterministically by build_benchmark.py.
set -e
cd /root
OUT=/root/xuad_delivery
rm -rf $OUT && mkdir -p $OUT
cp -r xuad/xuad $OUT/
cp -r xuad/scripts $OUT/
cp -r xuad/configs $OUT/
cp -r xuad/tests $OUT/
cp xuad/README.md $OUT/
mkdir -p $OUT/paper
cp xuad/paper/xuad.tex xuad/paper/refs.bib $OUT/paper/
cp -r xuad/paper/figures $OUT/paper/
cp -r xuad/paper/tables $OUT/paper/
[ -f xuad/paper/xuad.pdf ] && cp xuad/paper/xuad.pdf $OUT/paper/
mkdir -p $OUT/results
cp xuad/results/*.json $OUT/results/ 2>/dev/null || true
cp xuad/results/reports.txt $OUT/results/ 2>/dev/null || true
mkdir -p $OUT/results/logs
cp xuad/results/*.log $OUT/results/logs/ 2>/dev/null || true
find $OUT -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cat > $OUT/requirements.txt <<'EOF'
torch>=2.1
torchvision>=0.16
numpy>=1.24
opencv-python>=4.8
scikit-learn>=1.3
matplotlib>=3.7
EOF
cd /root && zip -qr /root/X-UAD.zip xuad_delivery && ls -la /root/X-UAD.zip
