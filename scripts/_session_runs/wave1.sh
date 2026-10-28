set -e
cd /root/xuad
python3 scripts/run_experiment.py --model vit_tiny --protocol P1 --epochs 6 --threads 1 --workers 1 > results/vit_tiny_P1.log 2>&1
python3 scripts/run_experiment.py --model cdcn     --protocol P1 --epochs 6 --threads 1 --workers 1 > results/cdcn_P1.log 2>&1
python3 scripts/run_experiment.py --model resnet18 --protocol P1 --epochs 6 --threads 1 --workers 1 > results/resnet18_P1.log 2>&1
