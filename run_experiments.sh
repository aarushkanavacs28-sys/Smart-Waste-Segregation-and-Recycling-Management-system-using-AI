#!/usr/bin/env bash
set -e

python scripts/download_realwaste.py
python scripts/prepare_realwaste.py --source data/raw/realwaste --output data/processed --seed 42 --overwrite

python -m robustwaste.train --data-dir data/processed --model resnet50 --run-name resnet50_baseline --epochs 20
python -m robustwaste.train --data-dir data/processed --model efficientnet_b0 --run-name efficientnet_b0_baseline --epochs 20
python -m robustwaste.train --data-dir data/processed --model mobilenet_v3_large --run-name mobilenet_v3_baseline --epochs 20
python -m robustwaste.train --data-dir data/processed --model mobilenet_v3_large --run-name robustwaste_proposed --epochs 25 --degradation-aware

python -m robustwaste.evaluate --data-dir data/processed --checkpoint outputs/robustwaste_proposed/best.pt --split test
python -m robustwaste.threshold --data-dir data/processed --checkpoint outputs/robustwaste_proposed/best.pt --target-accepted-accuracy 0.95
python -m robustwaste.robustness --data-dir data/processed --checkpoint outputs/robustwaste_proposed/best.pt
python -m robustwaste.ood --data-dir data/processed --checkpoint outputs/robustwaste_proposed/best.pt
python -m robustwaste.compare --data-dir data/processed --outputs-dir outputs
