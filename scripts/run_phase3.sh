#!/bin/bash
# Phase 3 in one go: features for the new clips (7, 7b; earlier results are reused), then the
# Step 8 models and the ViT in parallel on the GPU, then calibration + ensembles and the metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
python scripts/07_features.py > data/logs/07_features.log 2>&1
python scripts/07b_noiseprint.py > data/logs/07b_noiseprint.log 2>&1
SKIP_VIT=1 ./scripts/run_models.sh > data/logs/models.log 2>&1 &
models=$!
for protocol in realistic controlled; do
  for scheme in fixed folds; do
    python scripts/08b_vit.py --protocol $protocol --scheme $scheme
  done
done > data/logs/vit.log 2>&1
wait $models
python scripts/08c_ensemble.py > data/logs/08c_ensemble.log 2>&1
python scripts/09_evaluate.py > data/logs/09_evaluate.log 2>&1
echo "Phase 3 finished: data/results/metrics/results.md"
