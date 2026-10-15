#!/bin/bash
# Phase 4: ACID attacks + more genuine platform copies (Step 5), their features (7, 7b), every
# Step 8 model, the ViT at 5 frames (all setups) and 10 frames (realistic), calibration /
# ensembles / safe operating point (8c) and the metrics (9). Step 10 (camera identification)
# runs on its own: `python scripts/10_camera_id.py`; the ViT waits for it (one GPU job at a time).
# Logs in data/logs/. Earlier results are kept in data/phase3/.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
while pgrep -f "scripts/05_attacks[.]py" > /dev/null; do sleep 60; done      # a Step 5 run started earlier
python scripts/05_attacks.py > data/logs/05_attacks_finish.log 2>&1         # completes anything missing
python scripts/07_features.py > data/logs/07_features.log 2>&1
python scripts/07b_noiseprint.py > data/logs/07b_noiseprint.log 2>&1
SKIP_VIT=1 ./scripts/run_models.sh > data/logs/models.log 2>&1 &
models=$!
while pgrep -f "scripts/10_camera_id[.]py" > /dev/null; do sleep 60; done
for protocol in realistic controlled; do
  for scheme in fixed folds; do
    python scripts/08b_vit.py --protocol $protocol --scheme $scheme
  done
done > data/logs/vit.log 2>&1
for scheme in fixed folds; do
  python scripts/08b_vit.py --protocol realistic --scheme $scheme --window 10
done > data/logs/vit_w10.log 2>&1
wait $models
python scripts/08c_ensemble.py > data/logs/08c_ensemble.log 2>&1
python scripts/09_evaluate.py > data/logs/09_evaluate.log 2>&1
echo "Phase 4 finished: data/results/metrics/results.md"
