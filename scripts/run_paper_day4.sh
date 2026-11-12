#!/bin/bash
# Paper, day 4 (~3 h, GPU): fix the adaptive attack's compression shortcut and test the fix.
#   1. attacks at CRF 18 / 23 / 28 for the VISION training + validation cameras, with features   (paper_crf_aug.py)
#   2. the final HVPF retrained with them, saved as data/paper/bundle_aug_fixed_w10            (paper_bundle.py)
#   3. the same 3,300 adaptive-attack clips re-scored with the retrained HVPF                  (paper_adaptive.py)
# Waits until day 3 has finished (one GPU job at a time). Writes only to data/paper/.
# Result: data/paper/adaptive/summary_aug.txt (vs summary.txt)   Logs: data/logs/paper_aug*.log
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
echo "$(date '+%F %T') day 4 queued; waiting for day 3 to finish"
while pgrep -f "run_paper_day[23].sh" > /dev/null; do sleep 120; done
ls ~/DATASETS/Vision-Dataset > /dev/null || { echo "external drive not mounted (~/DATASETS)"; exit 1; }
echo "$(date '+%F %T') 1/3 attacks at CRF 18/23/28 for training cameras"
[ -f data/paper/features_aug/noiseprint_videos.parquet ] || python scripts/paper_crf_aug.py >> data/logs/paper_aug.log 2>&1
echo "$(date '+%F %T') 2/3 retraining the final HVPF with them"
[ -f data/paper/bundle_aug_fixed_w10/predictions.parquet ] || python scripts/paper_bundle.py \
  --features data/paper/features_aug --out data/paper/bundle_aug_fixed_w10 >> data/logs/paper_aug_bundle.log 2>&1
echo "$(date '+%F %T') 3/3 re-scoring the adaptive attack"
python scripts/paper_adaptive.py --predict-only --bundle data/paper/bundle_aug_fixed_w10 --suffix _aug \
  >> data/logs/paper_aug_adaptive.log 2>&1
echo "$(date '+%F %T') day 4 done: data/paper/adaptive/summary_aug.txt"
