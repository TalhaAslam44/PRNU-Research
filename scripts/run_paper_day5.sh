#!/bin/bash
# Paper, day 5 (~3.5 h, GPU): second hardening round against the adaptive removal attack.
#   1. round-2 clips (weak removal +- noise, injection after own-PRNU removal, matching genuine re-encodes) at
#      CRF 22 / 30 for the VISION training + validation cameras, with features    (paper_crf_aug.py --round 2)
#   2. the final HVPF retrained on round 1 + 2, saved as data/paper/bundle_aug2_fixed_w10   (paper_bundle.py)
#   3. the same 3,300 adaptive-attack clips re-scored                                     (paper_adaptive.py)
# None of the round-2 settings is in the adaptive attacker's grid. Writes only to data/paper/.
# Result: data/paper/adaptive/summary_aug2.txt   Logs: data/logs/paper_aug2*.log
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
while pgrep -f "run_paper_day[234].sh" > /dev/null; do sleep 120; done
ls ~/DATASETS/Vision-Dataset > /dev/null || { echo "external drive not mounted (~/DATASETS)"; exit 1; }
echo "$(date '+%F %T') 1/3 round-2 clips for training cameras"
[ -f data/paper/features_aug2/noiseprint_videos.parquet ] || python scripts/paper_crf_aug.py --round 2 >> data/logs/paper_aug2.log 2>&1
echo "$(date '+%F %T') 2/3 retraining the final HVPF (rounds 1 + 2)"
[ -f data/paper/bundle_aug2_fixed_w10/predictions.parquet ] || python scripts/paper_bundle.py \
  --features data/paper/features_aug2 --out data/paper/bundle_aug2_fixed_w10 >> data/logs/paper_aug2_bundle.log 2>&1
echo "$(date '+%F %T') 3/3 re-scoring the adaptive attack"
python scripts/paper_adaptive.py --predict-only --bundle data/paper/bundle_aug2_fixed_w10 --suffix _aug2 \
  >> data/logs/paper_aug2_adaptive.log 2>&1
echo "$(date '+%F %T') day 5 done: data/paper/adaptive/summary_aug2.txt"
