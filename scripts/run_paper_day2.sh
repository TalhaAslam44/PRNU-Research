#!/bin/bash
# Paper, day 2 (overnight, ~7 h, GPU): adaptive attack against the final HVPF.
#   1. Noiseprint++ features recomputed on this PC for all clips   (scripts/paper_noiseprint.py, ~4.7 h, resumes)
#   2. final HVPF trained once on the fixed split and saved           (scripts/paper_bundle.py, ~1 h)
#   3. adaptive attack on the 55 test-camera VISION videos            (scripts/paper_adaptive.py, ~45 min)
# Needs the external drive "New Volume" (raw VISION videos for step 3). Writes only to data/paper/.
# Result: data/paper/adaptive/summary.txt   Logs: data/logs/paper_*.log
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs data/paper
ls ~/DATASETS/Vision-Dataset > /dev/null || { echo "external drive not mounted (~/DATASETS)"; exit 1; }
echo "$(date '+%F %T') day 2 started"
if [ ! -f data/paper/features_newpc/noiseprint_videos.parquet ]; then
  echo "$(date '+%F %T') 1/3 Noiseprint++ on this PC"
  python scripts/paper_noiseprint.py >> data/logs/paper_noiseprint.log 2>&1
fi
if [ ! -f data/paper/bundle_fixed_w10/predictions.parquet ]; then
  echo "$(date '+%F %T') 2/3 training and saving the final HVPF"
  python scripts/paper_bundle.py >> data/logs/paper_bundle.log 2>&1
fi
echo "$(date '+%F %T') 3/3 adaptive attack"
python scripts/paper_adaptive.py >> data/logs/paper_adaptive.log 2>&1
echo "$(date '+%F %T') day 2 done: data/paper/adaptive/summary.txt"
