#!/bin/bash
# Phase 5 (experiment): changes 1-3 of hvpf/phase5.py - processing-matched training, reliability-aware
# fusion, inconclusive output. New scripts only (scripts/p5_*.py); the Phase 3 / Phase 4 code and results
# are not changed. Reads data/phase4/features, writes data/phase5/. Finished parts are skipped on re-runs.
#
#   ./scripts/run_phase5.sh          feature models SVM/XGB/GRU/CNN/Transformer (~1.5 h, GPU) + fusion + comparison
#   ./scripts/run_phase5.sh --vit    also the processing-matched ViT (~7 h more, GPU), then fusion + comparison again
# Result: data/phase5/metrics/phase5_comparison.txt (logs: data/logs/p5_*.log)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs data/phase5
echo "$(date '+%F %T') Phase 5 started"
for w in 5 10; do
  for s in folds fixed; do
    [ -f data/phase5/models/realistic_${s}_all_w${w}.parquet ] && continue
    echo "$(date '+%F %T') feature models ${s} w${w}"
    python scripts/p5_models.py --scheme $s --window $w >> data/logs/p5_models.log 2>&1
  done
done
if [ "${1:-}" = "--vit" ]; then
  for w in 5 10; do
    for s in folds fixed; do
      [ -f data/phase5/models/vit/realistic_${s}_w${w}.parquet ] && continue
      echo "$(date '+%F %T') ViT ${s} w${w}"
      python scripts/p5_vit.py --scheme $s --window $w >> data/logs/p5_vit.log 2>&1
    done
  done
fi
echo "$(date '+%F %T') fusion + comparison"
python scripts/p5_fusion.py > data/logs/p5_fusion.log 2>&1
python scripts/p5_compare.py > data/logs/p5_compare.log 2>&1
echo "$(date '+%F %T') Phase 5 done: data/phase5/metrics/phase5_comparison.txt"
