#!/bin/bash
# Phase 5 (experiment): changes of hvpf/phase5.py - 1 processing-matched training, 2 reliability-aware fusion,
# 3 inconclusive output, 4 camera-calibrated PRNU evidence. New scripts only (scripts/p5_*.py); the Phase 3 /
# Phase 4 code and results are not changed. Reads data/phase4/features, writes data/phase5/. Finished parts are
# skipped on re-runs, so the script can simply be started again after an interruption.
#
#   ./scripts/run_phase5.sh          changes 1-3: feature models (~1.5 h, GPU) + fusion + comparison
#   ./scripts/run_phase5.sh --c4     also change 4, alone and with change 1 (~1.5 h, GPU)
#   ./scripts/run_phase5.sh --vit    also the processing-matched ViT (~7 h, GPU)
# Result: data/phase5/metrics/phase5_comparison.txt (logs: data/logs/p5_*.log)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs data/phase5
echo "$(date '+%F %T') Phase 5 started ($*)"
for w in 5 10; do
  for s in folds fixed; do
    [ -f data/phase5/models/realistic_${s}_all_w${w}.parquet ] && continue
    echo "$(date '+%F %T') change 1: feature models ${s} w${w}"
    python scripts/p5_models.py --scheme $s --window $w >> data/logs/p5_models.log 2>&1
  done
done
if [[ " $* " == *" --c4 "* ]]; then
  for w in 5 10; do
    for s in folds fixed; do
      for run in c4 c4m; do
        [ -f data/phase5/models_${run}/realistic_${s}_all_w${w}.parquet ] && continue
        echo "$(date '+%F %T') change 4 (${run}): feature models ${s} w${w}"
        python scripts/p5_c4_models.py --scheme $s --window $w $([ $run = c4m ] && echo --matched) \
          >> data/logs/p5_c4_models.log 2>&1
      done
    done
  done
fi
if [[ " $* " == *" --vit "* ]]; then
  for w in 5 10; do
    for s in folds fixed; do
      [ -f data/phase5/models/vit/realistic_${s}_w${w}.parquet ] && continue
      echo "$(date '+%F %T') change 1: ViT ${s} w${w}"
      python scripts/p5_vit.py --scheme $s --window $w >> data/logs/p5_vit.log 2>&1
    done
  done
fi
echo "$(date '+%F %T') fusion + comparison"
python scripts/p5_fusion.py > data/logs/p5_fusion.log 2>&1
python scripts/p5_compare.py > data/logs/p5_compare.log 2>&1
echo "$(date '+%F %T') Phase 5 done: data/phase5/metrics/phase5_comparison.txt"
