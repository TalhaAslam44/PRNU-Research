#!/bin/bash
# Phase 2, stage 2: genuine re-encodes + platform-like copies (Step 5), synthetic platform
# copies (Step 6), features for the 5- and 10-frame windows (Steps 7, 7b), models and
# results (Steps 8-9). Phase 1 features/results are kept in data/phase1/.
# Everything is also written to data/logs/stage2.log.
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
run() { echo; echo "=== $(date '+%d %H:%M') $*"; python "$@"; }
{
  while pgrep -f "scripts/run_stage[1].sh" > /dev/null; do sleep 30; done  # Stage 1 must finish first ([1]: never matches this line itself)
  if [ ! -d data/phase1 ]; then
    mkdir -p data/phase1
    mv data/features data/phase1/features
    mv data/results/models data/phase1/models
    mv data/results/metrics data/phase1/metrics
    echo "phase 1 features and results moved to data/phase1/"
  fi
  run scripts/05_attacks.py
  run scripts/06_synthetic.py
  echo; echo "=== $(date '+%d %H:%M') Step 7 + Step 7b in parallel"
  touch data/logs/.stage2_step7_start
  python scripts/07_features.py > data/logs/stage2_07.log 2>&1 &
  p7=$!
  until [ data/features/samples.csv -nt data/logs/.stage2_step7_start ]; do sleep 10; done
  python scripts/07b_noiseprint.py > data/logs/stage2_07b.log 2>&1
  wait $p7
  tail -40 data/logs/stage2_07.log | tr '\r' '\n' | grep -v "features:" | tail -30
  tail -30 data/logs/stage2_07b.log | tr '\r' '\n' | grep -v "noiseprint:" | tail -20
  echo; echo "=== $(date '+%d %H:%M') Steps 8-9"
  ./scripts/run_models.sh
  echo; echo "=== $(date '+%d %H:%M') STAGE 2 DONE"
} 2>&1 | tee data/logs/stage2.log
