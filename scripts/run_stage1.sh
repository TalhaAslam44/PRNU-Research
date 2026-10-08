#!/bin/bash
# Phase 2, stage 1: inventory -> splits -> I-frames -> fingerprints (ACID, FloreView).
# Everything is also written to data/logs/stage1.log.
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
run() { echo; echo "=== $(date '+%H:%M') $*"; python "$@"; }
{
  run scripts/01_inventory.py
  run scripts/02_splits.py
  run scripts/03_extract_frames.py
  run scripts/04_prnu_baseline.py --datasets ACID
  run scripts/04_prnu_baseline.py --datasets FloreView
  echo; echo "=== $(date '+%H:%M') STAGE 1 DONE"
} 2>&1 | tee data/logs/stage1.log
