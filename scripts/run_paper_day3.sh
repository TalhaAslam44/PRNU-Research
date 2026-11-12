#!/bin/bash
# Paper, day 3 (~4-5 h, GPU): same-protocol baselines (scripts/paper_baselines.py), 5-fold CV + fixed split.
# Waits until the day 2 run has finished (one GPU job at a time). CLIP / DINOv2 run only when the
# `transformers` package is installed. Finished parts are skipped on re-runs.
# Results: data/paper/baselines/<model>/   Log: data/logs/paper_baselines.log
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
echo "$(date '+%F %T') day 3 queued; waiting for day 2 to finish"
while pgrep -f "run_paper_day2.sh" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') day 3 started"
models="effnet_b4"
python -c "import transformers" 2> /dev/null && models="effnet_b4 clip_b16 dinov2_b" \
  || echo "$(date '+%F %T') transformers not installed: CLIP / DINOv2 baselines skipped"
for m in $models; do
  for s in folds fixed; do
    [ -f data/paper/baselines/$m/realistic_${s}_w10.parquet ] && continue
    echo "$(date '+%F %T') baseline $m, $s"
    python scripts/paper_baselines.py --model $m --scheme $s >> data/logs/paper_baselines.log 2>&1 \
      || echo "$(date '+%F %T') baseline $m $s FAILED (see data/logs/paper_baselines.log)"
  done
done
echo "$(date '+%F %T') day 3 done: data/paper/baselines/"
