#!/bin/bash
# Thesis: PDC-ViT exactly as described in the paper (3-channel PDC branches, Adam, lr 3e-5) with a longer
# training budget (30 epochs, patience 6 instead of 15 / 3), so that the baseline is not under-trained:
#   1. fixed split  (validation score, fixed-split test cameras, unseen datasets)   ~1 h
#   2. 5 camera folds (unseen cameras)                                              up to ~9 h
#   3. HVPF vs every PDC-ViT variant again (data/thesis/pdcvit/compare.txt)
# Output tag: paper_long. Finished parts are skipped on a re-run; a second copy refuses to start.
#
#   setsid nohup ./scripts/run_thesis_pdcvit_long.sh >> data/logs/run_thesis_pdcvit_long.log 2>&1 < /dev/null &
# Training log: data/logs/thesis_pdcvit_long.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs data/thesis/pdcvit
exec 9> data/logs/.run_thesis_pdcvit_long.lock
flock -n 9 || { echo "$(date '+%F %T') already running - not started again"; exit 1; }
source .venv/bin/activate
ARGS="--tag paper_long --recipe paper --channels 3 --epochs 30 --patience 6"
echo "$(date '+%F %T') PDC-ViT (paper, long budget) started"
echo "$(date '+%F %T') 1/3 fixed split"
python scripts/thesis_pdcvit.py --scheme fixed $ARGS >> data/logs/thesis_pdcvit_long.log 2>&1
echo "$(date '+%F %T') 2/3 five folds"
python scripts/thesis_pdcvit.py --scheme folds $ARGS >> data/logs/thesis_pdcvit_long.log 2>&1
echo "$(date '+%F %T') 3/3 comparison with HVPF"
python scripts/thesis_pdcvit_compare.py > data/logs/thesis_pdcvit_compare.log 2>&1
echo "$(date '+%F %T') PDC-ViT long done: data/thesis/pdcvit/compare.txt"
