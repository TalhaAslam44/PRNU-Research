#!/bin/bash
# Experiment (separate from the Phase 3 / Phase 4 runs, which never use it): dataset-balanced training. Inside every
# class each data source (VISION, ACID, AI) gets an equal share of the training weight, so the larger ACID attack set
# cannot make genuine ACID clips look suspicious. XGBoost + GRU, realistic, 5-fold, 5 frames (~15 min, GPU).
# Output: data/results/models/realistic_folds_all_w5_baldata.parquet (Phase 4 results are not touched).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
python scripts/08_models.py --protocol realistic --scheme folds --models baseline xgb gru --balance-datasets \
  > data/logs/balanced.log 2>&1
python scripts/compare_training.py | tee data/results/metrics/compare_training.txt
