#!/bin/bash
# Which Phase 4 addition lowered the scores? XGBoost + GRU retrained four times (realistic, 5-fold, 5 frames), each time
# leaving Phase 4 additions out of training and validation, then compared on the clips Phase 3 tested (~40 min, GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
for ex in acid_attacks new_genuine new_generators "acid_attacks new_genuine new_generators"; do
  python scripts/08_models.py --protocol realistic --scheme folds --models baseline xgb gru --exclude-train $ex
done > data/logs/diagnosis.log 2>&1
python scripts/diagnose_phase4.py | tee data/results/metrics/diagnosis_phase4.txt
