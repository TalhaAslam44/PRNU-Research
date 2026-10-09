#!/bin/bash
# Steps 8-9: every protocol x scheme for both frame windows, the ablations (5-frame window,
# 5-fold CV), then the metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
for protocol in realistic controlled; do
  for window in 5 10; do
    for scheme in fixed folds; do
      python scripts/08_models.py --protocol $protocol --scheme $scheme --window $window
    done
  done
  for group in prnu resid fft tsncs noiseprint; do      # ablation
    python scripts/08_models.py --protocol $protocol --scheme folds --window 5 --drop $group
  done
done
python scripts/09_evaluate.py
