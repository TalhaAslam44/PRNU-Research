#!/bin/bash
# Steps 8-9: every protocol x scheme with all features, the ablation runs, then the metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
for protocol in realistic controlled; do
  for scheme in fixed folds; do
    python scripts/08_models.py --protocol $protocol --scheme $scheme
  done
  for group in prnu resid fft tsncs noiseprint; do      # ablation (5-fold CV)
    python scripts/08_models.py --protocol $protocol --scheme folds --drop $group
  done
done
python scripts/09_evaluate.py
