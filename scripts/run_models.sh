#!/bin/bash
# Steps 8-9: every protocol x scheme for both frame windows, the ablations (5-frame window,
# 5-fold CV), the Vision Transformer (8b, 5 frames), calibration + ensembles (8c), the metrics.
# SKIP_VIT=1 ./scripts/run_models.sh leaves out the ViT (~3 h on an RTX 3080).
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
# length-matched check: only clips with >= 10 I-frames, so 5 vs 10 frames is a fair comparison
for window in 5 10; do
  python scripts/08_models.py --protocol realistic --scheme folds --window $window --min-frames 10
done
if [ "${SKIP_VIT:-0}" != 1 ]; then
  for protocol in realistic controlled; do
    for scheme in fixed folds; do
      python scripts/08b_vit.py --protocol $protocol --scheme $scheme
    done
  done
fi
python scripts/08c_ensemble.py
python scripts/09_evaluate.py
