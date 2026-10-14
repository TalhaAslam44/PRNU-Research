#!/bin/bash
# hvpf-2 in one go (~5-6 h on an RTX 3080), logs in data/logs/:
#   Step 8   provenance models, 80/20 (realistic + controlled, fixed split + 5 folds)    models.log
#   Step 8b  ViT, 80/20 (same 4 setups; runs on the GPU next to Step 8)                   vit.log
#   Step 10  trained camera identification (VISION, ACID, FloreView)                     camera_id.log
#   Step 8c  calibration + ensembles, Step 9 metrics, the report PDF
# Steps 1-7 are not re-run: hvpf-2 uses the same splits, features and frames as the original.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs
(
  for protocol in realistic controlled; do
    for scheme in fixed folds; do
      python scripts/08_models.py --protocol $protocol --scheme $scheme
    done
  done
) > data/logs/models.log 2>&1 &
models=$!
for protocol in realistic controlled; do
  for scheme in fixed folds; do
    python scripts/08b_vit.py --protocol $protocol --scheme $scheme
  done
done > data/logs/vit.log 2>&1
python scripts/10_camera_id.py > data/logs/camera_id.log 2>&1
wait $models
python scripts/08c_ensemble.py > data/logs/08c_ensemble.log 2>&1
python scripts/09_evaluate.py > data/logs/09_evaluate.log 2>&1
python scripts/make_hvpf2_report.py > data/logs/report.log 2>&1
~/.local/bin/tectonic -X compile report/results_hvpf2.tex >> data/logs/report.log 2>&1
echo "hvpf-2 finished: report/results_hvpf2.pdf"
