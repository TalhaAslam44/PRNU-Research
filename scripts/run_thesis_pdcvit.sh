#!/bin/bash
# Thesis: the base paper's method (PDC-ViT) on our task and splits (GPU, ~6 h in total).
#   1. three variants on the fixed split (~40 min each): "paper" = as described in the paper (PDC branches of
#      3 channels, Adam, lr 3e-5), "wide" = 32-channel PDC branches, "ours" = 32 channels + our ViT's recipe
#   2. the variant with the best VALIDATION macro F1 is trained on the 5 camera folds (~3-4 h)
#   3. HVPF vs PDC-ViT with paired significance tests
# Waits for the ablation run to finish first (one GPU job at a time). Finished parts are skipped on a re-run.
#
#   setsid nohup ./scripts/run_thesis_pdcvit.sh > data/logs/run_thesis_pdcvit.log 2>&1 < /dev/null &
# Result: data/thesis/pdcvit/compare.txt   (training log: data/logs/thesis_pdcvit.log)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs data/thesis/pdcvit
while pgrep -f "run_thesis_ablation.sh" > /dev/null; do sleep 60; done
declare -A ARGS=( [paper]="--recipe paper --channels 3" [wide]="--recipe paper --channels 32" [ours]="--recipe ours --channels 32" )
echo "$(date '+%F %T') PDC-ViT started"
for tag in paper wide ours; do
  [ -f data/thesis/pdcvit/${tag}/realistic_fixed_w10.parquet ] && continue
  echo "$(date '+%F %T') 1/3 variant ${tag}, fixed split"
  python scripts/thesis_pdcvit.py --scheme fixed --tag $tag ${ARGS[$tag]} >> data/logs/thesis_pdcvit.log 2>&1
done
best=$(python - <<'PY'
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from hvpf import phase5
stats = phase5.load_step("paper_stats.py")
score = {}
for tag in ("paper", "wide", "ours"):
    v = stats.system(phase5.ROOT / "data/thesis/pdcvit" / tag / "realistic_fixed_w10.parquet", "pdc_vit", "val")
    score[tag] = stats.macro_f1(v.y.to_numpy(), v.pred.to_numpy())
print(max(score, key=score.get))
PY
)
echo "$(date '+%F %T') 2/3 best variant on validation: ${best}; training it on the 5 folds"
python scripts/thesis_pdcvit.py --scheme folds --tag $best ${ARGS[$best]} >> data/logs/thesis_pdcvit.log 2>&1
echo "$(date '+%F %T') 3/3 comparison with HVPF"
python scripts/thesis_pdcvit_compare.py > data/logs/thesis_pdcvit_compare.log 2>&1
echo "$(date '+%F %T') PDC-ViT done: data/thesis/pdcvit/compare.txt"
