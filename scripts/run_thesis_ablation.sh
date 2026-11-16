#!/bin/bash
# Thesis ablation of the final HVPF (10 frames): feature groups left out / used alone (GPU, ~3 h), then the
# ablation tables with paired significance tests (CPU, ~10 min). New scripts only; Phases 3-5 are not changed.
# Finished runs are skipped, so the script can simply be started again after an interruption.
#
#   setsid nohup ./scripts/run_thesis_ablation.sh > data/logs/run_thesis_ablation.log 2>&1 < /dev/null &
# Result: data/thesis/ablation/ablation.txt and report/thesis/ablation.md (training log: data/logs/thesis_ablation_models.log)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p data/logs data/thesis/ablation
echo "$(date '+%F %T') thesis ablation started"
run() {   # run <tag> <arguments of thesis_ablation_models.py>
  local tag=$1; shift
  for s in folds fixed; do
    [ -f data/thesis/ablation/models/${tag}/realistic_${s}_all_w10.parquet ] && continue
    echo "$(date '+%F %T') ${tag} ${s}"
    python scripts/thesis_ablation_models.py "$@" --scheme $s >> data/logs/thesis_ablation_models.log 2>&1
  done
}
run all_rerun
for g in noiseprint prnu tsncs resid fft; do run no_${g} --drop $g; done
for g in noiseprint prnu tsncs resid fft; do run only_${g} --only $g; done
echo "$(date '+%F %T') tables + significance tests"
python scripts/thesis_ablation.py > data/logs/thesis_ablation.log 2>&1
echo "$(date '+%F %T') thesis ablation done: data/thesis/ablation/ablation.txt"
