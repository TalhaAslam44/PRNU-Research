"""Which Phase 4 addition lowered the scores? Summary of scripts/run_diagnosis.sh.

XGBoost and GRU (realistic, 5-fold, 5 frames) trained on: Phase 3 data, full Phase 4 data, and Phase 4 data with one
addition left out of training (acid_attacks / new_genuine / new_generators) or all three (control: should match Phase 3).
All are scored on the clips Phase 3 tested, so only the training data differs.

    python scripts/diagnose_phase4.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_phases import scores
from hvpf.config import load_config

RUNS = [("Phase 3 training data", "data/phase3", "all_w5"),
        ("Phase 4, all additions", "data/results", "all_w5"),
        ("Phase 4 without ACID attacks", "data/results", "all_w5_ex-acid_attacks"),
        ("Phase 4 without new genuine copies", "data/results", "all_w5_ex-new_genuine"),
        ("Phase 4 without new generators", "data/results", "all_w5_ex-new_generators"),
        ("Phase 4 without all three (control)", "data/results", "all_w5_ex-acid_attacks-new_generators-new_genuine")]


def main():
    warnings.filterwarnings("ignore")
    root = load_config()["paths"]["data_dir"].parent
    preds = {}
    for label, d, tag in RUNS:
        f = root / d / "models" / f"realistic_folds_{tag}.parquet"
        if f.exists():
            p = pd.read_parquet(f)
            preds[label] = p[p.part == "test"]
        else:
            print(f"missing (not run yet): {f.name}")
    base = preds["Phase 3 training data"]
    shared = set(base[base.model == "baseline"].sample_id)
    rows = []
    for label, p in preds.items():
        for m in ("xgb", "gru"):
            g = p[(p.model == m) & p.sample_id.isin(shared)]
            if len(g):
                s = scores(g)
                rows.append({"training data": label, "model": m, "F1": s["f1"], "AUC": s["auc"],
                             "genuine": s["recall_organic"], "manipulated": s["recall_manipulated"],
                             "AI": s["recall_synthetic"]})
    t = pd.DataFrame(rows).pivot_table(index="training data", columns="model", values=["F1", "AUC", "genuine", "manipulated"],
                                      sort=False)
    print(f"Scored on the {len(shared):,} clips Phase 3 tested (realistic, 5-fold, 5 frames):\n")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
