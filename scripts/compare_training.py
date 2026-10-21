"""Compare training settings on the same test clips: old clips (tested in Phase 3), new clips (added in Phase 4) and all.

Each setting is a Step 8 prediction file (realistic, 5-fold) that predicted every Phase 4 test clip; only the training
data / weighting differs. Rows that have not been run yet are skipped.

    python scripts/compare_training.py                 # XGBoost and GRU, 5 frames
    python scripts/compare_training.py --models xgb
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_phases import scores
from hvpf.config import load_config
from hvpf.model_data import CLASSES

SETTINGS = [("Phase 3 data only (control)", "all_w5_ex-acid_attacks-new_generators-new_genuine"),
            ("Phase 4, all additions", "all_w5"),
            ("Phase 4, without ACID attacks", "all_w5_ex-acid_attacks"),
            ("Phase 4, dataset-balanced (experiment)", "all_w5_baldata")]
NEW_GENERATORS = {"SVD", "Mora", "Text2Video-Zero", "Pika"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["xgb", "gru"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    paths = load_config()["paths"]
    root = paths["data_dir"].parent
    old_ids = set(pd.read_parquet(root / "data/phase3/models/realistic_folds_all_w5.parquet")
                  .query("part == 'test' and model == 'baseline'").sample_id)
    rows = []
    for label, tag in SETTINGS:
        f = paths["results_dir"] / "models" / f"realistic_folds_{tag}.parquet"
        if not f.exists():
            print(f"not run yet: {f.name}")
            continue
        p = pd.read_parquet(f)
        p = p[p.part == "test"]
        for m in args.models:
            g = p[p.model == m]
            if not len(g):
                continue
            ok = g[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1) == g.label.map(CLASSES.index).to_numpy()
            g = g.assign(ok=ok, is_old=g.sample_id.isin(old_ids))
            old, new = g[g.is_old], g[~g.is_old]
            gen = new.variant.astype(str).str.split("+").str[0].isin(NEW_GENERATORS) & (new.source == "GenVidBench")
            rows.append({"setting": label, "model": m,
                         "F1 old clips": scores(old)["f1"], "F1 new clips": scores(new)["f1"], "F1 all": scores(g)["f1"],
                         "genuine ok (old)": old[old.label == "organic"].ok.mean(),
                         "ACID genuine ok": g[(g.source == "ACID")].ok.mean(),
                         "VISION attacks caught": g[(g.source == "VISION-attack") & (g.label == "manipulated")].ok.mean(),
                         "ACID attacks caught": g[(g.source == "ACID-attack") & (g.label == "manipulated")].ok.mean(),
                         "new-generator AI caught": new[gen].ok.mean()})
    t = pd.DataFrame(rows)
    for m in args.models:
        print(f"\n{m} (realistic, 5-fold, 5 frames):")
        print(t[t.model == m].drop(columns="model").set_index("setting").T.round(3).to_string())
    out = paths["results_dir"] / "metrics" / "compare_training.csv"
    t.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
