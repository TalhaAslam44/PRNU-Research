"""Same-clips comparison of two phases.

Scores the test predictions of an earlier phase and of the current one only on the clips that both
tested, so a difference comes from the models (more / different training data), not from a harder
test set. The PCE rule is not trained: on the shared clips it must score the same in both phases,
which checks that the clips and features match. Clips that only the current phase tested are scored
separately (how hard is the new material?).

Writes data/results/metrics/phase_comparison_<protocol>_<tag>.csv.

Usage:
    python scripts/compare_phases.py                                  # Phase 3 vs Phase 4, realistic, 5 frames
    python scripts/compare_phases.py --protocol controlled
    python scripts/compare_phases.py --old data/phase2 --tag all_w5   # any phase backup vs the current results
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.model_data import CLASSES

MODELS = ["baseline", "svm", "xgb", "gru", "cnn", "transformer", "vit", "ens_all", "stack"]


def scores(g):
    """Macro F1, PVA, per-class recall, AUC and EER (genuine vs not) of one model's predictions."""
    y = g.label.map(CLASSES.index).to_numpy()
    genuine = (y == 0).astype(int)
    if g.model.iloc[0] == "baseline":
        s = g.score_organic.to_numpy()
        out = {}
    else:
        prob = g[[f"p_{c}" for c in CLASSES]].to_numpy()
        pred, s = prob.argmax(1), prob[:, 0]
        out = {"f1": f1_score(y, pred, average="macro"), "pva": (pred == y).mean(),
               **{f"recall_{c}": (pred[y == k] == k).mean() for k, c in enumerate(CLASSES)}}
    fpr, tpr, _ = roc_curve(genuine, s)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return {**out, "auc": roc_auc_score(genuine, s), "eer": (fpr[i] + 1 - tpr[i]) / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="data/phase3", help="earlier phase backup (contains models/)")
    ap.add_argument("--new", default="data/results", help="current results (contains models/)")
    ap.add_argument("--protocol", choices=["realistic", "controlled"], default="realistic")
    ap.add_argument("--tag", default="all_w5")
    ap.add_argument("--new-tag", help="tag of the new phase's file if it differs (e.g. all_w5_ex-acid_attacks)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    root = load_config()["paths"]["data_dir"].parent
    name = f"{args.protocol}_folds_{args.tag}.parquet"
    old = pd.read_parquet(root / args.old / "models" / name)
    new = pd.read_parquet(root / args.new / "models" / f"{args.protocol}_folds_{args.new_tag or args.tag}.parquet")
    old, new = old[old.part == "test"], new[new.part == "test"]
    ids_old = set(old[old.model == "baseline"].sample_id)
    ids_new = set(new[new.model == "baseline"].sample_id)
    shared, only_new = ids_old & ids_new, ids_new - ids_old
    print(f"{args.protocol} / 5-fold / {args.tag}: {len(shared):,} clips tested in both phases, "
          f"{len(only_new):,} only in the new phase")

    rows = []
    for m in MODELS:
        for phase, df, ids, part in (("old", old, shared, "shared"), ("new", new, shared, "shared"),
                                     ("new", new, only_new, "new clips only")):
            g = df[(df.model == m) & df.sample_id.isin(ids)]
            if len(g) and ids:
                rows.append({"model": m, "phase": phase, "clips": part, "n": len(g), **scores(g)})
    t = pd.DataFrame(rows)
    out = root / args.new / "metrics" / f"phase_comparison_{args.protocol}_{args.new_tag or args.tag}.csv"
    t.to_csv(out, index=False)

    sh = t[t.clips == "shared"].pivot_table(index="model", columns="phase", values=["f1", "auc"]).reindex(MODELS).dropna(how="all")
    sh[("f1", "change")] = sh[("f1", "new")] - sh[("f1", "old")]
    sh[("auc", "change")] = sh[("auc", "new")] - sh[("auc", "old")]
    print("\nSame clips (macro F1 and AUC genuine vs not; old = earlier phase, new = current):")
    print(sh[[("f1", "old"), ("f1", "new"), ("f1", "change"), ("auc", "old"), ("auc", "new"), ("auc", "change")]].round(3).to_string())
    rec = t[(t.clips == "shared") & (t.model == "stack")].set_index("phase")[[f"recall_{c}" for c in CLASSES]]
    print("\nStacking, recall per class on the same clips:\n" + rec.round(3).to_string())
    nw = t[t.clips == "new clips only"].set_index("model")[["n", "f1", "auc"]].reindex(MODELS).dropna(how="all")
    print("\nClips only in the new phase (how hard is the new material?):\n" + nw.round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
