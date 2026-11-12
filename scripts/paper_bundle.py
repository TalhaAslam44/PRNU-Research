"""Paper, step 2a - train the final HVPF once on the fixed split and save it (GPU, ~1 h; the ViT is most of it).

Saves data/paper/bundle_fixed_w10/ (hvpf/paper_bundle.Bundle) and checks it against the Phase 5 result of
the same split (fusion_c4m4_realistic_fixed_w10, stack_rel): the scores should agree within a few
thousandths (the ViT is retrained, so it is not bit-identical).

    .venv/bin/python scripts/paper_bundle.py
    .venv/bin/python scripts/paper_bundle.py --quick --out data/paper/bundle_check   # pipeline check (~10 min)
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.model_data import CLASSES
from hvpf.paper_bundle import train_bundle

P = [f"p_{c}" for c in CLASSES]


def report(d, name):
    for part in ("test", "crossdataset"):
        g = d[d.part == part]
        y = g.label.map(CLASSES.index).to_numpy()
        print(f"  {name:28s} {part:12s} F1 {f1_score(y, g[P].to_numpy().argmax(1), labels=np.unique(y), average='macro'):.3f}"
              f"  AUC {roc_auc_score(y == 0, g.p_organic):.3f}  (n = {len(g):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=phase5.ROOT / "data" / "paper" / "bundle_fixed_w10")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="1 ViT epoch on 5%% of the frames (pipeline check only)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    pred = train_bundle(args.out, window=args.window, quick=args.quick)
    print("\nsaved bundle -> " + str(args.out))
    ref = pd.read_parquet(phase5.PHASE5 / "models" / f"fusion_c4m4_realistic_fixed_w{args.window}.parquet")
    report(pred, "saved bundle")
    report(ref[ref.model == "stack_rel"], "Phase 5 (same split)")


if __name__ == "__main__":
    main()
