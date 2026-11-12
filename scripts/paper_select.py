"""Paper, step 1a - choose the final HVPF configuration on validation data only (no test data).

Candidates: the Phase 4 / Phase 5 base-model runs (p4, p5, c4, c4m with the matched ViT, c4m4 with the
Phase 4 ViT) x stacking (plain stack, reliability-aware stack_rel) x 5 or 10 frames. The stacking layer is
fitted on the validation cameras, so its in-sample validation score would favour the more flexible
stacker: here it is cross-fitted inside the validation set (3 folds over claimed cameras; fit on two,
score the third). The selected configuration is the one with the best cross-fitted validation macro F1
(mean over the 5 CV splits); its test score is only looked up afterwards.

Writes data/paper/selection.csv.

    .venv/bin/python scripts/paper_select.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.model_data import CLASSES, load

fusion = phase5.load_step("p5_fusion.py")
step8c = phase5.load_step("08c_ensemble.py")
P = [f"p_{c}" for c in CLASSES]
OUT = phase5.ROOT / "data" / "paper"
SOURCES = {"p4": "Phase 4 models", "p5": "+ matched training (1), matched ViT", "c4": "+ camera calibration (4)",
           "c4m": "+ 1 + 4, matched ViT", "c4m4": "+ 1 + 4, Phase 4 ViT"}


def crossfit_f1(z_val, y, groups, fit):
    """Macro F1 of out-of-fold predictions of a stacker inside the validation set."""
    pred = np.zeros(len(y), int)
    for a, b in GroupKFold(3).split(z_val, y, groups):
        pred[b] = fit(z_val[a], y[a], groups[a]).predict(z_val[b])
    return f1_score(y, pred, average="macro")


def plain(z, y, groups):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(z, y)


def rel(z, y, groups):
    return phase5.fit_rel_stack(z, y, groups)[0]


def main():
    warnings.filterwarnings("ignore")
    test = pd.read_csv(phase5.PHASE5 / "metrics" / "phase5_comparison.csv")
    test = test[test.test == "unseen cameras (5-fold)"]
    rows = []
    for w in (5, 10):
        per_video, _ = load(phase5.PHASE4 / "features", w)
        relf = per_video.set_index("sample_id")[phase5.RELIABILITY]
        for source, label in SOURCES.items():
            pred, _ = fusion.inputs(source, "realistic", "folds", w)
            if pred is None:
                continue
            scores = {"stack": [], "stack_rel": []}
            for _, split in pred[(pred.part == "val") & (pred.model != "baseline")].groupby("split_name"):
                models = sorted(split.model.unique())
                meta = split[split.model == models[0]].set_index("sample_id")
                probs = [split[split.model == m].set_index("sample_id").reindex(meta.index)[P].to_numpy() for m in models]
                y = meta.label.map(CLASSES.index).to_numpy()
                groups = meta.claimed_device.to_numpy()
                lp = np.concatenate([step8c.logp(p) for p in probs], 1)
                r = relf.reindex(meta.index).to_numpy(float)
                r = np.where(np.isnan(r), np.nanmedian(r, 0), r)
                r = (r - r.mean(0)) / (r.std(0) + 1e-9)
                scores["stack"].append(crossfit_f1(lp, y, groups, plain))
                scores["stack_rel"].append(crossfit_f1(phase5.rel_design(lp, r), y, groups, rel))
            for fused, s in scores.items():
                cfg = {"p4": {"stack": "Phase 4", "stack_rel": "+ reliability fusion (2)"},
                       "p5": {"stack": "+ matched training (1)", "stack_rel": "+ matched + reliability (1+2)"},
                       "c4": {"stack": "+ camera calibration (4)"}, "c4m": {"stack_rel": "+ all four (1+2+4)"},
                       "c4m4": {"stack_rel": "+ all four, Phase 4 ViT"}}.get(source, {}).get(fused)
                t = test[(test.window == w) & (test.config == cfg)] if cfg else test.iloc[:0]
                rows.append({"frames": w, "base models": label, "stacking": fused, "val F1 (cross-fitted)": np.mean(s),
                             "val F1 sd": np.std(s), "test F1": t.F1.iat[0] if len(t) else np.nan})
                print(f"w{w} {source:5s} {fused:9s} val F1 {np.mean(s):.4f}", flush=True)
    t = pd.DataFrame(rows).sort_values("val F1 (cross-fitted)", ascending=False).reset_index(drop=True)
    t["selected"] = ""
    t.loc[0, "selected"] = "<- best on validation"
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "selection.csv", index=False)
    print("\n" + t.round(4).to_string())


if __name__ == "__main__":
    main()
