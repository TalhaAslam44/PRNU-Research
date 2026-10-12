"""Step 8c - calibration and ensembles of the Step 8 / 8b models (CPU, ~1 min).

For every Step 8 prediction file data/results/models/<protocol>_<scheme>_all_w<N>.parquet:
  vit          Step 8b's ViT predictions (models/vit/<protocol>_<scheme>_w<N>.parquet) are added
  <model>+cal  calibration: three additive class offsets on the log-probabilities, tuned per
               split on that split's validation predictions (best val macro-F1). It moves the
               decision boundaries (e.g. a model that calls too many WhatsApp copies
               "manipulated"); nothing is retrained.
  ens_*        soft voting (mean class probability) over a model set, plus its +cal version:
                 ens_ml    svm, xgb
                 ens_seq   gru, cnn, transformer
                 ens_all   every learned model (incl. vit)
                 ens_top3  the 3 models with the best val macro-F1 in that split
  stack        multinomial logistic regression on the log-probabilities of every learned model,
               fit on the split's validation predictions (other camera models than test)
Validation data never sees the test set (val = held-out camera models; in CV the next fold).
Earlier vit / +cal / ens / stack rows are replaced, so the step can be re-run after Step 8.

Usage:
    python scripts/08c_ensemble.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.model_data import CLASSES

P = [f"p_{c}" for c in CLASSES]
DERIVED = r"vit|.*\+cal|ens_.*|stack"          # rows this step writes
GRID = np.round(np.arange(-3, 3.01, 0.1), 2)


def logp(p):
    return np.log(np.clip(p, 1e-6, 1))


def calibrate(p_val, y_val):
    """Class offsets b (b_organic = 0) maximising val macro-F1 of argmax(log p + b)."""
    lv = logp(p_val)
    b1, b2 = np.meshgrid(GRID, GRID, indexing="ij")
    b = np.stack([np.zeros(b1.size), b1.ravel(), b2.ravel()], 1)            # (G, 3)
    f1 = grid_f1(y_val, (lv[None] + b[:, None]).argmax(2))
    return b[int(np.argmax(f1))] if f1.max() > grid_f1(y_val, lv.argmax(1)[None])[0] else np.zeros(3)


def grid_f1(y, pred):
    """Macro-F1 of every row of pred (G, n) against y (n,), vectorised."""
    f1 = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum(1)
        f1.append(2 * tp / np.maximum((pred == c).sum(1) + (y == c).sum(), 1))
    return np.mean(f1, axis=0)


def shift(p, b):
    z = logp(p) + b
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def macro_f1(y, p):
    return f1_score(y, p.argmax(1), average="macro")


def derive(split):
    """All derived rows of one split (one Step 8 split_name)."""
    split = split[split.model != "baseline"]
    models = sorted(split.model.unique())
    meta = split[split.model == models[0]].drop(columns=["model", "score_organic", "threshold", *P],
                                                errors="ignore").set_index("sample_id")
    probs = {m: split[split.model == m].set_index("sample_id").reindex(meta.index)[P].to_numpy() for m in models}
    if any(np.isnan(p).any() for p in probs.values()):
        missing = [m for m, p in probs.items() if np.isnan(p).any()]
        print(f"    skipping models without predictions for every sample: {missing}")
        probs = {m: p for m, p in probs.items() if m not in missing}
        models = list(probs)
    y = meta.label.map(CLASSES.index).to_numpy()
    val = (meta.part == "val").to_numpy()

    val_f1 = {m: macro_f1(y[val], p[val]) for m, p in probs.items()}
    sets = {"ens_ml": ["svm", "xgb"], "ens_seq": ["gru", "cnn", "transformer"], "ens_all": models,
            "ens_top3": sorted(models, key=val_f1.get, reverse=True)[:3]}
    out = dict(probs)
    for name, members in sets.items():
        members = [m for m in members if m in probs]
        if len(members) >= 2:
            out[name] = np.mean([probs[m] for m in members], axis=0)
    for name in list(out):
        out[f"{name}+cal"] = shift(out[name], calibrate(out[name][val], y[val]))
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    z = np.concatenate([logp(probs[m]) for m in models], 1)
    out["stack"] = lr.fit(z[val], y[val]).predict_proba(z)

    rows = [meta.reset_index().assign(model=name, **{c: p[:, i] for i, c in enumerate(P)})
            for name, p in out.items() if name not in probs]
    test = ~val & (meta.part == "test").to_numpy()
    if test.any():
        best = sorted(((macro_f1(y[test], p[test]), name) for name, p in out.items()), reverse=True)[:5]
        print("    test macro-F1 top 5: " + ", ".join(f"{n} {f:.3f}" for f, n in best))
    return rows


def main():
    paths = load_config()["paths"]
    models_dir = paths["results_dir"] / "models"
    for f in sorted(models_dir.glob("*_all_w*.parquet")):
        m = re.fullmatch(r"(realistic|controlled)_(fixed|folds)_all_w(\d+)", f.stem)
        if not m:                      # *_min10 (length-matched check) and *_check files stay as they are
            continue
        protocol, scheme, window = m.groups()
        pred = pd.read_parquet(f)
        pred = pred[~pred.model.str.fullmatch(DERIVED)]
        vit = models_dir / "vit" / f"{protocol}_{scheme}_w{window}.parquet"
        if vit.exists():
            pred = pd.concat([pred, pd.read_parquet(vit)], ignore_index=True)
        if not (pred.part == "val").any():
            print(f"{f.name}: no validation predictions (written before Phase 3) - re-run Step 8; skipped")
            continue
        print(f"{f.name}: models {sorted(pred.model.unique())}")
        rows = []
        for name, split in pred.groupby("split_name"):
            print(f"  {name}")
            rows += derive(split)
        pd.concat([pred, *rows], ignore_index=True).to_parquet(f, index=False)
        print(f"  wrote {f.name}")


if __name__ == "__main__":
    main()
