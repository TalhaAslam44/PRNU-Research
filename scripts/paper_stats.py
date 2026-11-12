"""Paper, step 1b - 95% confidence intervals and paired significance tests for every headline claim.

Clips of one camera are not independent, so the bootstrap resamples whole cameras (claimed camera, with
replacement; 2,000 resamples) and every metric is recomputed on the resampled clips. Comparisons are
paired: both systems are scored on the same resampled clips; two-sided p = 2 x min(P(diff <= 0),
P(diff >= 0)). Scores here are pooled over the 5 test folds (so they can differ from fold means by ~0.001).

Systems: final HVPF (changes 1+2+4, Phase 4 ViT, 10 frames, chosen on validation: scripts/paper_select.py),
Phase 4 stacking, Phase 3 stacking, the PCE rule. Writes data/paper/stats.csv and stats.txt.

    .venv/bin/python scripts/paper_stats.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.model_data import CLASSES

P = [f"p_{c}" for c in CLASSES]
P3, P5 = phase5.ROOT / "data" / "phase3", phase5.PHASE5
OUT = phase5.ROOT / "data" / "paper"
B = 2000
rng = np.random.default_rng(0)


def system(path, model, part):
    """Per-clip arrays of one system: label y, genuine score s, decision pred (PCE rule: 0 genuine / -2 not)."""
    d = pd.read_parquet(path)
    d = d[(d.model == model) & (d.part == part)].set_index("sample_id")
    if model == "baseline":
        pred = np.where(d.score_organic > d.threshold, 0, -2)
        s = d.score_organic.to_numpy()
    else:
        pred, s = d[P].to_numpy().argmax(1), d.p_organic.to_numpy()
    return d.assign(y=d.label.map(CLASSES.index), s=s, pred=pred)


def macro_f1(y, pred):
    f = []
    for c in np.unique(y):
        tp = ((pred == c) & (y == c)).sum()
        f.append(2 * tp / max((pred == c).sum() + (y == c).sum(), 1))
    return np.mean(f)


METRICS = {
    "macro F1": lambda d: macro_f1(d.y.to_numpy(), d.pred.to_numpy()),
    "AUC genuine vs not": lambda d: roc_auc_score(d.y == 0, d.s),
    "genuine accepted %": lambda d: 100 * (d.pred[d.y == 0] == 0).mean(),
    "attacks caught %": lambda d: 100 * (d.pred[d.y == 1] == 1).mean(),
    "attacks accepted as genuine %": lambda d: 100 * (d.pred[d.y == 1] == 0).mean(),
    "AI caught %": lambda d: 100 * (d.pred[d.y == 2] == 2).mean(),
    "AI + injected PRNU accepted %": lambda d: 100 * (d.pred[d.variant.astype(str).str.contains(r"\+inj")] == 0).mean(),
    "injection attack (a=1) accepted %": lambda d: 100 * (d.pred[d.variant.astype(str).str.startswith("injection_a1")] == 0).mean(),
}


def boot(a, b, metrics, ids=None):
    """Point values, 95% CIs and paired p-values of metrics for systems a and b (b may be None)."""
    if ids is not None:
        a = a.loc[a.index.intersection(ids)]
        b = b.loc[a.index] if b is not None else None
    elif b is not None:
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
    cams = a.claimed_device.to_numpy()
    uniq, inv = np.unique(cams, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    res = []
    for name in metrics:
        fn = METRICS[name]
        try:
            pa, pb = fn(a), fn(b) if b is not None else np.nan
        except ValueError:
            continue
        if np.isnan(pa):
            continue
        da, db = [], []
        for _ in range(B):
            idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
            try:
                da.append(fn(a.iloc[idx]))
                if b is not None:
                    db.append(fn(b.iloc[idx]))
            except ValueError:
                da.append(np.nan)
                db.append(np.nan) if b is not None else None
        da, db = np.array(da), np.array(db)
        row = {"metric": name, "n clips": len(a), "n cameras": len(uniq), "A": pa,
               "A 95% CI": f"[{np.nanpercentile(da, 2.5):.3f}, {np.nanpercentile(da, 97.5):.3f}]"}
        if b is not None:
            diff = da - db
            p = 2 * min(np.nanmean(diff <= 0), np.nanmean(diff >= 0))
            row.update({"B": pb, "B 95% CI": f"[{np.nanpercentile(db, 2.5):.3f}, {np.nanpercentile(db, 97.5):.3f}]",
                        "A - B": pa - pb, "diff 95% CI": f"[{np.nanpercentile(diff, 2.5):.3f}, {np.nanpercentile(diff, 97.5):.3f}]",
                        "p": min(max(p, 1 / B), 1.0)})
        res.append(row)
    return res


def main():
    warnings.filterwarnings("ignore")
    f = lambda src, scheme, w: P5 / "models" / f"fusion_{src}_realistic_{scheme}_w{w}.parquet"
    final = system(f("c4m4", "folds", 10), "stack_rel", "test")
    final_x = system(f("c4m4", "fixed", 10), "stack_rel", "crossdataset")
    p4_10 = system(f("p4", "folds", 10), "stack", "test")
    p4_5 = system(f("p4", "folds", 5), "stack", "test")
    p4_x = system(f("p4", "fixed", 10), "stack", "crossdataset")
    p3 = system(P3 / "models" / "realistic_folds_all_w5.parquet", "stack", "test")
    p3_x = system(P3 / "models" / "realistic_fixed_all_w5.parquet", "stack", "crossdataset")
    pce = system(f("c4m4", "folds", 10), "baseline", "test")
    pce_x = system(f("c4m4", "fixed", 10), "baseline", "crossdataset")
    main_m = ["macro F1", "AUC genuine vs not", "genuine accepted %", "attacks caught %", "AI caught %"]
    attack_m = ["AUC genuine vs not", "attacks accepted as genuine %", "AI + injected PRNU accepted %",
                "injection attack (a=1) accepted %"]
    blocks = [
        ("Final HVPF vs PCE rule (unseen cameras, 42,083 clips)", final, pce, attack_m, None),
        ("Final HVPF vs PCE rule (unseen datasets)", final_x, pce_x, ["AUC genuine vs not", "AI + injected PRNU accepted %"], None),
        ("Final HVPF vs Phase 4 stacking, 10 frames (unseen cameras)", final, p4_10, main_m, None),
        ("Final HVPF vs Phase 4 stacking, 5 frames = reported Phase 4 (unseen cameras)", final, p4_5, main_m, None),
        ("Final HVPF vs Phase 3 stacking on Phase 3's test clips", final, p3, main_m, p3.index),
        ("Final HVPF vs Phase 3 stacking (unseen datasets)", final_x, p3_x, ["macro F1", "AUC genuine vs not", "genuine accepted %", "AI caught %"], None),
        ("Final HVPF vs Phase 4 stacking, 10 frames (unseen datasets)", final_x, p4_x, ["macro F1", "AUC genuine vs not", "genuine accepted %", "AI caught %"], None),
    ]
    rows, lines = [], []
    for title, a, b, metrics, ids in blocks:
        print(title, flush=True)
        r = pd.DataFrame(boot(a, b, metrics, ids)).assign(comparison=title)
        rows.append(r)
        show = r.drop(columns="comparison").set_index("metric")
        lines += [f"\n== {title}  (A = final HVPF, B = other; {int(r['n clips'].iat[0]):,} clips, "
                  f"{int(r['n cameras'].iat[0])} cameras)",
                  show.drop(columns=["n clips", "n cameras"]).to_string(float_format=lambda v: f"{v:.3f}")]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(OUT / "stats.csv", index=False)
    text = "\n".join(lines)
    (OUT / "stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
