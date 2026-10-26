"""Phase 5 comparison: Phase 4 stacking vs changes 1-3, on the same test clips.

    Phase 4                  stack of data/phase4 (re-derived by p5_fusion.py; equals Phase 4's)
    + reliability fusion     change 2 on the Phase 4 models
    + matched training       change 1 (data/phase5 models), plain stacking
    + matched + reliability  changes 1 and 2
    + camera calibration     change 4 (data/phase5/models_c4), plain stacking
    + all four               changes 1, 2 and 4 (data/phase5/models_c4m) with reliability-aware stacking
Each configuration is scored three ways: its default decision (argmax), the Phase 4 safe operating
point (+safe: >= 95% genuine accepted on validation), and the three-zone decision with an
inconclusive zone (change 3, thresholds tau_lo / tau_hi from validation).

Realistic protocol; 5-fold test (unseen cameras) and the fixed split's unseen datasets (FloreView +
GenBuster). F1 (over the classes present, as in Step 9) and AUC (genuine vs not) are means over
folds; the rates pool the folds' test clips. Writes data/phase5/metrics/phase5_comparison.csv / .txt.

Usage:
    python scripts/p5_compare.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.model_data import CLASSES

P = [f"p_{c}" for c in CLASSES]
CONFIGS = [("Phase 4", "p4", "stack"), ("+ reliability fusion (2)", "p4", "stack_rel"),
           ("+ matched training (1)", "p5", "stack"), ("+ matched + reliability (1+2)", "p5", "stack_rel"),
           ("+ camera calibration (4)", "c4", "stack"), ("+ all four (1+2+4)", "c4m", "stack_rel")]
ATTACKED = ("VISION-attack", "ACID-attack")


def groups(d):
    """Clip groups reported: name -> (mask, wanted class)."""
    base = d.variant.astype(str).str.split("+").str[0]
    genuine, reenc = d.label == "organic", d.source.isin(ATTACKED)
    inj = d.variant.astype(str).str.contains(r"\+inj")
    g = {"genuine (all)": genuine,
         "genuine originals": genuine & (base == "native"),
         "genuine real WhatsApp/YouTube": genuine & base.isin(["whatsapp", "youtube"]),
         "genuine re-encodes (all)": genuine & reenc,
         "genuine CRF-20 re-encodes": genuine & reenc & (base == "reencode") & (d.social == "none"),
         "genuine WhatsApp-like copies": genuine & reenc & (d.social == "wa"),
         "attacks": d.label == "manipulated",
         "AI": d.label == "synthetic",
         "FloreView genuine": genuine & (d.source == "FloreView"),
         "GenBuster AI": (d.source == "GenBuster") & ~inj,
         "GenBuster AI + injected PRNU": (d.source == "GenBuster") & inj}
    return {k: (m.to_numpy(), 0 if k.startswith(("genuine", "FloreView")) else 1 if k == "attacks" else 2)
            for k, m in g.items() if m.any()}


def macro_f1(y, pred):
    return f1_score(y, pred, labels=np.unique(y), average="macro")


def score(d, safe):
    """Metrics of one configuration (d: its rows of one part; safe: its +safe rows of the same clips)."""
    y = d.label.map(CLASSES.index).to_numpy()
    prob = d[P].to_numpy()
    pred = prob.argmax(1)
    zone = phase5.three_zone_decision(prob, d.tau_lo.to_numpy(), d.tau_hi.to_numpy())
    sp = safe.set_index("sample_id").reindex(d.sample_id)[P].to_numpy().argmax(1)
    folds = d.split_name.to_numpy()
    out = {"F1": np.mean([macro_f1(y[folds == f], pred[folds == f]) for f in np.unique(folds)])}
    if (y == 0).any() and (y != 0).any():
        out["AUC"] = np.mean([roc_auc_score(y[folds == f] == 0, prob[folds == f, 0]) for f in np.unique(folds)])
    decided = zone >= 0
    out["3-zone: decided %"] = 100 * decided.mean()
    out["3-zone: F1 on decided"] = macro_f1(y[decided], zone[decided])
    for name, (m, want) in groups(d).items():
        out[f"{name}: correct %"] = 100 * (pred[m] == want).mean()
        out[f"{name}: safe point correct %"] = 100 * (sp[m] == want).mean()
        out[f"{name}: 3-zone accepted %"] = 100 * (zone[m] == 0).mean()
        out[f"{name}: 3-zone inconclusive %"] = 100 * (zone[m] == -1).mean()
        out[f"{name}: 3-zone accused %"] = 100 * (zone[m] > 0).mean()
    return out


def table(g):
    """Readable text tables of one (window, test) block."""
    cfgs = list(g.index)
    rows = ["F1", "AUC"] + [c for c in g.columns if c.endswith(": correct %")]
    a = g[[c for c in rows if c in g]].T
    safe = g[[c for c in g.columns if c.endswith("safe point correct %")]].T
    zone_rows = {}
    for c in g.columns:
        if c.endswith(": 3-zone accepted %"):
            name = c.split(":")[0]
            zone_rows[name] = [f"{g.at[k, c]:5.1f} / {g.at[k, name + ': 3-zone inconclusive %']:5.1f} / "
                               f"{g.at[k, name + ': 3-zone accused %']:5.1f}" for k in cfgs]
    z = pd.DataFrame(zone_rows, index=cfgs).T
    z.loc["decided % | F1 on decided"] = [f"{g.at[k, '3-zone: decided %']:.1f} | {g.at[k, '3-zone: F1 on decided']:.3f}"
                                          for k in cfgs]
    return (f"Default decision (correct %: genuine accepted, attacks/AI caught as their class)\n{a.round(3).to_string()}\n\n"
            f"Safe operating point (>= 95% genuine accepted on validation)\n{safe.round(1).to_string()}\n\n"
            f"Three-zone decision (change 3): accepted / inconclusive / accused %\n{z.to_string()}")


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for window in (5, 10):
        for scheme, part in (("folds", "test"), ("fixed", "crossdataset")):
            for label, source, model in CONFIGS:
                path = phase5.PHASE5 / "models" / f"fusion_{source}_realistic_{scheme}_w{window}.parquet"
                if not path.exists():
                    continue
                f = pd.read_parquet(path)
                f = f[f.part == part]
                rows.append({"window": window, "test": "unseen cameras (5-fold)" if part == "test" else "unseen datasets",
                             "config": label, "vit": f.vit_source.iat[0], "n": int((f.model == model).sum()),
                             **score(f[f.model == model], f[f.model == f"{model}+safe"])})
    t = pd.DataFrame(rows)
    out = phase5.PHASE5 / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "phase5_comparison.csv", index=False)
    lines = []
    for (window, test), g in t.groupby(["window", "test"], sort=False):
        vits = ", ".join(f"{c}: {v}" for c, v in zip(g.config, g.vit) if "/phase5/" in v)
        lines.append(f"\n===== {test}, {window} frames, {g.n.iat[0]:,} clips" + (f" (ViT of {vits})" if vits else ""))
        lines.append(table(g.set_index("config").drop(columns=["window", "test", "vit", "n"]).dropna(axis=1, how="all")))
    text = "\n".join(lines)
    print(text)
    (out / "phase5_comparison.txt").write_text(text + "\n")
    print(f"\nwrote {out / 'phase5_comparison.csv'} and .txt")


if __name__ == "__main__":
    main()
