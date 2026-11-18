"""Thesis - results of the final HVPF per processing condition (RQ2), from saved predictions (CPU, seconds).

For every condition (genuine native / re-encoded at each CRF / platform copies, every attack and strength, every AI
generator with and without platform copies or injected PRNU): number of clips, share put in the right class by the
final HVPF (stack_rel, 10 frames), share HVPF accepts as genuine, and share the conventional PCE rule accepts.
Unseen cameras (5-fold test) and unseen datasets (fixed split, crossdataset part).
Writes data/thesis/conditions/conditions.csv, conditions.txt and report/thesis/conditions.md.

    .venv/bin/python scripts/thesis_conditions.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.model_data import CLASSES

P = [f"p_{c}" for c in CLASSES]
OUT = phase5.ROOT / "data" / "thesis" / "conditions"
FUSION = lambda scheme: phase5.PHASE5 / "models" / f"fusion_c4m4_realistic_{scheme}_w10.parquet"


def frame(scheme, part):
    f = pd.read_parquet(FUSION(scheme))
    f = f[f.part == part]
    hv = f[f.model == "stack_rel"].set_index("sample_id")
    pce = f[f.model == "baseline"].set_index("sample_id").reindex(hv.index)
    hv["pred"] = hv[P].to_numpy().argmax(1)
    hv["y"] = hv.label.map(CLASSES.index)
    hv["pce_accepts"] = (pce.score_organic > pce.threshold).to_numpy()
    return hv


def condition(d):
    lab, var, soc = d.label.astype(str), d.variant.astype(str), d.social.astype(str)
    stab = d.stabilized.fillna(False).astype(bool)
    cond = np.where(lab == "synthetic", "AI: " + var, np.where(lab == "manipulated", "attack: " + var, "genuine: " + var))
    cond = np.where((lab == "organic") & soc.isin(["wa-real", "yt-real"]), "genuine: real " + var + " copy", cond)
    return pd.Series(cond, index=d.index)


def table(d):
    d = d.assign(cond=condition(d), ok=d.pred == d.y, acc=d.pred == 0)
    g = d.groupby("cond").agg(clips=("y", "size"), correct=("ok", "mean"), hvpf_accepts=("acc", "mean"),
                              pce_accepts=("pce_accepts", "mean"))
    g[["correct", "hvpf_accepts", "pce_accepts"]] *= 100
    return g


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    lines, md, rows = [], ["# Thesis - final HVPF per condition", ""], []
    for name, (scheme, part) in {"unseen cameras (5-fold)": ("folds", "test"),
                                 "unseen datasets": ("fixed", "crossdataset")}.items():
        d = frame(scheme, part)
        t = table(d)
        rows.append(t.assign(test=name))
        lines += [f"== {name}", t.to_string(float_format=lambda v: f"{v:.1f}"), ""]
        md += [f"## {name}", "", t.round(1).to_markdown(), ""]
        if scheme == "folds":
            s = d[d.label == "organic"].assign(stab=d.stabilized.fillna(False).astype(bool))
            st = s.groupby(["source", "stab"]).apply(lambda x: pd.Series(
                {"clips": len(x), "accepted %": 100 * (x.pred == 0).mean(), "PCE accepts %": 100 * x.pce_accepts.mean()}))
            lines += ["-- genuine clips by source and stabilisation", st.to_string(float_format=lambda v: f"{v:.1f}"), ""]
            md += ["**Genuine clips by source and stabilisation**", "", st.round(1).to_markdown(), ""]
    pd.concat(rows).to_csv(OUT / "conditions.csv")
    text = "\n".join(lines)
    (OUT / "conditions.txt").write_text(text + "\n")
    (phase5.ROOT / "report" / "thesis" / "conditions.md").write_text("\n".join(md) + "\n")
    print(text)


if __name__ == "__main__":
    main()
