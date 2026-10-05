"""Step 9 - metrics from Step 8 predictions.

For every predictions file data/results/models/<protocol>_<scheme>_<tag>.parquet:
    3-class   PVA (= accuracy: videos assigned to their true provenance category),
              macro precision / recall / F1, one-vs-rest macro ROC-AUC
    binary    authentic (organic) vs not: ROC-AUC and EER of the organic score; the
              conventional PCE rule is evaluated here (it cannot tell manipulated from synthetic)
    folds     mean and std over folds
RQ2 / strength: per-variant detection rate (share of clips assigned to their true class)
for native / YouTube / WhatsApp / re-encode, each attack and alpha, each generator, and
stabilized vs not. Cross-dataset: the same on GenBuster (part = crossdataset).
Ablation: `all` vs each `no_<group>` run of the same protocol and scheme.

Writes data/results/metrics/{summary,per_variant,ablation}.csv and results.md.

Usage:
    python scripts/09_evaluate.py            # all prediction files
    python scripts/09_evaluate.py --check    # the *_check files
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.model_data import CLASSES


def eer(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, score)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def safe_auc(y, score, **kw):
    try:
        return float(roc_auc_score(y, score, **kw))
    except ValueError:
        return np.nan


def metrics(g):
    """Metrics for one model on one evaluation set."""
    y = g.label.map(CLASSES.index).to_numpy()
    authentic = (y == 0).astype(int)
    out = {"n": len(g)}
    if g.model.iloc[0].startswith("baseline"):
        score = g.score_organic.to_numpy()
        pred_auth = (score > g.threshold.iloc[0]).astype(int)
        p, r, f, _ = precision_recall_fscore_support(authentic, pred_auth, average="binary", zero_division=0)
        out.update(bin_acc=accuracy_score(authentic, pred_auth), bin_precision=p, bin_recall=r, bin_f1=f,
                   bin_auc=safe_auc(authentic, score), bin_eer=eer(authentic, score))
        return out
    prob = g[[f"p_{c}" for c in CLASSES]].to_numpy()
    pred = prob.argmax(1)
    present = np.unique(y)      # e.g. GenBuster holds only synthetic clips: macro over present classes
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=present, average="macro", zero_division=0)
    auc = safe_auc(y, prob[:, present] / prob[:, present].sum(1, keepdims=True), multi_class="ovr",
                   labels=present) if len(present) > 2 else safe_auc((y == present[-1]).astype(int), prob[:, present[-1]]) if len(present) == 2 else np.nan
    pred_auth = (pred == 0).astype(int)
    bp, br, bf, _ = precision_recall_fscore_support(authentic, pred_auth, average="binary", zero_division=0)
    out.update(pva=accuracy_score(y, pred), precision=p, recall=r, f1=f, auc_ovr=auc,
               bin_acc=accuracy_score(authentic, pred_auth), bin_precision=bp, bin_recall=br, bin_f1=bf,
               bin_auc=safe_auc(authentic, prob[:, 0]), bin_eer=eer(authentic, prob[:, 0]))
    return out


def correct(g):
    """Per-row correctness: predicted class == true class (baseline: authentic decision)."""
    if g.model.iloc[0].startswith("baseline"):
        return (g.score_organic > g.threshold) == (g.label == "organic")
    return g[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1) == g.label.map(CLASSES.index).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    paths = load_config()["paths"]
    files = sorted((paths["results_dir"] / "models").glob("*_check.parquet" if args.check else "*.parquet"))
    files = [f for f in files if args.check or not f.stem.endswith("_check")]
    out_dir = paths["results_dir"] / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, variants = [], []
    for f in files:
        protocol, scheme, *tag = f.stem.removesuffix("_check").split("_", 2)
        tag = tag[0] if tag else "all"
        pred = pd.read_parquet(f)
        for (model, part, split_name), g in pred.groupby(["model", "part", "split_name"]):
            summary.append({"protocol": protocol, "scheme": scheme, "tag": tag, "model": model, "part": part,
                            "split_name": split_name, **metrics(g)})
            g = g.assign(correct=correct(g), alpha=g.alpha.fillna(0))
            for keys, h in g.groupby(["label", "variant", "alpha", "stabilized"], dropna=False):
                variants.append({"protocol": protocol, "scheme": scheme, "tag": tag, "model": model, "part": part,
                                 "split_name": split_name, **dict(zip(["label", "variant", "alpha", "stabilized"], keys)),
                                 "n": len(h), "detection_rate": h.correct.mean()})

    summary = pd.DataFrame(summary)
    variants = pd.DataFrame(variants)
    keys = ["protocol", "scheme", "tag", "model", "part"]
    num = summary.columns.difference(keys + ["split_name"])
    table = summary.groupby(keys)[list(num)].agg(["mean", "std"])
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    table = table.reset_index()
    table.to_csv(out_dir / "summary.csv", index=False)
    var = variants.groupby(keys + ["label", "variant", "alpha", "stabilized"], dropna=False).agg(
        n=("n", "sum"), detection_rate=("detection_rate", "mean")).reset_index()
    var.to_csv(out_dir / "per_variant.csv", index=False)

    # ablation: all vs no_<group>, same protocol/scheme/model/part
    full = table[table.tag == "all"].set_index(["protocol", "scheme", "model", "part"])
    abl = []
    for _, r in table[table.tag != "all"].iterrows():
        k = (r.protocol, r.scheme, r.model, r.part)
        if k in full.index and "f1_mean" in r and pd.notna(r.f1_mean):
            abl.append({"protocol": r.protocol, "scheme": r.scheme, "model": r.model, "part": r.part,
                        "dropped": r.tag.removeprefix("no_"), "pva": r.pva_mean, "pva_drop": full.loc[k, "pva_mean"] - r.pva_mean,
                        "f1": r.f1_mean, "f1_drop": full.loc[k, "f1_mean"] - r.f1_mean,
                        "bin_auc_drop": full.loc[k, "bin_auc_mean"] - r.bin_auc_mean})
    ablation = pd.DataFrame(abl)
    ablation.to_csv(out_dir / "ablation.csv", index=False)

    cols = ["n_mean", "pva_mean", "pva_std", "f1_mean", "f1_std", "auc_ovr_mean", "bin_auc_mean", "bin_eer_mean",
            "bin_f1_mean"]
    lines = ["# HVPF results", ""]
    for (protocol, scheme, part), t in table[table.tag == "all"].groupby(["protocol", "scheme", "part"]):
        lines += [f"## {protocol} / {scheme} / {part}", "", t.set_index("model")[[c for c in cols if c in t]].round(3).to_markdown(), ""]
    if len(ablation):
        lines += ["## Ablation (drop in macro F1 / PVA when a feature group is removed)", "",
                  ablation.round(3).to_markdown(index=False), ""]
    (out_dir / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
