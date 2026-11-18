"""Thesis - validation spread and uncertainty measures of the final HVPF (CPU, from saved predictions only).

Mid-evaluation comments: "results of validation need to be completed", "consider the uncertainty measures". Nothing
is re-trained; this reads the final system's predictions (stack_rel of data/phase5/models/fusion_c4m4_*_w10, the
0.842 / 0.920 system) and its base models (data/phase5/models_c4m + Phase 4 ViT) and reports
    A. per-fold results (mean, std, min, max over the 5 camera folds) and validation vs test
    B. calibration of the predicted probabilities: ECE (15 bins, top label), Brier score, log-loss, and the
       reliability diagram; for HVPF and for every base model
    C. per-clip uncertainty: confidence (max probability), predictive entropy, and disagreement of the six base
       models; how well each one flags HVPF's own errors (AUROC), risk-coverage (error rate when the most uncertain
       clips are referred to an examiner) and AURC
    D. uncertainty and error rate per clip type (native, re-encoded, real WhatsApp / YouTube, attacks, AI, ...)
Every headline number has a 95% camera-level bootstrap interval (whole cameras resampled, as in paper_stats.py).
Writes data/thesis/uncertainty/{uncertainty.txt, *.csv}, report/thesis/uncertainty.md and
report/thesis/fig_reliability.{pdf,png}, report/thesis/fig_risk_coverage.{pdf,png}.

    .venv/bin/python scripts/thesis_uncertainty.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.model_data import CLASSES

stats = phase5.load_step("paper_stats.py")
P = [f"p_{c}" for c in CLASSES]
OUT = phase5.ROOT / "data" / "thesis" / "uncertainty"
REP = phase5.ROOT / "report" / "thesis"
FUSION = lambda scheme: phase5.PHASE5 / "models" / f"fusion_c4m4_realistic_{scheme}_w10.parquet"
BASE = ["svm", "xgb", "gru", "cnn", "transformer", "vit"]
NAMES = {"svm": "SVM", "xgb": "XGBoost", "gru": "GRU", "cnn": "CNN", "transformer": "Transformer", "vit": "ViT",
         "stack_rel": "HVPF (final)"}
BINS, B = 15, 1000
COVER = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def probs(d):
    p = d[P].to_numpy(float)
    return p / p.sum(1, keepdims=True)


def ece(p, y, bins=BINS):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def brier(p, y):
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean())


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1) / np.log(p.shape[1])


def risk_coverage(u, err):
    """Error rate on the kept clips when the most uncertain (1 - coverage) are referred; AURC over all coverages."""
    order = np.argsort(u, kind="stable")
    e = err[order].astype(float)
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    at = {c: cum[max(int(round(c * len(e))) - 1, 0)] for c in COVER}
    return at, float(cum.mean()), cum


def boot(df, fn, b=B):
    """Point value and 95% camera-level bootstrap interval of fn(df)."""
    rng = np.random.default_rng(0)
    cams = df.claimed_device.unique()
    groups = {c: g for c, g in df.groupby("claimed_device")}
    vals = []
    for _ in range(b):
        pick = rng.choice(cams, len(cams))
        vals.append(fn(pd.concat([groups[c] for c in pick])))
    return fn(df), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def fmt(t, k=3):
    v, lo, hi = t
    return f"{v:.{k}f} [{lo:.{k}f}, {hi:.{k}f}]"


def clip_type(d):
    lab, var, soc = d.label.astype(str), d.variant.astype(str), d.social.astype(str)
    out = np.select(
        [(lab == "organic") & (var == "native"),
         (lab == "organic") & soc.isin(["wa-real", "yt-real"]),
         (lab == "organic") & (soc == "none"),
         (lab == "organic"),
         (lab == "manipulated") & (soc == "none") & var.str.startswith("injection"),
         (lab == "manipulated") & (soc == "none"),
         (lab == "manipulated"),
         (lab == "synthetic") & var.str.contains(r"\+inj"),
         (lab == "synthetic") & (soc == "none"),
         (lab == "synthetic")],
        ["genuine: native", "genuine: real WhatsApp / YouTube copy", "genuine: re-encoded",
         "genuine: re-encoded + platform-like", "attacked: PRNU injection", "attacked: removal / denoise / re-encode",
         "attacked: + platform-like", "AI: + injected PRNU", "AI: plain", "AI: + platform-like"], "other")
    return out


def system_frame(scheme, part):
    """Final HVPF predictions joined with the six base models' probabilities."""
    f = pd.read_parquet(FUSION(scheme))
    f = f[(f.model == "stack_rel") & (f.part == part)].set_index("sample_id")
    base = pd.read_parquet(phase5.PHASE5 / "models_c4m" / f"realistic_{scheme}_all_w10.parquet")
    vit = pd.read_parquet(phase5.PHASE4 / "models" / "vit" / f"realistic_{scheme}_w10.parquet")
    base = pd.concat([base, vit], ignore_index=True)
    base = base[base.model.isin(BASE) & (base.part == part)]
    for m, g in base.groupby("model"):
        g = g.set_index("sample_id").reindex(f.index)
        p = probs(g)
        for k, c in enumerate(CLASSES):
            f[f"{m}_p_{c}"] = p[:, k]
    f["y"] = f.label.map(CLASSES.index)
    p = probs(f)
    f["pred"], f["conf"], f["ent"] = p.argmax(1), p.max(1), entropy(p)
    f["err"] = f.pred != f.y
    votes = np.stack([f[[f"{m}_p_{c}" for c in CLASSES]].to_numpy().argmax(1) for m in BASE], 1)
    f["disagree"] = (votes != f.pred.to_numpy()[:, None]).mean(1)
    f["p_org_std"] = np.stack([f[f"{m}_p_organic"] for m in BASE], 1).std(1)
    f["type"] = clip_type(f)
    return f.reset_index()


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    lines, md = [], ["# Thesis - validation spread and uncertainty of the final HVPF",
                     "", "From saved predictions (no re-training). 95% camera-level bootstrap intervals in brackets.", ""]
    sets = {"unseen cameras (5-fold, 42,083 clips)": system_frame("folds", "test"),
            "unseen datasets (4,716 clips)": system_frame("fixed", "crossdataset")}

    # A. per-fold results
    f = pd.read_parquet(FUSION("folds"))
    f = f[f.model == "stack_rel"]
    rows = []
    for (fold, part), g in f.groupby(["split_name", "part"]):
        d = g.assign(y=g.label.map(CLASSES.index), s=g.p_organic, pred=probs(g).argmax(1))
        rows.append(dict(fold=fold, part=part, clips=len(d), cameras=d.claimed_device.nunique(),
                         **{k: stats.METRICS[k](d) for k in ["macro F1", "AUC genuine vs not", "genuine accepted %",
                                                             "attacks caught %", "AI caught %"]}))
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "per_fold.csv", index=False)
    test = folds[folds.part == "test"].set_index("fold").drop(columns="part")
    summ = test.drop(columns=["clips", "cameras"]).agg(["mean", "std", "min", "max"])
    val = folds[folds.part == "val"].set_index("fold")["macro F1"]
    a = test.copy()
    a.insert(2, "val macro F1", val)
    lines += ["== A. Per-fold results, unseen cameras (test fold never seen; val = the fold the fusion is fitted on)",
              a.to_string(float_format=lambda v: f"{v:.3f}"), "", summ.to_string(float_format=lambda v: f"{v:.3f}"), ""]
    md += ["## A. Per-fold results (unseen cameras)", "", a.round(3).to_markdown(), "", summ.round(3).to_markdown(), "",
           "Validation F1 is optimistic: the fusion layer is fitted on the validation fold. Test folds are untouched.", ""]

    # B. calibration
    cal_rows = []
    for name, d in sets.items():
        y = d.y.to_numpy()
        for m in BASE + ["stack_rel"]:
            cols = P if m == "stack_rel" else [f"{m}_p_{c}" for c in CLASSES]
            p = d[cols].to_numpy()
            p = p / p.sum(1, keepdims=True)
            cal_rows.append(dict(test=name, model=NAMES[m], accuracy=(p.argmax(1) == y).mean(), ECE=ece(p, y),
                                 Brier=brier(p, y), **{"log-loss": nll(p, y)}))
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "calibration.csv", index=False)
    lines += ["== B. Calibration (ECE: gap between confidence and accuracy, 0 = perfect; Brier and log-loss: lower is better)"]
    md += ["## B. Calibration of the predicted probabilities", ""]
    for name, d in sets.items():
        e = boot(d, lambda x: ece(probs(x), x.y.to_numpy()))
        br = boot(d, lambda x: brier(probs(x), x.y.to_numpy()))
        t = cal[cal.test == name].drop(columns="test").set_index("model")
        lines += [f"-- {name}: HVPF ECE {fmt(e)}, Brier {fmt(br)}", t.to_string(float_format=lambda v: f"{v:.3f}"), ""]
        md += [f"**{name}** - HVPF ECE {fmt(e)}, Brier {fmt(br)}", "", t.round(3).to_markdown(), ""]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (name, d) in zip(axes, sets.items()):
        p, y = probs(d), d.y.to_numpy()
        conf, ok = p.max(1), p.argmax(1) == y
        edges = np.linspace(0, 1, BINS + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1]), 0, BINS - 1)
        xs = [conf[idx == b].mean() for b in range(BINS) if (idx == b).sum() >= 20]
        ys = [ok[idx == b].mean() for b in range(BINS) if (idx == b).sum() >= 20]
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
        ax.plot(xs, ys, "o-", color="#1f5fa8", label=f"HVPF (ECE {ece(p, y):.3f})")
        ax2 = ax.twinx()
        ax2.hist(conf, bins=edges, color="#1f5fa8", alpha=0.15)
        ax2.set_yticks([])
        ax.set(xlim=(0.2, 1), ylim=(0.2, 1), xlabel="confidence (max probability)", ylabel="accuracy",
               title=name.split(" (")[0])
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REP / f"fig_reliability.{ext}", dpi=200)
    plt.close(fig)

    # C. per-clip uncertainty
    unc = {"1 - confidence": lambda d: 1 - d.conf.to_numpy(), "entropy": lambda d: d.ent.to_numpy(),
           "base-model disagreement": lambda d: d.disagree.to_numpy() + 1e-3 * d.ent.to_numpy(),
           "spread of base p(genuine)": lambda d: d.p_org_std.to_numpy()}
    lines += ["== C. Does the uncertainty point at HVPF's own errors?  (AUROC: 0.5 = no, 1 = perfectly)"]
    md += ["## C. Per-clip uncertainty: does it point at the errors?", "",
           "AUROC of each uncertainty measure for separating HVPF's wrong decisions from its right ones; "
           "error rate when the most uncertain clips are referred to an examiner (coverage = share decided automatically).", ""]
    rc_rows, curves = [], {}
    for name, d in sets.items():
        err = d.err.to_numpy()
        for uname, ufn in unc.items():
            au = boot(d, lambda x: roc_auc_score(x.err, ufn(x)))
            at, aurc, cum = risk_coverage(ufn(d), err)
            rc_rows.append(dict(test=name, measure=uname, **{"AUROC errors": fmt(au)}, AURC=round(aurc, 4),
                                **{f"error @ {int(c * 100)}%": round(100 * at[c], 1) for c in COVER}))
            if uname in ("entropy", "base-model disagreement"):
                curves[(name, uname)] = cum
    rc = pd.DataFrame(rc_rows)
    rc.to_csv(OUT / "risk_coverage.csv", index=False)
    for name in sets:
        t = rc[rc.test == name].drop(columns="test").set_index("measure")
        lines += [f"-- {name}", t.to_string(), ""]
        md += [f"**{name}**", "", t.to_markdown(), ""]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, sets):
        for (n, u), cum in curves.items():
            if n == name:
                cov = np.arange(1, len(cum) + 1) / len(cum)
                ax.plot(cov * 100, cum * 100, label=u)
        d = sets[name]
        ax.axhline(100 * d.err.mean(), ls="--", color="grey", lw=1, label="no referral")
        ax.set(xlim=(20, 100), ylim=(0, 115 * d.err.mean()), xlabel="coverage: clips decided automatically (%)",
               ylabel="error rate on decided clips (%)", title=name.split(" (")[0])
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REP / f"fig_risk_coverage.{ext}", dpi=200)
    plt.close(fig)

    # D. uncertainty by clip type
    lines += ["== D. Uncertainty and errors by clip type"]
    md += ["## D. Uncertainty and errors by clip type", ""]
    for name, d in sets.items():
        g = d.groupby("type").agg(clips=("y", "size"), error_pct=("err", lambda e: 100 * e.mean()),
                                  mean_entropy=("ent", "mean"), mean_confidence=("conf", "mean"),
                                  models_disagreeing_pct=("disagree", lambda v: 100 * v.mean()))
        g = g.sort_values("mean_entropy", ascending=False)
        g.to_csv(OUT / f"by_type_{name.split(' (')[0].replace(' ', '_')}.csv")
        lines += [f"-- {name}", g.to_string(float_format=lambda v: f"{v:.3f}"), ""]
        md += [f"**{name}**", "", g.round(3).to_markdown(), ""]

    text = "\n".join(lines)
    (OUT / "uncertainty.txt").write_text(text + "\n")
    (REP / "uncertainty.md").write_text("\n".join(md) + "\n")
    print(text)


if __name__ == "__main__":
    main()
