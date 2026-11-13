"""Paper, day 11 - every table and figure of the paper, from the saved results (nothing is retrained).

  tab_main        HVPF vs PCE rule and three same-protocol baselines: unseen cameras (5-fold) + unseen datasets
  tab_spoof       PRNU-transfer spoofing: AI video / camera-to-camera injection accepted as genuine
  tab_adaptive    adaptive attacker vs HVPF, HVPF-R (compression-diverse training) and the round-2 variant
  tab_ablation    feature groups (XGBoost) and the framework's components (stacking, 10 frames)
  tab_zones       three-zone decision (accept / inconclusive / accuse)
  fig_roc         ROC curves, genuine vs not, unseen cameras
  fig_adaptive    evasion rates before / after compression-diverse training
Significance markers come from data/paper/compare.csv (camera-level paired bootstrap). Writes report/paper/
{tables.tex, tables.md, fig_*.pdf, fig_*.png}.

    .venv/bin/python scripts/paper_tables.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5  # noqa: E402

stats = phase5.load_step("paper_stats.py")
report = phase5.load_step("make_final_report.py")
ROOT, PAPER = phase5.ROOT, phase5.ROOT / "data" / "paper"
OUT = ROOT / "report" / "paper"
BASE = [("dinov2_b", "DINOv2 + linear [frozen]"), ("clip_b16", "UnivFD-style (CLIP ViT-B/16)"),
        ("effnet_b4", "EfficientNet-B4 (end-to-end)")]
INK, BLUE, ORANGE, GREY = "#2E4A66", "#1F5F99", "#D9822B", "#8A99A8"


def eer(d):
    fpr, tpr, _ = roc_curve(d.y == 0, d.s)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def systems():
    f = lambda scheme: phase5.PHASE5 / "models" / f"fusion_c4m4_realistic_{scheme}_w10.parquet"
    s = {"PCE rule (conventional)": (stats.system(f("folds"), "baseline", "test"), stats.system(f("fixed"), "baseline", "crossdataset"))}
    for key, name in BASE:
        b = PAPER / "baselines" / key
        s[name] = (stats.system(b / "realistic_folds_w10.parquet", key, "test"),
                   stats.system(b / "realistic_fixed_w10.parquet", key, "crossdataset"))
    s["HVPF (ours)"] = (stats.system(f("folds"), "stack_rel", "test"), stats.system(f("fixed"), "stack_rel", "crossdataset"))
    return s


def sig(name, metric, setting):
    """'*' if HVPF is significantly better than baseline `name` (p < 0.05, camera bootstrap)."""
    c = pd.read_csv(PAPER / "compare.csv")
    key = {"DINOv2": "DINOv2 linear", "UnivFD": "UnivFD-style CLIP", "EfficientNet": "EfficientNet-B4"}
    k = next((v for kk, v in key.items() if kk in name), None)
    r = c[(c.comparison == f"HVPF vs {k} ({setting})") & (c.metric == metric)] if k else c.iloc[:0]
    return "$^{*}$" if len(r) and r.p.iat[0] < 0.05 and r["A - B"].iat[0] > 0 else ""


def to_tex(df, caption, label, note=""):
    cols = "l" + "c" * (df.shape[1])
    body = df.to_latex(index=True, escape=False, column_format=cols).replace("\\toprule", "\\toprule").strip()
    return (f"\\begin{{table*}}[t]\\centering\\small\\caption{{{caption}}}\\label{{{label}}}\n{body}\n"
            + (f"\\par\\footnotesize {note}\n" if note else "") + "\\end{table*}\n")


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    m = stats.METRICS
    S = systems()
    tex, md = [], []

    # ---- main table
    rows = {}
    for name, (cam, ds) in S.items():
        pce = name.startswith("PCE")
        rows[name] = {
            "F1": "--" if pce else f"{m['macro F1'](cam):.3f}{sig(name, 'macro F1', 'unseen cameras')}",
            "AUC": f"{m['AUC genuine vs not'](cam):.3f}{'' if pce else sig(name, 'AUC genuine vs not', 'unseen cameras')}",
            "EER": f"{eer(cam):.3f}",
            "Genuine acc.": f"{m['genuine accepted %'](cam):.1f}",
            "Attacks caught": "--" if pce else f"{m['attacks caught %'](cam):.1f}",
            "AI caught": "--" if pce else f"{m['AI caught %'](cam):.1f}",
            "F1 ": "--" if pce else f"{m['macro F1'](ds):.3f}{sig(name, 'macro F1', 'unseen datasets')}",
            "AUC ": f"{m['AUC genuine vs not'](ds):.3f}{'' if pce else sig(name, 'AUC genuine vs not', 'unseen datasets')}"}
    t = pd.DataFrame(rows).T
    t.columns = pd.MultiIndex.from_tuples([("Unseen cameras (5-fold, 42,083 clips)", c) for c in t.columns[:6]]
                                          + [("Unseen datasets (4,716 clips)", c.strip()) for c in t.columns[6:]])
    tex.append(to_tex(t, "Detection on camera models never seen in training (realistic protocol, 10 I-frames). "
                         "Baselines are trained and tested on exactly the same splits and clips.", "tab:main",
                      "$^{*}$: HVPF significantly better (camera-level paired bootstrap, $p<0.05$). Genuine acc.: genuine "
                      "clips accepted; attacks / AI caught: recognised as their class (\\%)."))
    md += ["## Main results", t.to_markdown()]

    # ---- spoofing table
    rows = {}
    for name, (cam, ds) in S.items():
        v = cam.variant.astype(str)
        inj = lambda d, a: 100 * (d.pred[d.variant.astype(str).str.endswith(f"+inj{a}")] == 0).mean()
        cam_inj = lambda a: 100 * (cam.pred[v.str.startswith(f"injection_a{a}") & ~v.str.contains(r"\+")] == 0).mean()
        rows[name] = {"AI+PRNU seen gen.": f"{inj(cam, '1'):.1f}", "AI+PRNU unseen gen.": f"{inj(ds, '1'):.1f}",
                      "Camera inj. a=1": f"{cam_inj('1'):.1f}", "Camera inj. a=0.5": f"{cam_inj('0.5'):.1f}"}
    t = pd.DataFrame(rows).T
    tex.append(to_tex(t, "PRNU-transfer spoofing: share of clips accepted as genuine footage of the claimed camera "
                         "(\\%, lower is better). Injection strength $\\alpha$ in units of a 1\\% PRNU.", "tab:spoof",
                      "Khan et al.\\ report 85.5\\% for PRNU transfer into images; the conventional PCE rule behaves alike on video."))
    md += ["## Spoofing", t.to_markdown()]

    # ---- adaptive attack table
    rows, adapt = {}, {}
    for name, bundle, suffix in (("HVPF", "bundle_fixed_w10", ""), ("HVPF-R (ours, robust)", "bundle_aug_fixed_w10", "_aug"),
                                 ("HVPF-R + round 2 (not kept)", "bundle_aug2_fixed_w10", "_aug2")):
        pred = pd.read_parquet(PAPER / bundle / "predictions.parquet")
        cl = {p: stats.macro_f1(pred[pred.part == p].label.map(["organic", "manipulated", "synthetic"].index).to_numpy(),
                                pred[pred.part == p][["p_organic", "p_manipulated", "p_synthetic"]].to_numpy().argmax(1))
              for p in ("test", "crossdataset")}
        c = pd.read_parquet(PAPER / "adaptive" / f"candidates{suffix}.parquet")
        per = lambda kind, ok: 100 * c[c.kind == kind].assign(ok=ok).groupby("content_id").ok.any().mean()
        std = c[c.standard]
        v = {"Inj. non-adaptive": 100 * (std[std.kind == "injection"].pred == 0).mean(),
             "Inj. adaptive": per("injection", c.pred == 0),
             "Usable forgery": per("injection", (c.pred == 0) & c.pce_pass),
             "Rem. non-adaptive": 100 * (std[std.kind == "removal"].pred == 0).mean(),
             "Rem. adaptive": per("removal", (c.pred == 0) & ~c.pce_pass)}
        adapt[name] = v
        rows[name] = {"Clean F1 cameras": f"{cl['test']:.3f}", "Clean F1 datasets": f"{cl['crossdataset']:.3f}",
                      **{k: f"{x:.1f}" for k, x in v.items()},
                      "3-zone inj. / rem.": f"{per('injection', c.zone == 0):.1f} / {per('removal', c.zone == 0):.1f}"}
    t = pd.DataFrame(rows).T
    tex.append(to_tex(t, "Adaptive attacker with query access (55 videos of 5 unseen cameras; 36 injection and 24 removal "
                         "settings per video, the most genuine-looking one kept). Evasion rates in \\%.", "tab:adaptive",
                      "Usable forgery: HVPF says genuine \\emph{and} the PCE rule accepts the claimed camera. Removal counts "
                      "only clips whose camera link is broken (PCE rule fails). Clean F1 on the fixed split."))
    md += ["## Adaptive attack", t.to_markdown()]

    # ---- ablation
    s4 = pd.read_csv(phase5.PHASE4 / "metrics" / "summary.csv").query(
        "protocol == 'realistic' and scheme == 'folds' and part == 'test' and model == 'xgb'").set_index("tag")
    feats = {"all features": "all_w5", "without Noiseprint++": "no_noiseprint_w5", "without PRNU": "no_prnu_w5",
             "without temporal consistency": "no_tsncs_w5", "without residual stats": "no_resid_w5", "without FFT": "no_fft_w5"}
    ta = pd.DataFrame({"XGBoost F1": {k: f"{s4.at[v, 'f1_mean']:.3f}" for k, v in feats.items()},
                       "AUC": {k: f"{s4.at[v, 'bin_auc_mean']:.3f}" for k, v in feats.items()}})
    pc = pd.read_csv(phase5.PHASE5 / "metrics" / "phase5_comparison.csv")
    steps = [("hybrid stacking (PRNU, Noiseprint++, residual, FFT, TSNCS, ViT)", "Phase 4"),
             ("  only + reliability-aware fusion", "+ reliability fusion (2)"),
             ("  only + processing-matched training", "+ matched training (1)"),
             ("  only + camera-calibrated PRNU", "+ camera calibration (4)"),
             ("full HVPF (all three)", "+ all four, Phase 4 ViT")]
    tb = pd.DataFrame({"F1 cameras": {k: f"{pc.query('config == @c and window == 10 and test.str.startswith(\"unseen cameras\")').F1.iat[0]:.3f}" for k, c in steps},
                       "F1 datasets": {k: f"{pc.query('config == @c and window == 10 and test == \"unseen datasets\"').F1.iat[0]:.3f}" for k, c in steps}})
    tex.append(to_tex(ta, "Feature groups: XGBoost retrained without one group (unseen cameras, 5 I-frames).", "tab:abl_feat"))
    tex.append(to_tex(tb, "Framework components (10 I-frames).", "tab:abl_comp"))
    md += ["## Ablation: features", ta.to_markdown(), "## Ablation: components", tb.to_markdown()]

    # ---- three-zone
    z = report.decisions(report.final("folds", 10).query("model == 'stack_rel'"))
    zx = report.decisions(report.final("fixed", 10).query("model == 'stack_rel'"))
    zx = zx[zx.part == "crossdataset"]
    rows = {}
    for setting, d in (("unseen cameras", z), ("unseen datasets", zx)):
        for cls, nm in (("organic", "genuine"), ("manipulated", "attacked"), ("synthetic", "AI")):
            q = d[d.label == cls].zone
            if len(q):
                rows[f"{nm} ({setting})"] = {"accepted": f"{100 * (q == 0).mean():.1f}", "inconclusive": f"{100 * (q == -1).mean():.1f}",
                                             "accused / flagged": f"{100 * (q > 0).mean():.1f}"}
    t = pd.DataFrame(rows).T
    tex.append(to_tex(t, "Three-zone decision (thresholds set on validation cameras: at most 5\\% genuine accused and 5\\% "
                         "attacked / AI accepted). Shares in \\%.", "tab:zones"))
    md += ["## Three-zone decision", t.to_markdown()]
    (OUT / "tables.tex").write_text("\n".join(tex))
    (OUT / "tables.md").write_text("\n\n".join(md) + "\n")

    # ---- figures
    plt.rcParams.update({"font.family": "Noto Sans", "font.size": 9, "axes.edgecolor": GREY})
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    colors = {"PCE rule (conventional)": ORANGE, "DINOv2 + linear [frozen]": "#A0A9B3",
              "UnivFD-style (CLIP ViT-B/16)": "#6AA84F", "EfficientNet-B4 (end-to-end)": "#8E5BA8", "HVPF (ours)": BLUE}
    for name, (cam, _) in S.items():
        fpr, tpr, _ = roc_curve(cam.y == 0, cam.s)
        ax.plot(fpr, tpr, lw=2.0 if "HVPF" in name else 1.2, color=colors.get(name),
                label=f"{name.split(' (')[0].split(' +')[0]} ({stats.METRICS['AUC genuine vs not'](cam):.3f})")
    ax.plot([0, 1], [0, 1], ":", color=GREY, lw=0.8)
    ax.set_xlabel("False acceptance (not genuine accepted)")
    ax.set_ylabel("Genuine accepted")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_roc.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    keys = ["Inj. non-adaptive", "Inj. adaptive", "Usable forgery", "Rem. adaptive"]
    x = np.arange(len(keys))
    for i, (name, col, lab) in enumerate((("HVPF", GREY, "HVPF"), ("HVPF-R (ours, robust)", BLUE, "HVPF-R"))):
        vals = [adapt[name][k] for k in keys]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=col, label=lab)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, ["Injection\nnon-adaptive", "Injection\nadaptive", "Usable\nforgery", "Removal\nadaptive"], fontsize=7)
    ax.set_ylabel("Evasion (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_adaptive.{ext}", dpi=300)
    plt.close(fig)
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
