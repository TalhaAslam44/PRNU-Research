"""Results report of Phase 3 (mid-evaluation) and Phase 5 (final): report/results_phase3_phase5.pdf

Every number is computed here from the saved predictions with the same definitions for both phases
(realistic protocol, test cameras never seen in training; 5-fold scores are means over folds unless a
table says "pooled"). Phase 3 = data/phase3 (stacking); Phase 5 final = changes 1+2+4 with the Phase 4
ViT (data/phase5/models/fusion_c4m4_*, model stack_rel). The PDF is printed from HTML by headless Chrome.

    .venv/bin/python scripts/make_final_report.py
"""
import base64
import io
import shutil
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5  # noqa: E402
from hvpf.model_data import CLASSES  # noqa: E402

ROOT = phase5.ROOT
P3, P4, P5 = ROOT / "data" / "phase3", phase5.PHASE4, phase5.PHASE5
P = [f"p_{c}" for c in CLASSES]
step8c = phase5.load_step("08c_ensemble.py")
OUT = ROOT / "report" / "results_phase3_phase5"
MODELS = [("baseline", "PCE rule (conventional)"), ("svm", "SVM"), ("xgb", "XGBoost"), ("gru", "GRU"),
          ("cnn", "1D-CNN"), ("transformer", "Transformer"), ("vit", "ViT (SRM residuals)")]
BLUE, LIGHT, ORANGE, GREY, INK = "#1F5F99", "#9CC3E6", "#D9822B", "#B9C3CE", "#2E4A66"


# ---------------------------------------------------------------- data access
def rows(path, model=None, part=None):
    d = pd.read_parquet(path)
    if model:
        d = d[d.model == model]
    return d[d.part == part] if part else d


def p5_base(scheme, w):
    """Phase 5 base models (matched + camera-calibrated) plus the Phase 4 ViT."""
    d = pd.read_parquet(P5 / "models_c4m" / f"realistic_{scheme}_all_w{w}.parquet")
    return pd.concat([d, pd.read_parquet(P4 / "models" / "vit" / f"realistic_{scheme}_w{w}.parquet")], ignore_index=True)


def final(scheme, w):
    return rows(P5 / "models" / f"fusion_c4m4_realistic_{scheme}_w{w}.parquet")


def p3(scheme, w, protocol="realistic", tag="all"):
    return rows(P3 / "models" / f"{protocol}_{scheme}_{tag}_w{w}.parquet")


# ---------------------------------------------------------------- metrics
def eer(genuine, score):
    fpr, tpr, _ = roc_curve(genuine, score)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def scores(d):
    """F1 (classes present) and accuracy for learned models, AUC / EER genuine-vs-not; means over splits."""
    out = {"F1": [], "PVA": [], "AUC": [], "EER": []}
    for _, g in d.groupby("split_name"):
        y = g.label.map(CLASSES.index).to_numpy()
        s = g.score_organic.to_numpy() if g.model.iat[0] == "baseline" else g.p_organic.to_numpy()
        if g.model.iat[0] != "baseline":
            pred = g[P].to_numpy().argmax(1)
            out["F1"].append(f1_score(y, pred, labels=np.unique(y), average="macro"))
            out["PVA"].append((pred == y).mean())
        if (y == 0).any() and (y != 0).any():
            out["AUC"].append(roc_auc_score(y == 0, s))
            out["EER"].append(eer(y == 0, s))
    res = {k: (np.mean(v) if v else np.nan) for k, v in out.items()}
    res["F1 sd"] = np.std(out["F1"]) if len(out["F1"]) > 1 else np.nan
    return res


def predictions(d):
    """Class decision per clip (0/1/2); the PCE rule only says genuine (0) or not (-2)."""
    if d.model.iat[0] == "baseline":
        return np.where(d.score_organic > d.threshold, 0, -2)
    return d[P].to_numpy().argmax(1)


def decisions(d):
    """Default, safe-point and three-zone decisions of one fused model for its test / crossdataset clips;
    the safe boost and the three-zone thresholds are set on each split's validation clips (as in Phase 5)."""
    parts = []
    for _, g in d.groupby("split_name"):
        val = (g.part == "val").to_numpy()
        y = g.label.map(CLASSES.index).to_numpy()
        prob = g[P].to_numpy()
        lp = step8c.logp(prob[val & (y == 0)])
        boost = next((b for b in np.arange(0, 8.01, 0.05)
                      if ((lp + [b, 0, 0]).argmax(1) == 0).mean() >= step8c.SAFE_RECALL), 8.0)
        lo, hi = phase5.three_zone_thresholds(prob[val], y[val])
        parts.append(g[~val].assign(pred=prob[~val].argmax(1),
                                    safe=step8c.shift(prob[~val], np.array([boost, 0, 0])).argmax(1),
                                    zone=phase5.three_zone_decision(prob[~val], lo, hi)))
    return pd.concat(parts, ignore_index=True)


def groups(d, unseen=False):
    """Clip groups: name -> (mask, wanted class)."""
    base = d.variant.astype(str).str.split("+").str[0]
    inj = d.variant.astype(str).str.contains(r"\+inj")
    gen, man, syn = d.label == "organic", d.label == "manipulated", d.label == "synthetic"
    reenc = gen & d.source.isin(["VISION-attack", "ACID-attack"])
    if unseen:
        g = {"FloreView genuine videos (unseen phones)": (gen, 0), "GenBuster AI videos (unseen generators)": (syn & ~inj, 2),
             "GenBuster AI + injected PRNU": (syn & inj, 2)}
    else:
        g = {"Genuine: original camera videos": (gen & (base == "native"), 0),
             "Genuine: real WhatsApp / YouTube copies": (gen & base.isin(["whatsapp", "youtube"]), 0),
             "Genuine: re-encoded (CRF 18-28, platform-like)": (reenc, 0),
             "Genuine: CRF-20 re-encodes (same encoder as attacks)": (reenc & (base == "reencode") & (d.social == "none"), 0),
             "Attack: PRNU removal": (man & base.str.startswith("removal_a"), 1),
             "Attack: white-box removal (PCE driven to ~0)": (man & (base == "removal_opt"), 1),
             "Attack: PRNU injection": (man & base.str.startswith("injection"), 1),
             "Attack: denoising": (man & base.str.startswith("denoise"), 1),
             "Attacks on ACID videos": (man & (d.source == "ACID-attack"), 1),
             "AI video (seen generators)": (syn & ~inj, 2),
             "AI video + injected PRNU": (syn & inj, 2)}
    return {k: (m.to_numpy(), w) for k, (m, w) in g.items() if m.any()}


def correct(pred, want):
    """Share decided correctly: genuine accepted; attacks / AI recognised as their class (PCE rule: rejected)."""
    if want == 0:
        return 100 * (pred == 0).mean()
    return 100 * (np.isin(pred, [want, -2])).mean()


# ---------------------------------------------------------------- html helpers
CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: "Noto Sans", "DejaVu Sans", sans-serif; color: #1d2a36; font-size: 10.2pt; line-height: 1.38; }
h1 { font-size: 20pt; color: #1F5F99; margin: 0 0 2mm 0; }
h2 { font-size: 14pt; color: #1F5F99; border-bottom: 1.5px solid #1F5F99; padding-bottom: 1mm; margin-top: 7mm; }
h3 { font-size: 11.2pt; color: #2E4A66; margin: 5mm 0 1.5mm 0; break-after: avoid; page-break-after: avoid; }
.sub { color: #4A5B6C; font-size: 10.5pt; margin-bottom: 4mm; }
table { border-collapse: collapse; width: 100%; margin: 1.5mm 0 1mm 0; font-size: 9pt; page-break-inside: avoid; }
th { background: #E7EEF7; color: #2E4A66; text-align: center; padding: 1.4mm 1.6mm; border-bottom: 1px solid #9CC3E6; }
td { padding: 1.1mm 1.6mm; border-bottom: 0.5px solid #dde4ea; text-align: center; }
td:first-child, th:first-child { text-align: left; }
tr.hl td { background: #F3F8FD; font-weight: bold; }
.note { color: #4A5B6C; font-size: 8.6pt; margin: 0.5mm 0 3mm 0; }
.box { background: #F3F8FD; border-left: 3px solid #1F5F99; padding: 2.5mm 3.5mm; margin: 3mm 0; }
.break { page-break-before: always; }
img { width: 100%; }
ul { margin: 1mm 0 2mm 0; padding-left: 6mm; } li { margin: 0.6mm 0; }
"""


def fmt(v, kind="f3"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.3f}" if kind == "f3" else f"{v:.1f}%" if kind == "pct" else f"{v:,}" if kind == "int" else str(v)


def table(header, body, highlight=(), note=None):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr>"
    b = "".join(f"<tr{' class=hl' if i in highlight else ''}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                for i, r in enumerate(body))
    return f"<table>{h}{b}</table>" + (f"<p class=note>{note}</p>" if note else "")


def figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# ---------------------------------------------------------------- sections
def data_section():
    def counts(path):
        v = pd.read_parquet(path, columns=["label", "source", "variant", "split"])
        base = v.variant.astype(str).str.split("+").str[0]
        gen = v.label == "organic"
        syn = v.label == "synthetic"
        return {"Genuine VISION videos (originals + real WhatsApp/YouTube)": (gen & (v.source == "VISION")).sum(),
                "Genuine ACID videos": (gen & (v.source == "ACID")).sum(),
                "Genuine re-encoded copies (VISION + ACID)": (gen & v.source.isin(["VISION-attack", "ACID-attack"])).sum(),
                "Attacked clips (removal, injection, denoising)": (v.label == "manipulated").sum(),
                "&nbsp;&nbsp;&nbsp;&ndash; of which on ACID videos": ((v.label == "manipulated") & (v.source == "ACID-attack")).sum(),
                "AI clips, seen generators (GenVidBench)": (syn & (v.source == "GenVidBench")).sum(),
                "&nbsp;&nbsp;&nbsp;&ndash; number of generators": len(base[syn & (v.source == "GenVidBench")].unique()),
                "Unseen test: FloreView genuine videos": (gen & (v.source == "FloreView")).sum(),
                "Unseen test: GenBuster AI clips": (syn & (v.source == "GenBuster")).sum(),
                "Total clips": len(v)}
    a, b = counts(P3 / "features" / "videos.parquet"), counts(P4 / "features" / "videos.parquet")
    body = [[k, fmt(int(a[k]), "int"), fmt(int(b[k]), "int")] for k in a]
    return ("<h2>1. Data and evaluation protocol</h2>"
            + table(["", "Phase 3", "Phase 5"], body, highlight=(len(body) - 1,),
                    note="Phase 5 uses the Phase 4 data (ACID attacks, extra genuine platform-like copies, 4 more AI "
                         "generators: SVD, Mora, Text2Video-Zero, Pika). Each clip: first 5 (or 10) I-frames, 480 x 480 luma.")
            + "<div class=box><b>Protocol.</b> Realistic protocol, 5-fold cross-validation grouped by camera model: every "
              "test camera is unseen in training; validation uses other camera models. <b>Unseen datasets</b>: trained on "
              "VISION + ACID + GenVidBench, tested on FloreView (new phones) and GenBuster (new AI generators). "
              "<b>Metrics</b>: macro F1 over organic / manipulated / synthetic; AUC and EER for genuine vs. not genuine; "
              "the PCE rule (PCE&nbsp;&gt;&nbsp;60 with the claimed camera) is the conventional baseline.</div>")


def model_table(sources, highlight_last=True, note=None):
    """sources: [(label, {w: frame})]; one row per model, F1/AUC/EER at 5 and 10 frames."""
    body = []
    for label, frames in sources:
        r = {w: scores(f) for w, f in frames.items()}
        body.append([label] + sum([[fmt(r[w]["F1"]) + (f" ± {r[w]['F1 sd']:.3f}" if not np.isnan(r[w]["F1"]) else ""),
                                    fmt(r[w]["AUC"]), fmt(r[w]["EER"])] for w in (5, 10)], []))
    return table(["Model", "F1 (5 frames)", "AUC", "EER", "F1 (10 frames)", "AUC", "EER"], body,
                 highlight=(len(body) - 1,) if highlight_last else (), note=note)


def type_table(cols, unseen=False, note=None):
    """cols: [(label, test rows with a pred column)] -> % correct per clip group."""
    names = list(groups(cols[0][1], unseen))
    for _, d in cols[1:]:
        names += [n for n in groups(d, unseen) if n not in names]
    body = []
    for n in names:
        row = [n]
        for _, d in cols:
            g = groups(d, unseen)
            row.append(fmt(correct(d.pred.to_numpy()[g[n][0]], g[n][1]), "pct") if n in g else "–")
        body.append(row)
    return table(["Clips"] + [c for c, _ in cols], body, note=note)


def with_pred(d):
    return d.assign(pred=predictions(d))


def inj_table(cols):
    """% of AI clips with injected PRNU accepted as genuine. cols: [(label, folds test rows, crossdataset rows)]."""
    body = []
    for a in ("1", "0.5"):
        for where, k in (("seen generators", 1), ("unseen generators (GenBuster)", 2)):
            row = [f"Injection strength {a}, {where}"]
            for _, *frames in cols:
                d = frames[k - 1]
                m = d.variant.astype(str).str.endswith(f"+inj{a}").to_numpy()
                row.append(fmt(100 * (predictions(d)[m] == 0).mean(), "pct") if m.any() else "–")
            body.append(row)
    return table(["AI video + injected PRNU: accepted as genuine (lower is better)"] + [c for c, *_ in cols], body)


def zone_table(cols, note=None):
    """cols: [(label, decisions frame)] -> default / safe / three-zone outcomes per class."""
    body = []
    for cls, k in (("Genuine", 0), ("Attacked", 1), ("AI", 2)):
        for what, fn in (("default: correct", lambda d: (d.pred == k).mean()),
                         ("safe point: correct", lambda d: (d.safe == k).mean()),
                         ("3-zone: accepted as genuine", lambda d: (d.zone == 0).mean()),
                         ("3-zone: inconclusive", lambda d: (d.zone == -1).mean()),
                         ("3-zone: " + ("accused" if k == 0 else "flagged (not genuine)"), lambda d: (d.zone > 0).mean())):
            body.append([f"{cls}: {what}"] + [fmt(100 * fn(d[d.label == CLASSES[k]]), "pct") for _, d in cols])
    body.append(["Clips decided (3-zone)"] + [fmt(100 * (d.zone >= 0).mean(), "pct") for _, d in cols])
    body.append(["Macro F1 on decided clips (3-zone)"]
                + [fmt(f1_score(d.label.map(CLASSES.index)[d.zone >= 0], d.zone[d.zone >= 0],
                                labels=np.unique(d.label.map(CLASSES.index)), average="macro")) for _, d in cols])
    return table(["Decision rule"] + [c for c, _ in cols], body, note=note)


def phase3_section():
    html = ["<h2 class=break>2. Phase 3 results (mid-evaluation)</h2>",
            "<h3>2.1 All models, unseen cameras (realistic, 5-fold)</h3>"]
    srcs = [(n, {w: p3("folds", w).query("part == 'test' and model == @m") for w in (5, 10)}) for m, n in MODELS]
    srcs.append(("<b>HVPF stacking</b>", {w: p3("folds", w).query("part == 'test' and model == 'stack'") for w in (5, 10)}))
    html.append(model_table(srcs, note="Stacking = logistic regression on the class probabilities of every model, "
                                       "fitted on the validation cameras. EER: equal error rate, genuine vs. not. "
                                       "Phase 3 trained the ViT on 5 frames only, so its 10-frame stacking has no ViT."))
    t5 = with_pred(p3("folds", 5).query("part == 'test' and model == 'stack'"))
    html += ["<h3>2.2 Accuracy per kind of clip (5 frames)</h3>",
             type_table([("PCE rule", with_pred(p3("folds", 5).query("part == 'test' and model == 'baseline'"))),
                         ("XGBoost", with_pred(p3("folds", 5).query("part == 'test' and model == 'xgb'"))),
                         ("ViT", with_pred(p3("folds", 5).query("part == 'test' and model == 'vit'"))),
                         ("<b>HVPF stacking</b>", t5)],
                        note="Correct = genuine clip accepted, attacked / AI clip recognised (PCE rule: rejected, it cannot "
                             "name the class). Phase 3 has no ACID attacks.")]
    cr = [("Controlled protocol, 5-fold", "controlled", "folds", "test"), ("Unseen datasets (fixed split)", "realistic", "fixed", "crossdataset")]
    body = []
    for label, prot, scheme, part in cr:
        for w in (5, 10):
            d = p3(scheme, w, prot)
            s, b = scores(d.query("part == @part and model == 'stack'")), scores(d.query("part == @part and model == 'baseline'"))
            body.append([f"{label}, {w} frames", fmt(s["F1"]), fmt(s["AUC"]), fmt(s["EER"]), fmt(b["AUC"])])
    html += ["<h3>2.3 Encoder-controlled protocol and unseen datasets (stacking)</h3>",
             table(["Setting", "Macro F1", "AUC", "EER", "PCE rule AUC"], body,
                   note="Controlled: every clip re-encoded with libx264 so compression cannot give the class away. "
                        "Unseen datasets: FloreView + GenBuster, never used in training. The 10-frame rows have no ViT "
                        "(not trained at 10 frames in Phase 3), which is why they drop on the unseen datasets.")]
    cross = {w: with_pred(p3("fixed", w).query("part == 'crossdataset' and model == 'stack'")) for w in (5, 10)}
    html.append(type_table([(f"Stacking, {w} frames", cross[w]) for w in (5, 10)]
                           + [("PCE rule", with_pred(p3("fixed", 5).query("part == 'crossdataset' and model == 'baseline'")))],
                           unseen=True))
    html += ["<h3>2.4 AI video with injected camera PRNU (Khan-style attack)</h3>",
             inj_table([("PCE rule", p3("folds", 5).query("part == 'test' and model == 'baseline'"),
                          p3("fixed", 5).query("part == 'crossdataset' and model == 'baseline'")),
                         ("HVPF stacking", p3("folds", 5).query("part == 'test' and model == 'stack'"),
                          p3("fixed", 5).query("part == 'crossdataset' and model == 'stack'"))])]
    abl = pd.read_csv(P3 / "metrics" / "summary.csv").query("protocol == 'realistic' and scheme == 'folds' and part == 'test'")
    mods = [("svm", "SVM"), ("xgb", "XGBoost"), ("gru", "GRU"), ("cnn", "1D-CNN"), ("transformer", "Transformer")]
    body = []
    for tag, name in [("all_w5", "All features"), ("no_noiseprint_w5", "without Noiseprint++"), ("no_prnu_w5", "without PRNU"),
                      ("no_tsncs_w5", "without temporal noise consistency"), ("no_resid_w5", "without residual statistics"),
                      ("no_fft_w5", "without FFT")]:
        r = abl[abl.tag == tag].set_index("model")
        body.append([name] + [fmt(r.at[m, "f1_mean"]) if m in r.index else "–" for m, _ in mods])
    html += ["<h3>2.5 Which features matter (remove one group, retrain; macro F1, 5 frames)</h3>",
             table(["Feature set"] + [n for _, n in mods], body, highlight=(0,))]
    z = {w: decisions(p3("folds", w).query("model == 'stack'")) for w in (5, 10)}
    html += ["<h3>2.6 Decision rules (stacking)</h3>",
             zone_table([(f"{w} frames", z[w]) for w in (5, 10)],
                        note="Safe point: organic boost so that &ge; 95% of genuine validation clips are accepted. "
                             "3-zone: accept / inconclusive / accuse with both error rates capped at 5% on validation "
                             "(the Phase 5 rule, applied here to the Phase 3 models for comparison).")]
    return "\n".join(html), z


def phase5_section():
    t = pd.read_csv(P5 / "metrics" / "phase5_comparison.csv")
    order = ["Phase 4", "+ reliability fusion (2)", "+ matched training (1)", "+ matched + reliability (1+2)",
             "+ camera calibration (4)", "+ all four (1+2+4)", "+ all four, Phase 4 ViT"]
    names = {"Phase 4": "Phase 4 (starting point)", "+ reliability fusion (2)": "+ reliability-aware fusion (2)",
             "+ matched training (1)": "+ processing-matched training (1)", "+ matched + reliability (1+2)": "+ 1 + 2",
             "+ camera calibration (4)": "+ camera-calibrated PRNU (4)", "+ all four (1+2+4)": "+ 1 + 2 + 4, matched ViT",
             "+ all four, Phase 4 ViT": "<b>Phase 5 final: 1 + 2 + 4, Phase 4 ViT</b>"}
    body = []
    for c in order:
        r = {(w, s): t[(t.config == c) & (t.window == w) & (t.test == s)].iloc[0]
             for w in (5, 10) for s in ("unseen cameras (5-fold)", "unseen datasets")}
        cam10, ds10 = r[(10, "unseen cameras (5-fold)")], r[(10, "unseen datasets")]
        body.append([names[c], fmt(r[(5, "unseen cameras (5-fold)")]["F1"]), fmt(cam10["F1"]), fmt(cam10["AUC"]),
                     fmt(cam10["genuine (all): correct %"], "pct"), fmt(cam10["genuine CRF-20 re-encodes: correct %"], "pct"),
                     fmt(cam10["attacks: correct %"], "pct"), fmt(cam10["AI: correct %"], "pct"),
                     fmt(ds10["F1"]), fmt(ds10["AUC"])])
    html = ["<h2 class=break>3. Phase 5 results (final framework)</h2>",
            "<div class=box><b>Phase 5 changes</b> (same data and splits as Phase 4): <b>1</b> processing-matched training "
            "weights; <b>2</b> reliability-aware stacking; <b>3</b> three-zone decision (accept / inconclusive / accuse); "
            "<b>4</b> camera-calibrated PRNU evidence (the clip's PCE relative to the claimed camera's median over 10 "
            "genuine reference videos, never the clip's own source video).</div>",
            "<h3>3.1 What each change contributes</h3>",
            table(["Configuration", "F1 5 fr.", "F1 10 fr.", "AUC 10", "Genuine ok", "CRF-20 genuine ok",
                   "Attacks caught", "AI caught", "Unseen-data F1", "Unseen AUC"], body, highlight=(len(body) - 1,),
                  note="Unseen cameras, 5-fold, 10 frames unless marked; unseen-data columns: FloreView + GenBuster, "
                       "10 frames. Change 3 (three-zone) does not change these default decisions; see 3.5.")]
    srcs = []
    for m, n in MODELS:
        srcs.append((n, {w: p5_base("folds", w).query("part == 'test' and model == @m") for w in (5, 10)}))
    srcs.append(("Stacking (plain)", {w: final("folds", w).query("part == 'test' and model == 'stack'") for w in (5, 10)}))
    srcs.append(("<b>HVPF Phase 5 (reliability-aware stacking)</b>",
                 {w: final("folds", w).query("part == 'test' and model == 'stack_rel'") for w in (5, 10)}))
    html += ["<h3>3.2 All models, unseen cameras (realistic, 5-fold)</h3>",
             model_table(srcs, note="Feature models trained with changes 1 + 4; the ViT is the Phase 4 ViT "
                                    "(the matched ViT lowered F1, see 3.7).")]
    p4 = {w: with_pred(rows(P5 / "models" / f"fusion_p4_realistic_folds_w{w}.parquet", "stack", "test")) for w in (5, 10)}
    f5 = {w: with_pred(final("folds", w).query("part == 'test' and model == 'stack_rel'")) for w in (5, 10)}
    html += ["<h3>3.3 Accuracy per kind of clip</h3>",
             type_table([("PCE rule", with_pred(final("folds", 10).query("part == 'test' and model == 'baseline'"))),
                         ("Phase 4, 10 fr.", p4[10]), ("Phase 5, 5 fr.", f5[5]), ("<b>Phase 5, 10 fr.</b>", f5[10])])]
    body = []
    for w in (5, 10):
        d = final("fixed", w)
        s, b = scores(d.query("part == 'crossdataset' and model == 'stack_rel'")), scores(d.query("part == 'crossdataset' and model == 'baseline'"))
        body.append([f"Unseen datasets, {w} frames", fmt(s["F1"]), fmt(s["AUC"]), fmt(s["EER"]), fmt(b["AUC"])])
    cross = {w: with_pred(final("fixed", w).query("part == 'crossdataset' and model == 'stack_rel'")) for w in (5, 10)}
    html += ["<h3>3.4 Unseen datasets (FloreView + GenBuster)</h3>",
             table(["Setting", "Macro F1", "AUC", "EER", "PCE rule AUC"], body),
             type_table([(f"Phase 5, {w} frames", cross[w]) for w in (5, 10)]
                        + [("PCE rule", with_pred(final("fixed", 5).query("part == 'crossdataset' and model == 'baseline'")))],
                        unseen=True)]
    z = {w: decisions(final("folds", w).query("model == 'stack_rel'")) for w in (5, 10)}
    zc = decisions(final("fixed", 10).query("model == 'stack_rel'"))
    zc = zc[zc.part == "crossdataset"]
    html += ["<h3>3.5 Decision rules, incl. change 3 (three-zone)</h3>",
             zone_table([(f"Unseen cameras, {w} fr.", z[w]) for w in (5, 10)] + [("Unseen datasets, 10 fr.", zc)],
                        note="3-zone thresholds come from the validation cameras (at most 5% of genuine clips accused, at "
                             "most 5% of attacked and of AI clips accepted). Inconclusive clips go to an examiner.")]
    html += ["<h3>3.6 AI video with injected camera PRNU</h3>",
             inj_table([("PCE rule", final("folds", 10).query("part == 'test' and model == 'baseline'"),
                         final("fixed", 10).query("part == 'crossdataset' and model == 'baseline'")),
                        ("Phase 5, 10 frames", final("folds", 10).query("part == 'test' and model == 'stack_rel'"),
                         final("fixed", 10).query("part == 'crossdataset' and model == 'stack_rel'"))])]
    cam = pd.read_csv(P4 / "camera_id" / "summary.csv").set_index(["dataset", "group"])
    body = []
    for ds, name in (("VISION", "VISION"), ("ACID", "Video-ACID"), ("FloreView", "FloreView")):
        for grp in ("all", "usable PRNU", "no usable PRNU"):
            r = cam.loc[(ds, grp)]
            body.append([f"{name}: {grp} ({int(r.cameras)} cameras, {int(r.videos)} test videos)", fmt(100 * r.prnu_acc, "pct"),
                         fmt(100 * r.vit_acc, "pct"), fmt(100 * r.hybrid_val_acc, "pct"), fmt(100 * r.hybrid_val_model_acc, "pct")])
    html += ["<h3>3.7 Camera identification (Step 10, same data as Phase 5)</h3>",
             table(["Test videos", "PRNU", "Trained ViT", "<b>Hybrid PRNU + ViT</b>", "Hybrid: camera model"], body,
                   note="Top-1 accuracy among the dataset's cameras; scenes grouped so that near-duplicate recordings never "
                        "fall on both sides of the split."),
             "<h3>3.8 Tried in Phase 5 and not kept</h3><ul>"
             "<li><b>Processing-matched ViT</b> (overnight run): genuine CRF-20 re-encodes accepted 58% &rarr; 72%, but "
             "F1 0.842 &rarr; 0.833 (10 frames), so the final framework keeps the Phase 4 ViT.</li>"
             "<li><b>Block-level compression-aware PRNU predictor</b> (Chen et al., 2008): after video compression the "
             "block correlation is mostly noise (observed vs. expected r = 0.15); no gain, dropped.</li>"
             "<li><b>Single-threshold inconclusive rule</b>: needed 44% inconclusive for the same false-accusation rate; "
             "replaced by the three-zone rule.</li></ul>"]
    return "\n".join(html), z, f5


def comparison_section(z3, z5, f5):
    ids = set(p3("folds", 5).query("part == 'test' and model == 'baseline'").sample_id)
    cols = [("Phase 3, 5 frames", p3("folds", 5).query("part == 'test' and model == 'stack'"), z3[5]),
            ("Phase 3, 10 frames", p3("folds", 10).query("part == 'test' and model == 'stack'"), z3[10]),
            ("Phase 5, 5 frames", final("folds", 5).query("part == 'test' and model == 'stack_rel'"), z5[5]),
            ("<b>Phase 5, 10 frames</b>", final("folds", 10).query("part == 'test' and model == 'stack_rel'"), z5[10])]
    met = {}
    for label, d, z in cols:
        d, z = d[d.sample_id.isin(ids)], z[z.sample_id.isin(ids)]
        y, pred = d.label.map(CLASSES.index).to_numpy(), predictions(d)
        met[label] = {"Clips": fmt(len(d), "int"), "Macro F1 (pooled)": fmt(f1_score(y, pred, average="macro")),
                      "AUC genuine vs. not": fmt(roc_auc_score(y == 0, d.p_organic)),
                      "Genuine accepted": fmt(100 * (pred[y == 0] == 0).mean(), "pct"),
                      "Attacks caught": fmt(100 * (pred[y == 1] == 1).mean(), "pct"),
                      "AI caught": fmt(100 * (pred[y == 2] == 2).mean(), "pct"),
                      "3-zone: genuine accused": fmt(100 * (z.zone[z.label == "organic"] > 0).mean(), "pct"),
                      "3-zone: attacks accepted as genuine": fmt(100 * (z.zone[z.label == "manipulated"] == 0).mean(), "pct"),
                      "3-zone: clips decided": fmt(100 * (z.zone >= 0).mean(), "pct")}
    keys = list(next(iter(met.values())))
    t1 = table(["Same clips: Phase 3's test set"] + list(met), [[k] + [met[c][k] for c in met] for k in keys], highlight=(1,))
    body, vals = [], {}
    for label, fr in (("Phase 3", lambda w: p3("fixed", w)), ("Phase 5", lambda w: final("fixed", w))):
        model = "stack" if label == "Phase 3" else "stack_rel"
        for w in (5, 10):
            d = fr(w).query("part == 'crossdataset' and model == @model").assign(model=model)
            s = scores(d)
            g = groups(with_pred(d), unseen=True)
            pr = predictions(d)
            row = [f"{label}, {w} frames", fmt(s["F1"]), fmt(s["AUC"])] + [fmt(correct(pr[m], k), "pct") for m, k in g.values()]
            body.append(row)
            vals[(label, w)] = s
    t2 = table(["Unseen datasets (same 4,716 clips)", "Macro F1", "AUC", "FloreView genuine accepted", "GenBuster AI caught",
                "GenBuster AI + injected caught"], body, highlight=(3,))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    labels = ["Unseen cameras\n(Phase 3 test clips)", "Unseen datasets\n(FloreView + GenBuster)"]
    f1v = [[float(met["Phase 3, 5 frames"]["Macro F1 (pooled)"]), vals[("Phase 3", 5)]["F1"]],
           [float(met["<b>Phase 5, 10 frames</b>"]["Macro F1 (pooled)"]), vals[("Phase 5", 10)]["F1"]]]
    aucv = [[float(met["Phase 3, 5 frames"]["AUC genuine vs. not"]), vals[("Phase 3", 5)]["AUC"]],
            [float(met["<b>Phase 5, 10 frames</b>"]["AUC genuine vs. not"]), vals[("Phase 5", 10)]["AUC"]]]
    for ax, v, title, lo in ((axes[0], f1v, "Macro F1", 0.7), (axes[1], aucv, "AUC (genuine vs. not)", 0.85)):
        x = np.arange(2)
        for i, (name, c) in enumerate((("Phase 3 (mid-evaluation)", LIGHT), ("Phase 5 (final, 10 frames)", BLUE))):
            bars = ax.bar(x + (i - 0.5) * 0.36, v[i], 0.36, color=c, label=name)
            for b, val in zip(bars, v[i]):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(lo, 1.0)
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#E3E8ED")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig1 = figure(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    rows_ = []
    for label, z in (("Phase 3", z3[5]), ("Phase 5", z5[10])):
        for cls, name in (("organic", "genuine"), ("manipulated", "attacked"), ("synthetic", "AI")):
            zz = z[(z.label == cls) & z.sample_id.isin(ids)].zone
            rows_.append((f"{label}: {name}", (zz == 0).mean(), (zz == -1).mean(), (zz > 0).mean()))
    yl = [r[0] for r in rows_][::-1]
    acc, inc, acu = (np.array([r[i] for r in rows_][::-1]) * 100 for i in (1, 2, 3))
    ax.barh(yl, acc, color="#6AA84F", label="accepted as genuine")
    ax.barh(yl, inc, left=acc, color=GREY, label="inconclusive")
    ax.barh(yl, acu, left=acc + inc, color=ORANGE, label="flagged / accused")
    for i, (a, b, c) in enumerate(zip(acc, inc, acu)):
        for start, val in ((0, a), (a, b), (a + b, c)):
            if val >= 6:
                ax.text(start + val / 2, i, f"{val:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of clips (Phase 3's test set; Phase 3 at 5 frames, Phase 5 at 10 frames)", fontsize=9)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig2 = figure(fig)

    return ("<h2 class=break>4. Phase 3 vs. Phase 5</h2><h3>4.1 Same test clips</h3>" + t1 +
            "<p class=note>Phase 3's 22,867 test clips scored with both phases (pooled over folds). Phase 5 was trained on "
            "the broader Phase 4 data (ACID attacks, more generators) and still matches Phase 3 here.</p>"
            + "<h3>4.2 Unseen datasets</h3>" + t2 + "<h3>4.3 At a glance</h3>" + fig1 + fig2
            + "<p class=note>Three-zone outcomes on Phase 3's test clips. Inconclusive clips are mostly re-encoded or "
              "platform-compressed videos, where the camera evidence is weakest.</p>")


def summary_box(z3, z5):
    def s(path, model, part="test"):
        return scores(rows(path, model, part))
    a3, a5 = s(P3 / "models" / "realistic_folds_all_w5.parquet", "stack"), s(P5 / "models" / "fusion_c4m4_realistic_folds_w10.parquet", "stack_rel")
    u3 = s(P3 / "models" / "realistic_fixed_all_w5.parquet", "stack", "crossdataset")
    u5 = s(P5 / "models" / "fusion_c4m4_realistic_fixed_w10.parquet", "stack_rel", "crossdataset")
    pce = s(P5 / "models" / "fusion_c4m4_realistic_folds_w10.parquet", "baseline")
    pce3 = s(P3 / "models" / "realistic_folds_all_w5.parquet", "baseline")

    def data(path):
        v = pd.read_parquet(path, columns=["label", "source", "variant"])
        syn = v.variant[v.label == "synthetic"].astype(str).str.split("+").str[0]
        acid = ((v.label == "manipulated") & (v.source == "ACID-attack")).any()
        return f"{len(v):,} clips, {'VISION + ACID' if acid else 'VISION'} attacks, {syn.nunique()} AI generators"
    acc5 = 100 * (z5[10].zone[z5[10].label == "organic"] > 0).mean()
    pas5 = 100 * (z5[10].zone[z5[10].label == "manipulated"] == 0).mean()
    body = [["Unseen cameras: macro F1 / AUC", f"{a3['F1']:.3f} / {a3['AUC']:.3f}", f"{a5['F1']:.3f} / {a5['AUC']:.3f}"],
            ["Unseen datasets: macro F1 / AUC", f"{u3['F1']:.3f} / {u3['AUC']:.3f}", f"{u5['F1']:.3f} / {u5['AUC']:.3f}"],
            ["Conventional PCE rule: AUC", f"{pce3['AUC']:.3f}", f"{pce['AUC']:.3f}"],
            ["Data (incl. unseen test sets)", data(P3 / "features" / "videos.parquet"), data(P4 / "features" / "videos.parquet")],
            ["3-zone: genuine accused / attacks accepted", "–", f"{acc5:.1f}% / {pas5:.1f}%"]]
    return table(["", "Phase 3 (5 frames)", "Phase 5 final (10 frames)"], body,
                 note="Unseen-camera scores are on each phase's own test set (Phase 5's is larger and harder); section 4 "
                      "compares both on the same clips.")


def main():
    warnings.filterwarnings("ignore")
    plt.rcParams.update({"font.family": "Noto Sans", "axes.edgecolor": "#8A99A8", "xtick.color": INK, "ytick.color": INK})
    print("data ...", flush=True)
    data = data_section()
    print("phase 3 ...", flush=True)
    s3, z3 = phase3_section()
    print("phase 5 ...", flush=True)
    s5, z5, f5 = phase5_section()
    print("comparison ...", flush=True)
    comp = comparison_section(z3, z5, f5)
    head = (f"<h1>HVPF results: Phase 3 and Phase 5</h1><div class=sub>Robust Video Source Provenance: Distinguishing "
            f"Synthetic vs. Organic Sensor Noise to Counter Anti-Forensic Injection Attacks<br>Talha Aslam (24i-8067), "
            f"MS(DS), FAST-NUCES Islamabad &middot; {date.today():%d %B %Y}</div>"
            "<h3>Summary</h3>" + summary_box(z3, z5))
    html = f"<!doctype html><html><head><meta charset=utf-8><title>HVPF results</title><style>{CSS}</style></head><body>" \
           f"{head}{data}{s3}{s5}{comp}</body></html>"
    OUT.with_suffix(".html").write_text(html)
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT.with_suffix('.pdf')}", OUT.with_suffix(".html").as_uri()],
                   check=True, capture_output=True)
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
