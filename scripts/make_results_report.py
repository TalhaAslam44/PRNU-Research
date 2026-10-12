"""Write the Phase 3 results report (report/results_phase3.tex) straight from the result files.

Every number in the report is read from data/results (Phase 3) and data/phase2, so the PDF
can be regenerated after any re-run:

    python scripts/make_results_report.py
    ~/.local/bin/tectonic -X compile report/results_phase3.tex

(report/results_phase2.pdf is the frozen Phase 2 report; its data is kept in data/phase2.)
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.model_data import CLASSES

MODELS = ["baseline", "svm", "xgb", "gru", "cnn", "transformer"]            # Step 8 (ablation runs too)
LEARNED = ["svm", "xgb", "gru", "cnn", "transformer", "vit"]
COMBOS = ["ens_ml", "ens_seq", "ens_all", "ens_top3", "stack", "xgb+cal", "ens_all+cal", "ens_top3+cal"]
ENS = "ens_all"           # plain soft voting, nothing fitted on validation data
STACK = "stack"           # best combination (logistic regression on validation outputs)
TAB = ["baseline", "xgb", "gru", "vit", ENS, STACK]          # columns of the wide tables
TAB_HEAD = "PCE & XGB & GRU & ViT & Ens. & Stack"
NAMES = {"baseline": "PCE rule", "baseline_w": "PCE rule", "svm": "SVM", "xgb": "XGBoost", "gru": "GRU",
         "cnn": "1D-CNN", "transformer": "Transformer", "vit": "ViT (residual)", "ens_ml": "Ens.\\ SVM+XGB",
         "ens_seq": "Ens.\\ GRU+CNN+Transf.", "ens_all": "Ens.\\ all", "ens_top3": "Ens.\\ top-3 (val)",
         "stack": "Stacking (LR)"}


def name(m):
    return NAMES[m.removesuffix("+cal")] + (" + cal." if m.endswith("+cal") else "")


def tex(s):
    return str(s).replace("\\", "/").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def f3(x):
    return "--" if x is None or pd.isna(x) else f"{x:.3f}"


def pct(x):
    return "--" if x is None or pd.isna(x) else f"{100 * x:.0f}"


def pm(m, s):
    return "--" if pd.isna(m) else (f"{m:.3f}" if pd.isna(s) else f"{m:.3f} $\\pm$ {s:.3f}")


def get(r, key):
    return None if r is None else r.get(key)


def correct(g):
    if g.model.iloc[0].startswith("baseline"):
        return ((g.score_organic > g.threshold) == (g.label == "organic")).to_numpy()
    return g[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1) == g.label.map(CLASSES.index).to_numpy()


def accepted(g):
    """Clip accepted as genuine: PCE rule above threshold / model predicts organic."""
    if g.model.iloc[0].startswith("baseline"):
        return pd.Series((g.score_organic > g.threshold).to_numpy(), index=g.index)
    return pd.Series(g[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1) == 0, index=g.index)


def with_correct(pred):
    return pd.concat([g.assign(correct=correct(g), accepted=accepted(g)) for _, g in pred.groupby("model")])


def injected(variant):
    return variant.str.contains(r"\+inj", regex=True, na=False)


def condition(r):
    """(section, row label, sort key) for the per-condition table."""
    social = "" if r.social in ("none", None) or pd.isna(r.social) else f" + {r.social.upper()}-like"
    if r.source in ("VISION-attack", "ACID-attack"):
        base = r.variant.split("+")[0]
        ds = r.source.split("-")[0]
        if r.label == "organic":
            crf = base.removeprefix("reencode_crf") if "crf" in base else "20"
            return "Genuine (organic)", f"{ds} re-encode, CRF {crf}{social}", f"0b{ds}{crf}{social}"
        return "Manipulated", f"{tex(base)}{social}", f"1{base}{social}"
    if r.label == "organic":
        label = {"native": "native", "youtube": "real YouTube copy", "whatsapp": "real WhatsApp copy"}.get(r.variant, r.variant)
        return "Genuine (organic)", f"{r.source} {label}", f"0a{r.source}{r.variant}"
    gen, _, pf = r.variant.partition("+")
    if pf.startswith("inj"):              # Khan-style: a real camera's PRNU injected into the AI clip
        return ("AI-generated + injected camera PRNU", f"{gen}, injected $\\alpha$={pf.removeprefix('inj')}",
                f"3{gen}{pf}")
    return "AI-generated (synthetic)", f"{gen}" + (f" + {pf.upper()}-like" if pf else ""), f"2{gen}{pf}"


def main():
    paths = load_config()["paths"]
    res, p2dir = paths["results_dir"], paths["data_dir"] / "phase2"
    summ = pd.read_csv(res / "metrics" / "summary.csv")
    summ2 = pd.read_csv(p2dir / "metrics" / "summary.csv")
    abl = pd.read_csv(res / "metrics" / "ablation.csv")
    samples = pd.read_csv(paths["data_dir"] / "features" / "samples.csv")

    def row(df, proto, scheme, tag, part, model):
        g = df[(df.protocol == proto) & (df.scheme == scheme) & (df.tag == tag) & (df.part == part) & (df.model == model)]
        return g.iloc[0] if len(g) else None

    def main_row(model, df=summ):
        return row(df, "realistic", "folds", "all_w5", "test", model)

    L = []
    w = L.append

    # ---- composition
    kind = np.where(injected(samples.variant), "AI + injected camera PRNU", samples.label)
    comp = samples.assign(kind=kind).groupby(["kind", "source"]).size()
    total, n_inj = len(samples), int(injected(samples.variant).sum())

    # ---- table 1: main
    def t1_line(m):
        r = main_row(m)
        if r is None:
            return None
        return (f"{name(m)} & {pm(r.get('pva_mean'), r.get('pva_std'))} & {pm(r.get('f1_mean'), r.get('f1_std'))} & "
                f"{f3(r.get('auc_ovr_mean'))} & {f3(r.bin_auc_mean)} & {f3(r.bin_eer_mean)} \\\\")
    t1 = [x for x in map(t1_line, ["baseline", *LEARNED]) if x]
    t1c = [x for x in map(t1_line, COMBOS) if x]
    main_tab = summ[(summ.protocol == "realistic") & (summ.scheme == "folds") & (summ.tag == "all_w5") & (summ.part == "test")]
    best = main_tab[~main_tab.model.str.startswith("baseline")].sort_values("f1_mean").iloc[-1]
    best_single = main_tab[main_tab.model.isin(LEARNED)].sort_values("f1_mean").iloc[-1]
    base, vit, ens = main_row("baseline"), main_row("vit"), main_row(ENS)
    best2 = summ2[(summ2.protocol == "realistic") & (summ2.scheme == "folds") & (summ2.tag == "all_w5")
                  & (summ2.part == "test") & ~summ2.model.str.startswith("baseline")].sort_values("f1_mean").iloc[-1]

    # ---- table 2: other setups
    t2 = []
    for m in ["baseline", *LEARNED, *COMBOS]:
        cells = []
        for proto, scheme, part in (("controlled", "folds", "test"), ("realistic", "fixed", "crossdataset")):
            r = row(summ, proto, scheme, "all_w5", part, m)
            cells += [pct(get(r, "pva_mean")), f3(get(r, "f1_mean")), f3(get(r, "bin_auc_mean"))]
        if set(cells) != {"--"}:
            t2.append(("\\midrule " if m == COMBOS[0] else "") + f"{name(m)} & " + " & ".join(cells) + " \\\\")

    # ---- table 3: phase 2 vs phase 3
    t3 = []
    for m in ("baseline", "xgb", "gru", "transformer"):
        a, b = main_row(m, summ2), main_row(m)
        t3.append(f"{name(m)} & {pct(get(a, 'pva_mean'))} & {pct(get(b, 'pva_mean'))} & {f3(get(a, 'f1_mean'))} & "
                  f"{f3(get(b, 'f1_mean'))} & {f3(get(a, 'bin_auc_mean'))} & {f3(get(b, 'bin_auc_mean'))} \\\\")
    t3.append(f"\\midrule Best model & {pct(best2.pva_mean)} & {pct(best.pva_mean)} & {f3(best2.f1_mean)} & "
              f"{f3(best.f1_mean)} & {f3(best2.bin_auc_mean)} & {f3(best.bin_auc_mean)} \\\\")

    # ---- table: calibration in-distribution vs unseen sources (fixed split)
    tcal = []
    for m in ("xgb", "xgb+cal", ENS, f"{ENS}+cal", "ens_top3", "ens_top3+cal", "stack"):
        a, b = row(summ, "realistic", "fixed", "all_w5", "test", m), row(summ, "realistic", "fixed", "all_w5", "crossdataset", m)
        tcal.append(f"{name(m)} & {f3(get(a, 'f1_mean'))} & {pct(get(a, 'pva_mean'))} & {pct(get(b, 'pva_mean'))} & "
                    f"{f3(get(b, 'bin_auc_mean'))} \\\\")
    cal_in = get(row(summ, "realistic", "fixed", "all_w5", "test", "xgb+cal"), "f1_mean") - \
        get(row(summ, "realistic", "fixed", "all_w5", "test", "xgb"), "f1_mean")
    cal_x = get(row(summ, "realistic", "fixed", "all_w5", "crossdataset", "xgb+cal"), "pva_mean") - \
        get(row(summ, "realistic", "fixed", "all_w5", "crossdataset", "xgb"), "pva_mean")

    # ---- table 4: windows
    t4 = []
    for m in ("xgb", "gru", "cnn", "transformer"):
        cells = []
        for tag in ("all_w5", "all_w10", "all_w5_min10", "all_w10_min10"):
            r = row(summ, "realistic", "folds", tag, "test", m)
            cells += [pct(get(r, "pva_mean")), f3(get(r, "bin_auc_mean"))]
        t4.append(f"{name(m)} & " + " & ".join(cells) + " \\\\")
    n_min10 = row(summ, "realistic", "folds", "all_w5_min10", "test", "xgb").n_mean

    # ---- table 5: per condition (realistic, 5-fold, 5 frames; test parts only)
    pred = pd.read_parquet(res / "models" / "realistic_folds_all_w5.parquet")
    pred = with_correct(pred[pred.part == "test"])
    keys = pred.apply(condition, axis=1, result_type="expand")
    pred[["section", "cond", "order"]] = keys
    tab = pred.pivot_table(index=["section", "order", "cond"], columns="model", values="correct", aggfunc="mean")
    n = pred[pred.model == "xgb"].groupby(["section", "order", "cond"]).size()
    t5, last = [], None
    for (section, order, cond), r in tab.sort_index(level=1).iterrows():
        if section != last:
            t5.append(f"\\midrule\\multicolumn{{8}}{{l}}{{\\textbf{{{section}}}}} \\\\")
            last = section
        vals = [pct(r.get(m)) for m in TAB]
        flagged = [f"\\textbf{{{v}}}" if v != "--" and m != "baseline" and int(v) < 70 else v for m, v in zip(TAB, vals)]
        t5.append(f"{cond} & {n.loc[(section, order, cond)]} & " + " & ".join(flagged) + " \\\\")

    # ---- table 6: cross-dataset breakdown
    s = pd.read_parquet(res / "prnu_baseline" / "FloreView" / "scores.parquet")
    q = s.loc[s.groupby("query").pce0.idxmax()]
    acc = (q.candidate == q.device).groupby(q.device).mean()
    strong = set(acc[acc >= 0.5].index)
    cx = pd.read_parquet(res / "models" / "realistic_fixed_all_w5.parquet")
    cx = with_correct(cx[cx.part == "crossdataset"])
    inj_tag = cx.variant.str.extract(r"\+inj([\d.]+)")[0]
    cx["cond"] = np.where(cx.label == "organic",
                          "FloreView genuine, " + np.where(cx.claimed_device.isin(strong), "usable PRNU", "stabilised (no PRNU)"),
                          np.where(inj_tag.notna(), "GenBuster AI + injected PRNU, $\\alpha$=" + inj_tag.fillna(""),
                                   "GenBuster AI" + cx.social.fillna("none").map({"none": "", "wa": " + WA-like", "yt": " + YT-like"})))
    t6tab = cx.pivot_table(index="cond", columns="model", values="correct", aggfunc="mean")
    t6n = cx[cx.model == "xgb"].groupby("cond").size()
    t6 = [f"{c} & {t6n[c]} & " + " & ".join(pct(t6tab.loc[c].get(m)) for m in TAB) + " \\\\" for c in t6tab.index]
    fv_strong_cams, fv_cams = len(strong), len(acc)

    # ---- table 7 + figure: ablation
    a = abl[(abl.part == "test") & (abl.window == "w5")]
    groups = [("noiseprint", "Noiseprint"), ("prnu", "PRNU"), ("tsncs", "TSNCS"), ("resid", "Residual"), ("fft", "FFT")]
    t7 = []
    for proto in ("realistic", "controlled"):
        for gkey, gname in groups:
            vals = [a[(a.protocol == proto) & (a.dropped == gkey) & (a.model == m)].f1_drop for m in MODELS[1:]]
            t7.append(f"{proto} & {gname} & " + " & ".join(f3(v.iloc[0]) if len(v) else "--" for v in vals) + " \\\\")
    abl_r = a[a.protocol == "realistic"]
    fig_abl = {m: " ".join(f"({gname},{abl_r[(abl_r.dropped == g) & (abl_r.model == m)].f1_drop.iloc[0]:.4f})"
                           for g, gname in groups) for m in ("xgb", "gru")}

    # ---- figure: attack strength (XGBoost / ensemble vs PCE rule)
    vis = pred[pred.source == "VISION-attack"]

    def rate(variant, model):
        g = vis[(vis.variant == variant) & (vis.model == model)]
        return 100 * g.correct.mean()
    strength = {}
    for attack, grid in (("removal", ("0.25", "0.5", "1")), ("injection", ("0.25", "0.5", "1")), ("denoise", ("1", "2", "4"))):
        strength[attack] = {m: " ".join(f"({x},{rate(f'{attack}_a{x}', m):.1f})" for x in grid) for m in ("xgb", STACK, "baseline")}

    # ---- Khan-style: camera PRNU injected into AI clips
    ai_in = pred[(pred.label == "synthetic")]
    ai_x = cx[cx.label == "synthetic"]
    khan_cols = []
    for title, d in (("GenVidBench", ai_in), ("GenBuster", ai_x)):
        d = d.assign(inj=d.variant.str.extract(r"\+inj([\d.]+)")[0].fillna("none"))
        d = d[(d.inj != "none") | (d.social.fillna("none") == "none")]          # plain AI clips: original copies only
        t = d.pivot_table(index="model", columns="inj", values="accepted", aggfunc="mean")
        t.columns = pd.MultiIndex.from_product([[title], t.columns])
        khan_cols.append(t)
    khan = pd.concat(khan_cols, axis=1)
    khan_order = [(t, c) for t in ("GenVidBench", "GenBuster") for c in ("none", "0.5", "1")]
    tk = [f"{name(m)} & " + " & ".join(pct(khan.loc[m].get(c)) for c in khan_order) + " \\\\"
          for m in ["baseline", *LEARNED, ENS, "stack"] if m in khan.index]
    inj_ai = pred[(pred.label == "synthetic") & injected(pred.variant)]
    called = inj_ai[inj_ai.model == STACK][[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)
    called = np.bincount(called, minlength=3) / max(len(called), 1)
    khan_pce = khan.loc["baseline", ("GenVidBench", "1")]
    khan_stack = khan.loc[STACK, ("GenVidBench", "1")]
    khan_pce_x, khan_stack_x = khan.loc["baseline", ("GenBuster", "1")], khan.loc[STACK, ("GenBuster", "1")]

    # ---- extra numbers for the text: unseen phones, unseen generators + WhatsApp-like, 10 frames
    fv_ok = cx[cx.label == "organic"].groupby("model").correct.mean()
    gbwa = cx[(cx.label == "synthetic") & (cx.social == "wa") & (cx.model == STACK)]
    gbwa = np.bincount(gbwa[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1), minlength=3) / max(len(gbwa), 1)
    vre = pred[(pred.source == "VISION-attack") & (pred.label == "organic") & (pred.model == STACK)]
    vre_all, vre_wa = vre.correct.mean(), vre[vre.social == "wa"].correct.mean()
    w10 = summ[(summ.protocol == "realistic") & (summ.scheme == "folds") & (summ.tag == "all_w10") & (summ.part == "test")
               & summ.model.isin(COMBOS)].sort_values("f1_mean").iloc[-1]
    win = {m: (row(summ, "realistic", "folds", "all_w10_min10", "test", m).pva_mean
               - row(summ, "realistic", "folds", "all_w5_min10", "test", m).pva_mean) for m in ("xgb", "gru", "cnn", "transformer")}
    cal_x_vit = get(row(summ, "realistic", "fixed", "all_w5", "crossdataset", "vit+cal"), "pva_mean") - \
        get(row(summ, "realistic", "fixed", "all_w5", "crossdataset", "vit"), "pva_mean")

    # ---- comparison with the base papers
    camid = {}
    for ds in ("VISION", "ACID", "FloreView"):
        sc = pd.read_parquet(res / "prnu_baseline" / ds / "scores.parquet")
        sc = sc[sc.version == "native"]
        qq = sc.loc[sc.groupby("query").pce0.idxmax()]
        ok = qq.candidate == qq.device
        tr, fa = sc[sc.is_match], sc[~sc.is_match]
        fpr_c, tpr_c, _ = roc_curve(sc.is_match, sc.pce0)
        i = np.nanargmin(np.abs(fpr_c - (1 - tpr_c)))
        camid[ds] = {"dev": sc.candidate.nunique(), "acc": ok.mean(), "fnr": (tr.pce0 <= 60).mean(),
                     "fpr": (fa.pce0 > 60).mean(), "eer": (fpr_c[i] + 1 - tpr_c[i]) / 2}
        if ds == "VISION":
            nst = ~qq.stabilized
            camid["VISION_nostab"] = {"dev": qq[nst].device.nunique(), "acc": ok[nst].mean()}
    inj = pred[pred.variant.str.startswith("injection")]
    comp_rate = inj.pivot_table(index="model", columns="variant", values="accepted", aggfunc="mean")
    comp_rate["all"] = inj.groupby("model").accepted.mean()
    inj_cols = ["injection_a0.25", "injection_a0.5", "injection_a1", "injection_a0.5+wa", "injection_a0.5+yt", "all"]
    lowfpr = {}
    for m, g in pred.groupby("model"):
        y = (g.label != "organic").astype(int).to_numpy()
        score = -g.score_organic.to_numpy() if m == "baseline" else 1 - g.p_organic.to_numpy()
        fpr_c, tpr_c, _ = roc_curve(y, score)
        lowfpr[m] = [np.interp(f, fpr_c, tpr_c) for f in (0.01, 0.05, 0.10)]

    # =============================== document
    w(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage[margin=2.2cm]{geometry}\usepackage{microtype}
\usepackage{booktabs}\usepackage{tabularx}\usepackage{longtable}\usepackage{array}\usepackage{float}
\usepackage{caption}\usepackage{enumitem}\usepackage{xcolor}\usepackage{amsmath}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}\usepgfplotslibrary{groupplots}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\definecolor{seriesA}{HTML}{2A78D6}\definecolor{seriesB}{HTML}{EB6834}\definecolor{seriesC}{HTML}{3BA272}
\captionsetup{font=small,labelfont=bf}\setlist{itemsep=2pt,topsep=4pt}\setlength{\emergencystretch}{3em}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\pgfplotsset{every axis/.append style={axis line style={draw=black!40}, tick style={draw=black!40},
  ymajorgrids, grid style={draw=black!12}, legend style={draw=none, font=\small}}}
\title{\textbf{Hybrid Video Provenance Framework (HVPF)}\\[3pt]\large Phase 3 Results: Vision Transformer, Ensembles and PRNU Injection into AI Video}
\author{Talha Aslam\\ MS Thesis}""")
    w(f"\\date{{{date.today().strftime('%d %B %Y')}}}")
    w(r"""\begin{document}\maketitle
\begin{abstract}\noindent
Phase 3 adds a Vision Transformer on I-frame noise residuals, calibrated models, model ensembles and
equal-length inputs for the sequence models. It also adds a harder attack, taken from Khan et al.\ (2025):
a real camera's PRNU injected into AI-generated video, so that the clip passes conventional PRNU verification.
As in Phase 2, the task is a three-class decision for each video clip (organic / manipulated / synthetic),
compared with the conventional PRNU method (PCE rule).
All numbers are generated directly from the result files.
\end{abstract}""")

    w(r"\section{Key results}\begin{itemize}")
    w(f"\\item \\textbf{{Best result:}} {name(best.model)} reaches macro F1 {best.f1_mean:.3f}, PVA {best.pva_mean:.3f} and "
      f"authentic-vs-not AUC {best.bin_auc_mean:.3f} (EER {best.bin_eer_mean:.3f}) in 5-fold cross-validation on camera models "
      f"never seen in training (Phase 2 best: {name(best2.model)}, F1 {best2.f1_mean:.3f}). The conventional PCE rule reaches "
      f"AUC {base.bin_auc_mean:.3f} (RQ3); it dropped from Phase 2 because injected AI clips pass it.")
    w(f"\\item \\textbf{{Vision Transformer:}} the best single model (F1 {vit.f1_mean:.3f}) and by far the most robust on "
      f"unseen sources: it recognises {pct(fv_ok['vit'])}\\% of FloreView genuine videos (XGBoost {pct(fv_ok['xgb'])}\\%), "
      "including stabilised phones without a usable PRNU (Table~\\ref{tab:cross}). Plain averaging of all models "
      f"(F1 {ens.f1_mean:.3f}) does not beat the ViT alone; stacking, which learns how much to trust each model, does "
      f"({main_row(STACK).f1_mean:.3f}).")
    w(f"\\item \\textbf{{Calibration is not worth it:}} per-class threshold offsets change macro F1 on the test cameras by "
      f"{cal_in:+.3f} (XGBoost) but cost PVA on unseen sources (XGBoost {100 * cal_x:+.0f}, ViT {100 * cal_x_vit:+.0f} points). "
      "Stacking, also fitted on validation data, is the best combination on both (Table~\\ref{tab:cal}).")
    w(f"\\item \\textbf{{PRNU injected into AI video (Khan-style):}} {n_inj:,} AI clips carry a real camera's PRNU. "
      f"At full strength the PCE rule accepts {pct(khan_pce)}\\% of them as genuine footage of that camera "
      f"({pct(khan_pce_x)}\\% for unseen generators); stacking accepts {pct(khan_stack)}\\% ({pct(khan_stack_x)}\\%) "
      "(Table~\\ref{tab:khanai}).")
    w(f"\\item \\textbf{{Weakest spots:}} (1) AI clips from unseen generators that were also WhatsApp-like compressed: "
      f"stacking calls {pct(gbwa[2])}\\% synthetic, {pct(gbwa[1])}\\% manipulated and {pct(gbwa[0])}\\% organic. "
      f"(2) Genuine VISION videos that went through our own re-encoding: stacking recognises {pct(vre_all)}\\% of them "
      f"({pct(vre_wa)}\\% of the WhatsApp-like ones); almost all others are called manipulated, because the attacked clips went "
      "through the same encoder (Table~\\ref{tab:cond}).")
    w(f"\\item \\textbf{{10 frames:}} with equal-length inputs, 10 frames is slightly better than 5 even on length-matched "
      f"clips ({min(win.values()) * 100:+.0f} to {max(win.values()) * 100:+.0f} PVA points); combinations of the Step 8 models at "
      f"10 frames reach F1 {w10.f1_mean:.3f} without the ViT, so a 10-frame ViT and stack is the next likely gain.")
    w("\\item \\textbf{Features (RQ1):} removing Noiseprint statistics costs the most (about 0.05 macro F1), followed by "
      "TSNCS and PRNU (0.01--0.03); FFT and residual moments add almost nothing (Figure~\\ref{fig:ablation}).")
    w(r"\end{itemize}")

    w(r"\section{What changed in Phase 3}\begin{itemize}")
    w(r"\item \textbf{Vision Transformer (Step 8b).} Pretrained ViT-S/16 (21.7M parameters, ImageNet-21k) fine-tuned on "
      r"three SRM high-pass residuals of each I-frame (random 224$\times$224 crops and flips, AdamW, mixed precision, "
      r"class-balanced loss, early stopping on validation macro F1). A clip's prediction averages its first five I-frames "
      r"$\times$ four crops. This is the same input size as PDC-ViT.")
    w(r"\item \textbf{Calibration and ensembles (Step 8c).} Validation predictions of every model are now kept. "
      r"\emph{+cal}: three class offsets on the log-probabilities tuned for validation macro F1. "
      r"\emph{Ensembles}: mean class probability of SVM+XGB, of the three sequence models, of all learned models, "
      r"and of the three best models on validation data; \emph{stacking}: logistic regression on all models' "
      r"validation outputs. Validation data always comes from other camera models than the test data.")
    w(r"\item \textbf{Equal-length inputs.} Clips shorter than the window repeat their own frames, so sequence models "
      r"cannot use the number of I-frames as a shortcut (a Phase 2 limitation).")
    w(r"\item \textbf{Wider tuning.} XGBoost: depth 4/6/8 and a min-child-weight variant, up to 1500 trees; SVM: $C$ up to 30.")
    w(r"\item \textbf{Harder synthetic class.} Khan-style AI clips with an injected camera PRNU ($\alpha$ = 0.5 and 1, "
      r"same strength units as the manipulation attacks) are added to training and test. The attacker estimates the "
      r"fingerprint from the camera's own natural videos, never from the analyst's reference videos. Cameras without "
      r"a usable video PRNU are not impersonated.")
    w(r"\end{itemize}")

    w(r"\section{Data and protocol}")
    w(f"The evaluation uses {total:,} clips. Model inputs come from the first five I-frames of each clip. "
      "Results are averaged over five grouped cross-validation folds: no camera model is in both training and test. "
      "\\emph{Realistic} uses all clips; \\emph{encoder-controlled} uses only clips encoded by our libx264 pipeline "
      "(genuine re-encodes, attacks, AI clips), so encoder traces cannot separate the classes. "
      "\\emph{Cross-dataset} is FloreView (genuine) plus GenBuster (unseen AI generators), tested with the fixed-split models.")
    w(r"\begin{table}[H]\centering\small\caption{Clips per class and source.}\begin{tabular}{llr}\toprule Class & Source & Clips \\\midrule")
    for (label, source), cnt in comp.items():
        w(f"{label} & {tex(source)} & {cnt:,} \\\\")
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{Main results}")
    w(r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}\caption{Realistic setup, 5-fold cross-validation (mean $\pm$ std over folds), 5 frames. "
      r"PVA = share of clips assigned to their true provenance class; AUC/EER for genuine vs not genuine. "
      r"Lower block: calibrated models and combinations (Step 8c).}\label{tab:main}"
      r"\begin{tabular}{lccccc}\toprule Model & PVA & Macro F1 & AUC (3-class) & AUC genuine vs not & EER \\\midrule")
    L.extend(t1)
    w(r"\midrule")
    L.extend(t1c)
    w(r"\bottomrule\end{tabular}\end{table}")
    w(r"\begin{table}[H]\centering\small\caption{Encoder-controlled setup (5-fold) and cross-dataset test (FloreView + GenBuster), 5 frames.}"
      r"\begin{tabular}{lcccccc}\toprule & \multicolumn{3}{c}{Encoder-controlled} & \multicolumn{3}{c}{Cross-dataset} \\"
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-7} Model & PVA (\%) & Macro F1 & AUC & PVA (\%) & Macro F1 & AUC \\\midrule")
    L.extend(t2)
    w(r"\bottomrule\end{tabular}\end{table}")
    w(r"\begin{table}[H]\centering\small\caption{Phase 2 vs Phase 3 (realistic, 5-fold, 5 frames). Phase 3 also contains the "
      r"injected AI clips, so its test set is larger and harder.}"
      r"\begin{tabular}{lcccccc}\toprule & \multicolumn{2}{c}{PVA (\%)} & \multicolumn{2}{c}{Macro F1} & \multicolumn{2}{c}{AUC genuine vs not} \\"
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7} Model & Phase 2 & Phase 3 & Phase 2 & Phase 3 & Phase 2 & Phase 3 \\\midrule")
    L.extend(t3)
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{Calibration and ensembles on unseen sources}")
    w("Calibration moves the decision thresholds to where they work best on the validation cameras. That barely helps on "
      "test cameras from the same datasets, and FloreView phones and unseen AI generators look different, so the moved "
      "thresholds hurt there. Stacking is also fitted on validation data, but it learns how much weight each model gets "
      "(the ViT, the most robust model on unseen sources, is one of them) and is the best combination on both. Fixed split, 5 frames:")
    w(r"\begin{table}[H]\centering\small\caption{Effect of calibration and combination: same datasets vs unseen sources.}\label{tab:cal}"
      r"\begin{tabular}{lcccc}\toprule & \multicolumn{2}{c}{Test (VISION, ACID, GenVidBench)} & \multicolumn{2}{c}{Cross-dataset} \\"
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5} Model & Macro F1 & PVA (\%) & PVA (\%) & AUC \\\midrule")
    L.extend(tcal)
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{5 vs 10 frames}")
    w(f"The length-matched columns use only clips with at least 10 I-frames ({n_min10:,.0f} test clips per fold), so the "
      "clip length carries no information; since Phase 3 the sequence models get equal-length inputs on all clips as well. "
      "Unlike Phase 2, 10 frames is now slightly better than 5 even when lengths are matched: more frames give a better "
      "noise estimate. The larger gain on all clips comes from the long clips (VISION and attacks), whose 10-frame features "
      "are more reliable. The ViT and the main results use 5 frames, because ACID and most AI clips have only about six I-frames.")
    w(r"\begin{table}[H]\centering\small\caption{Effect of the frame window (realistic, 5-fold).}\label{tab:window}"
      r"\begin{tabular}{lcccccccc}\toprule & \multicolumn{4}{c}{All clips} & \multicolumn{4}{c}{Length-matched ($\geq$ 10 frames)} \\"
      r"\cmidrule(lr){2-5}\cmidrule(lr){6-9} & \multicolumn{2}{c}{5 frames} & \multicolumn{2}{c}{10 frames} & \multicolumn{2}{c}{5 frames} & \multicolumn{2}{c}{10 frames} \\"
      r"Model & PVA & AUC & PVA & AUC & PVA & AUC & PVA & AUC \\\midrule")
    L.extend(t4)
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{Results per condition (RQ2)}")
    w("Share of clips assigned to their true class (\\%), realistic setup, 5-fold, 5 frames. Values below 70\\% are in bold. "
      f"Ens.\\ = mean probability of all learned models; Stack = {name(STACK)}. The PCE rule only decides genuine vs not genuine: its "
      "high rates on attacks and AI clips come with rejecting most genuine clips.")
    w(r"{\small\setlength{\tabcolsep}{4pt}\begin{longtable}{p{7.0cm}rrrrrrr}")
    w(f"\\caption{{Detection rate per condition.}}\\label{{tab:cond}}\\\\\\toprule Condition & $n$ & {TAB_HEAD} \\\\\\endfirsthead"
      f"\\toprule Condition & $n$ & {TAB_HEAD} \\\\\\endhead\\bottomrule\\endfoot")
    L.extend(t5)
    w(r"\end{longtable}}")

    w(f"\\begin{{figure}}[H]\\centering\\begin{{tikzpicture}}\\begin{{groupplot}}[group style={{group size=3 by 1, horizontal sep=1.3cm}},"
      r"width=0.32\linewidth, height=5cm, ymin=0, ymax=105, ytick={0,25,50,75,100}]")
    for attack, xlabel in (("removal", r"removal strength $\alpha$"), ("injection", r"injection strength $\alpha$"),
                           ("denoise", "denoising strength")):
        extra = r", ylabel={detected (\%)}" if attack == "removal" else ""
        legend = r", legend to name=strengthlegend, legend columns=3" if attack == "removal" else ""
        ticks = "0.25,0.5,1" if attack != "denoise" else "1,2,4"
        w(f"\\nextgroupplot[title={{{attack}}}, xlabel={{{xlabel}}}, xmode=log, log basis x=2, xtick={{{ticks}}}, "
          f"xticklabels={{{ticks}}}{extra}{legend}]")
        w(f"\\addplot[color=seriesA, mark=*, mark size=2.2pt, line width=1.2pt] coordinates {{{strength[attack]['xgb']}}};")
        w(f"\\addplot[color=seriesC, mark=triangle*, mark size=2.4pt, line width=1.2pt] coordinates {{{strength[attack][STACK]}}};")
        w(f"\\addplot[color=seriesB, mark=square*, mark size=2.2pt, line width=1.2pt] coordinates {{{strength[attack]['baseline']}}};")
        if attack == "removal":
            w(f"\\legend{{XGBoost, {name(STACK)}, PCE rule}}")
    w(r"\end{groupplot}\end{tikzpicture}\\[2pt]\ref{strengthlegend}"
      r"\caption{Attack strength vs detection (VISION attacks, realistic, 5-fold). The PCE rule ``detects'' removal only because "
      r"low PCE is its sign of non-genuine; it misses injection, where the victim's fingerprint is matched.}\label{fig:strength}\end{figure}")

    w(r"\section{PRNU injected into AI-generated video}")
    w("Khan et al.\\ (2025) transfer a camera's PRNU into AI-generated \\emph{images} so that they pass source verification. "
      "We apply the same attack to video: the attacker estimates the fingerprint of a real camera from that camera's own "
      "videos and adds it to every frame of an AI clip, which is then encoded like all our clips. "
      "Table~\\ref{tab:khanai} gives the share of clips accepted as genuine (PCE rule: PCE$_0$ with the claimed camera "
      "above 60; models: predicted \\emph{organic}). GenVidBench clips are from the 5-fold test folds; GenBuster "
      "generators were never seen in training.")
    w(r"\begin{table}[H]\centering\small\caption{AI clips accepted as genuine camera footage (\%, lower is better for the defender).}\label{tab:khanai}"
      r"\begin{tabular}{lrrrrrr}\toprule & \multicolumn{3}{c}{GenVidBench (seen generators)} & \multicolumn{3}{c}{GenBuster (unseen generators)} \\"
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-7} Detector & plain & $\alpha$=0.5 & $\alpha$=1 & plain & $\alpha$=0.5 & $\alpha$=1 \\\midrule")
    w(r"Khan et al.~2025 \cite{khan2025}, AI images & \multicolumn{6}{c}{85.5 (images, average)} \\\midrule")
    L.extend(tk)
    w(r"\bottomrule\end{tabular}\end{table}")
    w(f"Injecting a camera's PRNU does not hide the AI origin: of the injected GenVidBench clips, {name(STACK)} calls "
      f"{100 * called[2]:.0f}\\% synthetic, {100 * called[1]:.0f}\\% manipulated and {100 * called[0]:.0f}\\% organic. "
      "The injection makes the clip match the camera for the PCE rule, but it does not remove the other traces the models "
      "use (Noiseprint statistics, residual and temporal patterns).")

    w(r"\section{Cross-dataset test}")
    w(f"FloreView cameras are split by their PRNU baseline: {fv_strong_cams} of {fv_cams} cameras have a usable video PRNU "
      "(attribution $\\geq$ 50\\%); the others are modern phones whose stabilisation removes it.")
    w(r"\begin{table}[H]\centering\small\caption{Cross-dataset detection rate (\%), models trained on VISION + ACID + GenVidBench (fixed split).}\label{tab:cross}"
      f"\\begin{{tabular}}{{p{{6.4cm}}rrrrrrr}}\\toprule Condition & $n$ & {TAB_HEAD} \\\\\\midrule")
    L.extend(t6)
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{Feature ablation (RQ1)}")
    w(r"\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[ybar, bar width=12pt, width=0.8\linewidth, height=5.4cm,"
      r"symbolic x coords={Noiseprint,PRNU,TSNCS,Residual,FFT}, xtick=data, ymin=-0.01, ylabel={macro-F1 drop}, enlarge x limits=0.15,"
      r"scaled y ticks=false, yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},"
      r"legend style={at={(0.98,0.95)}, anchor=north east}, legend image code/.code={\draw[#1, draw=none] (0cm,-0.1cm) rectangle (0.3cm,0.15cm);}]")
    w(f"\\addplot[fill=seriesA, draw=white] coordinates {{{fig_abl['xgb']}}};")
    w(f"\\addplot[fill=seriesB, draw=white] coordinates {{{fig_abl['gru']}}};")
    w(r"\legend{XGBoost, GRU}\end{axis}\end{tikzpicture}\caption{Drop in macro F1 when one feature group is removed "
      r"(realistic, 5-fold, 5 frames). Larger = more important.}\label{fig:ablation}\end{figure}")
    w(r"\begin{table}[H]\centering\small\caption{Macro-F1 drop per feature group and model (5-fold, 5 frames).}"
      r"\begin{tabular}{llccccc}\toprule Setup & Removed group & SVM & XGB & GRU & CNN & Transf. \\\midrule")
    L.extend(t7)
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{Comparison with the base papers}")
    w("The base papers solve related but different tasks, so each is compared on its own terms. "
      "Literature values are taken from the papers as reported; protocols differ (noted per table).")
    w(r"\subsection{Camera identification: PDC-ViT (Elharrouss et al.~2025)}")
    w("PDC-ViT is a learned camera-identification model (80/20 train-test split, every camera seen in training). "
      "The closest component of HVPF is the training-free PRNU baseline (Step 4): fingerprints from flat videos, "
      "tested on natural videos, closed-set over all cameras of the dataset; verification at PCE$_0>60$. "
      "Our ViT shares PDC-ViT's input size but answers a different question (provenance class, not camera identity), "
      "on camera models it never saw in training.")
    v, vn, a_, fv = camid["VISION"], camid["VISION_nostab"], camid["ACID"], camid["FloreView"]
    w(r"\begin{table}[H]\centering\small\caption{Camera identification. Accuracy = closed-set attribution; FNR/FPR = verification.}"
      r"\setlength{\tabcolsep}{4pt}\begin{tabular}{llrrrr}\toprule Method & Dataset (cameras) & Accuracy (\%) & FNR (\%) & FPR (\%) & EER (\%) \\\midrule")
    w(r"PDC-ViT \cite{pdcvit} & VISION (35) & 94.30 & -- & -- & -- \\")
    w(f"HVPF PRNU & VISION (35) & {100 * v['acc']:.1f} & {100 * v['fnr']:.1f} & {100 * v['fpr']:.2f} & {100 * v['eer']:.1f} \\\\")
    w(f"HVPF PRNU & VISION ({vn['dev']} non-stabilised) & {100 * vn['acc']:.1f} & -- & -- & -- \\\\")
    w(r"\midrule PRNU \cite{pdcvit} & Video-ACID + WA (26) & -- & 41.9 & 3.6 & -- \\")
    w(r"WA-PRNU \cite{pdcvit} & Video-ACID + WA (26) & -- & 28.06 & 2.8 & -- \\")
    w(r"PDC-ViT \cite{pdcvit} & Video-ACID + WA (26) & 74.77 & 23.4 & 0.92 & -- \\")
    w(f"HVPF PRNU & Video-ACID ({a_['dev']}) & {100 * a_['acc']:.1f} & {100 * a_['fnr']:.1f} & {100 * a_['fpr']:.2f} & {100 * a_['eer']:.1f} \\\\")
    w(f"\\midrule HVPF PRNU & FloreView ({fv['dev']}) & {100 * fv['acc']:.1f} & {100 * fv['fnr']:.1f} & {100 * fv['fpr']:.2f} & {100 * fv['eer']:.1f} \\\\")
    w(r"\bottomrule\end{tabular}\end{table}")
    w("PRNU matching needs no training and is perfect on non-stabilised VISION cameras, above PDC-ViT; electronic "
      "stabilisation is what lowers the all-camera figures. Our verification threshold is strict (FPR 0\\%), so FNR is "
      "higher than PDC-ViT's (FPR 0.92\\%); the EER gives a threshold-free view. PDC-ViT was not tested on manipulated "
      "or synthetic video, and no base paper uses FloreView.")

    w(r"\subsection{PRNU-transfer attack: Khan et al.~2025}")
    w("Khan et al.\\ report that 85.5\\% of PRNU-transferred images pass source verification. Table~\\ref{tab:khanai} "
      "covers their AI-image case on video; Table~\\ref{tab:khancam} covers transfer between real cameras (our "
      "\\emph{injection} attack). Compromise rate = share of injected clips accepted as genuine (realistic, 5-fold, 5 frames).")
    w(r"\begin{table}[H]\centering\small\caption{Compromise rate of PRNU injection between cameras (\%, lower is better for the defender).}\label{tab:khancam}"
      r"\begin{tabular}{lrrrrrr}\toprule & \multicolumn{3}{c}{Injection strength $\alpha$} & \multicolumn{2}{c}{$\alpha$=0.5, re-shared} & \\"
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-6} Detector & 0.25 & 0.5 & 1 & WA-like & YT-like & All \\\midrule")
    w(r"Khan et al.~2025 \cite{khan2025}, images & \multicolumn{5}{c}{--} & 85.5 \\\midrule")
    for m in ["baseline", *LEARNED, ENS, "stack"]:
        if m in comp_rate.index:
            w(f"{name(m)} & " + " & ".join(pct(comp_rate.loc[m].get(c)) for c in inj_cols) + " \\\\")
    w(r"\bottomrule\end{tabular}\end{table}")
    learned_rates = comp_rate.loc[comp_rate.index.isin([*LEARNED, ENS, "stack"]), "injection_a1"]
    w(f"A full-strength injection fools the PCE rule in {100 * comp_rate.loc['baseline', 'injection_a1']:.0f}\\% of the clips, "
      f"in line with Khan et al.; the best learned detector lets through only {100 * learned_rates.min():.1f}\\%. "
      "Weak injections ($\\alpha$=0.25) rarely fool PCE but are also harder for the models to spot.")

    w(r"\subsection{Low false-positive operation: Bao et al.~2024}")
    w("Bao et al.\\ detect frame deletion (a different manipulation) from inter-frame noise consistency, the idea behind "
      "TSNCS, and stress performance at low false-positive rates; their results are given as ROC curves only. "
      "Table~\\ref{tab:lowfpr} reports HVPF at the same kind of operating points: share of non-genuine clips detected "
      "while at most 1, 5 or 10\\% of genuine clips are flagged.")
    w(r"\begin{table}[H]\centering\small\caption{Detection rate of non-genuine clips at fixed false-positive rates (realistic, 5-fold, 5 frames).}\label{tab:lowfpr}"
      r"\begin{tabular}{lrrr}\toprule Detector & FPR 1\% & FPR 5\% & FPR 10\% \\\midrule")
    for m in ["baseline", *LEARNED, ENS, "stack"]:
        if m in lowfpr:
            w(f"{name(m)} & " + " & ".join(f"{100 * x:.1f}" for x in lowfpr[m]) + " \\\\")
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\subsection{What the base papers do not cover}\begin{itemize}")
    w(r"\item PRNU injection and removal attacks on \emph{video}, including platform-like re-sharing (Khan et al.\ test images only).")
    w(r"\item PRNU transfer into AI-generated \emph{video}, tested against unseen generators.")
    w(r"\item Camera identification under attack (PDC-ViT reports clean data only).")
    w(r"\item FloreView as an unseen-camera test set.")
    w(r"\item One model that separates genuine, manipulated and AI-generated video.")
    w(r"\end{itemize}")

    w(r"""\section{Limitations and next steps}\begin{itemize}
\item Phones whose electronic stabilisation removes the PRNU (most FloreView cameras): the feature-based models still
      miss about half of them; the ViT recognises most. More such phones in training would help every model.
\item Platform-like copies make attacks and unseen-generator AI clips much harder to see, WhatsApp-like re-encoding most
      of all (Tables~\ref{tab:cond} and \ref{tab:cross}). All platform copies of fakes are simulated; only genuine VISION
      videos have real WhatsApp/YouTube copies.
\item The manipulated class comes from 400 videos of the 35 VISION cameras, and all attacks are our own implementation;
      ACID cameras appear only as genuine re-encodes.
\item The ViT sees one I-frame at a time and was run at 5 frames; 10-frame and multi-frame versions are the natural next step.
\item Sora is represented by only 51 clips; more high-quality modern generators are needed.
\end{itemize}
\begin{thebibliography}{9}
\bibitem{pdcvit} O.~Elharrouss et al., ``PDC-ViT: source camera identification using pixel difference convolution and
  vision transformer,'' \emph{Neural Computing and Applications}, 2025, doi:10.1007/s00521-025-11004-z.
\bibitem{khan2025} Khan, Hasan, Islam, ``Spoofing camera source attribution via PRNU transfer attacks,''
  \emph{ISC 2025}, Springer LNCS 16186, pp.~279--299, 2025.
\bibitem{bao2024} Bao et al., ``Anti-forensics video forgery detection via noise transfer matrix,''
  \emph{Sensors} 24(16):5341, 2024, doi:10.3390/s24165341.
\end{thebibliography}
\end{document}""")

    out = paths["data_dir"].parent / "report" / "results_phase3.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
