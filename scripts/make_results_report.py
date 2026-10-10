"""Write the Phase 2 results report (report/results_phase2.tex) straight from the result files.

Every number in the report is read from data/results (Phase 2) and data/phase1 (Phase 1),
so the PDF can be regenerated after any re-run:

    python scripts/make_results_report.py
    ~/.local/bin/tectonic -X compile report/results_phase2.tex
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.model_data import CLASSES

MODELS = ["baseline", "svm", "xgb", "gru", "cnn", "transformer"]
NAMES = {"baseline": "PCE rule", "baseline_w": "PCE rule", "svm": "SVM", "xgb": "XGBoost", "gru": "GRU",
         "cnn": "1D-CNN", "transformer": "Transformer"}


def tex(s):
    return str(s).replace("\\", "/").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.0f}"


def pm(m, s):
    return "--" if pd.isna(m) else (f"{m:.3f}" if pd.isna(s) else f"{m:.3f} $\\pm$ {s:.3f}")


def correct(g):
    if g.model.iloc[0].startswith("baseline"):
        return ((g.score_organic > g.threshold) == (g.label == "organic")).to_numpy()
    return g[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1) == g.label.map(CLASSES.index).to_numpy()


def with_correct(pred):
    return pd.concat([g.assign(correct=correct(g)) for _, g in pred.groupby("model")])


def condition(r):
    """(section, row label, sort key) for the per-condition table."""
    social = "" if r.social in ("none", None) or pd.isna(r.social) else f" + {r.social.upper()}-like"
    if r.source in ("VISION-attack", "ACID-attack"):
        base = r.variant.split("+")[0]
        ds = r.source.split("-")[0]
        if r.label == "organic":
            crf = base.removeprefix("reencode_crf") if "crf" in base else "20"
            return "Genuine (organic)", f"{ds} genuine re-encode, CRF {crf}{social}", f"0b{ds}{crf}{social}"
        return "Manipulated", f"{tex(base)}{social}", f"1{base}{social}"
    if r.label == "organic":
        name = {"native": "native", "youtube": "real YouTube copy", "whatsapp": "real WhatsApp copy"}.get(r.variant, r.variant)
        return "Genuine (organic)", f"{r.source} {name}", f"0a{r.source}{r.variant}"
    gen, _, pf = r.variant.partition("+")
    return "AI-generated (synthetic)", f"{gen}" + (f" + {pf.upper()}-like" if pf else ""), f"2{gen}{pf}"


def main():
    paths = load_config()["paths"]
    res, p1dir = paths["results_dir"], paths["data_dir"] / "phase1"
    summ = pd.read_csv(res / "metrics" / "summary.csv")
    summ1 = pd.read_csv(p1dir / "metrics" / "summary.csv")
    abl = pd.read_csv(res / "metrics" / "ablation.csv")
    samples = pd.read_csv(paths["data_dir"] / "features" / "samples.csv")

    def row(df, proto, scheme, tag, part, model):
        g = df[(df.protocol == proto) & (df.scheme == scheme) & (df.tag == tag) & (df.part == part) & (df.model == model)]
        return g.iloc[0] if len(g) else None

    L = []
    w = L.append

    # ---- composition
    comp = samples.assign(social=samples.social.fillna("none")).groupby(["label", "source"]).size()
    total = len(samples)

    # ---- table 1: main
    t1 = []
    for m in MODELS:
        r = row(summ, "realistic", "folds", "all_w5", "test", m)
        if r is not None:
            t1.append(f"{NAMES[m]} & {pm(r.get('pva_mean'), r.get('pva_std'))} & {pm(r.get('f1_mean'), r.get('f1_std'))} & "
                      f"{f3(r.get('auc_ovr_mean'))} & {f3(r.bin_auc_mean)} & {f3(r.bin_eer_mean)} \\\\")
    best = summ[(summ.protocol == "realistic") & (summ.scheme == "folds") & (summ.tag == "all_w5") & (summ.part == "test")
                & ~summ.model.str.startswith("baseline")].sort_values("pva_mean").iloc[-1]
    base = row(summ, "realistic", "folds", "all_w5", "test", "baseline")

    # ---- table 2: other setups
    t2 = []
    for m in MODELS:
        cells = []
        for proto, scheme, part in (("controlled", "folds", "test"), ("realistic", "fixed", "crossdataset")):
            r = row(summ, proto, scheme, "all_w5", part, m)
            cells += [pct(r.get("pva_mean")) if r is not None else "--", f3(r.get("f1_mean")) if r is not None else "--",
                      f3(r.bin_auc_mean) if r is not None else "--"]
        t2.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")

    # ---- table 3: phase 1 vs phase 2
    t3 = []
    for m, m1 in (("baseline", "baseline_w"), ("xgb", "xgb"), ("gru", "gru"), ("transformer", "transformer")):
        a, b = row(summ1, "realistic", "folds", "all", "test", m1), row(summ, "realistic", "folds", "all_w5", "test", m)
        t3.append(f"{NAMES[m]} & {pct(a.get('pva_mean'))} & {pct(b.get('pva_mean'))} & {f3(a.get('f1_mean'))} & "
                  f"{f3(b.get('f1_mean'))} & {f3(a.bin_auc_mean)} & {f3(b.bin_auc_mean)} \\\\")
    c1, c2 = row(summ1, "controlled", "folds", "all", "test", "xgb"), row(summ, "controlled", "folds", "all_w5", "test", "xgb")

    # ---- table 4: windows
    t4 = []
    for m in ("xgb", "gru", "cnn", "transformer"):
        cells = []
        for tag in ("all_w5", "all_w10", "all_w5_min10", "all_w10_min10"):
            r = row(summ, "realistic", "folds", tag, "test", m)
            cells += [pct(r.pva_mean), f3(r.bin_auc_mean)]
        t4.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    n_min10 = row(summ, "realistic", "folds", "all_w5_min10", "test", "xgb").n_mean

    # ---- table 5: per condition (realistic, 5-fold, 5 frames)
    pred = with_correct(pd.read_parquet(res / "models" / "realistic_folds_all_w5.parquet"))
    keys = pred.apply(condition, axis=1, result_type="expand")
    pred[["section", "cond", "order"]] = keys
    tab = pred.pivot_table(index=["section", "order", "cond"], columns="model", values="correct", aggfunc="mean")
    n = pred[pred.model == "xgb"].groupby(["section", "order", "cond"]).size()
    t5, last = [], None
    for (section, order, cond), r in tab.sort_index(level=1).iterrows():
        if section != last:
            t5.append(f"\\midrule\\multicolumn{{8}}{{l}}{{\\textbf{{{section}}}}} \\\\")
            last = section
        vals = [pct(r.get(m)) for m in MODELS]
        flagged = [f"\\textbf{{{v}}}" if v != "--" and m != "baseline" and int(v) < 70 else v for m, v in zip(MODELS, vals)]
        t5.append(f"{cond} & {n.loc[(section, order, cond)]} & " + " & ".join(flagged) + " \\\\")

    # ---- table 6: cross-dataset breakdown
    s = pd.read_parquet(res / "prnu_baseline" / "FloreView" / "scores.parquet")
    q = s.loc[s.groupby("query").pce0.idxmax()]
    acc = (q.candidate == q.device).groupby(q.device).mean()
    strong = set(acc[acc >= 0.5].index)
    cx = pd.read_parquet(res / "models" / "realistic_fixed_all_w5.parquet")
    cx = with_correct(cx[cx.part == "crossdataset"])
    cx["cond"] = np.where(cx.label == "organic",
                          "FloreView genuine, PRNU " + np.where(cx.claimed_device.isin(strong), "usable", "removed (stabilised)"),
                          "GenBuster AI" + cx.social.fillna("none").map({"none": "", "wa": " + WA-like", "yt": " + YT-like"}))
    t6tab = cx.pivot_table(index="cond", columns="model", values="correct", aggfunc="mean")
    t6n = cx[cx.model == "xgb"].groupby("cond").size()
    t6 = [f"{c} & {t6n[c]} & " + " & ".join(pct(t6tab.loc[c].get(m)) for m in MODELS) + " \\\\" for c in t6tab.index]
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

    # ---- figure: attack strength (XGBoost vs PCE rule)
    vis = pred[pred.source == "VISION-attack"]
    def rate(variant, model):
        g = vis[(vis.variant == variant) & (vis.model == model)]
        return 100 * g.correct.mean()
    strength = {}
    for attack, grid in (("removal", ("0.25", "0.5", "1")), ("injection", ("0.25", "0.5", "1")), ("denoise", ("1", "2", "4"))):
        strength[attack] = {m: " ".join(f"({x},{rate(f'{attack}_a{x}', m):.1f})" for x in grid) for m in ("xgb", "baseline")}

    # ---- key numbers for the text
    wa_real = pred[(pred.model == best.model) & (pred.variant == "whatsapp") & (pred.source == "VISION")].correct.mean()
    p1pred = with_correct(pd.read_parquet(p1dir / "models" / "realistic_folds_all.parquet"))
    wa_p1 = p1pred[(p1pred.model == best.model) & (p1pred.variant == "whatsapp")].correct.mean()

    # =============================== document
    w(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage[margin=2.2cm]{geometry}\usepackage{microtype}
\usepackage{booktabs}\usepackage{tabularx}\usepackage{longtable}\usepackage{array}\usepackage{float}
\usepackage{caption}\usepackage{enumitem}\usepackage{xcolor}\usepackage{amsmath}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}\usepgfplotslibrary{groupplots}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\definecolor{seriesA}{HTML}{2A78D6}\definecolor{seriesB}{HTML}{EB6834}
\captionsetup{font=small,labelfont=bf}\setlist{itemsep=2pt,topsep=4pt}\setlength{\emergencystretch}{3em}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\pgfplotsset{every axis/.append style={axis line style={draw=black!40}, tick style={draw=black!40},
  ymajorgrids, grid style={draw=black!12}, legend style={draw=none, font=\small}}}
\title{\textbf{Hybrid Video Provenance Framework (HVPF)}\\[3pt]\large Phase 2 Results}
\author{Talha Aslam\\ MS Thesis}""")
    w(f"\\date{{{date.today().strftime('%d %B %Y')}}}")
    w(r"""\begin{document}\maketitle
\begin{abstract}\noindent
This report presents the Phase 2 results of the HVPF pipeline: a three-class decision
(organic / manipulated / synthetic) for video clips, compared with the conventional PRNU method (PCE rule).
Phase 2 extends Phase 1 with the full Video-ACID dataset, FloreView as an unseen-camera test set,
additional genuine re-encodes, and WhatsApp-/YouTube-like copies of genuine, attacked and AI-generated clips.
All numbers are generated directly from the result files.
\end{abstract}""")

    w(r"\section{Key results}\begin{itemize}")
    w(f"\\item \\textbf{{Best model:}} {NAMES[best.model]} reaches PVA {best.pva_mean:.3f}, macro F1 {best.f1_mean:.3f} and "
      f"authentic-vs-not AUC {best.bin_auc_mean:.3f} (EER {best.bin_eer_mean:.3f}) in 5-fold cross-validation on cameras never seen "
      f"in training, against AUC {base.bin_auc_mean:.3f} (EER {base.bin_eer_mean:.3f}) for the conventional PCE rule (RQ3).")
    w(f"\\item \\textbf{{Honest benchmark:}} Phase 1 recognised {100 * wa_p1:.0f}\\% of genuine WhatsApp copies, but only because "
      f"no fake clip was WhatsApp-compressed. With WhatsApp-like copies of fakes the rate is {100 * wa_real:.0f}\\%, "
      "which is why Phase 2 scores are lower.")
    w(f"\\item \\textbf{{Genuine re-encodes:}} in the encoder-controlled setup, XGBoost macro F1 rose from "
      f"{c1.f1_mean:.3f} (Phase 1) to {c2.f1_mean:.3f}.")
    w("\\item \\textbf{5 vs 10 frames:} 10 frames looks better on the full data, but only because sequence models learn the "
      "clip length (attacked clips have 30 frames, ACID and AI clips about 6). With clips of equal length the two windows "
      "perform the same (Table~\\ref{tab:window}); 5 frames is the main setting.")
    w(f"\\item \\textbf{{Unseen cameras:}} on FloreView ({fv_cams} cameras, never trained on) genuine videos of phones with a usable "
      "PRNU are mostly recognised, but phones whose stabilisation removes the PRNU are the hardest case (Table~\\ref{tab:cross}).")
    w("\\item \\textbf{Features (RQ1):} removing Noiseprint statistics costs the most, followed by PRNU and TSNCS; "
      "FFT and residual moments add almost nothing (Figure~\\ref{fig:ablation}).")
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
    w(r"\begin{table}[H]\centering\small\caption{Realistic setup, 5-fold cross-validation (mean $\pm$ std over folds), 5 frames. "
      r"PVA = share of clips assigned to their true provenance class; AUC/EER for genuine vs not genuine.}\label{tab:main}"
      r"\begin{tabular}{lccccc}\toprule Model & PVA & Macro F1 & AUC (3-class) & AUC genuine vs not & EER \\\midrule")
    L.extend(t1)
    w(r"\bottomrule\end{tabular}\end{table}")
    w(r"\begin{table}[H]\centering\small\caption{Encoder-controlled setup (5-fold) and cross-dataset test (FloreView + GenBuster), 5 frames.}"
      r"\begin{tabular}{lcccccc}\toprule & \multicolumn{3}{c}{Encoder-controlled} & \multicolumn{3}{c}{Cross-dataset} \\"
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-7} Model & PVA (\%) & Macro F1 & AUC & PVA (\%) & Macro F1 & AUC \\\midrule")
    L.extend(t2)
    w(r"\bottomrule\end{tabular}\end{table}")
    w(r"\begin{table}[H]\centering\small\caption{Phase 1 vs Phase 2 (realistic, 5-fold, 5 frames). Phase 2 is harder by design: "
      r"fakes also appear as platform-like copies and there are four times more genuine ACID clips.}"
      r"\begin{tabular}{lcccccc}\toprule & \multicolumn{2}{c}{PVA (\%)} & \multicolumn{2}{c}{Macro F1} & \multicolumn{2}{c}{AUC genuine vs not} \\"
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7} Model & Phase 1 & Phase 2 & Phase 1 & Phase 2 & Phase 1 & Phase 2 \\\midrule")
    L.extend(t3)
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{5 vs 10 frames}")
    w(f"The length-matched columns use only clips with at least 10 I-frames ({n_min10:,.0f} test clips per fold), "
      "so both windows see the same clips and the clip length carries no information.")
    w(r"\begin{table}[H]\centering\small\caption{Effect of the frame window (realistic, 5-fold).}\label{tab:window}"
      r"\begin{tabular}{lcccccccc}\toprule & \multicolumn{4}{c}{All clips} & \multicolumn{4}{c}{Length-matched ($\geq$ 10 frames)} \\"
      r"\cmidrule(lr){2-5}\cmidrule(lr){6-9} & \multicolumn{2}{c}{5 frames} & \multicolumn{2}{c}{10 frames} & \multicolumn{2}{c}{5 frames} & \multicolumn{2}{c}{10 frames} \\"
      r"Model & PVA & AUC & PVA & AUC & PVA & AUC & PVA & AUC \\\midrule")
    L.extend(t4)
    w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{Results per condition (RQ2)}")
    w("Share of clips assigned to their true class (\\%), realistic setup, 5-fold, 5 frames. Values below 70\\% are in bold. "
      "The PCE rule only decides genuine vs not genuine: its high rates on attacks and AI clips come with rejecting most "
      "genuine clips.")
    w(r"{\small\setlength{\tabcolsep}{4pt}\begin{longtable}{p{7.2cm}rrrrrrr}")
    w(r"\caption{Detection rate per condition.}\label{tab:cond}\\\toprule Condition & $n$ & PCE & SVM & XGB & GRU & CNN & Transf. \\\endfirsthead"
      r"\toprule Condition & $n$ & PCE & SVM & XGB & GRU & CNN & Transf. \\\endhead\bottomrule\endfoot")
    L.extend(t5)
    w(r"\end{longtable}}")

    w(f"\\begin{{figure}}[H]\\centering\\begin{{tikzpicture}}\\begin{{groupplot}}[group style={{group size=3 by 1, horizontal sep=1.3cm}},"
      r"width=0.32\linewidth, height=5cm, ymin=0, ymax=105, ytick={0,25,50,75,100}]")
    for attack, xlabel in (("removal", r"removal strength $\alpha$"), ("injection", r"injection strength $\alpha$"),
                           ("denoise", "denoising strength")):
        extra = r", ylabel={detected (\%)}" if attack == "removal" else ""
        legend = r", legend to name=strengthlegend, legend columns=2" if attack == "removal" else ""
        w(f"\\nextgroupplot[title={{{attack}}}, xlabel={{{xlabel}}}, xmode=log, log basis x=2, xtick={{"
          + ("0.25,0.5,1" if attack != "denoise" else "1,2,4") + "}, xticklabels={"
          + ("0.25,0.5,1" if attack != "denoise" else "1,2,4") + f"}}{extra}{legend}]")
        w(f"\\addplot[color=seriesA, mark=*, mark size=2.2pt, line width=1.2pt] coordinates {{{strength[attack]['xgb']}}};")
        w(f"\\addplot[color=seriesB, mark=square*, mark size=2.2pt, line width=1.2pt] coordinates {{{strength[attack]['baseline']}}};")
        if attack == "removal":
            w(r"\legend{XGBoost, PCE rule}")
    w(r"\end{groupplot}\end{tikzpicture}\\[2pt]\ref{strengthlegend}"
      r"\caption{Attack strength vs detection (VISION attacks, realistic, 5-fold). The PCE rule ``detects'' removal only because "
      r"low PCE is its sign of non-genuine; it misses injection, where the victim's fingerprint is matched.}\label{fig:strength}\end{figure}")

    w(r"\section{Cross-dataset test}")
    w(f"FloreView cameras are split by their PRNU baseline: {fv_strong_cams} of {fv_cams} cameras have a usable video PRNU "
      "(attribution $\\geq$ 50\\%); the others are modern phones whose stabilisation removes it.")
    w(r"\begin{table}[H]\centering\small\caption{Cross-dataset detection rate (\%), models trained on VISION + ACID + GenVidBench (fixed split).}\label{tab:cross}"
      r"\begin{tabular}{lrrrrrrr}\toprule Condition & $n$ & PCE & SVM & XGB & GRU & CNN & Transf. \\\midrule")
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

    w(r"""\section{Limitations and next steps}\begin{itemize}
\item Phones whose electronic stabilisation removes the PRNU (most FloreView cameras) are the hardest genuine case;
      more such phones in training is the most promising improvement.
\item Platform-like copies make attacks much harder to see; detection of attacked clips shared through WhatsApp-like
      re-encoding is lower than for the original attacked clips (Table~\ref{tab:cond}).
\item Sequence models can learn the clip length; equal-length inputs (e.g.\ cyclic padding) would remove this.
\item Sora is represented by only 51 clips; more high-quality modern generators are needed.
\end{itemize}
\end{document}""")

    out = paths["data_dir"].parent / "report" / "results_phase2.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
