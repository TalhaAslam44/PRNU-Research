"""Comparison of every phase run so far -> report/phase_comparison.tex (+ PDF) and
data/results/metrics/phase_comparison_all.csv.

Phases: 1, 2, 3 (data/phase1..3), HVPF-2 (80/20 branch, ~/hvpf-2/data/results), 4 (data/phase4; 5 and 10 frames),
plus the Phase 4 training experiments (diagnosis, dataset-balanced) and camera identification.
Three views: each phase on its own test set; all phases on the clips they share (the untrained PCE rule
must score the same, which checks that the clips match); and per-model / per-class detail.

    python scripts/make_phase_comparison.py
    ~/.local/bin/tectonic -X compile report/phase_comparison.tex
"""
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_phases import scores
from hvpf.config import load_config
from hvpf.model_data import CLASSES

ROOT = Path(__file__).resolve().parents[1]
PHASES = [  # name, results folder, tag of the main (5-frame) files, PCE-rule model name
    ("Phase 1", ROOT / "data/phase1", "all", "baseline_w"),
    ("Phase 2", ROOT / "data/phase2", "all_w5", "baseline"),
    ("Phase 3", ROOT / "data/phase3", "all_w5", "baseline"),
    ("HVPF-2 (80/20)", Path.home() / "hvpf-2/data/results", "all_w5", "baseline"),
    ("Phase 4", ROOT / "data/phase4", "all_w5", "baseline"),
    ("Phase 4, 10 frames", ROOT / "data/phase4", "all_w10", "baseline"),
]
WHAT = {
    "Phase 1": "first full pipeline; VISION + 40 ACID videos per camera; 4 generators",
    "Phase 2": "full Video-ACID, FloreView, platform-like copies of fakes, genuine re-encodes (honest benchmark)",
    "Phase 3": "PRNU injected into AI clips, ViT, ensembles + stacking, equal-length inputs",
    "HVPF-2 (80/20)": "Phase 3 data; train on 80\\% (tune on validation, then refit)",
    "Phase 4": "ACID attacks, 4 more generators, more genuine copies, safe rule",
    "Phase 4, 10 frames": "Phase 4 with 10 I-frames per clip",
}
LEARNED = ["svm", "xgb", "gru", "cnn", "transformer", "vit", "ens_all", "ens_top3", "stack"]
NAMES = {"svm": "SVM", "xgb": "XGBoost", "gru": "GRU", "cnn": "1D-CNN", "transformer": "Transformer", "vit": "ViT",
         "ens_all": "Ensemble (all)", "ens_top3": "Ensemble (top-3)", "stack": "Stacking", "baseline": "PCE rule"}


def f3(x):
    return "--" if x is None or pd.isna(x) else f"{x:.3f}"


def pct(x):
    return "--" if x is None or pd.isna(x) else f"{100 * x:.0f}\\%"


def summary(folder, tag):
    s = pd.read_csv(folder / "metrics/summary.csv")
    return s[(s.tag == tag)]


def pick(s, protocol, scheme, part, model):
    g = s[(s.protocol == protocol) & (s.scheme == scheme) & (s.part == part) & (s.model == model)]
    return g.iloc[0] if len(g) else None


def best_model(s, protocol="realistic", scheme="folds", part="test"):
    g = s[(s.protocol == protocol) & (s.scheme == scheme) & (s.part == part) & s.model.isin(LEARNED)]
    return g.sort_values("f1_mean").iloc[-1] if len(g) else None


def main():
    warnings.filterwarnings("ignore")
    paths = load_config()["paths"]
    rows, preds = [], {}
    for name, folder, tag, base in PHASES:
        s = summary(folder, tag)
        b = best_model(s)
        pce = pick(s, "realistic", "folds", "test", base)
        bc = best_model(s, "controlled")
        bx = best_model(s, "realistic", "fixed", "crossdataset")
        f = folder / f"models/realistic_folds_{tag}.parquet"
        p = pd.read_parquet(f)
        p = p[p.part == "test"].replace({"model": {"baseline_w": "baseline"}})
        preds[name] = p
        stamp = datetime.fromtimestamp((folder / "metrics/summary.csv").stat().st_mtime).strftime("%d %b")
        rows.append({"phase": name, "date": stamp, "test clips (5-fold)": p[p.model == "baseline"].sample_id.nunique(),
                     "best": b.model, "F1": b.f1_mean, "F1 std": b.f1_std, "PVA": b.pva_mean, "AUC": b.bin_auc_mean,
                     "EER": b.bin_eer_mean, "PCE AUC": pce.bin_auc_mean if pce is not None else None,
                     "controlled best F1": bc.f1_mean if bc is not None else None,
                     "unseen-sources best F1": bx.f1_mean if bx is not None else None,
                     "unseen-sources best AUC": bx.bin_auc_mean if bx is not None else None,
                     **{f"F1 {m}": (pick(s, "realistic", "folds", "test", m).f1_mean
                                    if pick(s, "realistic", "folds", "test", m) is not None else None) for m in LEARNED}})
    own = pd.DataFrame(rows)

    # ---- same clips
    ids = {k: set(p[p.model == "baseline"].sample_id) for k, p in preds.items()}
    sets = [("A", "clips tested in every phase", ["Phase 1", "Phase 2", "Phase 3", "HVPF-2 (80/20)", "Phase 4", "Phase 4, 10 frames"]),
            ("B", "clips tested in Phases 2--4", ["Phase 2", "Phase 3", "HVPF-2 (80/20)", "Phase 4", "Phase 4, 10 frames"]),
            ("C", "clips tested in Phases 3--4 (the Phase 3 test set)", ["Phase 3", "HVPF-2 (80/20)", "Phase 4", "Phase 4, 10 frames"])]
    same = []
    for key, label, phases in sets:
        shared = set.intersection(*(ids[ph] for ph in phases))
        if key == "A":   # also without the real WhatsApp / YouTube copies (the Phase 1 shortcut)
            p1 = preds["Phase 1"]
            plat = set(p1[(p1.model == "baseline") & p1.variant.isin(["whatsapp", "youtube"])].sample_id)
        for ph in phases:
            p = preds[ph]
            avail = [m for m in ["baseline", "xgb", "gru", "vit", "stack"] if (p.model == m).any()]
            for m in avail:
                g = p[(p.model == m) & p.sample_id.isin(shared)]
                r = {"set": key, "set label": label, "clips": len(shared), "phase": ph, "model": m, **scores(g)}
                if key == "A" and m != "baseline":
                    r["F1 without real platform copies"] = scores(g[~g.sample_id.isin(plat)])["f1"]
                same.append(r)
    same = pd.DataFrame(same)
    SOURCES = {"ACID / organic": "ACID genuine (originals)", "ACID-attack / organic": "ACID genuine re-encodes",
               "VISION / organic": "VISION genuine (native, real WhatsApp/YouTube)", "VISION-attack / organic": "VISION genuine re-encodes",
               "VISION-attack / manipulated": "VISION attacks", "GenVidBench / synthetic": "AI clips (GenVidBench)"}
    by_src = {}
    shared_c = set.intersection(*(ids[ph] for ph in sets[2][2]))
    for ph in sets[2][2]:
        g = preds[ph]
        g = g[(g.model == "stack") & g.sample_id.isin(shared_c)]
        ok = g[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1) == g.label.map(CLASSES.index).to_numpy()
        by_src[ph] = pd.Series(ok, index=g.index).groupby((g.source + " / " + g.label).map(SOURCES)).mean()
        n_src = (g.source + " / " + g.label).map(SOURCES).value_counts()
    by_src = pd.DataFrame(by_src).reindex(list(SOURCES.values()))
    out_csv = paths["results_dir"] / "metrics/phase_comparison_all.csv"
    pd.concat([own.assign(view="own benchmark"), same.assign(view="same clips")]).to_csv(out_csv, index=False)

    # ---- Phase 4 training experiments and camera identification
    ct_path = paths["results_dir"] / "metrics/compare_training.csv"
    ct = pd.read_csv(ct_path) if ct_path.exists() else pd.DataFrame()
    cam4 = pd.read_csv(ROOT / "data/phase4/camera_id/summary.csv")
    cam2 = pd.read_csv(Path.home() / "hvpf-2/data/results/camera_id/summary.csv")

    # =============================== LaTeX
    L = []
    w = L.append
    w(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage[margin=2cm]{geometry}\usepackage{microtype}
\usepackage{booktabs}\usepackage{tabularx}\usepackage{array}\usepackage{float}\usepackage{caption}\usepackage{enumitem}
\usepackage{xcolor}\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\captionsetup{font=small,labelfont=bf}\setlist{itemsep=2pt,topsep=3pt}\setlength{\emergencystretch}{3em}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\title{\textbf{HVPF: Comparison of All Phases}}
\author{Talha Aslam\\ \small Supervisor: Dr.\ Hasan Mujtaba \quad Co-supervisor: Dr.\ Muhammad Ishtiaq}""")
    w(f"\\date{{{date.today().strftime('%d %B %Y')}}}")
    w(r"\begin{document}\maketitle")

    p3, p4, p4w = (own.set_index("phase").loc[k] for k in ("Phase 3", "Phase 4", "Phase 4, 10 frames"))
    p1, p2 = own.set_index("phase").loc["Phase 1"], own.set_index("phase").loc["Phase 2"]
    shA = same[(same.set == "A")]
    shC = same[(same.set == "C")]

    def sv(df, phase, model, col="f1"):
        g = df[(df.phase == phase) & (df.model == model)]
        return g.iloc[0][col] if len(g) and col in g else None
    w(r"\section*{Summary}\begin{itemize}")
    w(f"\\item \\textbf{{Best result on its own test set:}} Phase 3 (stacking), macro F1 {p3['F1']:.3f}, AUC {p3['AUC']:.3f}; "
      f"the conventional PCE rule reaches AUC {p3['PCE AUC']:.3f}. Phase 1's F1 {p1['F1']:.3f} came from an easier test set "
      "with a shortcut (every WhatsApp clip was genuine).")
    pce5 = [sv(shA, ph, "baseline", "auc") for ph in ("Phase 1", "Phase 2", "Phase 3", "HVPF-2 (80/20)", "Phase 4")]
    w(f"\\item \\textbf{{Same clips:}} on the {int(shA.clips.iloc[0]):,} clips every phase tested, the PCE rule scores the same in "
      f"all 5-frame runs (AUC {min(pce5):.3f}--{max(pce5):.3f}), so the comparison is fair. "
      f"Without the real WhatsApp/YouTube copies, Phase 3 stacking ({sv(shA, 'Phase 3', 'stack', 'F1 without real platform copies'):.3f}) "
      f"beats Phase 1's GRU ({sv(shA, 'Phase 1', 'gru', 'F1 without real platform copies'):.3f}).")
    w(f"\\item \\textbf{{Phase 4:}} on the Phase 3 test clips, Phase 4 stacking scores {sv(shC, 'Phase 4', 'stack'):.3f} vs.\\ "
      f"{sv(shC, 'Phase 3', 'stack'):.3f} (10 frames: {sv(shC, 'Phase 4, 10 frames', 'stack'):.3f}). The change is uneven "
      f"(Table~\\ref{{tab:source}}): genuine VISION videos are recognised far more often "
      f"({pct(by_src.loc['VISION genuine (native, real WhatsApp/YouTube)', 'Phase 3'])} $\\rightarrow$ "
      f"{pct(by_src.loc['VISION genuine (native, real WhatsApp/YouTube)', 'Phase 4'])}), but genuine ACID re-encodes are now often "
      f"called manipulated ({pct(by_src.loc['ACID genuine re-encodes', 'Phase 3'])} $\\rightarrow$ "
      f"{pct(by_src.loc['ACID genuine re-encodes', 'Phase 4'])}): the new ACID attacks went through the same encoder. "
      f"On the clips every phase tested (VISION and GenVidBench), Phase 4 is the best phase ({sv(shA, 'Phase 4', 'stack'):.3f} vs.\\ "
      f"{sv(shA, 'Phase 3', 'stack'):.3f}). Own, larger test set: F1 {p4['F1']:.3f} (5 frames), {p4w['F1']:.3f} (10 frames).")
    w(f"\\item \\textbf{{HVPF-2 (80/20):}} same test clips as Phase 3; stacking {sv(shC, 'HVPF-2 (80/20)', 'stack'):.3f} vs.\\ "
      f"{sv(shC, 'Phase 3', 'stack'):.3f}: the split does not change the conclusions.")
    w(r"\end{itemize}")

    w(r"\section{What each phase changed}")
    w(r"\begin{table}[H]\centering\small\begin{tabularx}{\linewidth}{llrY}\toprule Phase & Results & Test clips & Main change \\\midrule")
    for _, r in own.iterrows():
        w(f"{r.phase} & {r.date} & {r['test clips (5-fold)']:,} & {WHAT[r.phase]} \\\\")
    w(r"\bottomrule\end{tabularx}\caption{Phases. Test clips = clips scored in 5-fold cross-validation (realistic protocol); "
      r"every clip is tested once, by a model that never saw its camera model.}\end{table}")

    w(r"\section{Each phase on its own test set}")
    w(r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}\begin{tabular}{llcccccccc}\toprule"
      r" & & \multicolumn{4}{c}{Realistic, 5-fold} & PCE rule & Controlled & \multicolumn{2}{c}{Unseen sources} \\"
      r"\cmidrule(lr){3-6}\cmidrule(lr){9-10} Phase & Best model & Macro F1 & PVA & AUC & EER & AUC & best F1 & F1 & AUC \\\midrule")
    for _, r in own.iterrows():
        w(f"{r.phase} & {NAMES[r.best]} & {r.F1:.3f} $\\pm$ {r['F1 std']:.3f} & {f3(r.PVA)} & {f3(r.AUC)} & {f3(r.EER)} & "
          f"{f3(r['PCE AUC'])} & {f3(r['controlled best F1'])} & {f3(r['unseen-sources best F1'])} & {f3(r['unseen-sources best AUC'])} \\\\")
    w(r"\bottomrule\end{tabular}\caption{Not directly comparable across rows: each phase has a different (mostly harder) test "
      r"set. AUC = genuine vs.\ not genuine. Unseen sources = FloreView + GenBuster, fixed split (Phase 1: GenBuster only).}\end{table}")
    w(r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}\begin{tabular}{l" + "c" * len(own) + r"}\toprule Model & "
      + " & ".join(own.phase.str.replace("HVPF-2 (80/20)", "HVPF-2").str.replace("Phase 4, 10 frames", "P4, 10 fr.")) + r" \\\midrule")
    for m in LEARNED:
        w(f"{NAMES[m]} & " + " & ".join(f3(v) for v in own[f"F1 {m}"]) + " \\\\")
    w(r"\bottomrule\end{tabular}\caption{Macro F1 per model and phase (realistic, 5-fold; each phase on its own test set).}\end{table}")

    w(r"\section{All phases on the same clips}")
    w("Each block uses only clips that all listed phases tested. The PCE rule is not trained, so it must give the same score "
      "in every 5-frame run; it does, which confirms that the clips and features match (the 10-frame run scores PCE on 10 frames).")
    for key, label, phases in sets:
        g = same[same.set == key]
        w(f"\\begin{{table}}[H]\\centering\\small\\setlength{{\\tabcolsep}}{{4pt}}\\begin{{tabular}}{{l" + "c" * len(phases)
          + f"}}\\toprule \\multicolumn{{{len(phases) + 1}}}{{l}}{{\\textbf{{{key}: {label}}} ({int(g.clips.iloc[0]):,} clips)}} \\\\\\midrule Model & "
          + " & ".join(p.replace("HVPF-2 (80/20)", "HVPF-2").replace("Phase 4, 10 frames", "P4, 10 fr.") for p in phases) + r" \\\midrule")
        for m in ["baseline", "xgb", "gru", "vit", "stack"]:
            vals = [sv(g, ph, m, "auc" if m == "baseline" else "f1") for ph in phases]
            if all(v is None for v in vals):
                continue
            label_m = "PCE rule (AUC)" if m == "baseline" else NAMES[m] + " (F1)"
            w(f"{label_m} & " + " & ".join(f3(v) for v in vals) + " \\\\")
        if key == "A":
            for m in ["xgb", "gru", "stack"]:
                vals = [sv(g, ph, m, "F1 without real platform copies") for ph in phases]
                if any(v is not None for v in vals):
                    w(f"{NAMES[m]} (F1, no real platform copies) & " + " & ".join(f3(v) for v in vals) + " \\\\")
        w(r"\bottomrule\end{tabular}\end{table}")
    w(r"\begin{table}[H]\centering\small\begin{tabular}{lccc}\toprule Stacking on the Phase 3 test clips & Genuine & Manipulated & AI \\\midrule")
    for ph in ["Phase 3", "HVPF-2 (80/20)", "Phase 4", "Phase 4, 10 frames"]:
        w(f"{ph} & " + " & ".join(pct(sv(shC, ph, 'stack', f'recall_{c}')) for c in ("organic", "manipulated", "synthetic")) + " \\\\")
    w(r"\bottomrule\end{tabular}\caption{Share of each class recognised (recall), same clips.}\end{table}")
    w(r"\begin{table}[H]\centering\small\begin{tabular}{lrcccc}\toprule Stacking, Phase 3 test clips & Clips & Phase 3 & HVPF-2 & "
      r"Phase 4 & P4, 10 fr. \\\midrule")
    for src, r in by_src.iterrows():
        w(f"{src} & {n_src.get(src, 0):,} & " + " & ".join(pct(r[ph]) for ph in sets[2][2]) + " \\\\")
    w(r"\bottomrule\end{tabular}\caption{Share recognised correctly by source (same clips). Phase 4 helps genuine VISION videos "
      r"and hurts genuine ACID re-encodes and VISION attacks.}\label{tab:source}\end{table}")

    if len(ct):
        w(r"\section{Phase 4 training experiments}")
        w("XGBoost and GRU retrained on Phase 4 data with parts left out, or with dataset-balanced weights; all predicted every "
          "Phase 4 test clip. Old clips = the Phase 3 test set; new clips = added in Phase 4.")
        cols = [("F1 old clips", "F1 old"), ("F1 new clips", "F1 new"), ("F1 all", "F1 all"), ("ACID genuine ok", "ACID genuine"),
                ("ACID attacks caught", "ACID attacks"), ("new-generator AI caught", "new-gen.\\ AI")]
        for m in ("xgb", "gru"):
            g = ct[ct.model == m]
            w(f"\\begin{{table}}[H]\\centering\\small\\setlength{{\\tabcolsep}}{{3.5pt}}\\begin{{tabular}}{{l{'c' * len(cols)}}}\\toprule "
              f"{NAMES[m]}: training data & " + " & ".join(c for _, c in cols) + r" \\\midrule")
            for _, r in g.iterrows():
                w(f"{r.setting.replace(' (experiment)', '').replace(' (control)', '')} & " + " & ".join(f3(r[c]) if c.startswith("F1") else pct(r[c]) for c, _ in cols) + " \\\\")
            w(r"\bottomrule\end{tabular}\end{table}")

    w(r"\section{Camera identification (trained ViT + PRNU)}")
    w(r"\begin{table}[H]\centering\small\begin{tabular}{lcccc}\toprule & \multicolumn{2}{c}{HVPF-2 (no scene grouping)} & "
      r"\multicolumn{2}{c}{Phase 4 (scene-grouped)} \\\cmidrule(lr){2-3}\cmidrule(lr){4-5} Dataset & ViT & Hybrid & ViT & Hybrid \\\midrule")
    for ds in ("VISION", "ACID", "FloreView"):
        a = cam2[(cam2.dataset == ds) & (cam2.group == "all")].iloc[0]
        b = cam4[(cam4.dataset == ds) & (cam4.group == "all")].iloc[0]
        w(f"{ds} & {pct(a.vit_acc)} & {pct(a.hybrid_val_acc)} & {pct(b.vit_acc)} & {pct(b.hybrid_val_acc)} \\\\")
    w(r"\bottomrule\end{tabular}\caption{Top-1 camera identification on test videos. Scene grouping removes near-duplicate "
      r"scenes from the test side; the FloreView drop shows that some of the earlier score came from recognising scenes.}\end{table}")

    w(r"\section{Conclusions}\begin{itemize}")
    w(r"\item Phase 1's high score was a shortcut; Phase 2 made the benchmark honest; Phase 3 is the best and most reliable result "
      r"and is the one used for the mid-evaluation.")
    w(r"\item Training on 80\% instead of 60\% of the camera models (HVPF-2) changes the results by less than the fold-to-fold spread.")
    w(r"\item Phase 4 adds coverage that Phase 3 lacks (attacks on a second dataset, more generators) and recognises genuine "
      r"VISION videos much better, but calls more genuine ACID re-encodes manipulated; 10 frames recovers part of the loss and "
      r"dataset-balanced training does not help. The open problem is separating genuine re-encodes from attacked clips.")
    w(r"\item In every phase the conventional PCE rule stays near AUC 0.55 for genuine vs.\ not genuine, far below the learned models.")
    w(r"\end{itemize}")
    w(f"\\noindent\\raggedright\\small Generated by \\texttt{{scripts/make\\_phase\\_comparison.py}}; numbers in \\texttt{{{out_csv.relative_to(ROOT).as_posix().replace('_', chr(92) + '_')}}}.")
    w(r"\end{document}")
    out = ROOT / "report/phase_comparison.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} and {out_csv}")
    print(own[["phase", "best", "F1", "AUC", "PCE AUC", "controlled best F1", "unseen-sources best F1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
