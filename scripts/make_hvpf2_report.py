"""Write the hvpf-2 report (report/results_hvpf2.tex) from the result files.

  1. provenance models trained 80/20 (hvpf-2) vs 60/20/20 (original project, Phase 3) on the
     same test sets (same camera-model folds and fixed test split)
  2. source camera identification by training (Step 10) vs the PRNU baseline and PDC-ViT

    python scripts/make_hvpf2_report.py
    ~/.local/bin/tectonic -X compile report/results_hvpf2.tex
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config

MODELS = ["baseline", "svm", "xgb", "gru", "cnn", "transformer", "vit", "ens_all", "ens_top3", "stack"]
NAMES = {"baseline": "PCE rule", "svm": "SVM", "xgb": "XGBoost", "gru": "GRU", "cnn": "1D-CNN",
         "transformer": "Transformer", "vit": "ViT (residual)", "ens_all": "Ens.\\ all",
         "ens_top3": "Ens.\\ top-3 (val)", "stack": "Stacking (LR)"}
DS_NAMES = {"VISION": "VISION", "ACID": "Video-ACID", "FloreView": "FloreView"}


def f3(x):
    return "--" if x is None or pd.isna(x) else f"{x:.3f}"


def pct(x, nd=1):
    return "--" if x is None or pd.isna(x) else f"{100 * x:.{nd}f}"


def diff(a, b):
    return "--" if a is None or b is None or pd.isna(a) or pd.isna(b) else f"{b - a:+.3f}"


def main():
    cfg = load_config()
    paths = cfg["paths"]
    new = pd.read_csv(paths["results_dir"] / "metrics" / "summary.csv")
    old = pd.read_csv(Path(paths["compare_results"]) / "metrics" / "summary.csv")
    cam_path = paths["results_dir"] / "camera_id" / "summary.csv"
    cam = pd.read_csv(cam_path) if cam_path.exists() else pd.DataFrame()

    def get(df, proto, scheme, part, model, key):
        g = df[(df.protocol == proto) & (df.scheme == scheme) & (df.tag == "all_w5") & (df.part == part) & (df.model == model)]
        return g.iloc[0][key] if len(g) and key in g else None

    L = []
    w = L.append
    w(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage[margin=2.2cm]{geometry}\usepackage{microtype}
\usepackage{booktabs}\usepackage{array}\usepackage{float}\usepackage{caption}\usepackage{enumitem}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\captionsetup{font=small,labelfont=bf}\setlist{itemsep=2pt,topsep=4pt}\setlength{\emergencystretch}{3em}
\title{\textbf{HVPF-2: 80/20 Training and Trained Source Camera Identification}}
\author{Talha Aslam\\ MS Thesis}""")
    w(f"\\date{{{date.today().strftime('%d %B %Y')}}}")
    w(r"""\begin{document}\maketitle
\begin{abstract}\noindent
HVPF-2 is a copy of the HVPF pipeline with two changes. (1) The provenance models (organic /
manipulated / synthetic) are trained on 80\% of the camera models and tested on 20\%, instead of
60/20/20; they are compared with the Phase 3 models on the same test sets. (2) Source camera
identification is also learned: a Vision Transformer is trained to name the camera of a video
(80/20 split of each camera's recordings) and compared with training-free PRNU matching and with
PDC-ViT. Data, features and frames are identical to Phase 3.
\end{abstract}""")

    # ---------------------------------------------------------------- key findings (numbers from the files)
    single = ["svm", "xgb", "gru", "cnn", "transformer", "vit"]
    gain = [get(new, "realistic", "folds", "test", m, "f1_mean") - get(old, "realistic", "folds", "test", m, "f1_mean")
            for m in single if get(new, "realistic", "folds", "test", m, "f1_mean") is not None]
    st_old, st_new = (get(d, "realistic", "folds", "test", "stack", "f1_mean") for d in (old, new))
    sd = get(new, "realistic", "folds", "test", "stack", "f1_std")
    cx_old, cx_new = (get(d, "realistic", "fixed", "crossdataset", "stack", "f1_mean") for d in (old, new))
    w(r"\section*{Key findings}\begin{itemize}")
    if gain:
        w(f"\\item \\textbf{{80/20 vs 60/20/20:}} on the same test folds, single models gain {min(gain):+.3f} to "
          f"{max(gain):+.3f} macro F1 (mean {sum(gain) / len(gain):+.3f}); the best model, stacking, stays at "
          f"{st_new:.3f} (60/20/20: {st_old:.3f}). The gains are smaller than the fold-to-fold spread "
          f"(std {sd:.3f}): more training cameras help a little, not decisively. On unseen sources (FloreView + "
          f"GenBuster, fixed split) stacking reaches {cx_new:.3f} ({cx_old:.3f}).")
    if len(cam):
        a = cam[cam.group == "all"].set_index("dataset")
        nu = cam[cam.group == "no usable PRNU"].set_index("dataset")
        if {"ACID", "FloreView", "VISION"} <= set(a.index):
            w(f"\\item \\textbf{{Trained camera identification:}} the ViT names the camera of {pct(a.vit_acc['ACID'])}\\% "
              f"(Video-ACID) and {pct(a.vit_acc['FloreView'])}\\% (FloreView) of the test videos, against "
              f"{pct(a.prnu_acc['ACID'])}\\% and {pct(a.prnu_acc['FloreView'])}\\% for PRNU matching; on VISION it reaches "
              f"only {pct(a.vit_acc['VISION'])}\\% (PRNU {pct(a.prnu_acc['VISION'])}\\%).")
            w(f"\\item \\textbf{{Stabilised phones:}} where PRNU fails ({pct(nu.prnu_acc['ACID'])}\\% on Video-ACID, "
              f"{pct(nu.prnu_acc['FloreView'])}\\% on FloreView), the ViT still identifies {pct(nu.vit_acc['ACID'])}\\% "
              f"and {pct(nu.vit_acc['FloreView'])}\\%. For these phones the sensor pattern is unusable, so the ViT must "
              "rely on other traces of the camera, such as its image processing, stabilisation and encoder settings.")
            w(f"\\item \\textbf{{Model vs device:}} on Video-ACID the ViT finds the right camera \\emph{{model}} for "
              f"{pct(a.vit_model_acc['ACID'])}\\% of the videos but the right device for {pct(a.vit_acc['ACID'])}\\%: devices of "
              "the same model are what it confuses. Telling those apart is exactly what PRNU does when it is available.")
            w(f"\\item \\textbf{{Hybrid:}} using PRNU when it is confident and the ViT otherwise gives the best accuracy on "
              f"every dataset: {pct(a.hybrid_val_acc['VISION'])}\\% (VISION), {pct(a.hybrid_val_acc['ACID'])}\\% (Video-ACID), "
              f"{pct(a.hybrid_val_acc['FloreView'])}\\% (FloreView); PDC-ViT reports 94.3\\% and 74.77\\% on VISION and Video-ACID.")
            w("\\item \\textbf{Why VISION is hard for the ViT:} about 30 training videos per camera, a split that keeps each "
              "scene on one side, and test videos that include WhatsApp and YouTube copies. Video-ACID has about 160 "
              "training videos per camera, and ACID and FloreView have no scene labels, so their splits are less strict "
              "(see Notes).")
    w(r"\end{itemize}")

    # ---------------------------------------------------------------- provenance
    w(r"\section{Provenance models: 80/20 vs 60/20/20}")
    w("Both versions use the same camera-model groups. In 5-fold cross-validation, fold $f$ is the test set in both; "
      "60/20/20 trains on three folds and uses fold $f{+}1$ for validation, while 80/20 trains on all four other folds. "
      "To keep model selection honest, 80/20 still uses fold $f{+}1$ to choose settings (SVM $C$, XGBoost depth and number "
      "of trees, the networks' number of epochs; calibration and stacking weights), then retrains on all four folds with "
      "those settings. The fixed split merges the old train and validation camera models (82\\% of the models) and keeps "
      "the same test models.")
    for proto, scheme, part, title in (("realistic", "folds", "test", "Realistic setup, 5-fold cross-validation (main result)"),
                                       ("controlled", "folds", "test", "Encoder-controlled setup, 5-fold cross-validation"),
                                       ("realistic", "fixed", "test", "Realistic setup, fixed split, test cameras"),
                                       ("realistic", "fixed", "crossdataset", "Unseen sources (FloreView + GenBuster), fixed split")):
        w(f"\\begin{{table}}[H]\\centering\\small\\caption{{{title}, 5 frames: 60/20/20 (Phase 3) vs 80/20 (HVPF-2).}}"
          r"\begin{tabular}{lccccccc}\toprule & \multicolumn{3}{c}{Macro F1} & \multicolumn{2}{c}{PVA} & "
          r"\multicolumn{2}{c}{AUC genuine vs not} \\\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
          r"Model & 60/20/20 & 80/20 & change & 60/20/20 & 80/20 & 60/20/20 & 80/20 \\\midrule")
        for m in MODELS:
            a, b = get(old, proto, scheme, part, m, "f1_mean"), get(new, proto, scheme, part, m, "f1_mean")
            if a is None and b is None and m != "baseline":
                continue
            pa, pb = get(old, proto, scheme, part, m, "pva_mean"), get(new, proto, scheme, part, m, "pva_mean")
            ua, ub = get(old, proto, scheme, part, m, "bin_auc_mean"), get(new, proto, scheme, part, m, "bin_auc_mean")
            w(f"{NAMES[m]} & {f3(a)} & {f3(b)} & {diff(a, b)} & {f3(pa)} & {f3(pb)} & {f3(ua)} & {f3(ub)} \\\\")
        w(r"\bottomrule\end{tabular}\end{table}")

    # ---------------------------------------------------------------- camera identification
    w(r"\section{Source camera identification by training}")
    w("Each camera's recordings are split 80/20: 20\\% of its natural videos are test videos; all other videos, "
      "including the flat reference videos, are training videos. All versions of a recording (VISION native, YouTube and "
      "WhatsApp) stay on the same side, and VISION's moving, still and panning takes of one scene count as one recording, "
      "so a test scene is never seen in training (ACID and FloreView have no scene labels). 10\\% of the recordings are held "
      "out inside the training side to choose the number of epochs; the model is then retrained on the whole 80\\%. "
      "The model is the ViT-S/16 of Step 8b with one output per camera, on SRM noise residuals. "
      "\\emph{PRNU} is the training-free baseline of Step 4 on the same test videos (fingerprints from the reference videos, "
      "which are never test videos). \\emph{Hybrid} uses PRNU when it verifies the camera (PCE$_0>60$) and the ViT otherwise; "
      "\\emph{hybrid (val)} chooses that PCE$_0$ threshold on the validation videos.")
    present = [ds for ds in DS_NAMES if len(cam) and (cam.dataset == ds).any()]
    if present:
        w(r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}\caption{Top-1 camera identification accuracy (\%) "
          r"on the test videos. Model level: a prediction of another camera of the same model counts as correct.}"
          r"\begin{tabular}{llrrrrrrr}\toprule Dataset & Test videos & Cameras & ViT & PRNU & Hybrid & Hybrid (val) & "
          r"ViT model level \\\midrule")
        for ds in present:
            for _, r in cam[cam.dataset == ds].iterrows():
                label = DS_NAMES[ds] if r.group == "all" else f"\\quad {r.group}"
                w(f"{label} & {r.videos} & {r.cameras} & {pct(r.vit_acc)} & {pct(r.prnu_acc)} & {pct(r.hybrid_acc)} & "
                  f"{pct(r.hybrid_val_acc)} & {pct(r.vit_model_acc)} \\\\")
            if ds != present[-1]:
                w(r"\midrule")
        w(r"\bottomrule\end{tabular}\end{table}")

        w(r"\subsection{Comparison with PDC-ViT}")
        w("PDC-ViT (Elharrouss et al.\\ 2025) also trains a Vision Transformer with an 80/20 split; its numbers are taken "
          "from the paper (Video-ACID there includes WhatsApp copies, 26 cameras). FNR and FPR are macro averages over the "
          "cameras (one camera vs the rest).")
        w(r"\begin{table}[H]\centering\small\caption{Camera identification compared with PDC-ViT.}"
          r"\begin{tabular}{llrrr}\toprule Method & Dataset (cameras) & Accuracy (\%) & FNR (\%) & FPR (\%) \\\midrule")
        w(r"PDC-ViT (reported) & VISION (35) & 94.30 & -- & -- \\")
        w(r"PDC-ViT (reported) & Video-ACID + WA (26) & 74.77 & 23.4 & 0.92 \\\midrule")
        for ds in present:
            r = cam[(cam.dataset == ds) & (cam.group == "all")].iloc[0]
            for method, mname in (("vit", "HVPF-2 ViT"), ("prnu", "PRNU"), ("hybrid_val", "Hybrid (val)")):
                w(f"{mname} & {DS_NAMES[ds]} ({r.cameras}) & {pct(r[f'{method}_acc'], 2)} & {pct(r[f'{method}_fnr'])} & "
                  f"{pct(r[f'{method}_fpr'], 2)} \\\\")
            if ds != present[-1]:
                w(r"\midrule")
        w(r"\bottomrule\end{tabular}\end{table}")
    else:
        w("Step 10 has not been run yet.")

    w(r"""\section{Notes}\begin{itemize}
\item The provenance test sets are identical in both versions, so the differences come only from training on 80\%
      instead of 60\% of the camera models (and from the refit on all of it).
\item Camera identification is closed-set: every test camera is in training. Provenance stays open-set: test camera
      models are never seen in training.
\item ACID and FloreView have no scene labels, so two videos of the same scene by one camera may fall on different
      sides of the split; the SRM residual input makes scene content less useful to the model, but not useless.
\end{itemize}
\end{document}""")
    out = paths["data_dir"].parent / "report" / "results_hvpf2.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
