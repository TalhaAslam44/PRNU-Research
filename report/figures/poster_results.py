"""Results figure for the mid-evaluation poster (Phase 3), read from data/phase3.

    .venv/bin/python report/figures/poster_results.py   -> report/figures/poster_results.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "data" / "phase3"
CLASSES = ["organic", "manipulated", "synthetic"]
BLUE, LIGHT, ORANGE, INK = "#1F5F99", "#9CC3E6", "#D9822B", "#2E4A66"

summ = pd.read_csv(P3 / "metrics" / "summary.csv")
main = summ[(summ.protocol == "realistic") & (summ.scheme == "folds") & (summ.tag == "all_w5") & (summ.part == "test")]
main = main.set_index("model")
models = [("baseline", "PCE rule"), ("svm", "SVM"), ("xgb", "XGBoost"), ("vit", "ViT"), ("stack", "HVPF\n(stacking)")]
auc = [main.loc[m, "bin_auc_mean"] for m, _ in models]
f1 = main.loc["stack", "f1_mean"]


def accepted(pred, model):
    g = pred[(pred.model == model) & (pred.label == "synthetic") & pred.variant.astype(str).str.contains(r"\+inj1$")]
    if model == "baseline":
        return 100 * (g.score_organic > g.threshold).mean()
    return 100 * (g[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1) == 0).mean()


folds = pd.read_parquet(P3 / "models" / "realistic_folds_all_w5.parquet")
folds = folds[folds.part == "test"]
fixed = pd.read_parquet(P3 / "models" / "realistic_fixed_all_w5.parquet")
cross = fixed[fixed.part == "crossdataset"]
pce = [accepted(folds, "baseline"), accepted(cross, "baseline")]
hvpf = [accepted(folds, "stack"), accepted(cross, "stack")]

plt.rcParams.update({"font.family": "Noto Sans", "font.size": 15, "axes.edgecolor": "#8A99A8", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})

colors = [ORANGE] + [LIGHT] * 3 + [BLUE]
bars = a1.bar([n for _, n in models], auc, color=colors, width=0.62)
for b, v in zip(bars, auc):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=14, color=INK,
            fontweight="bold" if v == max(auc) else "normal")
a1.set_ylim(0.4, 1.0)
a1.set_ylabel("AUC (genuine vs. not genuine)")
a1.set_title("Figure 1: Detection on unseen cameras", fontsize=16, fontweight="bold", color=INK, pad=10)

x = np.arange(2)
w = 0.36
b1 = a2.bar(x - w / 2, pce, w, color=ORANGE, label="PCE rule (conventional)")
b2 = a2.bar(x + w / 2, hvpf, w, color=BLUE, label="HVPF (stacking)")
for b, v in list(zip(b1, pce)) + list(zip(b2, hvpf)):
    a2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=14, color=INK)
a2.set_xticks(x, ["Seen\ngenerators", "Unseen\ngenerators"])
a2.set_ylim(0, 128)
a2.set_yticks([0, 20, 40, 60, 80, 100])
a2.set_ylabel("Accepted as genuine (%)")
a2.set_title("Figure 2: AI video + injected PRNU", fontsize=16, fontweight="bold", color=INK, pad=10)
a2.legend(frameon=False, fontsize=12.5, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, columnspacing=1.0)

for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E3E8ED")
    ax.set_axisbelow(True)
fig.text(0.29, 0.015, f"5-fold CV, test cameras never seen in training; HVPF macro F1 = {f1:.3f}",
         ha="center", fontsize=12.5, style="italic", color="#4A5B6C")
fig.text(0.775, 0.015, "Lower is better (Khan et al. 2025: 85.5% on images)", ha="center", fontsize=12.5,
         style="italic", color="#4A5B6C")
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = Path(__file__).with_name("poster_results.png")
fig.savefig(out, dpi=260)
print(f"wrote {out}; AUC {[round(v, 3) for v in auc]}, injected AI accepted: PCE {pce}, HVPF {hvpf}")
