"""Paper, day 5 - significance of HVPF vs the baselines and of the compression-diverse fix.

  1. HVPF (final) vs each baseline (scripts/paper_baselines.py) on identical clips: unseen cameras (5-fold
     test) and unseen datasets; camera-level paired bootstrap as in scripts/paper_stats.py.
  2. The saved HVPF before vs after compression-diverse training (bundle_fixed_w10 vs bundle_aug_fixed_w10)
     on the clean fixed-split test set and the unseen datasets (camera-level paired bootstrap).
  3. The adaptive attack before vs after the fix (same 55 target videos): exact McNemar test on the paired
     per-video outcomes; only 5 target cameras, so the per-video test treats videos of a camera as
     independent (stated as a limitation).
Writes data/paper/compare.csv and compare.txt.

    .venv/bin/python scripts/paper_compare.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5

stats = phase5.load_step("paper_stats.py")
PAPER = phase5.ROOT / "data" / "paper"
BASELINES = {"effnet_b4": "EfficientNet-B4", "clip_b16": "UnivFD-style CLIP", "dinov2_b": "DINOv2 linear"}


def bundle_system(name, part):
    d = pd.read_parquet(PAPER / name / "predictions.parquet")
    d = d[d.part == part].assign(model="bundle", split_name="fixed")
    path = PAPER / f"_{name}_{part}.parquet"
    d.to_parquet(path, index=False)
    s = stats.system(path, "bundle", part)
    path.unlink()
    return s


def mcnemar(before, after):
    """Exact McNemar on paired booleans: p of the discordant split under H0 (50/50)."""
    b, c = int((before & ~after).sum()), int((~before & after).sum())
    return b, c, binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def main():
    warnings.filterwarnings("ignore")
    f = lambda scheme: phase5.PHASE5 / "models" / f"fusion_c4m4_realistic_{scheme}_w10.parquet"
    hv = stats.system(f("folds"), "stack_rel", "test")
    hv_x = stats.system(f("fixed"), "stack_rel", "crossdataset")
    m_cam = ["macro F1", "AUC genuine vs not", "genuine accepted %", "attacks caught %", "AI caught %",
             "injection attack (a=1) accepted %"]
    m_set = ["macro F1", "AUC genuine vs not", "genuine accepted %", "AI caught %"]
    rows, lines = [], []
    blocks = []
    for key, name in BASELINES.items():
        base = PAPER / "baselines" / key
        blocks.append((f"HVPF vs {name} (unseen cameras)", hv, stats.system(base / "realistic_folds_w10.parquet", key, "test"), m_cam))
        blocks.append((f"HVPF vs {name} (unseen datasets)", hv_x,
                       stats.system(base / "realistic_fixed_w10.parquet", key, "crossdataset"), m_set))
    for part, title in (("test", "clean unseen cameras, fixed split"), ("crossdataset", "unseen datasets")):
        blocks.append((f"HVPF after vs before the compression-diverse fix ({title})",
                       bundle_system("bundle_aug_fixed_w10", part), bundle_system("bundle_fixed_w10", part),
                       m_cam[:5] if part == "test" else m_set))
    for title, a, b, metrics in blocks:
        print(title, flush=True)
        r = pd.DataFrame(stats.boot(a, b, metrics)).assign(comparison=title)
        rows.append(r)
        lines += [f"\n== {title}  (A = first, B = second; {int(r['n clips'].iat[0]):,} clips, {int(r['n cameras'].iat[0])} cameras)",
                  r.drop(columns=["comparison", "n clips", "n cameras"]).set_index("metric")
                  .to_string(float_format=lambda v: f"{v:.3f}")]

    lines.append("\n== Adaptive attack, before vs after the fix (55 target videos; exact McNemar on paired outcomes)")
    before = pd.read_parquet(PAPER / "adaptive" / "candidates.parquet")
    after = pd.read_parquet(PAPER / "adaptive" / "candidates_aug.parquet")
    goals = {"injection": [("HVPF says genuine", lambda d: d.pred == 0),
                           ("usable forgery: HVPF genuine and PCE accepts", lambda d: (d.pred == 0) & d.pce_pass),
                           ("3-zone accepted", lambda d: d.zone == 0)],
             "removal": [("HVPF says genuine", lambda d: d.pred == 0),
                         ("camera link broken (PCE fails) and HVPF genuine", lambda d: (d.pred == 0) & ~d.pce_pass),
                         ("3-zone accepted", lambda d: d.zone == 0)]}
    for kind, gl in goals.items():
        for label, fn in gl:
            s = [d[d.kind == kind].assign(ok=fn(d[d.kind == kind])).groupby("content_id").ok.any() for d in (before, after)]
            b, c, p = mcnemar(s[0], s[1].reindex(s[0].index))
            lines.append(f"  {kind:9s} {label:50s} before {100 * s[0].mean():5.1f}%  after {100 * s[1].mean():5.1f}%  "
                         f"(videos fixed {b}, newly broken {c}; p = {p:.2g})")
            rows.append(pd.DataFrame([{"comparison": f"adaptive {kind}: {label}", "metric": "success %",
                                       "A": 100 * s[1].mean(), "B": 100 * s[0].mean(), "p": p}]))
    pd.concat(rows, ignore_index=True).to_csv(PAPER / "compare.csv", index=False)
    text = "\n".join(lines)
    (PAPER / "compare.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
