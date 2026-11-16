"""Thesis - final HVPF against the base paper's method (PDC-ViT, scripts/thesis_pdcvit.py) on identical clips.

Every PDC-ViT variant found under data/thesis/pdcvit/<tag>/: validation macro F1 (fixed split; the variant is
chosen on it), then HVPF vs that variant on the unseen cameras (5-fold test, if trained) and on the fixed
split's test cameras and unseen datasets, with the camera-level paired bootstrap of scripts/paper_stats.py.
Writes data/thesis/pdcvit/compare.csv and compare.txt.

    .venv/bin/python scripts/thesis_pdcvit_compare.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5

stats = phase5.load_step("paper_stats.py")
OUT = phase5.ROOT / "data" / "thesis" / "pdcvit"
CAM = ["macro F1", "AUC genuine vs not", "genuine accepted %", "attacks caught %", "AI caught %",
       "AI + injected PRNU accepted %", "injection attack (a=1) accepted %"]
SETS = ["macro F1", "AUC genuine vs not", "genuine accepted %", "AI caught %", "AI + injected PRNU accepted %"]


def main():
    warnings.filterwarnings("ignore")
    f = lambda scheme: phase5.PHASE5 / "models" / f"fusion_c4m4_realistic_{scheme}_w10.parquet"
    hv = {("folds", "test"): stats.system(f("folds"), "stack_rel", "test"),
          ("fixed", "test"): stats.system(f("fixed"), "stack_rel", "test"),
          ("fixed", "crossdataset"): stats.system(f("fixed"), "stack_rel", "crossdataset")}
    titles = {("folds", "test"): "unseen cameras, 5-fold", ("fixed", "test"): "unseen cameras, fixed split",
              ("fixed", "crossdataset"): "unseen datasets"}
    rows, lines = [], ["== PDC-ViT variants: validation macro F1 on the fixed split (the variant is chosen on this)"]
    tags = sorted(d.name for d in OUT.iterdir() if d.is_dir())
    for tag in tags:
        p = OUT / tag / "realistic_fixed_w10.parquet"
        if p.exists():
            v = stats.system(p, "pdc_vit", "val")
            lines.append(f"  {tag:10s} val F1 {stats.macro_f1(v.y.to_numpy(), v.pred.to_numpy()):.3f}")
    for tag in tags:
        for (scheme, part), title in titles.items():
            p = OUT / tag / f"realistic_{scheme}_w10.parquet"
            if not p.exists():
                continue
            name = f"HVPF vs PDC-ViT [{tag}] ({title})"
            print(name, flush=True)
            r = pd.DataFrame(stats.boot(hv[(scheme, part)], stats.system(p, "pdc_vit", part),
                                        SETS if part == "crossdataset" else CAM)).assign(comparison=name)
            rows.append(r)
            lines += [f"\n== {name}  (A = HVPF, B = PDC-ViT; {int(r['n clips'].iat[0]):,} clips, {int(r['n cameras'].iat[0])} cameras)",
                      r.drop(columns=["comparison", "n clips", "n cameras"]).set_index("metric")
                      .to_string(float_format=lambda v: f"{v:.3f}")]
    pd.concat(rows, ignore_index=True).to_csv(OUT / "compare.csv", index=False)
    text = "\n".join(lines)
    (OUT / "compare.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
