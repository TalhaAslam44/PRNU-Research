"""Paper - score the real WhatsApp / YouTube copies of the upload set (scripts/paper_upload_set.py).

For every clip in data/paper/real_upload/{originals, whatsapp, youtube}/V###.<ext> that exists: I-frames as in
Step 3 for real platform copies (a copy whose size differs from the 480 x 480 original is scaled back first,
then centre crop, luma; first 10 I-frames), Step 7 / 7b features (hvpf/paper_bundle.ClipFeatures), camera
calibration, and the scores of the saved HVPF (bundle_fixed_w10), HVPF-R (bundle_aug_fixed_w10, compression-
diverse training) and the PCE rule. Platforms re-encode with their own key-frame spacing, so a short copy can
have fewer than 10 I-frames (handled like any short clip). Missing copies are skipped.

Writes data/paper/real_upload/{frames/, scores.parquet, summary.txt}.

    .venv/bin/python scripts/paper_real_upload.py
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.attacks import social_filter
from hvpf.config import load_config
from hvpf.frames import extract_iframes
from hvpf.model_data import CLASSES, load, slog
from hvpf.paper_bundle import FEATURES, PAPER, Bundle, ClipFeatures, with_calibration

UP = PAPER / "real_upload"
PLATFORMS = ("originals", "whatsapp", "youtube")
MODELS = {"HVPF": PAPER / "bundle_fixed_w10", "HVPF-R": PAPER / "bundle_aug_fixed_w10"}
LABEL = {"genuine": "organic", "attacked": "manipulated", "AI": "synthetic"}
EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".3gp")


def size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def main():
    warnings.filterwarnings("ignore")
    cfg = load_config()
    crop = cfg["frames"]["crop"]
    feats = ClipFeatures(12)
    m = pd.read_csv(UP / "manifest.csv")
    rows = []
    for platform in PLATFORMS:
        for r in m.itertuples():
            src = next((UP / platform / f"{r.id}{e}" for e in EXTS if (UP / platform / f"{r.id}{e}").exists()), None)
            if src is None:
                continue
            out = UP / "frames" / platform / f"{r.id}.npy"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                w, h = size(src)
                vf = "format=gray" if (w, h) == (crop, crop) else f"scale={crop}:{crop}:flags=bicubic,format=gray"
                fr = extract_iframes(src, vf if (w, h) != (crop, crop) else social_filter(crop, crop), crop, 10)
                if len(fr) == 0:
                    continue
                np.save(out, fr)
            rows.append({"sample_id": f"{platform}_{r.id}", "label": LABEL[r[2]], "source": f"real-{platform}",
                         "variant": r.kind, "social": platform, "alpha": np.nan, "device": r.claimed_device,
                         "claimed_device": r.claimed_device, "claimed_dataset": "VISION", "split": "test", "fold": np.nan,
                         "stabilized": False, "source_rotation": 0.0, "frames_path": str(out),
                         "id": r.id, "class": r[2], "platform": platform})
    s = pd.DataFrame(rows)
    print(s.groupby(["platform", "class"]).size().unstack(fill_value=0).to_string(), flush=True)
    meta = ["id", "class", "platform", "variant"]
    pv, fr = feats.table(s.drop(columns=meta[:3]), 10)
    pv = with_calibration(pv.merge(s[["sample_id", *meta[:3]]], on="sample_id"), load(FEATURES, 10)[0])
    res = pv[["sample_id", *meta]].assign(pce_accepts=pv.w_pce0_claimed.to_numpy() > slog(60.0),
                                          n_iframes=pv.n_frames.to_numpy())
    for name, path in MODELS.items():
        if not path.exists():
            continue
        p, pred, zone, _ = Bundle(path).predict(pv, fr)
        res[f"{name} p_genuine"], res[f"{name} pred"], res[f"{name} zone"] = p[:, 0], pred, zone
    res.to_parquet(UP / "scores.parquet", index=False)

    lines = []
    for name in [n for n in MODELS if f"{n} pred" in res]:
        want = res["class"].map({"genuine": 0, "attacked": 1, "AI": 2})
        t = res.assign(correct=res[f"{name} pred"] == want, accepted=res[f"{name} pred"] == 0,
                       z_accepted=res[f"{name} zone"] == 0, z_inconclusive=res[f"{name} zone"] == -1)
        g = t.groupby(["class", "platform"])
        tab = pd.DataFrame({"clips": g.size(), "correct %": 100 * g.correct.mean(), "accepted as genuine %": 100 * g.accepted.mean(),
                            "3-zone accepted %": 100 * g.z_accepted.mean(), "3-zone inconclusive %": 100 * g.z_inconclusive.mean(),
                            "PCE accepts %": 100 * g.pce_accepts.mean(), "median I-frames": g.n_iframes.median()})
        lines += [f"\n== {name}", tab.round(1).to_string()]
    text = "\n".join(lines)
    (UP / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
