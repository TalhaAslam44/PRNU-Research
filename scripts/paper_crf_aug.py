"""Paper, day 4 - compression-diverse attack training (the fix for the adaptive attack's shortcut).

The adaptive attack (scripts/paper_adaptive.py) showed that HVPF accepts attacked clips re-encoded at CRF 26:
every training attack was encoded at CRF 20 (attacks.crf), while genuine re-encodes exist at CRF 18-28, so
"a compression level never seen in an attack" partly means "genuine" to the models. Here the VISION targets of
the TRAINING and VALIDATION cameras (never the test cameras the attack uses) get the removal and injection
attacks (alpha 0.25 / 0.5 / 1, same attacker fingerprints and victims as Step 5) also at CRF 18, 23 and 28,
the CRFs of the genuine controls. Their Step 7 / 7b features are added to data/paper/features_newpc ->
data/paper/features_aug; scripts/paper_bundle.py --features data/paper/features_aug then trains the
augmented HVPF. Raw ACID videos are not on this PC, so only VISION targets are added.

    .venv/bin/python scripts/paper_crf_aug.py               (~1.5 h: rendering ~20 min, features ~1 h)
    .venv/bin/python scripts/paper_crf_aug.py --limit 2     (pipeline check -> data/paper/aug_check)
"""
import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.attacks import attacker_fingerprint, encode, prnu_frames, read_clip
from hvpf.config import load_config
from hvpf.frames import extract_iframes
from hvpf.paper_bundle import FEATURES, PAPER, ClipFeatures

ALPHAS, CRFS = (0.25, 0.5, 1.0), (18, 23, 28)
WINDOW = 10


def targets(cfg):
    a = pd.read_csv(phase5.PHASE4 / "metadata" / "attacks.csv")
    t = a[(a.variant == "injection_a1") & (a.dataset == "VISION") & a.split.isin(["train", "val"])]
    s = phase5.load_step("05_attacks.py").load_samples(cfg["paths"])
    s["half"] = s.groupby("device").cumcount() % 2
    t = t[["content_id", "device", "claimed_device", "split", "fold", "stabilized"]].rename(columns={"claimed_device": "victim"})
    return t.merge(s[["content_id", "path", "fps", "half"]], on="content_id").reset_index(drop=True)


def render(t, cfg, frames_dir):
    acfg, crop = cfg["attacks"], cfg["frames"]["crop"]
    kdir = cfg["paths"]["fingerprints_dir"] / "attacker"
    k_own = attacker_fingerprint(np.load(kdir / f"{t.device}_h{1 - int(t.half)}.npy"))
    k_vic = attacker_fingerprint(np.load(kdir / f"{t.victim}_all.npy"))
    fps = float(t.fps) if t.fps and t.fps > 0 else 30.0
    clip, rows = None, []
    for attack, k, sign in (("removal", k_own, -1), ("injection", k_vic, +1)):
        for alpha in ALPHAS:
            for crf in CRFS:
                variant = f"{attack}_a{alpha:g}_crf{crf}"
                sid = f"{t.content_id}__{variant}"
                out = frames_dir / f"{sid}.npy"
                if not out.exists():
                    clip = clip if clip is not None else read_clip(t.path, crop, WINDOW + 1)
                    tmp = frames_dir / f"{sid}.mp4"
                    encode(prnu_frames(clip[0], k, alpha, sign), clip[1], tmp, crop, fps, crf, acfg["preset"], max(1, round(fps)))
                    np.save(out, extract_iframes(tmp, "format=gray", crop, WINDOW))
                    tmp.unlink()
                rows.append({"sample_id": sid, "label": "manipulated", "source": "VISION-attack", "variant": variant,
                             "social": "none", "alpha": alpha, "device": t.device,
                             "claimed_device": t.victim if attack == "injection" else t.device,
                             "claimed_dataset": "VISION", "split": t.split, "fold": t.fold, "stabilized": t.stabilized,
                             "source_rotation": 0.0, "frames_path": str(out)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="number of target videos (pipeline check)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = load_config()
    work = PAPER / ("aug_check" if args.limit else "aug")
    out_dir = PAPER / ("features_aug_check" if args.limit else "features_aug")
    (work / "frames").mkdir(parents=True, exist_ok=True)
    feats = ClipFeatures(args.workers)                      # forks its workers before any CUDA work
    t = targets(cfg)
    t = t.head(args.limit) if args.limit else t
    print(f"{len(t)} VISION targets (train / val cameras: {sorted(t.device.unique())}); rendering "
          f"{len(t) * 2 * len(ALPHAS) * len(CRFS)} clips ...", flush=True)
    with ThreadPoolExecutor(8) as ex:
        samples = pd.DataFrame([r for rows in ex.map(lambda x: render(x, cfg, work / "frames"), t.itertuples()) for r in rows])
    parts = []
    for i in range(0, len(samples), 480):
        part = work / f"features_{i // 480:03d}.pkl"
        if not part.exists():
            pd.to_pickle(feats.raw(samples.iloc[i:i + 480]), part)
        parts.append(pd.read_pickle(part))
        print(f"  features {min(i + 480, len(samples))}/{len(samples)}", flush=True)
    new = [pd.concat([p[k] for p in parts], ignore_index=True) for k in range(4)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, add in zip(("videos", "frames", "noiseprint_frames", "noiseprint_videos"), new):
        base = pd.read_parquet(FEATURES / f"{name}.parquet")
        base = base[~base.sample_id.isin(set(add.sample_id))] if "sample_id" in base else base
        pd.concat([base, add], ignore_index=True).to_parquet(out_dir / f"{name}.parquet", index=False)
    print(f"added {len(new[0])} attacked clips at CRF {CRFS} -> {out_dir}")
    print(new[0].groupby(["split"]).size().to_string())


if __name__ == "__main__":
    main()
