"""Paper, day 4 - compression-diverse attack training (the fix for the adaptive attack's shortcut).

The adaptive attack (scripts/paper_adaptive.py) showed that HVPF accepts attacked clips re-encoded at CRF 26:
every training attack was encoded at CRF 20 (attacks.crf), while genuine re-encodes exist at CRF 18-28, so
"a compression level never seen in an attack" partly means "genuine" to the models. Here the VISION targets of
the TRAINING and VALIDATION cameras (never the test cameras the attack uses) get the removal and injection
attacks (alpha 0.25 / 0.5 / 1, same attacker fingerprints and victims as Step 5) also at CRF 18, 23 and 28,
the CRFs of the genuine controls. Their Step 7 / 7b features are added to data/paper/features_newpc ->
data/paper/features_aug; scripts/paper_bundle.py --features data/paper/features_aug then trains the
augmented HVPF. Raw ACID videos are not on this PC, so only VISION targets are added.

Round 2 (--round 2, after the round-1 HVPF still let 49% of adaptive removals through, mostly weak removal at
CRF 26): weak removal (alpha 0.1 / 0.15 / 0.35, with and without Gaussian noise sigma 1.5), injection after
removing the own PRNU (alpha 0.35 / 0.75, beta 0 / 0.75), and genuine re-encodes with the same CRFs and noise
(so "unusual compression or noise" cannot become an attack cue), all at CRF 22 and 30. None of these settings is
in the adaptive attacker's grid (alpha 0.25-1.5, CRF 16 / 20 / 26, sigma 1), so a lower evasion rate measures
robustness, not memorised settings. Builds on features_aug -> features_aug2.

    .venv/bin/python scripts/paper_crf_aug.py               (~1.5 h: rendering ~20 min, features ~1 h)
    .venv/bin/python scripts/paper_crf_aug.py --round 2     (~2 h)
    .venv/bin/python scripts/paper_crf_aug.py --limit 2     (pipeline check -> data/paper/aug_check)
"""
import argparse
import sys
import warnings
import zlib
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

WINDOW = 10
# (attack, alpha, beta = own-PRNU removal before injection, crf, sigma = added noise)
ROUNDS = {
    1: [(a, al, 0.0, c, 0.0) for a in ("removal", "injection") for al in (0.25, 0.5, 1.0) for c in (18, 23, 28)],
    2: [("removal", al, 0.0, c, s) for al in (0.1, 0.15, 0.35) for c in (22, 30) for s in (0.0, 1.5)]
       + [("injection", al, b, c, 0.0) for al in (0.35, 0.75) for b in (0.0, 0.75) for c in (22, 30)]
       + [("reencode", 0.0, 0.0, c, s) for c in (22, 30) for s in (0.0, 1.5)],
}
BASE = {1: FEATURES, 2: PAPER / "features_aug"}


def targets(cfg):
    a = pd.read_csv(phase5.PHASE4 / "metadata" / "attacks.csv")
    t = a[(a.variant == "injection_a1") & (a.dataset == "VISION") & a.split.isin(["train", "val"])]
    s = phase5.load_step("05_attacks.py").load_samples(cfg["paths"])
    s["half"] = s.groupby("device").cumcount() % 2
    t = t[["content_id", "device", "claimed_device", "split", "fold", "stabilized"]].rename(columns={"claimed_device": "victim"})
    return t.merge(s[["content_id", "path", "fps", "half"]], on="content_id").reset_index(drop=True)


def variant_name(attack, alpha, beta, crf, sigma):
    if attack == "reencode":
        return f"reencode_crf{crf}" + (f"_s{sigma:g}" if sigma else "")
    return f"{attack}_a{alpha:g}" + (f"_b{beta:g}" if beta else "") + f"_crf{crf}" + (f"_s{sigma:g}" if sigma else "")


def render(t, cfg, frames_dir, settings):
    acfg, crop = cfg["attacks"], cfg["frames"]["crop"]
    kdir = cfg["paths"]["fingerprints_dir"] / "attacker"
    k_own = attacker_fingerprint(np.load(kdir / f"{t.device}_h{1 - int(t.half)}.npy"))
    k_vic = attacker_fingerprint(np.load(kdir / f"{t.victim}_all.npy"))
    fps = float(t.fps) if t.fps and t.fps > 0 else 30.0
    clip, rows = None, []
    for attack, alpha, beta, crf, sigma in settings:
        variant = variant_name(attack, alpha, beta, crf, sigma)
        sid = f"{t.content_id}__{variant}"
        out = frames_dir / f"{sid}.npy"
        if not out.exists():
            clip = clip if clip is not None else read_clip(t.path, crop, WINDOW + 1)
            y = clip[0]
            if attack == "removal":
                y = np.stack(list(prnu_frames(y, k_own, alpha, -1)))
            elif attack == "injection":
                if beta:
                    y = np.stack(list(prnu_frames(y, k_own, beta, -1)))
                y = np.stack(list(prnu_frames(y, k_vic, alpha, +1)))
            if sigma:
                rng = np.random.default_rng(zlib.crc32(sid.encode()))
                y = np.clip(np.rint(y + rng.normal(0, sigma, y.shape)), 0, 255).astype(np.uint8)
            tmp = frames_dir / f"{sid}.mp4"
            encode(y, clip[1], tmp, crop, fps, crf, acfg["preset"], max(1, round(fps)))
            np.save(out, extract_iframes(tmp, "format=gray", crop, WINDOW))
            tmp.unlink()
        rows.append({"sample_id": sid, "label": "organic" if attack == "reencode" else "manipulated",
                     "source": "VISION-attack", "variant": variant, "social": "none", "alpha": alpha, "device": t.device,
                     "claimed_device": t.victim if attack == "injection" else t.device,
                     "claimed_dataset": "VISION", "split": t.split, "fold": t.fold, "stabilized": t.stabilized,
                     "source_rotation": 0.0, "frames_path": str(out)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="number of target videos (pipeline check)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--round", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = load_config()
    tag = "aug" if args.round == 1 else "aug2"
    work = PAPER / (f"{tag}_check" if args.limit else tag)
    out_dir = PAPER / (f"features_{tag}_check" if args.limit else f"features_{tag}")
    settings = ROUNDS[args.round]
    (work / "frames").mkdir(parents=True, exist_ok=True)
    feats = ClipFeatures(args.workers)                      # forks its workers before any CUDA work
    t = targets(cfg)
    t = t.head(args.limit) if args.limit else t
    print(f"{len(t)} VISION targets (train / val cameras: {sorted(t.device.unique())}); rendering "
          f"{len(t) * len(settings)} clips (round {args.round}) ...", flush=True)
    with ThreadPoolExecutor(8) as ex:
        samples = pd.DataFrame([r for rows in ex.map(lambda x: render(x, cfg, work / "frames", settings), t.itertuples()) for r in rows])
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
        base = pd.read_parquet(BASE[args.round] / f"{name}.parquet")
        base = base[~base.sample_id.isin(set(add.sample_id))] if "sample_id" in base else base
        pd.concat([base, add], ignore_index=True).to_parquet(out_dir / f"{name}.parquet", index=False)
    print(f"added {len(new[0])} clips (round {args.round}) -> {out_dir}")
    print(new[0].groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
