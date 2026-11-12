"""Paper, step 2b - adaptive attack: an attacker who knows HVPF and can query it.

Targets: the VISION native videos of the fixed split's test cameras (55 videos, 5 cameras never seen in
training); raw ACID videos are not on this PC. For every target the attacker tries a grid of attack
settings, renders each candidate exactly like Step 5 (luma attack, libx264, 1 I-frame/s), runs the saved
final HVPF (scripts/paper_bundle.py) on it and keeps the candidate HVPF finds most genuine:

  injection (claim the victim camera of Step 5)  remove own PRNU first (beta 0 / 0.5 / 1) x inject victim
                                                 PRNU (alpha 0.25 / 0.5 / 1 / 1.5) x CRF 16 / 20 / 26     36 queries
  removal (hide the manipulation, own camera)    alpha 0.25 / 0.5 / 1 / 1.5 x CRF 16 / 20 / 26 x added
                                                 Gaussian noise (sigma 0 / 1)                              24 queries
The non-adaptive Step 5 setting (injection alpha 1, removal alpha 0.5, CRF 20) is in the grid, so the same
run gives the non-adaptive success rate. Success = HVPF calls the clip genuine (default decision; also the
three-zone "accepted"); for injection also "and the PCE rule accepts the claimed camera" (what a forger needs).

Writes data/paper/adaptive/{frames/, features_*.parquet, candidates.parquet, summary.txt}.

    .venv/bin/python scripts/paper_adaptive.py                                     # all targets (~1-2 h)
    .venv/bin/python scripts/paper_adaptive.py --limit 1 --bundle data/paper/bundle_check   # pipeline check
    .venv/bin/python scripts/paper_adaptive.py --predict-only                      # re-score saved features
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
from hvpf.model_data import CLASSES, load, slog
from hvpf.paper_bundle import FEATURES, Bundle, ClipFeatures, with_calibration

OUT = phase5.ROOT / "data" / "paper" / "adaptive"
WINDOW = 10
GRID = {"injection": [(b, a, c, 0.0) for b in (0.0, 0.5, 1.0) for a in (0.25, 0.5, 1.0, 1.5) for c in (16, 20, 26)],
        "removal": [(0.0, a, c, s) for a in (0.25, 0.5, 1.0, 1.5) for c in (16, 20, 26) for s in (0.0, 1.0)]}
STANDARD = {"injection": (0.0, 1.0, 20, 0.0), "removal": (0.0, 0.5, 20, 0.0)}


def name(kind, p):
    return f"adv_{kind[:3]}_b{p[0]:g}_a{p[1]:g}_crf{p[2]}_s{p[3]:g}"


def targets(cfg):
    paths = cfg["paths"]
    a = pd.read_csv(phase5.PHASE4 / "metadata" / "attacks.csv")
    t = a[(a.variant == "injection_a1") & (a.dataset == "VISION") & (a.split == "test")]
    step5 = phase5.load_step("05_attacks.py")
    s = step5.load_samples(paths)
    s["half"] = s.groupby("device").cumcount() % 2              # as in Step 5's attacker fingerprints
    t = t[["content_id", "device", "claimed_device", "fold", "stabilized"]].rename(columns={"claimed_device": "victim"})
    return t.merge(s[["content_id", "path", "fps", "half"]], on="content_id").reset_index(drop=True)


def render_target(t, cfg, frames_dir):
    """All candidates of one target; returns SAMPLE_COLS rows."""
    acfg, crop = cfg["attacks"], cfg["frames"]["crop"]
    kdir = cfg["paths"]["fingerprints_dir"] / "attacker"
    k_own = attacker_fingerprint(np.load(kdir / f"{t.device}_h{1 - int(t.half)}.npy"))
    k_vic = attacker_fingerprint(np.load(kdir / f"{t.victim}_all.npy"))
    fps = float(t.fps) if t.fps and t.fps > 0 else 30.0
    gop = max(1, round(fps))
    clip = None
    rows = []
    rng = np.random.default_rng(zlib.crc32(t.content_id.encode()))
    for kind, grid in GRID.items():
        for p in grid:
            beta, alpha, crf, sigma = p
            sid = f"{t.content_id}__{name(kind, p)}"
            out = frames_dir / f"{sid}.npy"
            if not out.exists():
                if clip is None:
                    clip = read_clip(t.path, crop, WINDOW + 1)
                y = clip[0]
                if kind == "injection":
                    if beta:
                        y = np.stack(list(prnu_frames(y, k_own, beta, -1)))
                    y = np.stack(list(prnu_frames(y, k_vic, alpha, +1)))
                else:
                    y = np.stack(list(prnu_frames(y, k_own, alpha, -1)))
                if sigma:
                    y = np.clip(np.rint(y + rng.normal(0, sigma, y.shape)), 0, 255).astype(np.uint8)
                tmp = frames_dir / f"{sid}.mp4"
                encode(y, clip[1], tmp, crop, fps, crf, acfg["preset"], gop)
                fr = extract_iframes(tmp, "format=gray", crop, WINDOW)
                tmp.unlink()
                np.save(out, fr)
            rows.append({"sample_id": sid, "label": "manipulated", "source": "VISION-attack", "variant": name(kind, p),
                         "social": "none", "alpha": alpha, "device": t.device,
                         "claimed_device": t.victim if kind == "injection" else t.device, "claimed_dataset": "VISION",
                         "split": "test", "fold": t.fold, "stabilized": t.stabilized, "source_rotation": 0.0,
                         "frames_path": str(out), "content_id": t.content_id, "kind": kind, "beta": beta, "crf": crf,
                         "sigma": sigma, "standard": p == STANDARD[kind]})
    return rows


def wilson(k, n):
    if n == 0:
        return "–"
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * p:.1f}% [{100 * max(c - h, 0):.1f}, {100 * min(c + h, 1):.1f}]"


def summary(c):
    lines = []
    for kind, g in c.groupby("kind"):
        per = g.groupby("content_id")
        std = g[g.standard].set_index("content_id")
        n = per.ngroups
        rows = [("Non-adaptive (Step 5 setting): HVPF says genuine", (std.pred == 0).sum()),
                ("Non-adaptive: 3-zone accepted as genuine", (std.zone == 0).sum()),
                ("Adaptive (best of grid): HVPF says genuine", per.apply(lambda d: (d.pred == 0).any()).sum()),
                ("Adaptive: 3-zone accepted as genuine", per.apply(lambda d: (d.zone == 0).any()).sum())]
        if kind == "injection":
            rows += [("Non-adaptive: PCE rule accepts the claimed camera", std.pce_pass.sum()),
                     ("Adaptive: HVPF genuine AND PCE accepts (successful forgery)",
                      per.apply(lambda d: ((d.pred == 0) & d.pce_pass).any()).sum()),
                     ("Adaptive: 3-zone accepted AND PCE accepts", per.apply(lambda d: ((d.zone == 0) & d.pce_pass).any()).sum())]
        best = per.p_organic.max()
        lines.append(f"\n== {kind} attack: {n} target videos, {len(g) // n} queries each "
                     f"(median best p_genuine {best.median():.3f}, max {best.max():.3f})")
        lines += [f"  {label:62s} {k:3d}/{n}  {wilson(k, n)}" for label, k in rows]
        by = g.groupby("device").apply(lambda d: d.groupby("content_id").apply(lambda x: (x.pred == 0).any()).mean())
        lines.append("  adaptive success by target camera: " + ", ".join(f"{d} {100 * v:.0f}%" for d, v in by.items()))
        top = g.groupby(["beta", "alpha", "crf", "sigma"]).apply(lambda d: (d.pred == 0).mean()).sort_values(ascending=False)
        lines.append("  best single setting: " + ", ".join(f"{k}={v:g}" for k, v in zip(["beta", "alpha", "crf", "sigma"], top.index[0]))
                     + f" fools HVPF on {100 * top.iloc[0]:.1f}% of targets")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=phase5.ROOT / "data" / "paper" / "bundle_fixed_w10")
    ap.add_argument("--limit", type=int, help="number of target videos (pipeline check)")
    ap.add_argument("--predict-only", action="store_true", help="re-score the saved features with --bundle")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = load_config()
    out = OUT if not args.limit else OUT.parent / "adaptive_check"
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not args.predict_only:
        feats = ClipFeatures(args.workers)        # forks its workers before any CUDA work
        t = targets(cfg)
        if args.limit:
            t = t.head(args.limit)
        print(f"{len(t)} target videos on cameras {sorted(t.device.unique())}; rendering candidates ...", flush=True)
        with ThreadPoolExecutor(8) as ex:
            samples = pd.DataFrame([r for rows in ex.map(lambda x: render_target(x, cfg, frames_dir), t.itertuples())
                                    for r in rows])
        print(f"{len(samples)} attacked clips rendered; features (Steps 7 + 7b) ...", flush=True)
        parts = []
        for i in range(0, len(samples), 240):
            parts.append(feats.table(samples.iloc[i:i + 240], WINDOW))
            print(f"  features {min(i + 240, len(samples))}/{len(samples)}", flush=True)
        pv = pd.concat([p for p, _ in parts], ignore_index=True)
        fr = pd.concat([f for _, f in parts], ignore_index=True)
        pv.to_parquet(out / "features_videos.parquet", index=False)
        fr.to_parquet(out / "features_frames.parquet", index=False)
    pv = pd.read_parquet(out / "features_videos.parquet")
    fr = pd.read_parquet(out / "features_frames.parquet")
    reference, _ = load(FEATURES, WINDOW)
    pv = with_calibration(pv, reference)
    print("scoring with the saved HVPF ...", flush=True)
    p, pred, zone, _ = Bundle(args.bundle).predict(pv, fr)
    c = pv[["sample_id", "content_id", "device", "claimed_device", "kind", "beta", "alpha", "crf", "sigma", "standard",
            "stabilized"]].assign(**{f"p_{k}": p[:, i] for i, k in enumerate(CLASSES)}, pred=pred, zone=zone,
                                  pce_pass=pv.w_pce0_claimed.to_numpy() > slog(60.0))
    c.to_parquet(out / "candidates.parquet", index=False)
    text = summary(c)
    (out / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
