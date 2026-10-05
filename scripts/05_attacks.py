"""Step 5 - manipulated class: PRNU removal, PRNU injection, heavy denoising.

Targets are VISION natural native videos (role = sample). For each target the
first `clip_seconds` are decoded once (sensor orientation, analysis crop) and
re-encoded with libx264 (one I-frame per second) in these variants:

    reencode            alpha = 0 control, label organic (same encoder as the attacks)
    removal_a<alpha>    Y' = Y - alpha * Y * K_own      claimed device = own device
    removal_opt         white-box attacker: still subtracts its own K_own, but tunes alpha per
                        clip on the analyst's own measurement (I-frames after encoding, NCC with
                        the flat-field reference) by secant steps from the alpha grid, so the
                        PRNU correlation ends near 0 - the hardest case for the PCE baseline
    injection_a<alpha>  Y' = Y + alpha * Y * K_victim   claimed device = victim device
    denoise_a<s>        hqdn3d at s x default strength (desync-style PRNU suppression)

Attacker fingerprints never reuse the analyst's reference (flat videos):
K_own comes from the *other half* of the device's natural videos, and K_victim
from all natural videos of a non-stabilized victim of a different model in the
same cross-validation fold. Fingerprints are scaled to a 1% PRNU, so alpha = 1
means a 1% multiplicative pattern.

I-frames of every variant go to data/frames/ATTACK/<device>/<content_id>__<variant>.npy
and the index to data/metadata/attacks.csv. Finished variants are skipped on re-runs.

Usage:
    python scripts/05_attacks.py --devices D01 D27 --limit 2   # quick check
    python scripts/05_attacks.py
"""
import argparse
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.attacks import attacker_fingerprint, encode, prnu_frames, read_clip
from hvpf.config import load_config
from hvpf.datasets import resolve_paths
from hvpf.frames import extract_iframes
from hvpf.prnu_utils import fingerprint


def load_samples(paths):
    videos = resolve_paths(pd.read_csv(paths["metadata_dir"] / "videos.csv"), paths)
    frames = pd.read_csv(paths["metadata_dir"] / "frames_index.csv")
    frames = frames[frames.status == "ok"][["content_id", "version", "frames_path"]]
    nat = videos[(videos.dataset == "VISION") & (videos.role == "sample") & (videos.version == "native")]
    return nat.merge(frames, on=["content_id", "version"]).sort_values(["device", "content_id"]).reset_index(drop=True)


def attacker_fingerprints(samples, out_dir, workers):
    """Per device: K from each half of its natural videos (even / odd rank) and from all of them."""
    samples = samples.assign(half=samples.groupby("device").cumcount() % 2)
    ks = {}
    for device, g in tqdm(samples.groupby("device"), desc="attacker K"):
        for name, part in (("h0", g[g.half == 0]), ("h1", g[g.half == 1]), ("all", g)):
            out = out_dir / f"{device}_{name}.npy"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                frames = np.concatenate([np.load(p) for p in part.frames_path])
                np.save(out, fingerprint(frames, processes=workers))
            ks[device, name] = attacker_fingerprint(np.load(out))
    return samples, ks


def pick_victim(target, devices):
    """Non-stabilized device of another model in the same fold (same split preferred), fixed per target."""
    other = devices[(devices.group != target.group) & ~devices.stabilized]
    cand = other[other.fold == target.fold]
    cand = cand if len(cand) else other
    same_split = cand[cand.split == target.split]
    cand = same_split if len(same_split) else cand
    rng = np.random.default_rng(zlib.crc32(target.content_id.encode()))
    return cand.iloc[rng.integers(len(cand))]


def variants(acfg):
    """(name, attack, alpha); alpha None = chosen per clip by the attacker (removal_opt)."""
    out = [("reencode", "reencode", 0.0)]
    for attack in acfg["types"]:
        grid = acfg["denoise_strengths"] if attack == "denoise" else acfg["alphas"]
        out += [(f"{attack}_a{a:g}", attack, float(a)) for a in grid]
        if attack == "removal" and acfg["adaptive_removal"]:
            out.append(("removal_opt", "removal", None))
    return out


def zero_shift_ncc(frames, k_ref):
    """Normalized correlation at zero shift between the clip's fingerprint and a reference."""
    a = fingerprint(frames)
    a, b = a - a.mean(), k_ref - k_ref.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def secant_alpha(measured):
    """Next alpha towards NCC = 0, from (alpha, ncc) points measured so far."""
    pts = sorted(measured.items())
    bracket = [(p, q) for p, q in zip(pts, pts[1:]) if p[1] * q[1] <= 0]
    (a1, n1), (a2, n2) = bracket[0] if bracket else sorted(pts, key=lambda p: abs(p[1]))[:2]
    return float(np.clip(a1 - n1 * (a2 - a1) / (n2 - n1), 0.0, 2.0)) if n2 != n1 else a1


def run_target(t, victim, ks, ref_dir, acfg, crop, max_frames, frames_dir, tmp_dir):
    clip = None
    fps = float(t.fps) if t.fps and t.fps > 0 else 30.0
    k_own = ks[t.device, f"h{1 - t.half}"]
    k_ref = np.load(ref_dir / f"{t.device}.npy")
    measured = {}   # removal alpha -> NCC of the attacked I-frames with the analyst's reference

    def render(name, y, vf=None):
        tmp = tmp_dir / f"{t.content_id}__{name}.mp4"
        encode(y, clip[1], tmp, crop, fps, acfg["crf"], acfg["preset"], max(1, round(fps)), vf)
        frames = extract_iframes(tmp, "format=gray", crop, max_frames)
        if acfg["keep_videos"]:
            tmp.rename(frames_dir / "ATTACK" / t.device / tmp.name)
        else:
            tmp.unlink()
        return frames

    rows = []
    for name, attack, alpha in variants(acfg):
        out = frames_dir / "ATTACK" / t.device / f"{t.content_id}__{name}.npy"
        alpha_file = out.with_suffix(".alpha")         # alpha picked by the white-box attacker
        row = {"content_id": t.content_id, "variant": name, "attack": attack, "alpha": alpha,
               "label": "organic" if attack == "reencode" else "manipulated",
               "device": t.device, "claimed_device": victim.device if attack == "injection" else t.device,
               "group": t.group, "split": t.split, "fold": t.fold, "stabilized": t.stabilized,
               "victim_split": victim.split if attack == "injection" else t.split, "frames_path": str(out)}
        try:
            if out.exists():
                frames = np.load(out)
                if alpha is None:
                    alpha = row["alpha"] = float(alpha_file.read_text())
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                if clip is None:
                    clip = read_clip(t.path, crop, acfg["clip_seconds"])
                y = clip[0]
                if attack == "removal" and alpha is None:
                    # white-box attacker: tunes alpha on the analyst's own measurement (I-frames,
                    # NCC with the flat-field reference) until the PRNU correlation is ~0
                    for _ in range(acfg["white_box_steps"]):
                        alpha = secant_alpha(measured)
                        frames = render(name, prnu_frames(y, k_own, alpha, -1))
                        measured[alpha] = zero_shift_ncc(frames, k_ref)
                    row["alpha"] = round(alpha, 4)
                    alpha_file.write_text(f"{alpha:.6f}")
                elif attack == "removal":
                    frames = render(name, prnu_frames(y, k_own, alpha, -1))
                elif attack == "injection":
                    frames = render(name, prnu_frames(y, ks[victim.device, "all"], alpha, +1))
                elif attack == "denoise":
                    frames = render(name, y, f"hqdn3d={4 * alpha:g}:3:{6 * alpha:g}:4.5")
                else:
                    frames = render(name, y)
                np.save(out.with_suffix(".tmp.npy"), frames)
                out.with_suffix(".tmp.npy").rename(out)
            if attack in ("reencode", "removal") and alpha not in measured and acfg["adaptive_removal"]:
                measured[alpha] = zero_shift_ncc(frames, k_ref)
            rows.append({**row, "n_saved": len(frames), "status": "ok", "error": ""})
        except Exception as e:  # keep going; failures are listed in attacks.csv
            rows.append({**row, "n_saved": 0, "status": "failed", "error": str(e)[-300:]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*")
    ap.add_argument("--limit", type=int, help="targets per device")
    args = ap.parse_args()

    cfg = load_config()
    paths, acfg, fcfg = cfg["paths"], cfg["attacks"], cfg["frames"]
    samples = load_samples(paths)
    samples, ks = attacker_fingerprints(samples, paths["fingerprints_dir"] / "attacker", fcfg["workers"])

    devices = samples.drop_duplicates("device")[["device", "group", "stabilized", "fold", "split"]]
    targets = samples.groupby("device").head(args.limit or acfg["videos_per_device"])
    if args.devices:
        targets = targets[targets.device.isin(args.devices)]

    tmp_dir = paths["data_dir"] / "staging" / "attacks"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(t, pick_victim(t, devices)) for t in targets.itertuples()]
    ref_dir = paths["fingerprints_dir"] / "VISION"
    with ThreadPoolExecutor(acfg["workers"]) as ex:
        results = list(tqdm(ex.map(lambda j: run_target(*j, ks, ref_dir, acfg, fcfg["crop"], fcfg["max_iframes"],
                                                        paths["frames_dir"], tmp_dir), jobs),
                            total=len(jobs), desc="attacks"))

    index = pd.DataFrame([r for rows in results for r in rows])
    index_path = paths["metadata_dir"] / "attacks.csv"
    if index_path.exists():   # merge with earlier (partial) runs
        index = pd.concat([pd.read_csv(index_path), index]).drop_duplicates(["content_id", "variant"], keep="last")
    index.to_csv(index_path, index=False)

    ok = index[index.status == "ok"]
    print(f"\n{len(ok)} clips ok, {(index.status == 'failed').sum()} failed, {ok.n_saved.sum()} frames")
    print(ok.groupby("variant").size().to_string())
    failed = index[index.status == "failed"]
    if len(failed):
        print(failed[["content_id", "variant", "error"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
