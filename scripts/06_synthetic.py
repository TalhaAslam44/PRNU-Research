"""Step 6 - synthetic class: AI-generated videos with a randomly claimed camera.

Sources (config `synthetic.sources`) are folders of generated videos. For each
generator up to `videos_per_generator` clips are sampled, probed (min side must
fit the analysis crop) and passed through the *same* pipeline as the Step 5
attacks: first `clip_seconds`, center crop, libx264 CRF/preset from `attacks`,
one I-frame per second. Every class is then analysed on x264 I-frames with the
same spacing, so frame type and encoder cannot give the class away. Native
generator files are useless as they are: e.g. Kling T2V clips hold one I-frame.

Each synthetic clip claims a camera so the PRNU features can be computed:
* `sample` sources: a random fold, then a random device of that fold (VISION or
  ACID devices with a reference fingerprint); split = that device's split.
* `crossdataset` sources (GenBuster): split "crossdataset", any device.

Phase 2: every clip also gets WhatsApp-/YouTube-like copies (config synthetic.social,
parameters in attacks.social), content_id `<id>+wa` / `<id>+yt`, and Khan-style PRNU-transfer
copies (synthetic.inject_alphas): the claimed camera's fingerprint, estimated by the attacker
from that camera's natural videos, injected into the AI clip (`<id>+inj<alpha>`).

Frames go to data/frames/SYNTH/<generator>/<content_id>.npy and the index to
data/metadata/synthetic.csv. Finished clips are skipped on re-runs.

Usage:
    python scripts/06_synthetic.py --generators Sora --limit 3   # quick check
    python scripts/06_synthetic.py
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
from hvpf.attacks import attacker_fingerprint, encode, prnu_frames, read_clip, social_filter, social_reencode
from hvpf.config import load_config
from hvpf.frames import extract_iframes
from hvpf.media import VIDEO_EXTS, probe
from hvpf.prnu_utils import fingerprint


def scan_sources(scfg):
    rows = []
    root = Path(scfg["root"]).expanduser()
    for src in scfg["sources"]:
        base = root / src["dir"]
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in VIDEO_EXTS or p.name.startswith("._"):
                continue
            generator = p.relative_to(base).parts[0] if src["generator"] == "from_subdir" else src["generator"]
            rows.append({"dataset": src["dataset"], "generator": generator, "role": src["role"], "path": str(p)})
    return pd.DataFrame(rows)


def sample_per_generator(files, scfg, seed):
    out = []
    for (role, gen), g in files.groupby(["role", "generator"]):
        n = scfg["videos_per_generator"] if role == "sample" else scfg["crossdataset_videos_per_generator"]
        out.append(g.sample(min(n, len(g)), random_state=seed))
    df = pd.concat(out).reset_index(drop=True)
    df["content_id"] = [f"{d}_{g}_{zlib.crc32(p.encode()):08x}" for d, g, p in zip(df.dataset, df.generator, df.path)]
    return df


def claim_devices(df, devices, n_folds, seed):
    """Random fold per clip (balanced per generator), then a random device of that fold."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    folds, claimed, splits = [], [], []
    for _, g in df.groupby("generator", sort=False):
        f = np.resize(np.arange(n_folds), len(g))
        rng.shuffle(f)
        folds += list(zip(g.index, f))
    fold_of = dict(folds)
    for i, r in df.iterrows():
        if r.role == "crossdataset":
            d = devices.iloc[rng.integers(len(devices))]
            claimed.append(d.device); splits.append("crossdataset")
            fold_of[i] = pd.NA
        else:
            cand = devices[devices.fold == fold_of[i]]
            d = cand.iloc[rng.integers(len(cand))]
            claimed.append(d.device); splits.append(d.split)
    df["fold"] = pd.array([fold_of[i] for i in df.index], dtype="Int64")
    df["claimed_device"], df["split"] = claimed, splits
    return df


def attacker_keys(videos, frames, devices, out_dir, workers, per_device=30):
    """Attacker's fingerprint of every claimable camera (Khan-style PRNU transfer): from the
    camera's natural sample videos, never from the analyst's reference videos. VISION cameras
    reuse Step 5's `<dev>_all.npy`; ACID cameras get one from up to `per_device` sample videos."""
    ok = frames.loc[frames.status == "ok", ["content_id", "version", "frames_path"]]
    smp = videos[(videos.role == "sample") & (videos.version == "native")].merge(ok, on=["content_id", "version"])
    ks = {}
    for dev in tqdm(devices.device, desc="attacker K"):
        out = out_dir / f"{dev}_all.npy"
        if not out.exists():
            paths = smp[smp.device == dev].sort_values("content_id").frames_path.head(per_device)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, fingerprint(np.concatenate([np.load(p) for p in paths]), processes=workers))
        ks[dev] = attacker_fingerprint(np.load(out))
    return ks


def injection_victims(df, devices, videos, cfg):
    """Camera each injected copy impersonates: the clip's claimed camera if its video PRNU is
    usable, else a usable camera of the same fold (same split preferred). An attacker copies a
    fingerprint that exists; stabilised VISION cameras and PRNU-weak ACID cameras have none."""
    stab = set(videos[videos.stabilized.astype(bool)].device)
    usable = devices[~devices.device.isin(stab | set(cfg["acid"]["prnu_weak_devices"]))]
    out = []
    for r in df.itertuples():
        if r.claimed_device in set(usable.device):
            out.append(r.claimed_device)
            continue
        cand = usable if pd.isna(r.fold) else usable[usable.fold == r.fold]
        same = cand[cand.split == r.split]
        cand = same if len(same) else cand
        rng = np.random.default_rng(zlib.crc32(r.content_id.encode()))
        out.append(cand.iloc[rng.integers(len(cand))].device)
    return out


def run_clip(r, acfg, scfg, crop, max_frames, frames_dir, tmp_dir, ks):
    """The x264 base clip plus its platform-like copies (scfg.social); returns one row each."""
    base_out = frames_dir / "SYNTH" / r.generator / f"{r.content_id}.npy"
    common = {"dataset": r.dataset, "generator": r.generator, "label": "synthetic", "base_content_id": r.content_id,
              "claimed_device": r.claimed_device, "split": r.split, "fold": r.fold, "source_path": r.path}
    jobs = [(r.content_id, None, base_out, None)] + [
        (f"{r.content_id}+{pf}", pf, base_out.with_name(f"{r.content_id}+{pf}.npy"), None) for pf in scfg["social"]] + [
        (f"{r.content_id}+inj{a:g}", None, base_out.with_name(f"{r.content_id}+inj{a:g}.npy"), float(a))
        for a in scfg.get("inject_alphas", [])]
    rows = []
    if all(out.exists() for _, _, out, _ in jobs):
        return [{**common, "content_id": cid, "social": pf or "none", "inj_alpha": inj, "frames_path": str(out),
                 **({"claimed_device": r.inj_victim} if inj is not None else {}),
                 "n_saved": int(np.load(out, mmap_mode="r").shape[0]), "status": "ok", "error": ""}
                for cid, pf, out, inj in jobs]
    info = probe(r.path)
    common.update({k: info.get(k) for k in ("width", "height", "fps", "duration_s", "codec")})
    tmp = tmp_dir / f"{r.content_id}.mp4"
    try:
        if not info["ok"]:
            raise RuntimeError(info["error"])
        if min(info["width"], info["height"]) < crop:
            raise RuntimeError(f"smaller than crop {crop}")
        y, uv = read_clip(r.path, crop, scfg["clip_seconds"])
        fps = info["fps"] if info["fps"] and info["fps"] > 0 else 30.0
        gop = max(1, round(fps))
        encode(y, uv, tmp, crop, fps, acfg["crf"], acfg["preset"], gop)
        for cid, pf, out, inj in jobs:
            row = {**common, "content_id": cid, "social": pf or "none", "inj_alpha": inj, "frames_path": str(out)}
            if inj is not None:
                row["claimed_device"] = r.inj_victim
            try:
                if out.exists():
                    frames = np.load(out)
                else:
                    if inj is not None:
                        # Khan-style transfer: the claimed camera's PRNU injected into the AI clip
                        itmp = tmp_dir / f"{cid}.mp4"
                        encode(prnu_frames(y, ks[r.inj_victim], inj, +1), uv, itmp, crop, fps,
                               acfg["crf"], acfg["preset"], gop)
                        frames = extract_iframes(itmp, "format=gray", crop, max_frames)
                        itmp.unlink()
                    elif pf is None:
                        frames = extract_iframes(tmp, "format=gray", crop, max_frames)
                    else:
                        stmp = tmp_dir / f"{cid}.mp4"
                        size = social_reencode(tmp, stmp, acfg["social"][pf], min(info["width"], info["height"]),
                                               fps, crop, gop)
                        frames = extract_iframes(stmp, social_filter(size, crop), crop, max_frames)
                        stmp.unlink()
                    if len(frames) < scfg["min_frames"]:
                        raise RuntimeError(f"only {len(frames)} I-frame(s): clip too short")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out.with_suffix(".tmp.npy"), frames)
                    out.with_suffix(".tmp.npy").rename(out)
                rows.append({**row, "n_saved": len(frames), "status": "ok", "error": ""})
            except Exception as e:
                rows.append({**row, "n_saved": 0, "status": "failed", "error": str(e)[-300:]})
    except Exception as e:  # keep going; failures are listed in synthetic.csv
        rows = [{**common, "content_id": cid, "social": pf or "none", "inj_alpha": inj, "frames_path": str(out),
                 **({"claimed_device": r.inj_victim} if inj is not None else {}), "n_saved": 0, "status": "failed", "error": str(e)[-300:]} for cid, pf, out, inj in jobs]
    tmp.unlink(missing_ok=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="*")
    ap.add_argument("--limit", type=int, help="clips per generator")
    args = ap.parse_args()

    cfg = load_config()
    paths, scfg, acfg, fcfg = cfg["paths"], cfg["synthetic"], cfg["attacks"], cfg["frames"]
    seed = cfg["splits"]["seed"]

    # devices a synthetic clip may claim: those with a reference fingerprint
    videos = pd.read_csv(paths["metadata_dir"] / "videos.csv")
    frames = pd.read_csv(paths["metadata_dir"] / "frames_index.csv")
    ok_frames = frames.loc[frames.status == "ok", ["content_id", "version"]]
    refs = videos[videos.role == "reference"].merge(ok_frames, on=["content_id", "version"])
    devices = refs.drop_duplicates("device")[["dataset", "device", "split", "fold"]].dropna(subset=["fold"])

    files = scan_sources(scfg)
    print(files.groupby(["dataset", "generator"]).size().to_string())
    df = claim_devices(sample_per_generator(files, scfg, seed), devices, cfg["splits"]["n_folds"], seed)
    ks = attacker_keys(videos, frames, devices, paths["fingerprints_dir"] / "attacker", fcfg["workers"]) \
        if scfg.get("inject_alphas") else {}
    df["inj_victim"] = injection_victims(df, devices, videos, cfg)
    if args.generators:
        df = df[df.generator.isin(args.generators)]
    if args.limit:
        df = df.groupby("generator").head(args.limit)

    tmp_dir = paths["data_dir"] / "staging" / "synthetic"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(acfg["workers"]) as ex:
        rows = list(tqdm(ex.map(lambda r: run_clip(r, acfg, scfg, fcfg["crop"], fcfg["max_iframes"],
                                                   paths["frames_dir"], tmp_dir, ks), df.itertuples()),
                         total=len(df), desc="synthetic"))

    index = pd.DataFrame([row for clip_rows in rows for row in clip_rows])
    index_path = paths["metadata_dir"] / "synthetic.csv"
    if index_path.exists():   # merge with earlier (partial) runs
        index = pd.concat([pd.read_csv(index_path), index]).drop_duplicates("content_id", keep="last")
    index.to_csv(index_path, index=False)
    ok = index[index.status == "ok"]
    print(f"\n{len(ok)} clips ok, {(index.status == 'failed').sum()} failed")
    print(ok.groupby(["dataset", "generator", "social", "split"]).agg(n=("content_id", "size"), frames=("n_saved", "median")).to_string())
    failed = index[index.status == "failed"]
    if len(failed):
        print(failed.groupby("generator").error.agg(lambda e: e.str[:80].value_counts().index[0]).to_string())


if __name__ == "__main__":
    main()
