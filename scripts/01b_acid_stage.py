"""Step 1b - Video-ACID: stream each model archive without unpacking it.

ACID ships as one tar.gz per camera model (~177 GB in total), more than the free
disk space, so for every archive this script
    1. streams it and keeps the first `videos_per_device` videos of each device,
    2. probes them (same columns as Step 1) and extracts their I-frames (same as Step 3),
    3. deletes the staged videos and marks the archive as done.
Results are appended to data/metadata/inventory_acid.csv (merged by 01_inventory.py)
and frames land in data/frames/ACID/<device>/. Re-running skips finished archives and
redoes an interrupted one from scratch, so the script can be stopped at any time.

Run 01_inventory.py -> 02_splits.py -> 03_extract_frames.py afterwards so ACID gets
splits/roles and its frames are registered in frames_index.csv.

Usage:
    python scripts/01b_acid_stage.py                     # all archives
    python scripts/01b_acid_stage.py --models M18 M05    # selected models
"""
import argparse
import shutil
import sys
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.datasets import ACID_FILE_RE, parse_acid
from hvpf.frames import build_filter, extract_iframes
from hvpf.media import VIDEO_EXTS, probe


def stage_archive(tar_path, dest, per_device):
    """Copy the first `per_device` videos of each device out of a streamed tar.gz."""
    counts, staged = Counter(), []
    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            p = Path(member.name)
            if not member.isfile() or p.suffix.lower() not in VIDEO_EXTS:
                continue
            m = ACID_FILE_RE.match(p.stem)
            if not m:
                continue
            device = f"{m['mid']}_D{m['dev']}"
            if counts[device] >= per_device:
                continue
            counts[device] += 1
            out = dest / member.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            staged.append(out)
    return staged


def process_video(path, split, frames_dir, crop, max_frames):
    row = parse_acid(path, split)
    row.update(probe(path))
    row.update(stabilized=False, errata=False)
    if not row["ok"]:
        return row
    w, h = int(row["width"]), int(row["height"])
    if min(w, h) < crop:
        row.update(ok=False, error=f"smaller than crop {crop}")
        return row
    out = frames_dir / "ACID" / row["device"] / f"{row['content_id']}__native.npy"
    if not out.exists():
        step = max(1, int(row["n_iframes"]) // max_frames)
        try:
            frames = extract_iframes(path, build_filter(w, h, w, h, 0, crop, step), crop, max_frames)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.npy")
            np.save(tmp, frames)
            tmp.rename(out)
        except RuntimeError as e:
            row.update(ok=False, error=str(e)[-300:])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="model ids, e.g. M18 M05")
    args = ap.parse_args()

    cfg = load_config()
    paths, fcfg, acfg = cfg["paths"], cfg["frames"], cfg["acid"]
    staging = paths["data_dir"] / "staging" / "acid"
    done_dir = paths["metadata_dir"] / "acid_done"
    done_dir.mkdir(parents=True, exist_ok=True)
    inv_path = paths["metadata_dir"] / "inventory_acid.csv"

    archives = sorted(Path(paths["acid_root"]).glob("*/M[0-9][0-9]_*.tar.gz"))
    if args.models:
        archives = [a for a in archives if a.name[:3] in args.models]
    for tar_path in archives:
        split = tar_path.parent.name                       # train / eval
        model_dir = tar_path.name.removesuffix(".tar.gz")
        marker = done_dir / f"{split}_{model_dir}"
        if marker.exists() or (tar_path.parent / model_dir).is_dir():
            continue                                       # finished, or already unpacked (Step 1 covers it)
        per_device = acfg["videos_per_device"] if split == "train" else acfg["eval_videos_per_device"]
        if per_device == 0:
            continue
        dest = staging / split
        # an unfinished archive (e.g. PC shut down mid-run) may have left partial frame files: redo it
        for d in (paths["frames_dir"] / "ACID").glob(f"{model_dir[:3]}_D*"):
            shutil.rmtree(d)
        shutil.rmtree(dest / model_dir, ignore_errors=True)
        print(f"\n{split}/{model_dir}: streaming {tar_path.stat().st_size / 1e9:.1f} GB")
        try:
            staged = stage_archive(tar_path, dest, per_device)
        except (tarfile.TarError, OSError, EOFError) as e:
            print(f"  archive error, skipped: {e}")
            shutil.rmtree(dest / model_dir, ignore_errors=True)
            continue
        if not staged:
            print(f"  no recognised videos in {tar_path.name} - check hvpf/media.py VIDEO_EXTS; not marked done")
            continue
        with ThreadPoolExecutor(fcfg["workers"]) as ex:
            rows = list(tqdm(ex.map(lambda p: process_video(p, split, paths["frames_dir"], fcfg["crop"],
                                                            fcfg["max_iframes"]), staged),
                             total=len(staged), desc="  probe + I-frames"))
        df = pd.DataFrame(rows)
        df.to_csv(inv_path, mode="a", header=not inv_path.exists(), index=False)
        shutil.rmtree(dest / model_dir, ignore_errors=True)
        marker.touch()
        print(f"  kept {len(df)} videos from {df.device.nunique()} device(s), {(~df.ok).sum()} failed")


if __name__ == "__main__":
    main()
