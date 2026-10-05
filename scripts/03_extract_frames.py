"""Step 3 - I-frame extraction.

For every reference/sample video in data/metadata/videos.csv, decode up to
`max_iframes` evenly spaced I-frames, align them to the sensor geometry
(see hvpf/frames.py), keep a center crop of the luma plane, and save
    data/frames/<dataset>/<device>/<content_id>__<version>.npy   uint8 (n, crop, crop)
plus data/metadata/frames_index.csv. Already-extracted videos are skipped, so
the script can be stopped and restarted.

Usage:
    python scripts/03_extract_frames.py                  # everything
    python scripts/03_extract_frames.py --devices D01 D04 --limit 10
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.datasets import resolve_paths
from hvpf.frames import build_filter, extract_iframes


def frame_path(frames_dir, row):
    return frames_dir / row.dataset / row.device / f"{row.content_id}__{row.version}.npy"


def plan(videos, crop, max_frames):
    """Attach target geometry + filter chain to each video."""
    native = videos[videos.version == "native"].set_index("content_id")
    jobs = []
    for row in videos.itertuples():
        if row.version == "native":
            nat_w, nat_h, rot = row.width, row.height, 0
        elif row.content_id in native.index:
            n = native.loc[row.content_id]
            nat_w, nat_h = n.width, n.height
            rot = int(row.rotation - n.rotation) % 360
        else:
            continue
        step = max(1, int(row.n_iframes) // max_frames)
        vf = build_filter(int(row.width), int(row.height), int(nat_w), int(nat_h), rot, crop, step)
        jobs.append((row, vf))
    return jobs


def run_one(job, frames_dir, crop, max_frames):
    row, vf = job
    out = frame_path(frames_dir, row)
    rec = {"content_id": row.content_id, "version": row.version, "device": row.device,
           "dataset": row.dataset, "frames_path": str(out), "filter": vf}
    if out.exists():
        rec.update(n_saved=int(np.load(out, mmap_mode="r").shape[0]), status="ok", error="")
        return rec
    try:
        frames = extract_iframes(row.path, vf, crop, max_frames)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, frames)
        tmp.rename(out)
        rec.update(n_saved=len(frames), status="ok", error="")
    except Exception as e:  # keep going; failures are listed in the index
        rec.update(n_saved=0, status="failed", error=str(e)[-300:])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()

    cfg = load_config()
    paths, fcfg = cfg["paths"], cfg["frames"]
    crop, max_frames = fcfg["crop"], fcfg["max_iframes"]
    videos = resolve_paths(pd.read_csv(paths["metadata_dir"] / "videos.csv"), paths)
    videos = videos[videos.role.isin(["reference", "sample"])]
    too_small = (videos.version == "native") & ((videos.width < crop) | (videos.height < crop))
    videos = videos[~too_small]   # their YT/WA copies drop out in plan() with them
    if args.devices:
        videos = videos[videos.device.isin(args.devices)]
    if args.datasets:
        videos = videos[videos.dataset.isin(args.datasets)]

    jobs = plan(videos, crop, max_frames)
    if args.limit:
        jobs = jobs[: args.limit]
    workers = args.workers or fcfg["workers"]
    with ThreadPoolExecutor(workers) as ex:
        recs = list(tqdm(ex.map(lambda j: run_one(j, paths["frames_dir"], crop, max_frames), jobs),
                         total=len(jobs), desc="I-frames"))

    index = pd.DataFrame(recs)
    index_path = paths["metadata_dir"] / "frames_index.csv"
    if index_path.exists():   # merge with earlier (partial) runs
        old = pd.read_csv(index_path)
        index = pd.concat([old, index]).drop_duplicates(["content_id", "version"], keep="last")
    index.to_csv(index_path, index=False)

    done = index[index.status == "ok"]
    print(f"\n{len(done)} videos extracted, {(index.status == 'failed').sum()} failed, "
          f"{done.n_saved.sum()} frames total, median {done.n_saved.median():.0f} per video")
    failed = index[index.status == "failed"]
    if len(failed):
        print(failed[["content_id", "version", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
