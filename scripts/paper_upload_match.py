"""Paper / thesis - give the received WhatsApp / YouTube copies their original names (V001.mp4 ... V150.mp4).

WhatsApp renames every file it delivers (VID-20261117-WA0001.mp4 ...), and the order can get mixed up, so renaming
150 files by hand is error-prone. This script matches every received file to its original by its picture content:
    1. all files are decoded at 2 frames per second, scaled to 480 x 480, luma
    2. distance = mean squared error of 8 x 8 block means (60 x 60 per frame) against every original; this
       also separates the clips that show the same scene (a genuine clip and its attacked version): the attack's
       noise survives the platform at this scale, while full-resolution detail is dominated by the re-encoding
       (checked on simulated WhatsApp / YouTube copies: 92/92 same-scene clips right on their own, 150/150 overall)
    3. one-to-one assignment (Hungarian algorithm) over all copies
The received files are COPIED (never moved or changed) to data/paper/real_upload/<platform>/V###.<ext>; a report
data/paper/real_upload/match_<platform>.csv lists every match with its margin over the second-best original.

    .venv/bin/python scripts/paper_upload_match.py --platform whatsapp --incoming ~/Downloads/whatsapp_received
    .venv/bin/python scripts/paper_upload_match.py --platform youtube  --incoming ~/Downloads/youtube_received
"""
import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.paper_bundle import PAPER

UP = PAPER / "real_upload"
EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".3gp")
SIZE, FPS, MAXF, SMALL = 480, 2, 24, 60


def frames(path):
    """Luma frames at FPS frames per second, SIZE x SIZE (n, SIZE, SIZE) float32."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          f"fps={FPS},scale={SIZE}:{SIZE}:flags=bicubic,format=gray", "-frames:v", str(MAXF),
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, SIZE, SIZE).astype(np.float32)


def small(f):
    k = SIZE // SMALL
    return f.reshape(len(f), SMALL, k, SMALL, k).mean(axis=(2, 4))


def mse(a, b):
    n = min(len(a), len(b))
    return float(((a[:n] - b[:n]) ** 2).mean()) + 50.0 * abs(len(a) - len(b)) / max(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["whatsapp", "youtube"], required=True)
    ap.add_argument("--incoming", type=Path, required=True, help="folder with the received files (any names)")
    ap.add_argument("--originals", type=Path, default=UP / "originals")
    ap.add_argument("--out", type=Path, help="default data/paper/real_upload/<platform>")
    args = ap.parse_args()
    out_dir = args.out or UP / args.platform

    origs = sorted(p for p in args.originals.iterdir() if p.suffix.lower() in EXTS)
    copies = sorted(p for p in args.incoming.expanduser().iterdir() if p.suffix.lower() in EXTS)
    print(f"{len(copies)} received files, {len(origs)} originals; decoding ...", flush=True)
    with ThreadPoolExecutor(12) as ex:
        fo = list(ex.map(frames, origs))
        fc = list(ex.map(frames, copies))
    bad = [c.name for c, f in zip(copies, fc) if len(f) == 0]
    if bad:
        print(f"could not decode (skipped): {bad}")
    so, sc = [small(f) for f in fo], [small(f) for f in fc]

    cost = np.full((len(copies), len(origs)), 1e9)
    for i, f in enumerate(sc):
        if len(f):
            cost[i] = [mse(f, o) for o in so]
    rows_i, cols_j = linear_sum_assignment(cost)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for i, j in zip(rows_i, cols_j):
        if cost[i, j] >= 1e9:
            continue
        others = np.delete(cost[i], j)
        second = others.min()
        margin = second / max(cost[i, j], 1e-6)
        target = out_dir / f"{origs[j].stem}{copies[i].suffix.lower()}"
        status = "exists - not overwritten" if target.exists() else "copied"
        if not target.exists():
            shutil.copy2(copies[i], target)
        report.append(dict(received=copies[i].name, original=origs[j].stem, mse=round(cost[i, j], 2),
                           second_best=round(second, 2) if second < 1e9 else None,
                           margin=round(margin, 3) if second < 1e9 else None, status=status))
    rep = pd.DataFrame(report).sort_values("original")
    rep["check"] = np.where(rep.margin.fillna(99) < 1.02, "CHECK BY EYE", "")
    path = UP / f"match_{args.platform}.csv"
    rep.to_csv(path, index=False)
    unmatched = sorted(set(o.stem for o in origs) - set(rep.original))
    print(rep.to_string(index=False))
    print(f"\n{len(rep)} matched and copied to {out_dir}; {int((rep.check != '').sum())} close calls (margin < 1.02); "
          f"{len(unmatched)} originals without a copy{': ' + ', '.join(unmatched) if unmatched else ''}")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
