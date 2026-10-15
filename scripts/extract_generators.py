"""Unpack a fixed random sample of clips per GenVidBench generator archive and report resolutions
(Phase 4). Only generators whose clips are at least `frames.crop` pixels on the short side can
be used without upscaling (upscaling only AI clips would give the detector an "upscaled = AI"
shortcut), so the report says which ones to add to `synthetic.sources`.

Needs 7-Zip (~/.local/bin/7zz). Archives: <synthetic root>/../genvidbench/<name>.rar; clips go to
<synthetic root>/<name>/<name>/*.mp4 (the archives' own folder layout).

Usage:
    python scripts/extract_generators.py ms svd mora t2vz pika vc2
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config

SEVEN_ZIP = str(Path("~/.local/bin/7zz").expanduser())


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    try:
        info = json.loads(out)
        s = info["streams"][0]
        return int(s["width"]), int(s["height"]), float(info["format"].get("duration", "nan"))
    except (KeyError, IndexError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="archive names without .rar")
    ap.add_argument("--clips", type=int, default=400, help="clips to unpack per generator")
    args = ap.parse_args()
    cfg = load_config()
    root = Path(cfg["synthetic"]["root"]).expanduser()
    archives = root.parent / "genvidbench"
    crop, seed = cfg["frames"]["crop"], cfg["splits"]["seed"]

    for name in args.names:
        archive = archives / f"{name}.rar"
        listing = subprocess.run([SEVEN_ZIP, "l", "-ba", "-slt", str(archive)], capture_output=True, text=True).stdout
        clips = sorted(line[7:] for line in listing.splitlines()
                       if line.startswith("Path = ") and line.lower().endswith(".mp4"))
        pick = sorted(np.random.default_rng(seed).choice(clips, min(args.clips, len(clips)), replace=False))
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        todo = [c for c in pick if not (dest / c).exists()]
        if todo:
            lst = dest.parent / f".{name}_extract_list.txt"
            lst.write_text("\n".join(todo) + "\n")
            subprocess.run([SEVEN_ZIP, "x", "-y", "-bso0", "-bsp0", f"-o{dest}", str(archive), f"@{lst}"], check=True)
            lst.unlink()
        info = [probe(dest / c) for c in pick]
        ok = [i for i in info if i]
        short = np.array([min(w, h) for w, h, _ in ok])
        sizes = {f"{w}x{h}": 0 for w, h, _ in ok}
        for w, h, _ in ok:
            sizes[f"{w}x{h}"] += 1
        top = ", ".join(f"{k} ({v})" for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:3])
        usable = (short >= crop).mean() if len(short) else 0
        dur = np.nanmedian([d for _, _, d in ok]) if ok else float("nan")
        print(f"{name:6s} {len(clips):6d} clips in archive, {len(pick)} unpacked, readable {len(ok)}; "
              f"sizes {top}; median {dur:.1f} s; >= {crop}px short side: {100 * usable:.0f}%", flush=True)


if __name__ == "__main__":
    main()
