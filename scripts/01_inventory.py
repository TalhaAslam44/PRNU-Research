"""Step 1 - dataset inventory.

Probes every video with ffprobe and writes:
    data/metadata/inventory.csv          one row per video file
    data/metadata/inventory_devices.csv  one row per device (Chapter 5 table)
    data/metadata/inventory_broken.csv   files that failed to probe

Usage:
    python scripts/01_inventory.py                 # VISION + already-extracted ACID
    python scripts/01_inventory.py --limit 20      # quick smoke test
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.datasets import scan_acid_extracted, scan_vision
from hvpf.media import probe

COLUMNS = [
    "dataset", "device", "brand", "model", "model_key", "stabilized", "scene", "scene_group",
    "motion", "version", "content_id", "width", "height", "rotation", "codec", "profile",
    "pix_fmt", "fps", "n_frames", "n_iframes", "duration_s", "bitrate_kbps", "size_mb",
    "container", "errata", "ok", "error", "path",
]


def probe_all(rows, workers):
    with ThreadPoolExecutor(workers) as ex:
        results = list(tqdm(ex.map(lambda r: probe(r["path"]), rows), total=len(rows), desc="ffprobe"))
    return [{**r, **res} for r, res in zip(rows, results)]


def device_table(df):
    ok = df[df.ok & ~df.errata]
    g = ok.groupby(["dataset", "device"])
    table = g.agg(
        brand=("brand", "first"),
        model=("model", "first"),
        stabilized=("stabilized", "first"),
        n_videos=("path", "size"),
        n_native=("version", lambda v: (v == "native").sum()),
        n_youtube=("version", lambda v: (v == "youtube").sum()),
        n_whatsapp=("version", lambda v: (v == "whatsapp").sum()),
        n_flat=("scene_group", lambda v: (v == "flat").sum()),
        n_natural=("scene_group", lambda v: (v == "natural").sum()),
        codecs=("codec", lambda v: "/".join(sorted(set(v)))),
        total_duration_min=("duration_s", lambda v: round(v.sum() / 60, 1)),
    ).reset_index()
    native = ok[ok.version == "native"]
    res = native.groupby(["dataset", "device"]).apply(
        lambda d: "/".join(sorted({f"{w}x{h}" for w, h in zip(d.width, d.height)})), include_groups=False
    )
    fps = native.groupby(["dataset", "device"]).fps.median().round(2)
    table["native_resolution"] = table.set_index(["dataset", "device"]).index.map(res)
    table["native_fps"] = table.set_index(["dataset", "device"]).index.map(fps)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="probe only the first N files (smoke test)")
    ap.add_argument("--no-acid", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    out = paths["metadata_dir"]
    out.mkdir(parents=True, exist_ok=True)

    rows, unparsed = scan_vision(paths["vision_root"])
    print(f"VISION: {len(rows)} videos ({len(unparsed)} unparsed: {unparsed[:5]})")
    if not args.no_acid and paths["acid_root"].exists():
        acid = scan_acid_extracted(paths["acid_root"])
        print(f"ACID (extracted on disk): {len(acid)} videos")
        rows += acid
    acid_streamed = paths["metadata_dir"] / "inventory_acid.csv"
    if not args.no_acid and acid_streamed.exists():
        streamed = pd.read_csv(acid_streamed)
        print(f"ACID (streamed by 01b_acid_stage.py, already probed): {len(streamed)} videos")
    else:
        streamed = None
    if paths["floreview_root"].exists():
        print("FloreView folder found but its parser is not written yet - skipping")

    if args.limit:
        rows = rows[: args.limit]

    stabilized = set(cfg["vision"]["stabilized_devices"])
    errata = set(cfg["vision"]["errata"])
    rows = probe_all(rows, cfg["inventory"]["workers"])
    df = pd.DataFrame(rows)
    df["stabilized"] = (df.dataset == "VISION") & df.device.isin(stabilized)
    df["errata"] = df.path.map(lambda p: Path(p).name in errata)
    if streamed is not None and not args.limit:
        df = pd.concat([df, streamed]).drop_duplicates("content_id", keep="first")
    df = df.reindex(columns=COLUMNS)

    suffix = "_smoke" if args.limit else ""
    df.to_csv(out / f"inventory{suffix}.csv", index=False)
    df[~df.ok].to_csv(out / f"inventory_broken{suffix}.csv", index=False)
    table = device_table(df)
    table.to_csv(out / f"inventory_devices{suffix}.csv", index=False)

    print(f"\n{len(df)} videos probed, {(~df.ok).sum()} broken, {df.errata.sum()} errata")
    print(df.groupby(["dataset", "version"]).size().to_string())
    if (~df.ok).any():
        print("\nBroken files:")
        print(df.loc[~df.ok, ["path", "error"]].to_string(index=False))
    print(f"\nWrote {out}/inventory{suffix}.csv and inventory_devices{suffix}.csv")


if __name__ == "__main__":
    main()
