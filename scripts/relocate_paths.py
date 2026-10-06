"""Rewrite absolute paths in the metadata / feature tables after moving data to a new system.

The tables store absolute paths (video `path`, `frames_path`, `source_path`, ...). If the
home folder or the data disk's mount point changes, rewrite the old prefix to the new one:

    python scripts/relocate_paths.py /media/talhaaslam/Disk /mnt/data          # dry run
    python scripts/relocate_paths.py /media/talhaaslam/Disk /mnt/data --apply

Touches data/metadata/*.csv and data/features/*.{csv,parquet}; keeps a .bak copy of each changed file.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_prefix")
    ap.add_argument("new_prefix")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = ap.parse_args()

    paths = load_config()["paths"]
    files = sorted(paths["metadata_dir"].glob("*.csv")) + sorted((paths["data_dir"] / "features").glob("*.csv")) \
        + sorted((paths["data_dir"] / "features").glob("*.parquet"))
    for f in files:
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
        hits = {c: int(df[c].astype(str).str.startswith(args.old_prefix).sum()) for c in cols}
        hits = {c: n for c, n in hits.items() if n}
        if not hits:
            continue
        print(f"{f.relative_to(paths['data_dir'])}: {hits}")
        if args.apply:
            for c in hits:
                df[c] = df[c].astype(str).str.replace(args.old_prefix, args.new_prefix, n=1, regex=False)
            shutil.copy2(f, f.with_name(f.name + ".bak"))
            df.to_parquet(f, index=False) if f.suffix == ".parquet" else df.to_csv(f, index=False)
    if not args.apply:
        print("\ndry run - add --apply to write the changes")


if __name__ == "__main__":
    main()
