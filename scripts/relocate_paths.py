"""Rewrite absolute paths in the metadata / feature tables after moving data to a new system.

The tables store absolute paths (video `path`, `frames_path`, `source_path`, ...). If the
home folder or the data disk's mount point changes, rewrite the old prefix to the new one:

    python scripts/relocate_paths.py /media/talhaaslam/Disk /mnt/data          # dry run
    python scripts/relocate_paths.py /media/talhaaslam/Disk /mnt/data --apply

Touches data/metadata/*.csv and data/features/*.{csv,parquet}; with --all every table under data/ (phase
backups, results, camera_id; frames and fingerprints are skipped). Keeps a .bak copy of each changed file.

    python scripts/relocate_paths.py /home/talhaaslam /home/i248067 --all --apply
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
    ap.add_argument("--all", action="store_true", help="every table under data/, not only metadata/ and features/")
    args = ap.parse_args()

    paths = load_config()["paths"]
    if args.all:
        dirs = [d for d in sorted(paths["data_dir"].iterdir())
                if d.is_dir() and not d.is_symlink() and d.name not in ("frames", "fingerprints", "staging")]
        files = sorted(f for d in dirs for ext in ("*.csv", "*.parquet") for f in d.rglob(ext))
    else:
        files = sorted(paths["metadata_dir"].glob("*.csv")) + sorted((paths["data_dir"] / "features").glob("*.csv")) \
            + sorted((paths["data_dir"] / "features").glob("*.parquet"))
    for f in files:
        try:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith(("str", "category"))]
        hits = {c: int(df[c].astype(str).str.startswith(args.old_prefix).sum()) for c in cols}
        hits = {c: n for c, n in hits.items() if n}
        if not hits:
            continue
        print(f"{f.relative_to(paths['data_dir'])}: {hits}")
        if args.apply and f.suffix == ".csv":      # plain text replacement: every other value stays byte-identical
            shutil.copy2(f, f.with_name(f.name + ".bak"))
            f.write_text(f.read_text().replace(args.old_prefix, args.new_prefix))
        elif args.apply:
            for c in hits:          # only the path text changes; empty cells stay empty
                col = df[c]
                if isinstance(col.dtype, pd.CategoricalDtype):
                    df[c] = col.cat.rename_categories(
                        [x.replace(args.old_prefix, args.new_prefix, 1) if str(x).startswith(args.old_prefix) else x
                         for x in col.cat.categories])
                else:
                    m = col.notna()
                    df.loc[m, c] = col[m].astype(str).str.replace(args.old_prefix, args.new_prefix, n=1, regex=False)
            shutil.copy2(f, f.with_name(f.name + ".bak"))
            df.to_parquet(f, index=False)
    if not args.apply:
        print("\ndry run - add --apply to write the changes")


if __name__ == "__main__":
    main()
