"""List the FloreView videos that are not downloaded yet (in any of the given folders).

Reads data/metadata/floreview_video_urls.txt and writes data/metadata/floreview_missing_urls.txt
with the URLs whose file is missing. The most recently written file in each folder is checked
against the server's size, because an interrupted download leaves it incomplete.

Usage:
    python scripts/floreview_missing_urls.py /media/talhaaslam/Disk/datasets/FloreView ~/Data/datasets/FloreView
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config

PREFIX = "/FloreView/Dataset/"     # wget --cut-dirs=2 drops "FloreView/Dataset"


def remote_size(url):
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    sizes = [int(line.split(":")[1]) for line in out.lower().splitlines() if line.startswith("content-length:")]
    return sizes[-1] if sizes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="folders that already hold FloreView downloads")
    args = ap.parse_args()
    meta = load_config()["paths"]["metadata_dir"]
    urls = (meta / "floreview_video_urls.txt").read_text().split()
    folders = [Path(f).expanduser() for f in args.folders]

    incomplete = set()
    for folder in folders:
        videos = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".mp4", ".mov")]
        if videos:
            newest = max(videos, key=lambda p: p.stat().st_mtime)
            url = next(u for u in urls if u.endswith("/" + str(newest.relative_to(folder))))
            if remote_size(url) != newest.stat().st_size:
                incomplete.add(url)
                print(f"incomplete: {newest}")

    missing = [u for u in urls
               if u in incomplete or not any((f / u.split(PREFIX, 1)[1]).is_file() for f in folders)]
    (meta / "floreview_missing_urls.txt").write_text("\n".join(missing) + "\n")
    print(f"{len(urls) - len(missing)} of {len(urls)} videos present; {len(missing)} to download "
          f"-> {meta / 'floreview_missing_urls.txt'}")


if __name__ == "__main__":
    main()
