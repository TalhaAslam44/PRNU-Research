"""Filename parsers: one metadata dict per video file.

Common columns:
    dataset, device, brand, model, model_key, scene, scene_group, motion,
    version (native / youtube / whatsapp / attacked variants later), content_id, path

content_id ties every version of the same recording together (native, YT, WA,
and later the attacked copies) so they always land in the same split.
"""
import re
from pathlib import Path

from .media import VIDEO_EXTS

VISION_RE = re.compile(
    r"^(?P<device>D\d{2})_V_(?P<scene>flat|indoor|outdoor)(?P<social>WA|YT)?"
    r"_(?P<motion>still|move|panrot)_(?P<idx>\d{4})$"
)
VISION_VERSION = {None: "native", "YT": "youtube", "WA": "whatsapp"}
VISION_DEVICE_RE = re.compile(r"/dataset/(D\d{2})_([^_/]+)_([^/]+)/")


def vision_device_names(root):
    """D01 -> ('Samsung', 'GalaxyS3Mini'), read from the download list shipped with VISION."""
    names = {}
    for txt in ("VISION_video_files.txt", "VISION_files.txt"):
        f = Path(root) / txt
        if f.exists():
            for m in VISION_DEVICE_RE.finditer(f.read_text()):
                names[m.group(1)] = (m.group(2), m.group(3))
            break
    return names


def scan_vision(root):
    root = Path(root)
    names = vision_device_names(root)
    rows, unparsed = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        m = VISION_RE.match(p.name.split(".")[0])   # a few WA files are named *.mov.mp4
        if not m:
            unparsed.append(p.name)
            continue
        d = m.groupdict()
        brand, model = names.get(d["device"], ("unknown", "unknown"))
        rows.append({
            "dataset": "VISION",
            "device": d["device"],
            "brand": brand,
            "model": model,
            "model_key": f"{brand}_{model}",
            "scene": d["scene"],
            "scene_group": "flat" if d["scene"] == "flat" else "natural",
            "motion": d["motion"],
            "version": VISION_VERSION[d["social"]],
            "content_id": f"{d['device']}_{d['scene']}_{d['motion']}_{d['idx']}",
            "path": str(p),
        })
    return rows, unparsed


ACID_FILE_RE = re.compile(r"^(?P<mid>M\d{2})_D(?P<dev>[A-Z])_(?P<idx>\w+)$")


def parse_acid(path, split):
    """ACID layout: <train|eval>/M18_Moto_E4/DeviceA/M18_DA_T0000.mp4 (no flat videos)."""
    p = Path(path)
    m = ACID_FILE_RE.match(p.stem)
    if not m:
        return None
    model_dir = p.parent.parent.name            # M18_Moto_E4
    _, brand, *model = model_dir.split("_")
    return {
        "dataset": "ACID",
        "device": f"{m['mid']}_D{m['dev']}",
        "brand": brand,
        "model": "_".join(model),
        "model_key": model_dir,
        "scene": "natural",
        "scene_group": "natural",
        "motion": "unknown",
        "version": "native",
        "content_id": f"ACID_{split}_{p.stem}",
        "path": str(p),
    }


def scan_acid_extracted(root):
    """ACID models that are already extracted on disk (tarballs are handled by 01b_acid_stage.py)."""
    rows = []
    for split in ("train", "eval"):
        for p in sorted(Path(root, split).glob("M*/Device*/*")):
            if p.suffix.lower() in VIDEO_EXTS:
                row = parse_acid(p, split)
                if row:
                    rows.append(row)
    return rows


FLOREVIEW_DIR_RE = re.compile(r"^(?P<dev>D\d{2})_(?P<brand>[^_]+)_(?P<model>.+)$")


def scan_floreview(root):
    """FloreView layout: <Dxx_Brand_Model>/<Flat|Nat>/<jpeg-h264|heic-h265>/[L*/S*/]<file>.

    Some cameras recorded the same scene with both codec settings, so the codec folder is
    part of content_id. Devices are named FV_Dxx (VISION also uses D01..D35).
    """
    rows = []
    root = Path(root)
    for p in sorted(root.glob("D[0-9][0-9]_*/*/*/**/*")):
        if p.suffix.lower() not in VIDEO_EXTS or not p.is_file():
            continue
        dev_dir, kind, fmt = p.relative_to(root).parts[:3]
        m = FLOREVIEW_DIR_RE.match(dev_dir)
        if not m or kind not in ("Flat", "Nat"):
            continue
        scene = "flat" if kind == "Flat" else "natural"
        rows.append({
            "dataset": "FloreView",
            "device": f"FV_{m['dev']}",
            "brand": m["brand"],
            "model": m["model"],
            "model_key": dev_dir,
            "scene": scene,
            "scene_group": scene,
            "motion": "unknown",
            "version": "native",
            "content_id": f"FV_{m['dev']}_{fmt}_{p.stem}",
            "path": str(p),
        })
    return rows


def resolve_paths(df, paths):
    """Point VISION file paths at the configured `vision_root`, so the dataset can be
    moved (e.g. to another disk) by editing config.yaml instead of re-running Step 1."""
    df = df.copy()
    vision = df.dataset == "VISION"
    df.loc[vision, "path"] = [str(paths["vision_root"] / Path(p).name) for p in df.loc[vision, "path"]]
    return df
