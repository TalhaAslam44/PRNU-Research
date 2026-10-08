import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    paths = {}
    for key, value in cfg["paths"].items():
        p = Path(value).expanduser()
        paths[key] = p if p.is_absolute() else ROOT / p
    cfg["paths"] = paths
    return cfg


def is_fresh(out, inputs):
    """True if cached file `out` was built from exactly `inputs` (recorded in `out`.inputs).

    Lets steps reuse expensive results (fingerprints, reference spectra) and rebuild them
    automatically when, e.g., the set of reference videos changes.
    """
    sig = hashlib.sha1("\n".join(sorted(map(str, inputs))).encode()).hexdigest()
    stamp = Path(str(out) + ".inputs")
    return Path(out).exists() and stamp.exists() and stamp.read_text().strip() == sig


def mark_fresh(out, inputs):
    sig = hashlib.sha1("\n".join(sorted(map(str, inputs))).encode()).hexdigest()
    Path(str(out) + ".inputs").write_text(sig + "\n")
