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
