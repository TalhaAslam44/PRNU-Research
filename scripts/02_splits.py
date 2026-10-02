"""Step 2 - device-disjoint train/val/test splits.

* The split unit is the camera model, so no test camera (and no camera of the
  same model) is ever seen in training. VISION and ACID devices of the same
  phone model are grouped together.
* Every version of a recording (native / YouTube / WhatsApp / later attacked
  copies) inherits the split of its device, so versions never straddle splits.
* FloreView is held out entirely (split = "crossdataset").

Each video also gets a role:
    reference          native flat video (VISION) or first N videos (ACID) -> estimates K
    reference_derived  YT/WA copy of a reference video -> unused (would leak the reference)
    sample             everything else -> classifier / evaluation data
    unused             ACID videos beyond `acid.videos_per_device`
    excluded           broken file or VISION errata

A grouped k-fold assignment (column `fold`) is written too: with 35 VISION devices a
single 20% test split holds only ~5 cameras, so reporting mean +- std over folds is
the more reliable protocol for the thesis.

Writes data/metadata/model_splits.csv and data/metadata/videos.csv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config

SPLITS = ("train", "val", "test")


def norm_model(brand, model, aliases):
    key = re.sub(r"[^a-z0-9]", "", f"{brand}{model}".lower())
    return aliases.get(key, key)


def acid_models(acid_root):
    """ACID model folders (M18_Moto_E4) from tarballs or extracted folders."""
    names = set()
    for split in ("train", "eval"):
        for p in Path(acid_root, split).glob("M[0-9][0-9]_*"):
            names.add(p.name.removesuffix(".tar.gz"))
    rows = []
    for name in sorted(names):
        _, brand, *model = name.split("_")
        rows.append({"dataset": "ACID", "model_key": name, "brand": brand, "model": "_".join(model),
                     "device": name.split("_")[0], "stabilized": False})
    return pd.DataFrame(rows)


def deal(groups, fractions, seed):
    """Deal model groups into labelled bins, keeping each bin's device share near its
    target fraction within every stratum (dataset mix x stabilization).

    Groups are shuffled within each stratum, then each group goes to the bin with
    the largest deficit (ties go to the smallest bin). Returns a label per group.
    """
    rng = np.random.default_rng(seed)
    shuffled = groups.assign(_r=rng.random(len(groups))).sort_values(["stratum", "_r"])
    order = sorted(fractions, key=lambda s: fractions[s])
    label = {}
    for _, stratum in shuffled.groupby("stratum"):
        assigned = dict.fromkeys(fractions, 0)
        total = 0
        for idx, g in stratum.iterrows():
            total += g.weight
            pick = max(order, key=lambda s: fractions[s] * total - assigned[s])
            assigned[pick] += g.weight
            label[idx] = pick
    return pd.Series(label)


def main():
    cfg = load_config()
    paths, scfg = cfg["paths"], cfg["splits"]
    aliases = scfg.get("model_aliases") or {}
    inv = pd.read_csv(paths["metadata_dir"] / "inventory.csv")

    # one row per (dataset, device) for every dataset that takes part in the split
    vision_dev = (inv[inv.dataset == "VISION"]
                  .groupby("device").agg(brand=("brand", "first"), model=("model", "first"),
                                         stabilized=("stabilized", "first"))
                  .reset_index().assign(dataset="VISION"))
    devices = pd.concat([vision_dev, acid_models(paths["acid_root"])], ignore_index=True)
    devices["group"] = [norm_model(b, m, aliases) for b, m in zip(devices.brand, devices.model)]

    groups = devices.groupby("group").agg(
        datasets=("dataset", lambda d: "+".join(sorted(set(d)))),
        stabilized=("stabilized", "any"),
        devices=("device", lambda d: " ".join(sorted(d))),
        weight=("device", "size"),
    ).reset_index()
    groups["stratum"] = groups.datasets + "|" + groups.stabilized.map({True: "stab", False: "nostab"})
    groups["split"] = deal(groups, {s: scfg[s] for s in SPLITS}, scfg["seed"])
    # grouped k-fold alternative: fold f is test, fold f+1 is val, the rest train
    k = scfg["n_folds"]
    groups["fold"] = deal(groups, dict.fromkeys(range(k), 1 / k), scfg["seed"] + 1)
    groups.drop(columns="stratum").to_csv(paths["metadata_dir"] / "model_splits.csv", index=False)

    # per-video table
    inv["group"] = [norm_model(b, m, aliases) for b, m in zip(inv.brand, inv.model)]
    split_of = dict(zip(groups.group, groups.split))
    inv["split"] = inv.group.map(split_of).fillna("crossdataset")
    inv["fold"] = inv.group.map(dict(zip(groups.group, groups.fold))).astype("Int64")
    inv.loc[inv.dataset == "FloreView", "split"] = "crossdataset"

    role = np.where(inv.scene_group == "flat",
                    np.where(inv.version == "native", "reference", "reference_derived"), "sample")
    inv["role"] = role
    acid = inv.dataset == "ACID"
    n_ref, n_keep = scfg["acid_reference_videos"], cfg["acid"]["videos_per_device"]
    rank = inv[acid].sort_values("content_id").groupby("device").cumcount()
    inv.loc[rank.index, "role"] = np.select([rank < n_ref, rank < n_keep], ["reference", "sample"], "unused")
    inv.loc[~inv.ok.astype(bool) | inv.errata.astype(bool), "role"] = "excluded"
    inv.to_csv(paths["metadata_dir"] / "videos.csv", index=False)

    # leakage checks
    per_device = inv.groupby(["dataset", "device"]).split.nunique()
    per_content = inv.groupby("content_id").split.nunique()
    assert per_device.max() == 1, "a device spans several splits"
    assert per_content.max() == 1, "versions of one recording span several splits"
    for s in SPLITS:
        others = set(inv.loc[inv.split != s, "group"])
        assert not (set(inv.loc[inv.split == s, "group"]) & others), f"model leak into {s}"

    print("Model groups per split:")
    print(groups.groupby("split").agg(models=("group", "size"), devices=("weight", "sum"),
                                       stabilized_models=("stabilized", "sum")).to_string())
    print("\nVideos by split / role / version:")
    print(pd.crosstab([inv.split, inv.role], inv.version, margins=True).to_string())
    print("\nVISION devices per split:")
    for s in SPLITS:
        devs = sorted(inv.loc[(inv.split == s) & (inv.dataset == "VISION"), "device"].unique())
        print(f"  {s:5s} ({len(devs)}): {' '.join(devs)}")
    vis = inv[inv.dataset == "VISION"].drop_duplicates("device")
    print("\nVISION devices per fold (stabilized in brackets):")
    for f, d in vis.groupby("fold"):
        print(f"  fold {f}: " + " ".join(f"[{x}]" if st else x for x, st in zip(d.device, d.stabilized)))
    print("\nNo device, model, or recording crosses splits. Wrote model_splits.csv and videos.csv")


if __name__ == "__main__":
    main()
