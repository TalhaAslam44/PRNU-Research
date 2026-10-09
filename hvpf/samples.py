"""The analysis sample table shared by Step 7 (PRNU/FFT/TSNCS) and Step 7b (Noiseprint)."""
import numpy as np
import pandas as pd

SAMPLE_COLS = ["sample_id", "label", "source", "variant", "social", "alpha", "device", "claimed_device",
               "claimed_dataset", "split", "fold", "stabilized", "source_rotation", "frames_path"]

# real platform copies (VISION) vs our platform-like re-encodes (`wa` / `yt`)
REAL_SOCIAL = {"native": "none", "youtube": "yt-real", "whatsapp": "wa-real"}


def build_samples(paths):
    meta = paths["metadata_dir"]
    videos = pd.read_csv(meta / "videos.csv")
    frames = pd.read_csv(meta / "frames_index.csv")
    frames = frames.loc[frames.status == "ok", ["content_id", "version", "frames_path"]]
    native_rot = videos[videos.version == "native"].set_index("content_id").rotation

    org = videos[videos.role == "sample"].merge(frames, on=["content_id", "version"])
    org = org.assign(sample_id=org.content_id + "__" + org.version, label="organic", source=org.dataset,
                     variant=org.version, social=org.version.map(REAL_SOCIAL), alpha=np.nan,
                     claimed_device=org.device, source_rotation=org.content_id.map(native_rot))
    parts = [org]

    if (meta / "attacks.csv").exists():
        att = pd.read_csv(meta / "attacks.csv").query("status == 'ok'")
        dataset = att.get("dataset", pd.Series("VISION", index=att.index)).fillna("VISION")
        att = att.assign(sample_id=att.content_id + "__" + att.variant, source=dataset + "-attack",
                         social=att.get("social", pd.Series("none", index=att.index)).fillna("none"),
                         source_rotation=att.content_id.map(native_rot))
        parts.append(att)
    if (meta / "synthetic.csv").exists():
        syn = pd.read_csv(meta / "synthetic.csv").query("status == 'ok'")
        social = syn.get("social", pd.Series("none", index=syn.index)).fillna("none")
        syn = syn.assign(sample_id=syn.content_id, source=syn.dataset, social=social,
                         variant=syn.generator + np.where(social == "none", "", "+" + social),
                         alpha=np.nan, device=pd.NA, stabilized=False, source_rotation=0)
        parts.append(syn)

    df = pd.concat(parts, ignore_index=True)
    dataset_of = videos.drop_duplicates("device").set_index("device").dataset
    df["claimed_dataset"] = df.claimed_device.map(dataset_of)
    return df.reindex(columns=SAMPLE_COLS)
