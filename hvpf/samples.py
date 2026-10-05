"""The analysis sample table shared by Step 7 (PRNU/FFT/TSNCS) and Step 7b (Noiseprint)."""
import numpy as np
import pandas as pd

SAMPLE_COLS = ["sample_id", "label", "source", "variant", "alpha", "device", "claimed_device",
               "claimed_dataset", "split", "fold", "stabilized", "source_rotation", "frames_path"]


def build_samples(paths):
    meta = paths["metadata_dir"]
    videos = pd.read_csv(meta / "videos.csv")
    frames = pd.read_csv(meta / "frames_index.csv")
    frames = frames.loc[frames.status == "ok", ["content_id", "version", "frames_path"]]
    native_rot = videos[videos.version == "native"].set_index("content_id").rotation

    org = videos[videos.role == "sample"].merge(frames, on=["content_id", "version"])
    org = org.assign(sample_id=org.content_id + "__" + org.version, label="organic", source=org.dataset,
                     variant=org.version, alpha=np.nan, claimed_device=org.device,
                     source_rotation=org.content_id.map(native_rot))
    parts = [org]

    if (meta / "attacks.csv").exists():
        att = pd.read_csv(meta / "attacks.csv").query("status == 'ok'")
        att = att.assign(sample_id=att.content_id + "__" + att.variant, source="VISION-attack",
                         source_rotation=att.content_id.map(native_rot))
        parts.append(att)
    if (meta / "synthetic.csv").exists():
        syn = pd.read_csv(meta / "synthetic.csv").query("status == 'ok'")
        syn = syn.assign(sample_id=syn.content_id, source=syn.dataset, variant=syn.generator, alpha=np.nan,
                         device=pd.NA, stabilized=False, source_rotation=0)
        parts.append(syn)

    df = pd.concat(parts, ignore_index=True)
    dataset_of = videos.drop_duplicates("device").set_index("device").dataset
    df["claimed_dataset"] = df.claimed_device.map(dataset_of)
    return df.reindex(columns=SAMPLE_COLS)
