"""Step 7 - features for every analysis sample (organic / manipulated / synthetic).

Samples:
    organic      VISION natural videos (native / YouTube / WhatsApp), ACID samples,
                 and the Step 5 `reencode` controls; claimed device = own device
    manipulated  Step 5 attacks; claimed device = own (removal, denoise) or victim (injection)
    synthetic    Step 6 clips; claimed device assigned at random

For each sample (see hvpf/features.py):
    per frame  PRNU NCC / zero-shift PCE with the claimed camera (W vs I*K), residual
               std / skew / kurtosis, FFT statistics, TSNCS (NCC of consecutive residuals)
    per video  the clip's MLE fingerprint from all frames (v_*) and from the first
               `features.window` frames (w_*), scored against the claimed camera and
               against every other known camera (max_other: a second camera's PRNU,
               e.g. after injection), plus mean/std of the per-frame values

Writes data/features/samples.csv, frames.parquet, videos.parquet.

Usage:
    python scripts/07_features.py --limit 20     # quick check (20 samples per label)
    python scripts/07_features.py
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.features import clip_features, finish_fingerprint
from hvpf.prnu_utils import FingerprintBank

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


_BANK = None
_CFG = None


def _init(ref_paths, cfg):
    global _BANK, _CFG
    _BANK = FingerprintBank(list(ref_paths.index), [np.load(p) for p in ref_paths.values])
    _BANK.fft = _BANK.fft.astype(np.complex64)      # halves memory per worker
    _CFG = cfg


def _scores(k_query, claimed, prefix):
    pce, pce0, ncc = _BANK.score(k_query)
    i = _BANK.names.index(claimed)
    others = np.arange(len(pce)) != i
    j = np.argmax(np.where(others, pce0, -np.inf))
    return {f"{prefix}_pce_claimed": float(pce[i]), f"{prefix}_pce0_claimed": float(pce0[i]),
            f"{prefix}_ncc_claimed": float(ncc[i]), f"{prefix}_pce0_max_other": float(pce0[j]),
            f"{prefix}_ncc_max_other": float(ncc[others].max()), f"{prefix}_max_other_device": _BANK.names[j]}


def process(sample):
    frames = np.load(sample["frames_path"])
    claimed = sample["claimed_device"]
    rows, (rp, nn), (rp_w, nn_w) = clip_features(frames, _BANK.ks[_BANK.names.index(claimed)], _CFG["window"])
    per_frame = pd.DataFrame(rows).assign(sample_id=sample["sample_id"])
    video = {"sample_id": sample["sample_id"], "n_frames": len(frames),
             **_scores(finish_fingerprint(rp, nn), claimed, "v"),
             **_scores(finish_fingerprint(rp_w, nn_w), claimed, "w")}
    for col in per_frame.columns.drop(["frame", "sample_id"]):
        video[f"{col}_mean"] = float(per_frame[col].mean())
        video[f"{col}_std"] = float(per_frame[col].std())
    video["tsncs_min"] = float(per_frame.tsncs.min())
    return per_frame, video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="samples per label (quick check)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    out = paths["data_dir"] / "features"
    out.mkdir(parents=True, exist_ok=True)

    samples = build_samples(paths)
    refs = {}
    for dataset in ("VISION", "ACID"):
        for p in sorted((paths["fingerprints_dir"] / dataset).glob("*.npy")):
            refs[p.stem] = p
    samples = samples[samples.claimed_device.isin(refs.keys())]
    if args.limit:
        samples = samples.groupby("label").head(args.limit)
    suffix = "_check" if args.limit else ""
    samples.to_csv(out / f"samples{suffix}.csv", index=False)
    print(samples.groupby(["label", "source"]).size().to_string())

    frame_parts, video_rows = [], []
    with Pool(args.workers, initializer=_init, initargs=(pd.Series(refs).map(str), cfg["features"])) as pool:
        for per_frame, video in tqdm(pool.imap_unordered(process, samples.to_dict("records"), chunksize=4),
                                     total=len(samples), desc="features"):
            frame_parts.append(per_frame)
            video_rows.append(video)

    frames_df = pd.concat(frame_parts, ignore_index=True)
    videos_df = samples.merge(pd.DataFrame(video_rows), on="sample_id")
    frames_df.to_parquet(out / f"frames{suffix}.parquet", index=False)
    videos_df.to_parquet(out / f"videos{suffix}.parquet", index=False)
    print(f"\n{len(videos_df)} samples, {len(frames_df)} frames -> {out}")
    cols = ["v_pce0_claimed", "v_pce0_max_other", "tsncs_mean", "res_std_mean", "fft_flatness_mean"]
    print(videos_df.groupby(["label", "variant"])[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
