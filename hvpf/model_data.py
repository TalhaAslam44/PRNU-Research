"""Assemble Step 7 features into model inputs (Step 8) with fixed-length windows.

Clips differ in length (VISION ~30 I-frames, ACID ~6, synthetic 2-20), so every input
is computed from the first `window` frames only: per-frame features are re-aggregated
here and the clip-fingerprint scores are the `w_*` ones (first `window` frames).
Clip length is never a feature.
"""
import numpy as np
import pandas as pd

CLASSES = ["organic", "manipulated", "synthetic"]

# per-frame features by group (Noiseprint ones exist only after Step 7b)
FRAME_GROUPS = {
    "prnu": ["prnu_ncc", "prnu_pce0"],
    "resid": ["res_std", "res_skew", "res_kurt"],
    "fft": ["fft_flatness", "fft_hf_ratio", "fft_peak_ratio", "fft_n_peaks", "fft_block8"],
    "tsncs": ["tsncs"],
    "noiseprint": ["np_std", "np_kurt", "np_flatness", "np_hf_ratio", "np_peak_ratio", "np_n_peaks", "np_block8"],
}
# clip-level features by group (computed in Step 7 from the first `window` frames)
VIDEO_GROUPS = {
    "prnu": ["w_pce0_claimed", "w_ncc_claimed", "w_pce_claimed", "w_pce0_max_other", "w_ncc_max_other"],
    "noiseprint": ["np_spec_ncc_claimed", "np_spec_ncc_max_other"],
}
SIGNED_LOG = ("pce",)   # PCE spans -1e4..1e4: sign(x) * log1p(|x|)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load(feat_dir, window, suffix=""):
    """Samples table, per-video window features, per-frame window features (first `window` frames)."""
    samples = pd.read_parquet(feat_dir / f"videos{suffix}.parquet")
    frames = pd.read_parquet(feat_dir / f"frames{suffix}.parquet")
    np_frames = feat_dir / f"noiseprint_frames{suffix}.parquet"
    np_videos = feat_dir / f"noiseprint_videos{suffix}.parquet"
    if np_frames.exists():
        frames = frames.merge(pd.read_parquet(np_frames), on=["sample_id", "frame"], how="left")
        samples = samples.merge(pd.read_parquet(np_videos).drop(columns=["label", "variant"]), on="sample_id", how="left")
    frames = frames[frames.frame < window].sort_values(["sample_id", "frame"])
    groups = {g: [c for c in cols if c in frames] for g, cols in FRAME_GROUPS.items()}

    agg = {}
    for cols in groups.values():
        for c in cols:
            agg[f"{c}_mean"] = (c, "mean")
            agg[f"{c}_std"] = (c, "std")
    per_video = frames.groupby("sample_id").agg(**agg)
    if "tsncs" in frames:
        per_video["tsncs_min"] = frames.groupby("sample_id").tsncs.min()
    # Step 7's all-frame averages share these names: the window versions replace them
    samples = samples.drop(columns=[c for c in per_video.columns if c in samples])
    per_video = samples.set_index("sample_id").join(per_video, how="inner").reset_index()
    for c in per_video.columns:
        if any(k in c for k in SIGNED_LOG) and per_video[c].dtype.kind == "f":
            per_video[c] = slog(per_video[c])
    for c in ("prnu_pce0",):
        if c in frames:
            frames[c] = slog(frames[c])
    return per_video, frames


def feature_columns(per_video, groups):
    """Video-level columns for the selected feature groups."""
    cols = []
    for g in groups:
        for c in FRAME_GROUPS.get(g, []):
            cols += [f"{c}_mean", f"{c}_std"]
        if g == "tsncs":
            cols.append("tsncs_min")
        cols += VIDEO_GROUPS.get(g, [])
    return [c for c in cols if c in per_video]


def frame_columns(frames, groups):
    return [c for g in groups for c in FRAME_GROUPS.get(g, []) if c in frames]


def protocol_mask(per_video, protocol):
    """realistic: everything; controlled: only libx264 clips (reencode controls, attacks, synthetic)."""
    if protocol == "realistic":
        return np.ones(len(per_video), bool)
    return (per_video.source.isin(["VISION-attack", "GenVidBench", "GenBuster"])).to_numpy()


def split_masks(per_video, scheme):
    """Yield (name, train, val, test) boolean masks; crossdataset samples are test-only (fixed scheme)."""
    if scheme == "fixed":
        s = per_video.split
        yield "fixed", (s == "train").to_numpy(), (s == "val").to_numpy(), (s == "test").to_numpy()
        return
    folds = per_video.fold.astype("Int64")
    k = int(folds.max()) + 1
    for f in range(k):
        test = (folds == f).fillna(False).to_numpy()
        val = (folds == (f + 1) % k).fillna(False).to_numpy()
        train = folds.notna().to_numpy() & ~test & ~val
        yield f"fold{f}", train, val, test


def sequences(frames, sample_ids, cols, window):
    """(n, window, d) float32 array + (n, window) mask, in the order of `sample_ids`."""
    idx = {s: i for i, s in enumerate(sample_ids)}
    x = np.zeros((len(sample_ids), window, len(cols)), np.float32)
    mask = np.zeros((len(sample_ids), window), bool)
    sub = frames[frames.sample_id.isin(idx)]
    rows = sub.sample_id.map(idx).to_numpy()
    pos = sub.frame.to_numpy()
    x[rows, pos] = np.nan_to_num(sub[cols].to_numpy(np.float32))
    mask[rows, pos] = True
    return x, mask
