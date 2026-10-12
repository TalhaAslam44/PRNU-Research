"""Step 7 - features for every analysis sample (organic / manipulated / synthetic).

Samples:
    organic      VISION natural videos (native / YouTube / WhatsApp), ACID samples,
                 and the Step 5 `reencode` controls; claimed device = own device
    manipulated  Step 5 attacks; claimed device = own (removal, denoise) or victim (injection)
    synthetic    Step 6 clips; claimed device assigned at random

For each sample (see hvpf/features.py):
    per frame  PRNU NCC / zero-shift PCE with the claimed camera (W vs I*K), residual
               std / skew / kurtosis, FFT statistics, TSNCS (NCC of consecutive residuals)
    per video  the clip's MLE fingerprint from the first 5 and 10 frames (features.windows:
               w5_*, w10_*), scored against the claimed camera and
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
from hvpf.samples import SAMPLE_COLS, build_samples

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
    rows, sums = clip_features(frames, _BANK.ks[_BANK.names.index(claimed)], _CFG["windows"])
    per_frame = pd.DataFrame(rows).assign(sample_id=sample["sample_id"])
    video = {"sample_id": sample["sample_id"], "n_frames": len(frames)}
    for win, (rp, nn) in sums.items():           # clip fingerprint of the first `win` frames
        video.update(_scores(finish_fingerprint(rp, nn), claimed, f"w{win}"))
    for col in per_frame.columns.drop(["frame", "sample_id"]):
        video[f"{col}_mean"] = float(per_frame[col].mean())
        video[f"{col}_std"] = float(per_frame[col].std())
    video["tsncs_min"] = float(per_frame.tsncs.min())
    return per_frame, video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="samples per label (quick check)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--fresh", action="store_true", help="recompute every sample instead of reusing features.parquet")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    out = paths["data_dir"] / "features"
    out.mkdir(parents=True, exist_ok=True)

    samples = build_samples(paths)
    refs = {}
    for dataset in ("VISION", "ACID", "FloreView"):
        for p in sorted((paths["fingerprints_dir"] / dataset).glob("*.npy")):
            refs[p.stem] = p
    samples = samples[samples.claimed_device.isin(refs.keys())]
    if args.limit:
        samples = pd.concat(g.sample(min(args.limit, len(g)), random_state=0) for _, g in samples.groupby("label"))
    suffix = "_check" if args.limit else ""
    samples.to_csv(out / f"samples{suffix}.csv", index=False)
    print(samples.groupby(["label", "source"]).size().to_string())

    # reuse features of samples whose frames and claimed camera are unchanged
    todo, keep_v, keep_f = samples, None, None
    if not (args.limit or args.fresh) and (out / "videos.parquet").exists() and (out / "frames.parquet").exists():
        old_v = pd.read_parquet(out / "videos.parquet")
        same = old_v.merge(samples[["sample_id", "claimed_device", "frames_path"]],
                           on=["sample_id", "claimed_device", "frames_path"])
        done = set(same.sample_id)
        keep_v = old_v[old_v.sample_id.isin(done)].drop(columns=[c for c in SAMPLE_COLS if c != "sample_id"], errors="ignore")
        keep_f = pd.read_parquet(out / "frames.parquet").query("sample_id in @done")
        todo = samples[~samples.sample_id.isin(done)]
        print(f"reusing features of {len(done)} samples, computing {len(todo)}")

    frame_parts, video_rows = [], []
    with Pool(args.workers, initializer=_init, initargs=(pd.Series(refs).map(str), cfg["features"])) as pool:
        for per_frame, video in tqdm(pool.imap_unordered(process, todo.to_dict("records"), chunksize=4),
                                     total=len(todo), desc="features"):
            frame_parts.append(per_frame)
            video_rows.append(video)

    frames_df = pd.concat([keep_f, *frame_parts], ignore_index=True)
    videos_df = samples.merge(pd.concat([keep_v, pd.DataFrame(video_rows)], ignore_index=True), on="sample_id")
    frames_df.to_parquet(out / f"frames{suffix}.parquet", index=False)
    videos_df.to_parquet(out / f"videos{suffix}.parquet", index=False)
    print(f"\n{len(videos_df)} samples, {len(frames_df)} frames -> {out}")
    cols = ["w5_pce0_claimed", "w10_pce0_claimed", "w5_pce0_max_other", "tsncs_mean", "res_std_mean"]
    print(videos_df.groupby(["label", "source", "social"])[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
