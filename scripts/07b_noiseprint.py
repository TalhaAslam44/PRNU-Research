"""Step 7b - Noiseprint++ output statistics (GPU).

Run third_party/get_noiseprintpp.sh once, and Step 7 first (it writes data/features/samples.csv).

Per frame, the Noiseprint++ map gets content-independent statistics: std, kurtosis and
the FFT statistics of Step 7 (flatness, high-frequency share, peak ratio, periodic peaks,
8-px block energy), prefixed `np_`. Per video: their mean/std, plus the spectral similarity
of the clip's map to the claimed camera's reference and the best other camera (`np_spec_*`):
log power spectra above 0.1 cycles/px, centred on the mean over all reference cameras.

Noiseprint++ is weak for attribution on our data: on 10 VISION cameras the spectral (or
spatial) similarity picks the right camera for only ~20% of clips (chance 10%). It was
trained on RGB stills; our frames are luma-only, heavily compressed video I-frames. Its
statistics may still separate camera-made from re-synthesised content - the ablation decides.

Writes data/features/noiseprint_frames.parquet, noiseprint_videos.parquet and
data/fingerprints/noiseprint/<device>.npy (reference spectra).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.features import fft_stats
from hvpf.noiseprint import load_model, noiseprint_maps

LOW_CUT = 0.1   # cycles/px; lower frequencies carry scene content


class Spectra:
    def __init__(self, n):
        f = np.fft.fftfreq(n)
        fy, fx = np.meshgrid(f, f, indexing="ij")
        self.mask = np.sqrt(fx ** 2 + fy ** 2) > LOW_CUT

    def __call__(self, maps):
        p = np.mean(np.abs(np.fft.fft2(maps - maps.mean(axis=(1, 2), keepdims=True))) ** 2, axis=0)
        s = np.log(p[self.mask] + 1e-8)
        return (s - s.mean()).astype(np.float32)


def _ncc(a, b):
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def reference_spectra(paths, net, dev, spectra, out_dir, per_device=100):
    videos = pd.read_csv(paths["metadata_dir"] / "videos.csv")
    frames = pd.read_csv(paths["metadata_dir"] / "frames_index.csv").query("status == 'ok'")
    refs = videos[(videos.role == "reference")].merge(frames[["content_id", "version", "frames_path"]],
                                                      on=["content_id", "version"])
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {}
    for device, g in tqdm(refs.groupby("device"), desc="reference spectra"):
        f = out_dir / f"{device}.npy"
        if not f.exists():
            fr = np.concatenate([np.load(p) for p in g.frames_path])
            fr = fr[np.linspace(0, len(fr) - 1, min(per_device, len(fr))).astype(int)]
            np.save(f, spectra(noiseprint_maps(net, dev, fr)))
        specs[device] = np.load(f)
    common = np.mean(list(specs.values()), axis=0)
    return {d: s - common for d, s in specs.items()}, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="samples per label (quick check)")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    feat_dir = paths["data_dir"] / "features"
    suffix = "_check" if args.limit else ""
    samples = pd.read_csv(feat_dir / f"samples{suffix}.csv")
    if args.limit:
        samples = pd.concat(g.sample(min(args.limit, len(g)), random_state=0) for _, g in samples.groupby("label"))

    net, dev = load_model()
    spectra = Spectra(cfg["frames"]["crop"])
    refs, common = reference_spectra(paths, net, dev, spectra, paths["fingerprints_dir"] / "noiseprint")
    names = list(refs)
    bank = np.stack([refs[d] for d in names])

    frame_parts, video_rows = [], []
    for s in tqdm(samples.itertuples(), total=len(samples), desc="noiseprint"):
        maps = noiseprint_maps(net, dev, np.load(s.frames_path))
        rows = [{"frame": i, "np_std": float(m.std()), "np_kurt": float(kurtosis(m, axis=None)),
                 **{f"np_{k.removeprefix('fft_')}": v for k, v in fft_stats(m).items()}}
                for i, m in enumerate(maps)]
        per_frame = pd.DataFrame(rows).assign(sample_id=s.sample_id)
        frame_parts.append(per_frame)
        spec = spectra(maps) - common
        sims = np.array([_ncc(spec, b) for b in bank])
        i = names.index(s.claimed_device)
        video = {"sample_id": s.sample_id, "np_spec_ncc_claimed": float(sims[i]),
                 "np_spec_ncc_max_other": float(np.delete(sims, i).max())}
        for col in per_frame.columns.drop(["frame", "sample_id"]):
            video[f"{col}_mean"] = float(per_frame[col].mean())
            video[f"{col}_std"] = float(per_frame[col].std())
        video_rows.append(video)

    pd.concat(frame_parts, ignore_index=True).to_parquet(feat_dir / f"noiseprint_frames{suffix}.parquet", index=False)
    videos = samples[["sample_id", "label", "variant"]].merge(pd.DataFrame(video_rows), on="sample_id")
    videos.to_parquet(feat_dir / f"noiseprint_videos{suffix}.parquet", index=False)
    print(videos.groupby(["label", "variant"])[["np_std_mean", "np_block8_mean", "np_flatness_mean",
                                                "np_spec_ncc_claimed"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
