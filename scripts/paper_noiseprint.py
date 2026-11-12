"""Paper, step 2 prerequisite - Step 7b (Noiseprint++) recomputed on this PC for all 46,799 clips (GPU, ~3-4 h).

Why: the stored Noiseprint++ features were computed on the old PC's RTX 3080, whose reduced-precision
(TF32) convolutions give ~6% weaker maps than this PC's full-precision ones (same code, weights and torch;
checked on the same frames). Clips attacked now must be described exactly like the clips the saved HVPF is
trained on, so both are computed here. Same computation as scripts/07b_noiseprint.py (reference spectra from
the reference videos, then per-clip statistics), via hvpf/paper_bundle.ClipFeatures. Resumes from finished
chunks. Nothing outside data/paper/ is written.

Writes data/paper/noiseprint_refs/<device>.npy and data/paper/features_newpc/ (Phase 4 videos/frames
tables + the new noiseprint_frames / noiseprint_videos).

    .venv/bin/python scripts/paper_noiseprint.py
"""
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.config import load_config
from hvpf.noiseprint import load_model, noiseprint_maps
from hvpf.paper_bundle import FEATURES, NP_REFS, ClipFeatures, step7b

CHUNK = 2000


def reference_spectra(paths, net, dev, spectra, per_device=100):
    """As scripts/07b_noiseprint.reference_spectra, written to NP_REFS."""
    videos = pd.read_csv(paths["metadata_dir"] / "videos.csv")
    frames = pd.read_csv(paths["metadata_dir"] / "frames_index.csv").query("status == 'ok'")
    refs = videos[videos.role == "reference"].merge(frames[["content_id", "version", "frames_path"]],
                                                    on=["content_id", "version"])
    NP_REFS.mkdir(parents=True, exist_ok=True)
    for device, g in refs.groupby("device"):
        f = NP_REFS / f"{device}.npy"
        if not f.exists():
            fr = np.concatenate([np.load(p) for p in g.frames_path])
            fr = fr[np.linspace(0, len(fr) - 1, min(per_device, len(fr))).astype(int)]
            np.save(f, spectra(noiseprint_maps(net, dev, fr)))
    print(f"reference spectra: {len(list(NP_REFS.glob('*.npy')))} cameras in {NP_REFS}", flush=True)


def main():
    warnings.filterwarnings("ignore")
    cfg = load_config()
    feats = ClipFeatures(workers=1)              # before any CUDA work (its worker is forked)
    net, dev = load_model()
    reference_spectra(cfg["paths"], net, dev, step7b.Spectra(cfg["frames"]["crop"]))
    del net
    src = phase5.PHASE4 / "features"
    samples = pd.read_parquet(src / "videos.parquet", columns=["sample_id", "label", "variant", "frames_path",
                                                                "claimed_device"])
    parts_dir = FEATURES / "noiseprint_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(samples), CHUNK):
        out = parts_dir / f"part_{i // CHUNK:03d}.parquet"
        if out.with_suffix(".videos.parquet").exists():
            continue
        f, v = feats._noiseprint(samples.iloc[i:i + CHUNK])
        f.to_parquet(out, index=False)
        v.to_parquet(out.with_suffix(".videos.parquet"), index=False)
        done = min(i + CHUNK, len(samples))
        print(f"  {done:,}/{len(samples):,} clips ({(time.time() - t0) / 60:.0f} min)", flush=True)
    frames = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("part_???.parquet"))], ignore_index=True)
    videos = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("part_*.videos.parquet"))], ignore_index=True)
    frames.to_parquet(FEATURES / "noiseprint_frames.parquet", index=False)
    videos.to_parquet(FEATURES / "noiseprint_videos.parquet", index=False)
    for name in ("videos.parquet", "frames.parquet"):
        shutil.copy2(src / name, FEATURES / name)
    old = pd.read_parquet(src / "noiseprint_videos.parquet").set_index("sample_id")
    new = videos.set_index("sample_id").loc[old.index]
    cols = ["np_std_mean", "np_kurt_mean", "np_flatness_mean", "np_spec_ncc_claimed_w10"]
    print("\nmedian new / stored ratio: " + ", ".join(f"{c} {np.nanmedian(new[c] / old[c]):.3f}" for c in cols))
    print(f"wrote {FEATURES}")


if __name__ == "__main__":
    main()
