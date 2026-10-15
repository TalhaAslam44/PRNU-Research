"""Step 8b - Vision Transformer on I-frame noise residuals (GPU).

A pretrained ViT-S/16 (timm vit_small_patch16_224.augreg_in21k_ft_in1k, 21.7M parameters) is
fine-tuned on the high-pass noise residual of single I-frames (hvpf/vit.py): three SRM filters
(5x5 KV, 3x3 second order, horizontal second order, truncated at +-4 grey levels) turn the luma
frame into 3 channels, so the network looks at sensor / compression / generator noise rather
than scene content. The ImageNet weights are still a good starting point for patch embedding and
attention. PDC-ViT (Elharrouss et al. 2025), the base paper's Transformer, also works on
224x224 crops.

Training: random 224x224 crops of the first `window` I-frames of every training clip, random
flips; AdamW, one-cycle cosine schedule, mixed precision, class-balanced loss with label
smoothing; the epoch with the best validation macro-F1 (clip level, centre crops) is kept.
Prediction: softmax averaged over the first `window` frames x 4 crops (2x2 tiles) of a clip.

Same samples, protocols and splits as Step 8 (hvpf/model_data). Writes
data/results/models/vit/<protocol>_<scheme>_w<window>.parquet (part = val | test | crossdataset);
Step 8c merges them into Step 8's prediction files and builds the ensembles.

Usage:
    python scripts/08b_vit.py --protocol realistic --scheme fixed
    python scripts/08b_vit.py --protocol realistic --scheme folds --window 10
    python scripts/08b_vit.py --limit 300 --epochs 1        # quick check (writes *_check)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import vit
from hvpf.config import load_config
from hvpf.model_data import CLASSES, load, protocol_mask, split_masks


def frame_items(per_video, idx, y, window):
    """(path, frame, label, k) for the first `window` frames of clip per_video.iloc[idx[k]]."""
    items = []
    for k, i in enumerate(idx):
        n = min(window, int(per_video.n_frames.iat[i]))
        items += [(per_video.frames_path.iat[i], f, int(y[i]), k) for f in range(n)]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["realistic", "controlled"], default="realistic")
    ap.add_argument("--scheme", choices=["fixed", "folds"], default="fixed")
    ap.add_argument("--window", type=int, help="frames per clip (default: features.window)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2, help="stop after this many epochs without a better val F1")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, help="clips per label (quick check; writes *_check)")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    window = args.window or cfg["features"]["window"]
    per_video, _ = load(paths["data_dir"] / "features", window)
    per_video = per_video[protocol_mask(per_video, args.protocol)].reset_index(drop=True)
    if args.limit:
        per_video = pd.concat(g.sample(min(args.limit, len(g)), random_state=0)
                              for _, g in per_video.groupby("label")).reset_index(drop=True)
    y = per_video.label.map(CLASSES.index).to_numpy()
    print(f"ViT {args.protocol}/{args.scheme}/w{window}: {len(per_video)} samples; "
          f"classes {np.bincount(y, minlength=3).tolist()}", flush=True)

    crop = cfg["frames"]["crop"]          # frames are crop x crop luma
    meta_cols = ["sample_id", "label", "source", "variant", "social", "alpha", "stabilized", "claimed_device"]
    out_rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        t0 = time.time()
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        evaluate = te | cross | va
        print(f"  {name}: train {tr.sum()}, val {va.sum()}, evaluate {evaluate.sum()}", flush=True)
        va_idx = np.flatnonzero(va)
        val_items = frame_items(per_video, va_idx, y, window)
        score = lambda net, srm: f1_score(
            y[va_idx], vit.predict(net, srm, val_items, len(va_idx), 3, vit.centre(crop), args.workers).argmax(1),
            average="macro")
        net, srm, _ = vit.train(frame_items(per_video, np.flatnonzero(tr), y, window), n_classes=3, epochs=args.epochs,
                                lr=args.lr, batch=args.batch, workers=args.workers, val_score=score,
                                patience=args.patience)
        idx = np.flatnonzero(evaluate)
        p = vit.predict(net, srm, frame_items(per_video, idx, y, window), len(idx), 3, vit.tiles(crop), args.workers)
        meta = per_video.loc[evaluate, meta_cols].assign(
            split_name=name, part=np.select([cross[evaluate], va[evaluate]], ["crossdataset", "val"], "test"))
        out_rows.append(meta.assign(model="vit", **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        test = ~va[evaluate] & ~cross[evaluate]
        f1 = f1_score(y[idx][test], p[test].argmax(1), average="macro")
        print(f"  {name} vit: test macro-F1 {f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        del net
        torch.cuda.empty_cache()

    out_dir = paths["results_dir"] / "models" / "vit"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.protocol}_{args.scheme}_w{window}{'_check' if args.limit else ''}.parquet"
    out = pd.concat(out_rows, ignore_index=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
