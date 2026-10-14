"""Step 8b - Vision Transformer on I-frame noise residuals (GPU), 80/20 training (hvpf-2).

Model and input: hvpf/vit.py (pretrained ViT-S/16 on three SRM residuals of single I-frames,
random 224x224 crops and flips, AdamW, one-cycle cosine schedule, mixed precision,
class-balanced loss with label smoothing). A clip's prediction averages the softmax over its
first `window` I-frames x 4 crops (2x2 tiles).

80/20 (hvpf-2): the 80% training side of each split is used twice.
  1. tuning: train on the tuning part, keep the epoch with the best validation macro-F1
     (validation part = the held-out camera models of the old 60/20/20 val split, or fold f+1);
     its validation predictions are written as part = val (Step 8c fits calibration / stacking on them)
  2. final: retrain on the whole 80% (tuning + validation part) for that number of epochs;
     its test / crossdataset predictions are the results
Same samples, protocols and splits as Step 8 (hvpf/model_data). Writes
data/results/models/vit/<protocol>_<scheme>_w<window>.parquet; Step 8c merges them into Step 8's files.

Usage:
    python scripts/08b_vit.py --protocol realistic --scheme folds
    python scripts/08b_vit.py --limit 300 --epochs 2        # quick check (writes *_check)
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
    ap.add_argument("--scheme", choices=["fixed", "folds"], default="folds")
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
    common = dict(n_classes=3, epochs=args.epochs, lr=args.lr, batch=args.batch, workers=args.workers)
    out_rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        t0 = time.time()
        full = tr | va                                   # the 80% training side
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        out = te | cross
        va_idx, out_idx = np.flatnonzero(va), np.flatnonzero(out)
        print(f"  {name}: train {full.sum()} (tuning {tr.sum()} + validation {va.sum()}), "
              f"test {te.sum()}, crossdataset {cross.sum()}", flush=True)

        # 1. tuning: best epoch on the validation part
        val_items = frame_items(per_video, va_idx, y, window)
        score = lambda net, srm: f1_score(
            y[va_idx], vit.predict(net, srm, val_items, len(va_idx), 3, vit.centre(crop), args.workers).argmax(1),
            average="macro")
        net, srm, best_epoch = vit.train(frame_items(per_video, np.flatnonzero(tr), y, window), val_score=score,
                                         patience=args.patience, **common)
        p = vit.predict(net, srm, val_items, len(va_idx), 3, vit.tiles(crop), args.workers)
        out_rows.append(per_video.loc[va, meta_cols].assign(
            split_name=name, part="val", model="vit", **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        del net
        torch.cuda.empty_cache()

        # 2. final: the whole 80% for best_epoch epochs
        print(f"  {name}: refit on {full.sum()} clips for {best_epoch} epochs", flush=True)
        net, srm, _ = vit.train(frame_items(per_video, np.flatnonzero(full), y, window), stop_after=best_epoch, **common)
        p = vit.predict(net, srm, frame_items(per_video, out_idx, y, window), len(out_idx), 3, vit.tiles(crop),
                        args.workers)
        out_rows.append(per_video.loc[out, meta_cols].assign(
            split_name=name, part=np.where(cross[out], "crossdataset", "test"), model="vit",
            **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        test = te[out]
        f1 = f1_score(y[out_idx][test], p[test].argmax(1), average="macro")
        print(f"  {name} vit: test macro-F1 {f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        del net
        torch.cuda.empty_cache()

    out_dir = paths["results_dir"] / "models" / "vit"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.protocol}_{args.scheme}_w{window}{'_check' if args.limit else ''}.parquet"
    res = pd.concat(out_rows, ignore_index=True)
    res.to_parquet(path, index=False)
    print(f"wrote {path} ({len(res)} rows)")


if __name__ == "__main__":
    main()
