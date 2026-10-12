"""Step 8b - Vision Transformer on I-frame noise residuals (GPU).

A pretrained ViT-S/16 (timm vit_small_patch16_224.augreg_in21k_ft_in1k, 21.7M parameters) is
fine-tuned on the high-pass noise residual of single I-frames: three SRM filters (5x5 KV,
3x3 second order, horizontal second order, truncated at +-4 grey levels) turn the luma frame
into 3 channels, so the network looks at sensor / compression / generator noise rather than
scene content. The ImageNet weights are still a good starting point for patch embedding and
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
    python scripts/08b_vit.py --limit 300 --epochs 1        # quick check (writes *_check)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.model_data import CLASSES, load, protocol_mask, split_masks

ARCH = "vit_small_patch16_224.augreg_in21k_ft_in1k"
DEV = "cuda"
SIZE = 224 + 4          # crop read from disk; the 5x5 SRM filters (no padding) leave 224x224


class SRM(nn.Module):
    """uint8 luma (B, 1, H, W) -> 3 truncated SRM residuals (B, 3, H-4, W-4) in [-1, 1]."""

    def __init__(self, t=4.0):
        super().__init__()
        k = np.zeros((3, 1, 5, 5), np.float32)
        k[0, 0] = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                            [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12
        k[1, 0, 1:4, 1:4] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4
        k[2, 0, 2, 1:4] = np.array([1, -2, 1]) / 2
        self.register_buffer("k", torch.from_numpy(k))
        self.t = t

    def forward(self, x):
        return F.conv2d(x.float(), self.k).clamp(-self.t, self.t) / self.t


class Frames(Dataset):
    """items: (frames_path, frame, label, clip). crops None -> one random crop + flips;
    otherwise every (y, x) crop offset of the list."""

    def __init__(self, items, crops=None):
        self.items, self.crops = items, crops

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, f, label, clip = self.items[i]
        a = np.load(path, mmap_mode="r")[f]
        if self.crops is None:          # torch RNG: numpy's would repeat across DataLoader workers
            y, x = (int(v) for v in torch.randint(0, a.shape[0] - SIZE + 1, (2,)))
            c = np.array(a[y:y + SIZE, x:x + SIZE])
            if torch.rand(1) < 0.5:
                c = c[:, ::-1]
            if torch.rand(1) < 0.5:
                c = c[::-1]
            x = torch.from_numpy(c.copy())[None]
        else:
            x = torch.stack([torch.from_numpy(np.array(a[y:y + SIZE, x:x + SIZE])) for y, x in self.crops])[:, None]
        return x, label, clip


def frame_items(per_video, idx, y, window):
    """(path, frame, label, k) for the first `window` frames of clip per_video.iloc[idx[k]]."""
    items = []
    for k, i in enumerate(idx):
        n = min(window, int(per_video.n_frames.iat[i]))
        items += [(per_video.frames_path.iat[i], f, int(y[i]), k) for f in range(n)]
    return items


@torch.no_grad()
def predict(net, srm, items, n_clips, crops, workers, batch=64):
    """Clip-level class probabilities: mean softmax over the clip's frames and crops."""
    net.eval()
    sums, count = np.zeros((n_clips, 3)), np.zeros(n_clips)
    for x, _, clip in DataLoader(Frames(items, crops), batch_size=batch, num_workers=workers, pin_memory=True):
        b, k = x.shape[:2]
        x = srm(x.to(DEV, non_blocking=True).flatten(0, 1))
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(net(x).float(), 1)
        p = p.view(b, k, 3).mean(1).cpu().numpy()
        np.add.at(sums, clip.numpy(), p)
        np.add.at(count, clip.numpy(), 1)
    return sums / np.maximum(count, 1)[:, None]


def fit(per_video, y, tr, va, window, args):
    torch.manual_seed(0)
    net = timm.create_model(ARCH, pretrained=True, num_classes=3, drop_path_rate=0.1).to(DEV)
    srm = SRM().to(DEV)
    train_items = frame_items(per_video, np.flatnonzero(tr), y, window)
    va_idx = np.flatnonzero(va)
    val_items = frame_items(per_video, va_idx, y, window)
    centre = [((args.crop - SIZE) // 2,) * 2]
    w = compute_class_weight("balanced", classes=np.arange(3), y=np.array([it[2] for it in train_items]))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV), label_smoothing=0.1)
    dl = DataLoader(Frames(train_items), batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl),
                                                pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler()
    best, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        net.train()
        t0, total = time.time(), 0.0
        for x, label, _ in dl:
            x, label = srm(x.to(DEV, non_blocking=True)), label.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(net(x), label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        f1 = f1_score(y[va_idx], predict(net, srm, val_items, len(va_idx), centre, args.workers).argmax(1),
                      average="macro")
        print(f"    epoch {epoch + 1}: {len(train_items)} frames, loss {total / len(dl):.3f}, "
              f"val macro-F1 {f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    return net, srm


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

    args.crop = cfg["frames"]["crop"]         # frames are crop x crop luma
    tiles = [(a, b) for a in (0, args.crop - SIZE) for b in (0, args.crop - SIZE)]
    out_rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        t0 = time.time()
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        evaluate = te | cross | va
        print(f"  {name}: train {tr.sum()}, val {va.sum()}, evaluate {evaluate.sum()}", flush=True)
        net, srm = fit(per_video, y, tr, va, window, args)
        idx = np.flatnonzero(evaluate)
        p = predict(net, srm, frame_items(per_video, idx, y, window), len(idx), tiles, args.workers)
        meta = per_video.loc[evaluate, ["sample_id", "label", "source", "variant", "social", "alpha", "stabilized",
                                        "claimed_device"]].assign(
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
