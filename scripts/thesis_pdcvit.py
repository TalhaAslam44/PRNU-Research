"""Thesis baseline - the base paper's method (PDC-ViT) trained and tested on HVPF's task and splits (GPU).

PDC-ViT (Elharrouss et al., "PDC-ViT: Source Camera Identification using Pixel Difference Convolution and
Vision Transformer", Neural Computing and Applications 2025, arXiv:2501.16227) was proposed for camera
identification. Here its network is trained for our three classes (genuine / manipulated / AI) on exactly the
clips, camera splits and 10 frames of the final HVPF, so the comparison is clip by clip.

As in the paper (Sec. 3, Fig. 1, Sec. 4.1): 224 x 224 crops; two feature branches, each two blocks of
[pixel difference convolution -> ReLU -> 2 x 2 max pooling] (output H/4 x W/4) - angular PDC with 3 x 3 kernels
and radial PDC with 5 x 5 kernels (Su et al., PiDiNet, ICCV 2021: the convolution is applied to differences of
neighbouring pixels, done by transforming the kernel) - concatenated and cut into patches for a Vision
Transformer trained from scratch (dim 1024, depth 6, 16 heads, MLP 2048, dropout 0.1, embedding dropout 0.1);
cross-entropy loss, Adam, learning rate 3e-5 (--recipe paper).
Not given in the paper, chosen here (state them with the result):
  * channels of the PDC branches: Fig. 1 draws C x H x W -> C x H/4 x W/4 (--channels 3); a wider branch can
    be tried (--channels 32),
  * patch size on the 56 x 56 feature map: the paper's "64 x 64" does not divide a 224 x 224 input, here
    8 x 8 feature cells = 32 x 32 input pixels, 49 tokens (--patch),
  * our frames are luma only (as for every model of the thesis), repeated to 3 channels,
  * class-balanced loss weights and the best epoch by validation macro F1 (as for our ViT and EfficientNet-B4),
    because the three classes are not balanced and the paper gives no number of epochs,
  * --recipe ours: the training recipe of our ViT / EfficientNet-B4 (AdamW, one-cycle, label smoothing) in
    case the paper's learning rate under-trains the network on this task.
A clip's probabilities = mean over 10 frames x 4 crops. Writes
data/thesis/pdcvit/<tag>/realistic_<scheme>_w10.parquet (Step 8 format, model = pdc_vit).

    .venv/bin/python scripts/thesis_pdcvit.py --scheme fixed --tag paper
    .venv/bin/python scripts/thesis_pdcvit.py --scheme folds --tag paper
    .venv/bin/python scripts/thesis_pdcvit.py --scheme fixed --limit 300 --epochs 1 --tag check     # pipeline check
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5, vit
from hvpf.model_data import CLASSES, load, protocol_mask, split_masks

step8b = phase5.load_step("08b_vit.py")
baselines = phase5.load_step("paper_baselines.py")
OUT = phase5.ROOT / "data" / "thesis" / "pdcvit"
DEV = "cuda"
WINDOW = 10


class PDConv(nn.Module):
    """Pixel difference convolution (PiDiNet): 'angular' = 3 x 3 kernel on differences of neighbours taken
    clockwise around the centre, 'radial' = 5 x 5 kernel on differences outer ring - inner ring."""

    def __init__(self, kind, cin, cout):
        super().__init__()
        self.kind = kind
        self.weight = nn.Parameter(torch.empty(cout, cin, 3, 3))
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        w = self.weight
        if self.kind == "angular":
            flat = w.flatten(2)
            k = (flat - flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view_as(w)        # w1 - w4, w2 - w1, w3 - w2 ...
            return F.conv2d(x, k, self.bias, padding=1)
        flat = w.flatten(2)
        k = torch.zeros(w.shape[0], w.shape[1], 25, device=w.device, dtype=w.dtype)
        k[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flat[:, :, [0, 1, 2, 3, 5, 6, 7, 8]]
        k[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flat[:, :, [0, 1, 2, 3, 5, 6, 7, 8]]
        return F.conv2d(x, k.view(w.shape[0], w.shape[1], 5, 5), self.bias, padding=2)


def branch(kind, channels):
    return nn.Sequential(PDConv(kind, 3, channels), nn.ReLU(), nn.MaxPool2d(2),
                         PDConv(kind, channels, channels), nn.ReLU(), nn.MaxPool2d(2))


class PDCViT(nn.Module):
    def __init__(self, n_classes=3, channels=3, patch=8, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.angular, self.radial = branch("angular", channels), branch("radial", channels)
        self.patch = patch
        n = (224 // 4 // patch) ** 2
        self.embed = nn.Sequential(nn.LayerNorm(2 * channels * patch * patch), nn.Linear(2 * channels * patch * patch, dim),
                                   nn.LayerNorm(dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def forward(self, x):
        f = torch.cat([self.angular(x), self.radial(x)], 1)                         # (B, 2C, 56, 56)
        p = self.patch
        t = f.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).flatten(3).flatten(1, 2)   # (B, n, 2C p p)
        t = torch.cat([self.cls.expand(len(t), -1, -1), self.embed(t)], 1) + self.pos
        return self.head(self.encoder(self.drop(t))[:, 0])


def train(items, val_score, args, seed=0):
    torch.manual_seed(seed)
    net, pre = PDCViT(channels=args.channels, patch=args.patch).to(DEV), baselines.Raw().to(DEV)
    labels = np.array([it[2] for it in items])
    w = torch.tensor(compute_class_weight("balanced", classes=np.arange(3), y=labels), dtype=torch.float32, device=DEV)
    dl = DataLoader(vit.Frames(items), batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                    pin_memory=True, persistent_workers=True)
    if args.recipe == "paper":
        loss_fn = nn.CrossEntropyLoss(weight=w)
        opt, sched = torch.optim.Adam(net.parameters(), lr=args.lr or 3e-5), None
    else:
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr or 3e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr or 3e-4, total_steps=args.epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    best, state, bad = -np.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, total = time.time(), 0.0
        for x, label, _ in dl:
            x, label = pre(x.to(DEV, non_blocking=True)), label.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(net(x).float(), label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
            total += loss.item()
        score = val_score(net, pre)
        print(f"    epoch {epoch}: loss {total / len(dl):.3f}, val F1 {score:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, bad, state = score, 0, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(state)
    return net, pre, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["folds", "fixed"], default="folds")
    ap.add_argument("--tag", default="paper", help="output folder under data/thesis/pdcvit/")
    ap.add_argument("--recipe", choices=["paper", "ours"], default="paper")
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--lr", type=float, help="default 3e-5 (paper recipe) / 3e-4 (ours)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, help="clips per label (pipeline check)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = OUT / args.tag / f"realistic_{args.scheme}_w{WINDOW}.parquet"
    if out.exists():
        print(f"{out} exists - skipped")
        return
    per_video, _ = load(phase5.PHASE4 / "features", WINDOW)
    per_video = per_video[protocol_mask(per_video, "realistic")].reset_index(drop=True)
    if args.limit:
        per_video = pd.concat(g.sample(min(args.limit, len(g)), random_state=0)
                              for _, g in per_video.groupby(["label", "split"])).reset_index(drop=True)
    y = per_video.label.map(CLASSES.index).to_numpy()
    n_par = sum(p.numel() for p in PDCViT(channels=args.channels, patch=args.patch).parameters())
    print(f"PDC-ViT ({args.tag}: recipe {args.recipe}, channels {args.channels}, patch {args.patch}, "
          f"{n_par / 1e6:.1f} M parameters) realistic/{args.scheme}/w{WINDOW}: {len(per_video)} clips", flush=True)
    rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        t0 = time.time()
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        ev = te | cross | va
        idx, va_idx = np.flatnonzero(ev), np.flatnonzero(va)
        val_items = step8b.frame_items(per_video, va_idx, y, WINDOW)
        score = lambda net, pre: f1_score(y[va_idx], vit.predict(net, pre, val_items, len(va_idx), 3, vit.centre(480),
                                                                 args.workers).argmax(1), average="macro")
        net, pre, best = train(step8b.frame_items(per_video, np.flatnonzero(tr), y, WINDOW), score, args)
        p = vit.predict(net, pre, step8b.frame_items(per_video, idx, y, WINDOW), len(idx), 3, vit.tiles(480), args.workers)
        del net
        torch.cuda.empty_cache()
        meta = per_video.loc[ev, baselines.META].assign(
            split_name=name, part=np.select([cross[ev], va[ev]], ["crossdataset", "val"], "test"))
        rows.append(meta.assign(model="pdc_vit", **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        test = (~va[ev]) & (~cross[ev])
        print(f"  {name} pdc_vit: best val F1 {best:.3f}, test macro-F1 "
              f"{f1_score(y[idx][test], p[test].argmax(1), average='macro'):.3f} ({time.time() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
