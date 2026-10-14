"""Vision Transformer on I-frame noise residuals, shared by Step 8b (provenance) and Step 10
(camera identification).

Input: crops of the luma I-frames, turned into three SRM high-pass residuals (5x5 KV, 3x3
second order, horizontal second order, truncated at +-4 grey levels), so the network looks at
sensor / compression / generator noise rather than scene content. Model: pretrained ViT-S/16
(timm vit_small_patch16_224.augreg_in21k_ft_in1k, 21.7M parameters) with a new head.

80/20 training (hvpf-2): `train(..., val_score=...)` fits on the tuning part and keeps the
epoch with the best validation score; `train(..., stop_after=E)` then refits on all training
data (tuning + validation part) with the same learning-rate schedule, stopping after epoch E.
"""
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

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
    """items: (frames_path, frame, label, clip). crops None -> one random crop (+ random flips
    if `flips`); otherwise every (y, x) crop offset of the list."""

    def __init__(self, items, crops=None, flips=True):
        self.items, self.crops, self.flips = items, crops, flips

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, f, label, clip = self.items[i]
        a = np.load(path, mmap_mode="r")[f]
        if self.crops is None:          # torch RNG: numpy's would repeat across DataLoader workers
            y, x = (int(v) for v in torch.randint(0, a.shape[0] - SIZE + 1, (2,)))
            c = np.array(a[y:y + SIZE, x:x + SIZE])
            if self.flips and torch.rand(1) < 0.5:
                c = c[:, ::-1]
            if self.flips and torch.rand(1) < 0.5:
                c = c[::-1]
            x = torch.from_numpy(c.copy())[None]
        else:
            x = torch.stack([torch.from_numpy(np.array(a[y:y + SIZE, x:x + SIZE])) for y, x in self.crops])[:, None]
        return x, label, clip


def tiles(crop):
    """2x2 crop offsets covering a crop x crop frame."""
    return [(a, b) for a in (0, crop - SIZE) for b in (0, crop - SIZE)]


def centre(crop):
    return [((crop - SIZE) // 2,) * 2]


@torch.no_grad()
def predict(net, srm, items, n_clips, n_classes, crops, workers, batch=64):
    """Clip-level class probabilities: mean softmax over the clip's frames and crops."""
    net.eval()
    sums, count = np.zeros((n_clips, n_classes)), np.zeros(n_clips)
    for x, _, clip in DataLoader(Frames(items, crops), batch_size=batch, num_workers=workers, pin_memory=True):
        b, k = x.shape[:2]
        x = srm(x.to(DEV, non_blocking=True).flatten(0, 1))
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(net(x).float(), 1)
        p = p.view(b, k, n_classes).mean(1).cpu().numpy()
        np.add.at(sums, clip.numpy(), p)
        np.add.at(count, clip.numpy(), 1)
    return sums / np.maximum(count, 1)[:, None]


def train(items, n_classes, epochs, lr, batch, workers, val_score=None, stop_after=None, patience=2,
          flips=True, seed=0):
    """Fine-tune on `items` (path, frame, label, clip). With `val_score(net, srm) -> float`: keep
    the best epoch, stop after `patience` epochs without improvement. With `stop_after`: train
    exactly that many epochs of the same `epochs`-long schedule. Returns (net, srm, best_epoch)."""
    torch.manual_seed(seed)
    net = timm.create_model(ARCH, pretrained=True, num_classes=n_classes, drop_path_rate=0.1).to(DEV)
    srm = SRM().to(DEV)
    labels = np.array([it[2] for it in items])
    present = np.unique(labels)
    w = np.ones(n_classes, np.float32)
    w[present] = compute_class_weight("balanced", classes=present, y=labels)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV), label_smoothing=0.1)
    dl = DataLoader(Frames(items, flips=flips), batch_size=batch, shuffle=True, drop_last=True,
                    num_workers=workers, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl),
                                                pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler()
    best, best_state, best_epoch, bad = -np.inf, None, 0, 0
    for epoch in range(1, (stop_after or epochs) + 1):
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
        msg = f"    epoch {epoch}: {len(items)} frames, loss {total / len(dl):.3f}"
        if val_score is None:
            print(f"{msg} ({time.time() - t0:.0f}s)", flush=True)
            best_epoch = epoch
            continue
        score = val_score(net, srm)
        print(f"{msg}, val {score:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, srm, best_epoch
