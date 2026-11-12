"""Paper, day 3 - same-protocol baselines: published detector designs trained and tested on HVPF's splits.

  effnet_b4   end-to-end CNN on the raw I-frames (EfficientNet-B4, ImageNet weights; the backbone of the
              DFDC-winning deepfake detectors). Same training recipe as the HVPF ViT (Step 8b): random
              224 x 224 crops of the first 10 I-frames + flips, AdamW, one-cycle schedule, class-balanced loss
              with label smoothing, best epoch by validation macro F1; a clip = mean over 10 frames x 4 crops.
  clip_b16    UnivFD-style (Ojha et al., CVPR 2023): frozen CLIP ViT-B/16 image features of each frame,
              mean over the clip's frames, logistic regression (C chosen on the validation cameras).
  dinov2_b    the same with frozen DINOv2-base features.
No PRNU, Noiseprint++ or noise residual is used: these are the "learn it from pixels" alternatives to HVPF.
Same samples, protocol (realistic), splits (5-fold CV + the fixed split with its unseen datasets) and window
(10 frames) as the final HVPF. Writes data/paper/baselines/<model>/realistic_<scheme>_w10.parquet in Step 8's
format (part = val | test | crossdataset), so every comparison is clip by clip.

    .venv/bin/python scripts/paper_baselines.py --model effnet_b4 --scheme folds
    .venv/bin/python scripts/paper_baselines.py --model clip_b16 --scheme fixed
    .venv/bin/python scripts/paper_baselines.py --model effnet_b4 --scheme fixed --limit 200 --epochs 1   # check
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5, vit
from hvpf.model_data import CLASSES, load, protocol_mask, split_masks

step8b = phase5.load_step("08b_vit.py")
OUT = phase5.ROOT / "data" / "paper" / "baselines"
DEV = "cuda"
WINDOW = 10
META = ["sample_id", "label", "source", "variant", "social", "alpha", "stabilized", "claimed_device"]
HF = Path.home() / ".cache" / "huggingface" / "hub"


class Raw(nn.Module):
    """uint8 luma crops (B, 1, 228, 228) -> centre 224 x 224, 3 channels, ImageNet-normalised."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = x[..., 2:-2, 2:-2].float().div(255).expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std


def effnet():
    import torchvision
    net = torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 3)
    return net


def train_cnn(items, val_score, epochs, lr, batch, workers, patience=2, seed=0):
    """vit.train with an EfficientNet-B4 on raw frames instead of a ViT on SRM residuals."""
    torch.manual_seed(seed)
    net, pre = effnet().to(DEV), Raw().to(DEV)
    labels = np.array([it[2] for it in items])
    w = compute_class_weight("balanced", classes=np.arange(3), y=labels)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV), label_smoothing=0.1)
    dl = DataLoader(vit.Frames(items), batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                    pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    best, state, bad = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        net.train()
        t0, total = time.time(), 0.0
        for x, label, _ in dl:
            x, label = pre(x.to(DEV, non_blocking=True)), label.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(net(x), label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        score = val_score(net, pre)
        print(f"    epoch {epoch}: loss {total / len(dl):.3f}, val F1 {score:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, bad, state = score, 0, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net, pre


def local_model(repo):
    """Folder with the cached config + weights of a Hugging Face model (they can sit in different snapshot
    folders of the cache); links them together under data/paper/models/, nothing is downloaded."""
    snaps = sorted((HF / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"))
    files = {f.name: f for d in snaps for f in d.iterdir()}
    out = OUT.parent / "models" / repo.replace("/", "--")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors"):
        if not (out / name).exists():
            (out / name).symlink_to(files[name].resolve())
    return out


class Frozen(nn.Module):
    """Frozen foundation-model encoder: uint8 luma crops -> one embedding per crop."""

    def __init__(self, kind):
        super().__init__()
        from transformers import AutoModel
        repo = {"clip_b16": "openai/clip-vit-base-patch16", "dinov2_b": "facebook/dinov2-base"}[kind]
        model = AutoModel.from_pretrained(local_model(repo), local_files_only=True)
        self.kind = kind
        self.enc = model.vision_model if kind == "clip_b16" else model
        self.proj = model.visual_projection if kind == "clip_b16" else None
        mean, std = (((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)) if kind == "clip_b16"
                     else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        self.pre = Raw(mean, std)

    @torch.no_grad()
    def forward(self, x):
        out = self.enc(pixel_values=self.pre(x))
        if self.kind == "clip_b16":
            return self.proj(out.pooler_output)
        return torch.cat([out.last_hidden_state[:, 0], out.last_hidden_state[:, 1:].mean(1)], 1)


@torch.no_grad()
def embed(kind, per_video, workers, cache):
    """Mean frozen embedding over each clip's first 10 frames (centre crop); cached per model."""
    if cache.exists():
        e = pd.read_parquet(cache)
        if set(per_video.sample_id) <= set(e.index):
            return e.loc[per_video.sample_id].to_numpy()
    net = Frozen(kind).to(DEV).eval()
    items = step8b.frame_items(per_video, np.arange(len(per_video)), np.zeros(len(per_video), int), WINDOW)
    sums, count = None, np.zeros(len(per_video))
    for x, _, clip in DataLoader(vit.Frames(items, vit.centre(480)), batch_size=128, num_workers=workers, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.float16):
            f = net(x.to(DEV).flatten(0, 1)).float().cpu().numpy()
        sums = np.zeros((len(per_video), f.shape[1])) if sums is None else sums
        np.add.at(sums, clip.numpy(), f)
        np.add.at(count, clip.numpy(), 1)
    e = sums / np.maximum(count, 1)[:, None]
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(e, index=per_video.sample_id).rename(columns=str).to_parquet(cache)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["effnet_b4", "clip_b16", "dinov2_b"], required=True)
    ap.add_argument("--scheme", choices=["folds", "fixed"], default="folds")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, help="clips per label (pipeline check; writes *_check)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    per_video, _ = load(phase5.PHASE4 / "features", WINDOW)
    per_video = per_video[protocol_mask(per_video, "realistic")].reset_index(drop=True)
    if args.limit:
        per_video = pd.concat(g.sample(min(args.limit, len(g)), random_state=0)
                              for _, g in per_video.groupby("label")).reset_index(drop=True)
    y = per_video.label.map(CLASSES.index).to_numpy()
    print(f"baseline {args.model} realistic/{args.scheme}/w{WINDOW}: {len(per_video)} clips", flush=True)
    emb = None
    if args.model != "effnet_b4":
        emb = embed(args.model, per_video, args.workers, OUT / args.model / f"embeddings{'_check' if args.limit else ''}.parquet")
    rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        t0 = time.time()
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        ev = te | cross | va
        idx = np.flatnonzero(ev)
        if args.model == "effnet_b4":
            va_idx = np.flatnonzero(va)
            val_items = step8b.frame_items(per_video, va_idx, y, WINDOW)
            score = lambda net, pre: f1_score(y[va_idx], vit.predict(net, pre, val_items, len(va_idx), 3, vit.centre(480),
                                                                     args.workers).argmax(1), average="macro")
            net, pre = train_cnn(step8b.frame_items(per_video, np.flatnonzero(tr), y, WINDOW), score, args.epochs,
                                 args.lr, args.batch, args.workers)
            p = vit.predict(net, pre, step8b.frame_items(per_video, idx, y, WINDOW), len(idx), 3, vit.tiles(480), args.workers)
            del net
            torch.cuda.empty_cache()
        else:
            sc = StandardScaler().fit(emb[tr])
            best = None
            for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
                lr = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(sc.transform(emb[tr]), y[tr])
                f1 = f1_score(y[va], lr.predict(sc.transform(emb[va])), average="macro")
                best = (f1, lr) if best is None or f1 > best[0] else best
            p = best[1].predict_proba(sc.transform(emb[idx]))
        meta = per_video.loc[ev, META].assign(split_name=name, part=np.select([cross[ev], va[ev]], ["crossdataset", "val"], "test"))
        rows.append(meta.assign(model=args.model, **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        test = (~va[ev]) & (~cross[ev])
        print(f"  {name} {args.model}: test macro-F1 {f1_score(y[idx][test], p[test].argmax(1), average='macro'):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out = OUT / args.model / f"realistic_{args.scheme}_w{WINDOW}{'_check' if args.limit else ''}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
