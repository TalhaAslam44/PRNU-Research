"""Phase 5, change 1 for the ViT - Step 8b with processing-matched training (optional, GPU, ~7 h).

Same network, training and prediction as Step 8b (hvpf/vit.py and scripts/08b_vit.py, imported
unchanged). hvpf.vit.train balances classes by item count, so processing matching is done by
repeating items: a genuine clip of a condition that also holds attacks is repeated
round(n_attacks / n_genuine) times (different random crops / flips each time), which gives the
same weighting as hvpf.phase5.matched_weights. Writes data/phase5/models/vit/<protocol>_<scheme>_w<N>.parquet;
scripts/p5_fusion.py uses it instead of the Phase 4 ViT when it exists.

Usage:
    python scripts/p5_vit.py --scheme folds --window 5
    python scripts/p5_vit.py --scheme fixed --limit 300 --epochs 1 --out-dir /tmp/check   # quick check
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
from hvpf import phase5, vit
from hvpf.config import load_config
from hvpf.model_data import CLASSES, load, protocol_mask, split_masks

step8b = phase5.load_step("08b_vit.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["realistic", "controlled"], default="realistic")
    ap.add_argument("--scheme", choices=["fixed", "folds"], default="folds")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, help="clips per label (quick check)")
    ap.add_argument("--features", type=Path, default=phase5.PHASE4 / "features")
    ap.add_argument("--out-dir", type=Path, default=phase5.PHASE5 / "models" / "vit")
    args = ap.parse_args()

    cfg = load_config()
    per_video, _ = load(args.features, args.window)
    per_video = per_video[protocol_mask(per_video, args.protocol)].reset_index(drop=True)
    if args.limit:
        per_video = pd.concat(g.sample(min(args.limit, len(g)), random_state=0)
                              for _, g in per_video.groupby("label")).reset_index(drop=True)
    y = per_video.label.map(CLASSES.index).to_numpy()
    condition = phase5.processing_condition(per_video, cfg["attacks"]["crf"])
    print(f"Phase 5 ViT {args.protocol}/{args.scheme}/w{args.window} (processing-matched): {len(per_video)} samples",
          flush=True)

    crop = cfg["frames"]["crop"]
    meta_cols = ["sample_id", "label", "source", "variant", "social", "alpha", "stabilized", "claimed_device"]
    out_rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        t0 = time.time()
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        evaluate = te | cross | va
        tr_idx = np.flatnonzero(tr)
        repeats = np.rint(phase5.matched_raw_weights(y[tr], condition[tr])).astype(int)
        items = []
        for i, r in zip(tr_idx, repeats):
            items += step8b.frame_items(per_video, [i], y, args.window) * r
        print(f"  {name}: train {tr.sum()} clips -> {len(items)} frames with repeats, val {va.sum()}", flush=True)
        va_idx = np.flatnonzero(va)
        val_items = step8b.frame_items(per_video, va_idx, y, args.window)
        score = lambda net, srm: f1_score(
            y[va_idx], vit.predict(net, srm, val_items, len(va_idx), 3, vit.centre(crop), args.workers).argmax(1),
            average="macro")
        net, srm, _ = vit.train(items, n_classes=3, epochs=args.epochs, lr=args.lr, batch=args.batch,
                                workers=args.workers, val_score=score, patience=args.patience)
        idx = np.flatnonzero(evaluate)
        p = vit.predict(net, srm, step8b.frame_items(per_video, idx, y, args.window), len(idx), 3, vit.tiles(crop),
                        args.workers)
        meta = per_video.loc[evaluate, meta_cols].assign(
            split_name=name, part=np.select([cross[evaluate], va[evaluate]], ["crossdataset", "val"], "test"))
        out_rows.append(meta.assign(model="vit", **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        test = ~va[evaluate] & ~cross[evaluate]
        print(f"  {name} vit: test macro-F1 {f1_score(y[idx][test], p[test].argmax(1), average='macro'):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del net
        torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.protocol}_{args.scheme}_w{args.window}.parquet"
    pd.concat(out_rows, ignore_index=True).to_parquet(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
