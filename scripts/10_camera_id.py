"""Step 10 - source camera identification by training (hvpf-2; GPU, ~1 h).

Closed-set camera identification per dataset, like PDC-ViT (Elharrouss et al. 2025): every
camera is in training and the model names which one recorded a test video.

Split (80/20 per camera, seed splits.seed): 20% of each camera's recordings are test, drawn
from its natural (sample) videos only; the rest - including all flat/reference videos - are
training. All versions of a recording (VISION native / YouTube / WhatsApp) stay together, and
VISION's move / still / panrot recordings of the same scene count as one recording, so a test
scene is never seen in training. ACID and FloreView have no scene labels (one video = one recording).
Tuning: 10% of the recordings are a validation part inside the training side; the epoch with the
best validation accuracy is kept, then the model is refit on the whole 80% for that many epochs.

Model: hvpf/vit.py (pretrained ViT-S/16 on three SRM noise residuals) with one output per camera,
trained on random 224x224 crops of up to `--frames` evenly spaced I-frames per video (no flips:
they would mirror the sensor pattern). A video's prediction averages its frames x 4 crops.

Compared on the same test videos with:
  prnu     the training-free PRNU baseline of Step 4 (fingerprints from the reference videos,
           which are never test videos): the camera with the highest PCE0
  hybrid   PRNU when it is confident (best PCE0 > 60, the verification threshold), otherwise the ViT
  hybrid_val  the same with the PCE0 threshold chosen on the validation videos
Metrics: top-1 accuracy (also at camera-model level), macro FNR / FPR (one-vs-rest, as PDC-ViT
reports for Video-ACID), per version (VISION) and per PRNU usability.

Writes data/results/camera_id/{predictions_<DATASET>.parquet, summary.csv}.

Usage:
    python scripts/10_camera_id.py                         # VISION, ACID, FloreView
    python scripts/10_camera_id.py --datasets VISION --limit-devices 5 --epochs 2   # quick check
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import vit
from hvpf.config import load_config

THRESHOLD = 60.0


def recording(content_id, dataset):
    """Split unit: VISION's move/still/panrot takes of one scene are one recording."""
    if dataset == "VISION" and (m := re.fullmatch(r"(D\d+)_(\w+?)_(?:move|still|panrot)_(\d+)", content_id)):
        return f"{m[1]}_{m[2]}_{m[3]}"
    return content_id


def split_videos(v, seed, test_share=0.2, val_share=0.1):
    """part = test | val | train per video (val is the tuning holdout inside the training side)."""
    rng = np.random.default_rng(seed)
    part = pd.Series("train", index=v.index)
    for _, g in v.groupby("device", sort=True):
        recs = sorted(g.rec.unique())
        natural = sorted(g[g.role == "sample"].rec.unique())
        n_test = max(1, round(test_share * len(recs)))
        n_val = max(1, round(val_share * len(recs)))
        order = list(rng.permutation(natural))
        test, val = set(order[:n_test]), set(order[n_test:n_test + n_val])
        part[g.index[g.rec.isin(test)]] = "test"
        part[g.index[g.rec.isin(val)]] = "val"
    return part


def items_of(v, idx, max_frames):
    """(path, frame, camera, k) for up to `max_frames` evenly spaced frames of video v.iloc[idx[k]]."""
    out = []
    for k, i in enumerate(idx):
        n = int(v.n_saved.iat[i])
        frames = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
        out += [(v.frames_path.iat[i], int(f), int(v.label.iat[i]), k) for f in frames]
    return out


def rates(y, pred, n_classes):
    """Top-1 accuracy, macro FNR and FPR (one-vs-rest over the cameras present)."""
    fnr, fpr = [], []
    for c in np.unique(y):
        pos, hit = y == c, pred == c
        fnr.append(1 - (hit & pos).sum() / pos.sum())
        fpr.append((hit & ~pos).sum() / max((~pos).sum(), 1))
    return (pred == y).mean(), float(np.mean(fnr)), float(np.mean(fpr))


def prnu_top1(scores, queries):
    """Best candidate and its PCE0 per query from Step 4's scores."""
    s = scores[scores["query"].isin(queries)]
    best = s.loc[s.groupby("query").pce0.idxmax(), ["query", "candidate", "pce0"]]
    return best.set_index("query").reindex(queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["VISION", "ACID", "FloreView"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--frames", type=int, default=10, help="I-frames per video for training and prediction")
    ap.add_argument("--limit-devices", type=int, help="quick check on the first N cameras (writes *_check)")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    crop = cfg["frames"]["crop"]
    meta = paths["metadata_dir"]
    videos = pd.read_csv(meta / "videos.csv")
    frames = pd.read_csv(meta / "frames_index.csv").query("status == 'ok'")
    out_dir = paths["results_dir"] / "camera_id"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_check" if args.limit_devices else ""
    stab_vision = set(cfg["vision"]["stabilized_devices"])
    weak_acid = set(cfg["acid"]["prnu_weak_devices"])

    summary = []
    for ds in args.datasets:
        t0 = time.time()
        v = videos[(videos.dataset == ds) & videos.role.isin(["reference", "sample"])].merge(
            frames[["content_id", "version", "frames_path", "n_saved"]], on=["content_id", "version"])
        if args.limit_devices:
            v = v[v.device.isin(sorted(v.device.unique())[: args.limit_devices])]
        v = v.reset_index(drop=True)
        cams = sorted(v.device.unique())
        v["label"] = v.device.map({c: i for i, c in enumerate(cams)})
        v["rec"] = [recording(c, ds) for c in v.content_id]
        v["part"] = split_videos(v, cfg["splits"]["seed"])
        scores = pd.read_parquet(paths["results_dir"] / "prnu_baseline" / ds / "scores.parquet")
        if ds == "VISION":
            v["prnu_usable"] = ~v.device.isin(stab_vision)
        elif ds == "ACID":
            v["prnu_usable"] = ~v.device.isin(weak_acid)
        else:   # FloreView: cameras with >= 50% PRNU attribution in Step 4 (defined from the baseline itself)
            q = scores[scores.version == "native"]
            q = q.loc[q.groupby("query").pce0.idxmax()]
            acc = (q.candidate == q.device).groupby(q.device).mean()
            v["prnu_usable"] = v.device.map(acc >= 0.5).fillna(False)
        n_part = v.groupby("part").size().to_dict()
        print(f"\n{ds}: {len(cams)} cameras, videos {n_part} (training = train + val)", flush=True)

        common = dict(n_classes=len(cams), epochs=args.epochs, lr=args.lr, batch=args.batch, workers=args.workers,
                      flips=False)
        tr, va, te = (np.flatnonzero(v.part == p) for p in ("train", "val", "test"))
        y = v.label.to_numpy()
        val_items = items_of(v, va, args.frames)
        acc = lambda net, srm: float((vit.predict(net, srm, val_items, len(va), len(cams), vit.centre(crop),
                                                  args.workers).argmax(1) == y[va]).mean())
        net, srm, best_epoch = vit.train(items_of(v, tr, args.frames), val_score=acc, patience=args.patience, **common)
        # PCE0 threshold of the hybrid, chosen on the validation videos (tuning model)
        pv = np.array(cams)[vit.predict(net, srm, val_items, len(va), len(cams), vit.tiles(crop), args.workers).argmax(1)]
        bv = prnu_top1(scores, (v.content_id.iloc[va] + "__" + v.version.iloc[va]).tolist())
        grid = [0, 5, 10, 20, 30, 40, 60, 100, 200, np.inf]
        hyb = [(np.where(bv.pce0.fillna(-np.inf).to_numpy() > t, bv.candidate.to_numpy(), pv) == v.device.iloc[va].to_numpy()).mean()
               for t in grid]
        t_val = grid[int(np.argmax(hyb))]
        print(f"  hybrid threshold on validation: PCE0 > {t_val} (val accuracy {max(hyb):.3f})", flush=True)
        del net
        torch.cuda.empty_cache()
        print(f"  refit on train + val for {best_epoch} epochs", flush=True)
        net, srm, _ = vit.train(items_of(v, np.concatenate([tr, va]), args.frames), stop_after=best_epoch, **common)
        p = vit.predict(net, srm, items_of(v, te, args.frames), len(te), len(cams), vit.tiles(crop), args.workers)
        del net
        torch.cuda.empty_cache()

        test = v.iloc[te].copy()
        test["vit_pred"] = np.array(cams)[p.argmax(1)]
        test["vit_conf"] = p.max(1)
        best = prnu_top1(scores, (test.content_id + "__" + test.version).tolist())
        test["prnu_pred"] = best.candidate.to_numpy()
        test["prnu_pce0"] = best.pce0.to_numpy()
        test["hybrid_pred"] = np.where(test.prnu_pce0 > THRESHOLD, test.prnu_pred, test.vit_pred)
        test["hybrid_val_pred"] = np.where(test.prnu_pce0 > t_val, test.prnu_pred, test.vit_pred)
        test["hybrid_val_threshold"] = t_val
        model_of = videos.drop_duplicates("device").set_index("device").model_key
        test.drop(columns=["label"]).to_parquet(out_dir / f"predictions_{ds}{suffix}.parquet", index=False)

        groups = [("all", np.ones(len(test), bool)), ("usable PRNU", test.prnu_usable.to_numpy()),
                  ("no usable PRNU", ~test.prnu_usable.to_numpy())]
        if ds == "VISION":
            groups += [(f"version {x}", (test.version == x).to_numpy()) for x in ("native", "youtube", "whatsapp")]
        for gname, m in groups:
            if not m.any():
                continue
            g = test[m]
            row = {"dataset": ds, "group": gname, "cameras": g.device.nunique(), "videos": len(g)}
            for method in ("vit", "prnu", "hybrid", "hybrid_val"):
                pred = g[f"{method}_pred"].fillna("none").to_numpy()
                a, fnr, fpr = rates(g.device.to_numpy(), pred, len(cams))
                row |= {f"{method}_acc": a, f"{method}_fnr": fnr, f"{method}_fpr": fpr,
                        f"{method}_model_acc": (g.device.map(model_of).to_numpy() == pd.Series(pred).map(model_of).to_numpy()).mean()}
            summary.append(row)
            print(f"  {gname:16s} n={len(g):5d}  ViT {row['vit_acc']:.3f}  PRNU {row['prnu_acc']:.3f}  "
                  f"hybrid {row['hybrid_acc']:.3f}  hybrid_val {row['hybrid_val_acc']:.3f}", flush=True)
        print(f"  {ds} done in {time.time() - t0:.0f}s (best epoch {best_epoch})", flush=True)

    path = out_dir / f"summary{suffix}.csv"
    if path.exists() and not args.limit_devices:          # keep other datasets' rows from earlier runs
        old = pd.read_csv(path)
        summary = old[~old.dataset.isin(args.datasets)].to_dict("records") + summary
    pd.DataFrame(summary).to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
