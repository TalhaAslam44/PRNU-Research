"""Step 4 - PRNU fingerprints and the conventional PCE baseline.

1. Reference fingerprint K per device from its reference videos
   (VISION: native flat videos; ACID: first N videos), saved to
   data/fingerprints/<dataset>/<device>.npy
2. Query fingerprint per sample video (native / YouTube / WhatsApp) from its
   I-frames, cached under data/fingerprints/queries/
3. PCE + normalized correlation of every query against every device of the same dataset
   -> data/results/prnu_baseline/scores.parquet (one row per query x candidate device)
4. Summary: closed-set attribution accuracy and PCE-threshold verification
   (TPR / FPR / AUC / EER) by version and stabilization, for PCE with a blind
   peak search (`pce`) and PCE at zero shift (`pce0`, frames are pre-aligned)
   -> data/results/prnu_baseline/summary.csv, summary_by_rotation.csv

Usage:
    python scripts/04_prnu_baseline.py
    python scripts/04_prnu_baseline.py --devices D01 D04 D11   # quick check on a subset
"""
import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.prnu_utils import FingerprintBank, fingerprint


def load_table(paths):
    videos = pd.read_csv(paths["metadata_dir"] / "videos.csv")
    frames = pd.read_csv(paths["metadata_dir"] / "frames_index.csv")
    frames = frames[frames.status == "ok"][["content_id", "version", "frames_path", "n_saved"]]
    return videos.merge(frames, on=["content_id", "version"])


def query_fingerprint(args):
    frames_path, out_path = args
    out = Path(out_path)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, fingerprint(np.load(frames_path), processes=1))
    return out_path


def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def summarize(scores, threshold, by=("version", "stabilized")):
    rows = []
    for key, g in scores.groupby(list(by)):
        true = g[g.is_match]
        false = g[~g.is_match]
        row = dict(zip(by, key), n_queries=g["query"].nunique())
        for s in ("pce", "pce0"):
            q = g.loc[g.groupby("query")[s].idxmax()]          # best-matching device per query
            row[f"acc_{s}"] = (q.candidate == q.device).mean()
            row[f"tpr@{s}"] = (true[s] > threshold).mean()
            row[f"fpr@{s}"] = (false[s] > threshold).mean()
            row[f"auc_{s}"] = roc_auc_score(g.is_match, g[s])
            row[f"eer_{s}"] = eer(g.is_match, g[s])
        row["auc_ncc"] = roc_auc_score(g.is_match, g.ncc)
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*")
    ap.add_argument("--datasets", nargs="*", default=["VISION"])
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--threshold", type=float, default=60.0, help="PCE decision threshold")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    fp_dir = paths["fingerprints_dir"]
    out_dir = paths["results_dir"] / "prnu_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(paths)
    table = table[table.dataset.isin(args.datasets)]
    if args.devices:
        table = table[table.device.isin(args.devices)]

    # 1. reference fingerprints
    refs = table[(table.role == "reference") & (table.version == "native")]
    bank_rows = []
    for (dataset, device), g in tqdm(refs.groupby(["dataset", "device"]), desc="reference K"):
        out = fp_dir / dataset / f"{device}.npy"
        if not out.exists():
            frames = np.concatenate([np.load(p) for p in g.frames_path])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, fingerprint(frames, processes=args.workers))
        bank_rows.append({"dataset": dataset, "device": device, "k_path": str(out),
                          "n_ref_videos": len(g), "n_ref_frames": int(g.n_saved.sum())})
    bank_df = pd.DataFrame(bank_rows)
    bank_df.to_csv(out_dir / "reference_fingerprints.csv", index=False)

    # 2. query fingerprints
    queries = table[table.role == "sample"].copy()
    # rotation flag of the original recording: social copies of rotated clips come out
    # slightly rescaled/shifted, which breaks pixel alignment (an RQ2 factor)
    native_rot = table[table.version == "native"].set_index("content_id").rotation
    queries["source_rotation"] = queries.content_id.map(native_rot).fillna(0).astype(int)
    queries["k_path"] = [str(fp_dir / "queries" / d / dev / Path(p).name)
                         for d, dev, p in zip(queries.dataset, queries.device, queries.frames_path)]
    with Pool(args.workers) as pool:
        list(tqdm(pool.imap_unordered(query_fingerprint, zip(queries.frames_path, queries.k_path)),
                  total=len(queries), desc="query K"))

    # 3. PCE of each query against every reference of its dataset
    records = []
    for dataset, qd in queries.groupby("dataset"):
        b = bank_df[bank_df.dataset == dataset]
        bank = FingerprintBank(b.device, [np.load(p) for p in b.k_path])
        for q in tqdm(qd.itertuples(), total=len(qd), desc=f"PCE {dataset}"):
            pce, pce0, ncc = bank.score(np.load(q.k_path))
            for cand, p, p0, c in zip(bank.names, pce, pce0, ncc):
                records.append((dataset, f"{q.content_id}__{q.version}", q.device, q.version, q.split,
                                bool(q.stabilized), q.source_rotation, cand, float(p), float(p0), float(c)))
    scores = pd.DataFrame(records, columns=["dataset", "query", "device", "version", "split", "stabilized",
                                            "source_rotation", "candidate", "pce", "pce0", "ncc"])
    scores["is_match"] = scores.device == scores.candidate
    scores.to_parquet(out_dir / "scores.parquet", index=False)

    summary = summarize(scores, args.threshold)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nPCE baseline (threshold {args.threshold:g}), {len(bank_df)} reference fingerprints:")
    print(summary.to_string(index=False))

    nonstab = scores[~scores.stabilized].assign(rotated=lambda d: d.source_rotation != 0)
    by_rot = summarize(nonstab, args.threshold, by=("version", "rotated"))
    by_rot.to_csv(out_dir / "summary_by_rotation.csv", index=False)
    print("\nNon-stabilized devices, by whether the original clip carried a rotation flag:")
    print(by_rot.to_string(index=False))


if __name__ == "__main__":
    main()
