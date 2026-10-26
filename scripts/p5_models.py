"""Phase 5, change 1 - Step 8 models with processing-matched training (hvpf/phase5.py).

Same models, features, splits and hyper-parameter grids as Step 8 (functions imported from
scripts/08_models.py, unchanged); the only difference is the per-clip training weight
(hvpf.phase5.matched_weights). Reads the frozen Phase 4 features and writes
data/phase5/models/<protocol>_<scheme>_all_w<N>.parquet (same format as Step 8; part = test |
crossdataset | val). Phase 3 / Phase 4 files are not touched.

Usage:
    python scripts/p5_models.py --scheme folds --window 5
    python scripts/p5_models.py --scheme fixed --window 10
    python scripts/p5_models.py --scheme fixed --models baseline cnn --out-dir /tmp/check   # quick check
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.config import load_config
from hvpf.model_data import (CLASSES, FRAME_GROUPS, feature_columns, frame_columns, load, protocol_mask, sequences,
                             slog, split_masks)

step8 = phase5.load_step("08_models.py")


def fit_split(per_video, frames, vcols, fcols, y, tr, va, evaluate, cross, name, window, models, sw):
    """Step 8's models on one split with per-clip training weights sw (None = Step 8's class weights);
    prediction rows for the evaluated clips (also used by scripts/p5_c4_models.py)."""
    rows = []
    meta = per_video.loc[evaluate, ["sample_id", "label", "source", "variant", "social", "alpha", "stabilized",
                                    "claimed_device"]].assign(
        split_name=name, part=np.select([cross[evaluate], va[evaluate]], ["crossdataset", "val"], "test"))
    if "baseline" in models:
        rows.append(meta.assign(model="baseline", score_organic=per_video.loc[evaluate, "w_pce0_claimed"].to_numpy(),
                                threshold=float(slog(60.0))))
    imp = SimpleImputer(strategy="median").fit(per_video.loc[tr, vcols])
    sc = StandardScaler().fit(imp.transform(per_video.loc[tr, vcols]))
    X = np.nan_to_num(sc.transform(imp.transform(per_video[vcols])))
    for model in [m for m in models if m in ("svm", "xgb")]:
        t0 = time.time()
        _, prob = (step8.fit_svm if model == "svm" else step8.fit_xgb)(X[tr], y[tr], X[va], y[va], sw=sw)
        p = prob(X[evaluate])
        rows.append(meta.assign(model=model, **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        print(f"  {name} {model:11s} {time.time() - t0:6.1f}s", flush=True)
    seq_models = [m for m in models if m in ("gru", "cnn", "transformer")]
    if seq_models:
        S, M = sequences(frames, per_video.sample_id.to_numpy(), fcols, window)
        mu, sd = S[tr][M[tr]].mean(0), S[tr][M[tr]].std(0) + 1e-6
        S = np.where(M[..., None], (S - mu) / sd, 0).astype(np.float32)
        data = {"train": (S[tr], M[tr], X[tr]), "val": (S[va], M[va], X[va])}
        for model in seq_models:
            t0 = time.time()
            _, predict = step8.fit_seq(model, data, y[tr], y[va], window, sw=sw)
            p = predict(S[evaluate], M[evaluate], X[evaluate])
            rows.append(meta.assign(model=model, **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
            print(f"  {name} {model:11s} {time.time() - t0:6.1f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["realistic", "controlled"], default="realistic")
    ap.add_argument("--scheme", choices=["fixed", "folds"], default="folds")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=["baseline", "svm", "xgb", "gru", "cnn", "transformer"])
    ap.add_argument("--features", type=Path, default=phase5.PHASE4 / "features")
    ap.add_argument("--out-dir", type=Path, default=phase5.PHASE5 / "models")
    args = ap.parse_args()

    cfg = load_config()
    per_video, frames = load(args.features, args.window)
    per_video = per_video[protocol_mask(per_video, args.protocol)].reset_index(drop=True)
    groups = list(FRAME_GROUPS)
    vcols, fcols = feature_columns(per_video, groups), frame_columns(frames, groups)
    y = per_video.label.map(CLASSES.index).to_numpy()
    condition = phase5.processing_condition(per_video, cfg["attacks"]["crf"])
    tag = f"all_w{args.window}"
    print(f"Phase 5 {args.protocol}/{args.scheme}/{tag} (processing-matched): {len(per_video)} samples, "
          f"{len(vcols)} clip features, {len(fcols)} frame features; classes {np.bincount(y, minlength=3).tolist()}",
          flush=True)

    out_rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        evaluate = te | cross | va
        sw = phase5.matched_weights(y[tr], condition[tr])
        before = phase5.attack_share(y[tr], condition[tr], phase5.matched_weights(y[tr], np.full(tr.sum(), None)))
        after = phase5.attack_share(y[tr], condition[tr], sw)
        print(f"  {name}: attack share of the training weight per matched condition (Phase 4 -> Phase 5): "
              + ", ".join(f"{k} {before[k]:.2f}->{after[k]:.2f}" for k in after), flush=True)
        out_rows += fit_split(per_video, frames, vcols, fcols, y, tr, va, evaluate, cross, name, args.window,
                              args.models, sw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.protocol}_{args.scheme}_{tag}.parquet"
    out = pd.concat(out_rows, ignore_index=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
