"""Step 8 - models: organic / manipulated / synthetic.

    baseline     conventional PCE rule on the claimed camera (binary: organic if PCE0 > 60),
                 on the same first `window` frames as the models
    svm, xgb     classical ML on clip-level features
    gru, cnn, transformer
                 temporal models on per-frame sequences (first `window` frames) whose pooled
                 output is joined with the clip-level features

Protocols (hvpf/model_data.protocol_mask): realistic | controlled (libx264 clips only).
Schemes: fixed (train/val/test, plus FloreView + GenBuster as unseen sources) | folds (5-fold grouped CV).

80/20 training (hvpf-2): every split has an 80% training side (fixed: train + val camera
models; folds: the 4 folds that are not tested) and a 20% test side. The training side is used twice:
  1. tuning: fit on the tuning part, pick settings on the validation part (SVM C, XGB depth and
     number of trees, the networks' best epoch); validation predictions -> part = val (Step 8c)
  2. final: refit on the whole 80% with those settings; test / crossdataset predictions -> results

Writes data/results/models/<protocol>_<scheme>_<tag>.parquet: one row per evaluated sample and model
with class probabilities (baseline rows carry only a score); part = test | crossdataset | val.
Validation rows only tune calibration and ensembles (Step 8c); Step 9 turns the rest into metrics.

Usage:
    python scripts/08_models.py --protocol realistic --scheme fixed
    python scripts/08_models.py --protocol controlled --scheme folds --drop tsncs   # ablation
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf.config import load_config
from hvpf.model_data import (CLASSES, FRAME_GROUPS, feature_columns, frame_columns, load, protocol_mask,
                             sequences, slog, split_masks)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_svm(xtr, ytr, xva, yva):
    """Best C on the validation part -> (model, prob, refit(x, y) -> prob)."""
    best = None
    for c in (0.3, 1, 3, 10, 30):
        m = SVC(C=c, gamma="scale", class_weight="balanced").fit(xtr, ytr)
        f1 = f1_score(yva, m.predict(xva), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, m, c)
    _, m, c = best

    def refit(x, y):
        final = SVC(C=c, gamma="scale", class_weight="balanced").fit(x, y)
        return lambda z: softmax(final.decision_function(z))
    return m, lambda x: softmax(m.decision_function(x)), refit


def fit_xgb(xtr, ytr, xva, yva):
    w = compute_class_weight("balanced", classes=np.arange(3), y=ytr)
    best = None
    for depth, mcw in ((4, 1), (6, 1), (8, 1), (6, 5)):
        m = XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=depth, min_child_weight=mcw,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=DEV,
                          eval_metric="mlogloss", early_stopping_rounds=60)
        m.fit(xtr, ytr, sample_weight=w[ytr], eval_set=[(xva, yva)], verbose=False)
        f1 = f1_score(yva, m.predict(xva), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, m, depth, mcw)
    _, m, depth, mcw = best

    def refit(x, y):
        wf = compute_class_weight("balanced", classes=np.arange(3), y=y)
        final = XGBClassifier(n_estimators=m.best_iteration + 1, learning_rate=0.05, max_depth=depth,
                              min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              device=DEV, eval_metric="mlogloss")
        final.fit(x, y, sample_weight=wf[y], verbose=False)
        return final.predict_proba
    return m, m.predict_proba, refit


class SeqNet(nn.Module):
    def __init__(self, kind, d_frame, d_video, window, hidden=64):
        super().__init__()
        self.kind = kind
        self.inp = nn.Linear(d_frame, hidden)
        if kind == "gru":
            self.enc = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
            out = 2 * hidden
        elif kind == "cnn":
            self.enc = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU())
            out = hidden
        else:
            layer = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.pos = nn.Parameter(torch.zeros(1, window, hidden))
            out = hidden
        self.head = nn.Sequential(nn.Linear(out + d_video, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 3))

    def forward(self, x, mask, v):
        h = self.inp(x)
        if self.kind == "gru":
            lengths = mask.sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            h, _ = self.enc(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        elif self.kind == "cnn":
            h = self.enc((h * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
        else:
            h = self.enc(h + self.pos[:, : h.shape[1]], src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([pooled, v], 1))


def fit_seq(kind, data, ytr, yva, window, epochs=80, patience=10, seed=0, stop_after=None):
    """Early stopping on data["val"]; with `stop_after`, train exactly that many epochs (refit).
    Returns (net, predict, best_epoch)."""
    str_, mtr, vtr = data["train"]
    torch.manual_seed(seed)
    net = SeqNet(kind, str_.shape[2], vtr.shape[1], window).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)   # thesis: Adam, lr 0.001, L2
    w = torch.tensor(compute_class_weight("balanced", classes=np.arange(3), y=ytr), dtype=torch.float32, device=DEV)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=DEV)
    xs, ms, vs, ys = t(str_), t(mtr, torch.bool), t(vtr), t(ytr, torch.long)

    def predict(s, m, v):
        net.eval()
        with torch.no_grad():
            out = [torch.softmax(net(t(s[i:i + 1024]), t(m[i:i + 1024], torch.bool), t(v[i:i + 1024])), 1)
                   for i in range(0, len(s), 1024)]
        return torch.cat(out).cpu().numpy()

    best, best_state, best_epoch, bad = -1, None, 0, 0
    for epoch in range(1, (stop_after or epochs) + 1):
        net.train()
        for b in torch.randperm(len(xs), device=DEV).split(128):
            opt.zero_grad()
            loss_fn(net(xs[b], ms[b], vs[b]), ys[b]).backward()
            opt.step()
        if stop_after:
            best_epoch = epoch
            continue
        f1 = f1_score(yva, predict(*data["val"]).argmax(1), average="macro")
        if f1 > best:
            best, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, predict, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["realistic", "controlled"], default="realistic")
    ap.add_argument("--scheme", choices=["fixed", "folds"], default="fixed")
    ap.add_argument("--models", nargs="*", default=["baseline", "svm", "xgb", "gru", "cnn", "transformer"])
    ap.add_argument("--drop", nargs="*", default=[], help="feature groups to leave out (ablation)")
    ap.add_argument("--check", action="store_true", help="use the *_check feature files")
    ap.add_argument("--window", type=int, help="frames per clip (default: features.window); one of features.windows")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="keep only clips with at least this many I-frames (length-matched 5 vs 10 comparison)")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    window = args.window or cfg["features"]["window"]
    per_video, frames = load(paths["data_dir"] / "features", window, "_check" if args.check else "")
    per_video = per_video[protocol_mask(per_video, args.protocol)].reset_index(drop=True)
    if args.min_frames:
        per_video = per_video[per_video.n_frames >= args.min_frames].reset_index(drop=True)
    groups = [g for g in FRAME_GROUPS if g not in args.drop]
    vcols, fcols = feature_columns(per_video, groups), frame_columns(frames, groups)
    y = per_video.label.map(CLASSES.index).to_numpy()
    tag = ("all" if not args.drop else "no_" + "_".join(sorted(args.drop))) + f"_w{window}" \
        + (f"_min{args.min_frames}" if args.min_frames else "")
    print(f"{args.protocol}/{args.scheme}/{tag}: {len(per_video)} samples, {len(vcols)} clip features, "
          f"{len(fcols)} frame features; classes {np.bincount(y, minlength=3).tolist()}")

    probs = lambda p: {f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}
    meta_cols = ["sample_id", "label", "source", "variant", "social", "alpha", "stabilized", "claimed_device"]

    def features(mask):
        """Clip features with imputer + scaler fitted on `mask`."""
        imp = SimpleImputer(strategy="median").fit(per_video.loc[mask, vcols])
        sc = StandardScaler().fit(imp.transform(per_video.loc[mask, vcols]))
        return np.nan_to_num(sc.transform(imp.transform(per_video[vcols])))

    out_rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        full = tr | va                      # the 80% training side: tuning part + validation part
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        out = te | cross
        meta_va = per_video.loc[va, meta_cols].assign(split_name=name, part="val")
        meta_out = per_video.loc[out, meta_cols].assign(split_name=name, part=np.where(cross[out], "crossdataset", "test"))
        print(f"  {name}: train {full.sum()} (tuning {tr.sum()} + validation {va.sum()}), test {te.sum()}, "
              f"crossdataset {cross.sum()}")

        def add(model, p_va, p_out):        # tuning-stage validation predictions + final-model results
            out_rows.append(meta_va.assign(model=model, **probs(p_va)))
            out_rows.append(meta_out.assign(model=model, **probs(p_out)))
        if "baseline" in args.models:
            for meta, mask in ((meta_va, va), (meta_out, out)):
                out_rows.append(meta.assign(model="baseline", score_organic=per_video.loc[mask, "w_pce0_claimed"].to_numpy(),
                                            threshold=float(slog(60.0))))
        X_t, X_f = features(tr), features(full)
        for model in [m for m in args.models if m in ("svm", "xgb")]:
            t0 = time.time()
            _, prob, refit = (fit_svm if model == "svm" else fit_xgb)(X_t[tr], y[tr], X_t[va], y[va])
            add(model, prob(X_t[va]), refit(X_f[full], y[full])(X_f[out]))
            print(f"  {name} {model:11s} {time.time() - t0:6.1f}s")
        seq_models = [m for m in args.models if m in ("gru", "cnn", "transformer")]
        if seq_models:
            S0, M = sequences(frames, per_video.sample_id.to_numpy(), fcols, window)

            def norm(mask):
                mu, sd = S0[mask][M[mask]].mean(0), S0[mask][M[mask]].std(0) + 1e-6
                return np.where(M[..., None], (S0 - mu) / sd, 0).astype(np.float32)
            S_t, S_f = norm(tr), norm(full)
            data_t = {"train": (S_t[tr], M[tr], X_t[tr]), "val": (S_t[va], M[va], X_t[va])}
            data_f = {"train": (S_f[full], M[full], X_f[full])}
            for model in seq_models:
                t0 = time.time()
                _, predict, best_epoch = fit_seq(model, data_t, y[tr], y[va], window)
                _, predict_f, _ = fit_seq(model, data_f, y[full], None, window, stop_after=best_epoch)
                add(model, predict(S_t[va], M[va], X_t[va]), predict_f(S_f[out], M[out], X_f[out]))
                print(f"  {name} {model:11s} {time.time() - t0:6.1f}s (best epoch {best_epoch})")

    out_dir = paths["results_dir"] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.concat(out_rows, ignore_index=True)
    path = out_dir / f"{args.protocol}_{args.scheme}_{tag}{'_check' if args.check else ''}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
