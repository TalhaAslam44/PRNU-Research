"""Step 8 - models: organic / manipulated / synthetic.

    baseline     conventional PCE rule on the claimed camera (binary: organic if PCE0 > 60);
                 `w_` uses the first `window` frames like the models, `v_` all frames
    svm, xgb     classical ML on clip-level features
    gru, cnn, transformer
                 temporal models on per-frame sequences (first `window` frames) whose pooled
                 output is joined with the clip-level features

Protocols (hvpf/model_data.protocol_mask): realistic | controlled (libx264 clips only).
Schemes: fixed (train/val/test, plus GenBuster as an unseen-source test) | folds (5-fold grouped CV).
Validation data picks hyper-parameters (SVM C, XGB depth) and early-stops the networks.

Writes data/results/models/<protocol>_<scheme>_<tag>.parquet: one row per test sample and model
with class probabilities (baseline rows carry only a score). Step 9 turns them into metrics.

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
    best = None
    for c in (0.3, 1, 3, 10):
        m = SVC(C=c, gamma="scale", class_weight="balanced").fit(xtr, ytr)
        f1 = f1_score(yva, m.predict(xva), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, m)
    return best[1], lambda x: softmax(best[1].decision_function(x))


def fit_xgb(xtr, ytr, xva, yva):
    w = compute_class_weight("balanced", classes=np.arange(3), y=ytr)
    best = None
    for depth in (4, 6):
        m = XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=depth, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", device=DEV, eval_metric="mlogloss",
                          early_stopping_rounds=50)
        m.fit(xtr, ytr, sample_weight=w[ytr], eval_set=[(xva, yva)], verbose=False)
        f1 = f1_score(yva, m.predict(xva), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, m)
    return best[1], best[1].predict_proba


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


def fit_seq(kind, data, ytr, yva, window, epochs=80, patience=10, seed=0):
    (str_, mtr, vtr), (sva, mva, vva) = data["train"], data["val"]
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

    best, best_state, bad = -1, None, 0
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(xs), device=DEV).split(128):
            opt.zero_grad()
            loss_fn(net(xs[b], ms[b], vs[b]), ys[b]).backward()
            opt.step()
        f1 = f1_score(yva, predict(sva, mva, vva).argmax(1), average="macro")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["realistic", "controlled"], default="realistic")
    ap.add_argument("--scheme", choices=["fixed", "folds"], default="fixed")
    ap.add_argument("--models", nargs="*", default=["baseline", "svm", "xgb", "gru", "cnn", "transformer"])
    ap.add_argument("--drop", nargs="*", default=[], help="feature groups to leave out (ablation)")
    ap.add_argument("--check", action="store_true", help="use the *_check feature files")
    args = ap.parse_args()

    cfg = load_config()
    paths, window = cfg["paths"], cfg["features"]["window"]
    per_video, frames = load(paths["data_dir"] / "features", window, "_check" if args.check else "")
    per_video = per_video[protocol_mask(per_video, args.protocol)].reset_index(drop=True)
    groups = [g for g in FRAME_GROUPS if g not in args.drop]
    vcols, fcols = feature_columns(per_video, groups), frame_columns(frames, groups)
    y = per_video.label.map(CLASSES.index).to_numpy()
    tag = "all" if not args.drop else "no_" + "_".join(sorted(args.drop))
    print(f"{args.protocol}/{args.scheme}/{tag}: {len(per_video)} samples, {len(vcols)} clip features, "
          f"{len(fcols)} frame features; classes {np.bincount(y, minlength=3).tolist()}")

    out_rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        evaluate = te | cross
        meta = per_video.loc[evaluate, ["sample_id", "label", "source", "variant", "alpha", "stabilized",
                                        "claimed_device"]].assign(split_name=name, part=np.where(cross[evaluate], "crossdataset", "test"))
        if "baseline" in args.models:
            for col in ("w_pce0_claimed", "v_pce0_claimed"):
                out_rows.append(meta.assign(model=f"baseline_{col[0]}", score_organic=per_video.loc[evaluate, col].to_numpy(),
                                            threshold=float(slog(60.0))))
        imp = SimpleImputer(strategy="median").fit(per_video.loc[tr, vcols])
        sc = StandardScaler().fit(imp.transform(per_video.loc[tr, vcols]))
        X = sc.transform(imp.transform(per_video[vcols]))
        X = np.nan_to_num(X)
        for model in [m for m in args.models if m in ("svm", "xgb")]:
            t0 = time.time()
            _, prob = (fit_svm if model == "svm" else fit_xgb)(X[tr], y[tr], X[va], y[va])
            p = prob(X[evaluate])
            out_rows.append(meta.assign(model=model, **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
            print(f"  {name} {model:11s} {time.time() - t0:6.1f}s")
        seq_models = [m for m in args.models if m in ("gru", "cnn", "transformer")]
        if seq_models:
            ids = per_video.sample_id.to_numpy()
            S, M = sequences(frames, ids, fcols, window)
            mu = S[tr][M[tr]].mean(0)
            sd = S[tr][M[tr]].std(0) + 1e-6
            S = np.where(M[..., None], (S - mu) / sd, 0).astype(np.float32)
            data = {"train": (S[tr], M[tr], X[tr]), "val": (S[va], M[va], X[va])}
            for model in seq_models:
                t0 = time.time()
                _, predict = fit_seq(model, data, y[tr], y[va], window)
                p = predict(S[evaluate], M[evaluate], X[evaluate])
                out_rows.append(meta.assign(model=model, **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
                print(f"  {name} {model:11s} {time.time() - t0:6.1f}s")

    out_dir = paths["results_dir"] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.concat(out_rows, ignore_index=True)
    path = out_dir / f"{args.protocol}_{args.scheme}_{tag}{'_check' if args.check else ''}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
