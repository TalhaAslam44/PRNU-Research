"""The final HVPF as one saved model (for attacks that query it clip by clip) and feature extraction for new clips.

Final HVPF = Phase 5 changes 1+2+4 with a standard (Phase 4 recipe) ViT, 10 frames, chosen on validation
(scripts/paper_select.py): SVM / XGBoost / GRU / 1D-CNN / Transformer on the Phase 4 features plus camera
calibration, trained with processing-matched weights; ViT-S/16 on SRM residuals; reliability-aware
stacking fitted on the validation cameras; three-zone thresholds from validation. The pipeline steps never
kept trained models (they predicted and moved on), so `train_bundle` retrains them on one split and saves
them (scripts/paper_bundle.py).

`ClipFeatures` computes exactly Step 7 (scripts/07_features.process) and Step 7b (Noiseprint++) for new
clips and returns them in model_data.load format, so a new clip goes through the same code path as the
benchmark clips.
"""
import multiprocessing as mp
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from . import phase5, vit
from .config import load_config
from .features import fft_stats
from .model_data import (CLASSES, FRAME_GROUPS, feature_columns, frame_columns, load, protocol_mask, sequences,
                         split_masks)
from .noiseprint import load_model, noiseprint_maps

step7 = phase5.load_step("07_features.py")
step7b = phase5.load_step("07b_noiseprint.py")
step8 = phase5.load_step("08_models.py")
step8b = phase5.load_step("08b_vit.py")
step8c = phase5.load_step("08c_ensemble.py")
SEQ = ("gru", "cnn", "transformer")
PAPER = phase5.ROOT / "data" / "paper"
FEATURES = PAPER / "features_newpc"      # Phase 4 features with Noiseprint++ recomputed on this PC (paper_noiseprint.py)
NP_REFS = PAPER / "noiseprint_refs"      # Noiseprint++ reference spectra computed on this PC
ORDER = sorted(["svm", "xgb", *SEQ, "vit"])          # the stacking input order used in Phase 5


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def with_calibration(per_video, reference):
    """Camera calibration of `per_video` rows against the genuine natives of `reference` (both load() tables)."""
    both = pd.concat([reference, per_video[~per_video.sample_id.isin(reference.sample_id)]], ignore_index=True)
    cal = phase5.camera_calibration(both).set_index(both.sample_id)
    return per_video.join(cal, on="sample_id")


class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        self.m = joblib.load(self.path / "bundle.joblib")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.seq = {}
        for kind in SEQ:
            net = step8.SeqNet(kind, len(self.m["fcols"]), len(self.m["vcols"]), self.m["window"]).to(self.dev)
            net.load_state_dict(torch.load(self.path / f"{kind}.pt", map_location=self.dev))
            self.seq[kind] = net.eval()
        import timm
        self.vit = timm.create_model(vit.ARCH, pretrained=False, num_classes=3).to(self.dev)
        self.vit.load_state_dict(torch.load(self.path / "vit.pt", map_location=self.dev))
        self.srm = vit.SRM().to(self.dev)

    def base_probs(self, per_video, frames, workers=8):
        """Class probabilities of every base model for load()-format rows (with calibration columns)."""
        m = self.m
        X = np.nan_to_num(m["scaler"].transform(m["imputer"].transform(per_video[m["vcols"]])))
        out = {"svm": softmax(m["svm"].decision_function(X)), "xgb": m["xgb"].predict_proba(X)}
        S, M = sequences(frames, per_video.sample_id.to_numpy(), m["fcols"], m["window"])
        S = np.where(M[..., None], (S - m["mu"]) / m["sd"], 0).astype(np.float32)
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.dev)
        with torch.no_grad():
            for kind, net in self.seq.items():
                out[kind] = np.concatenate([torch.softmax(net(t(S[i:i + 1024]), t(M[i:i + 1024], torch.bool),
                                                              t(X[i:i + 1024])), 1).cpu().numpy()
                                            for i in range(0, len(S), 1024)])
        items = step8b.frame_items(per_video, np.arange(len(per_video)), np.zeros(len(per_video), int), m["window"])
        out["vit"] = vit.predict(self.vit, self.srm, items, len(per_video), 3, vit.tiles(m["crop"]), workers)
        return out

    def predict(self, per_video, frames, workers=8):
        """Final probabilities (organic, manipulated, synthetic), default decision and three-zone decision."""
        m = self.m
        base = self.base_probs(per_video, frames, workers)
        r = per_video[phase5.RELIABILITY].to_numpy(float)
        r = np.where(np.isnan(r), m["rel_median"], r)
        r = (r - m["rel_mean"]) / m["rel_std"]
        z = phase5.rel_design(np.concatenate([step8c.logp(base[k]) for k in ORDER], 1), r)
        p = m["stacker"].predict_proba(z)
        return p, p.argmax(1), phase5.three_zone_decision(p, m["tau_lo"], m["tau_hi"]), base


def train_bundle(out, window=10, scheme="fixed", vit_epochs=8, quick=False, workers=10, features=FEATURES):
    """Train the final HVPF on one split (default: the fixed split) and save it to `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    per_video, frames = load(Path(features), window)
    per_video = per_video[protocol_mask(per_video, "realistic")].reset_index(drop=True)
    per_video = pd.concat([per_video, phase5.camera_calibration(per_video)], axis=1)
    groups = list(FRAME_GROUPS)
    vcols, fcols = feature_columns(per_video, groups) + phase5.CALIBRATION, frame_columns(frames, groups)
    y = per_video.label.map(CLASSES.index).to_numpy()
    condition = phase5.processing_condition(per_video, cfg["attacks"]["crf"])
    _, tr, va, te = next(split_masks(per_video, scheme))
    cross = (per_video.split == "crossdataset").to_numpy()
    sw = phase5.matched_weights(y[tr], condition[tr])
    imp = SimpleImputer(strategy="median").fit(per_video.loc[tr, vcols])
    sc = StandardScaler().fit(imp.transform(per_video.loc[tr, vcols]))
    X = np.nan_to_num(sc.transform(imp.transform(per_video[vcols])))
    b = {"vcols": vcols, "fcols": fcols, "window": window, "crop": cfg["frames"]["crop"], "imputer": imp, "scaler": sc}
    print("training SVM / XGBoost ...", flush=True)
    b["svm"], _ = step8.fit_svm(X[tr], y[tr], X[va], y[va], sw=sw)
    b["xgb"], _ = step8.fit_xgb(X[tr], y[tr], X[va], y[va], sw=sw)
    S, M = sequences(frames, per_video.sample_id.to_numpy(), fcols, window)
    b["mu"], b["sd"] = S[tr][M[tr]].mean(0), S[tr][M[tr]].std(0) + 1e-6
    Sn = np.where(M[..., None], (S - b["mu"]) / b["sd"], 0).astype(np.float32)
    data = {"train": (Sn[tr], M[tr], X[tr]), "val": (Sn[va], M[va], X[va])}
    for kind in SEQ:
        print(f"training {kind} ...", flush=True)
        net, _ = step8.fit_seq(kind, data, y[tr], y[va], window, sw=sw)
        torch.save(net.state_dict(), out / f"{kind}.pt")
    print("training ViT ...", flush=True)
    va_idx = np.flatnonzero(va)
    val_items = step8b.frame_items(per_video, va_idx, y, window)
    items = step8b.frame_items(per_video, np.flatnonzero(tr), y, window)
    if quick:                                   # pipeline check only: a fraction of the frames, 1 epoch
        items, val_items, va_idx = items[::20], [], va_idx
    score = lambda net, srm: f1_score(
        y[va_idx], vit.predict(net, srm, val_items, len(va_idx), 3, vit.centre(b["crop"]), workers).argmax(1),
        average="macro")
    net, _, _ = vit.train(items, 3, epochs=1 if quick else vit_epochs, lr=1e-4, batch=128, workers=workers,
                          val_score=None if quick else score, stop_after=1 if quick else None)
    torch.save(net.state_dict(), out / "vit.pt")
    del net
    torch.cuda.empty_cache()
    joblib.dump(b, out / "bundle.joblib")             # base models saved; stacker fitted below

    print("stacking on the validation cameras ...", flush=True)
    bundle = Bundle(out)
    rows = va | te | cross
    base = bundle.base_probs(per_video[rows].reset_index(drop=True), frames, workers)
    yv, val = y[rows], va[rows]
    r = per_video.loc[rows, phase5.RELIABILITY].to_numpy(float)
    b["rel_median"] = np.nanmedian(r[val], 0)
    r = np.where(np.isnan(r), b["rel_median"], r)
    b["rel_mean"], b["rel_std"] = r[val].mean(0), r[val].std(0) + 1e-9
    z = phase5.rel_design(np.concatenate([step8c.logp(base[k]) for k in ORDER], 1), (r - b["rel_mean"]) / b["rel_std"])
    b["stacker"], c = phase5.fit_rel_stack(z[val], yv[val], per_video.loc[rows, "claimed_device"].to_numpy()[val])
    p = b["stacker"].predict_proba(z)
    b["tau_lo"], b["tau_hi"] = phase5.three_zone_thresholds(p[val], yv[val])
    joblib.dump(b, out / "bundle.joblib")
    meta = per_video.loc[rows, ["sample_id", "label", "source", "variant", "claimed_device"]].assign(
        part=np.select([cross[rows], va[rows]], ["crossdataset", "val"], "test"),
        **{f"p_{k}": p[:, i] for i, k in enumerate(CLASSES)})
    meta.to_parquet(out / "predictions.parquet", index=False)
    return meta


def _worker_init(refs, fcfg):
    """Step 7 worker: one math thread each (12 workers x all cores would oversubscribe the CPU)."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    step7._init(refs, fcfg)


class ClipFeatures:
    """Steps 7 + 7b for new clips; `table(samples)` -> (per_video, frames) as model_data.load returns them.

    Create it before any CUDA work: the Step 7 workers are forked processes."""

    def __init__(self, workers=12, np_refs=None):
        cfg = load_config()
        self.np_refs = Path(np_refs) if np_refs else NP_REFS
        self.cfg, self.paths = cfg, cfg["paths"]
        refs = {p.stem: str(p) for ds in ("VISION", "ACID", "FloreView")
                for p in (self.paths["fingerprints_dir"] / ds).glob("*.npy")}
        self.pool = mp.get_context("fork").Pool(workers, initializer=_worker_init,
                                                initargs=(pd.Series(refs), cfg["features"]))
        self.windows = cfg["features"]["windows"]
        self.net = None

    def _noiseprint(self, samples):
        if self.net is None:
            self.net, self.dev = load_model()
            self.spectra = step7b.Spectra(self.cfg["frames"]["crop"])
            specs = {p.stem: np.load(p) for p in sorted(self.np_refs.glob("*.npy"))}
            self.common = np.mean(list(specs.values()), axis=0)
            self.names = list(specs)
            self.bank = np.stack([specs[d] - self.common for d in self.names])
        fparts, vrows = [], []
        for s in samples.itertuples():
            maps = noiseprint_maps(self.net, self.dev, np.load(s.frames_path)[: max(self.windows)])
            pf = pd.DataFrame([{"frame": i, "np_std": float(m.std()), "np_kurt": float(kurtosis(m, axis=None)),
                                **{f"np_{k.removeprefix('fft_')}": v for k, v in fft_stats(m).items()}}
                               for i, m in enumerate(maps)]).assign(sample_id=s.sample_id)
            fparts.append(pf)
            i = self.names.index(s.claimed_device)
            v = {"sample_id": s.sample_id, "label": s.label, "variant": s.variant}
            for win in self.windows:
                spec = self.spectra(maps[:win]) - self.common
                sims = np.array([step7b._ncc(spec, b) for b in self.bank])
                v[f"np_spec_ncc_claimed_w{win}"] = float(sims[i])
                v[f"np_spec_ncc_max_other_w{win}"] = float(np.delete(sims, i).max())
            for col in pf.columns.drop(["frame", "sample_id"]):
                v[f"{col}_mean"], v[f"{col}_std"] = float(pf[col].mean()), float(pf[col].std())
            vrows.append(v)
        return pd.concat(fparts, ignore_index=True), pd.DataFrame(vrows)

    def table(self, samples, window):
        """samples: DataFrame with hvpf.samples.SAMPLE_COLS; returns load(window)-format (per_video, frames)."""
        res = self.pool.map(step7.process, samples.to_dict("records"), chunksize=1)
        frames = pd.concat([f for f, _ in res], ignore_index=True)
        videos = samples.merge(pd.DataFrame([v for _, v in res]), on="sample_id")
        np_frames, np_videos = self._noiseprint(samples)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            videos.to_parquet(tmp / "videos.parquet", index=False)
            frames.to_parquet(tmp / "frames.parquet", index=False)
            np_frames.to_parquet(tmp / "noiseprint_frames.parquet", index=False)
            np_videos.to_parquet(tmp / "noiseprint_videos.parquet", index=False)
            return load(tmp, window)
