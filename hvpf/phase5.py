"""Phase 5 (experiment, separate from Phases 3 and 4): processing-matched training, reliability-aware
fusion, an "inconclusive" output and camera-calibrated PRNU evidence. The Phase 3 / Phase 4 code is
reused by import, never changed; Phase 5 reads the frozen Phase 4 features (data/phase4/features) and
writes only to data/phase5/.

1. Processing-matched training. Every attack is encoded like the genuine `reencode` control (libx264,
   CRF attacks.crf = 20, same WhatsApp-/YouTube-like copies), but in that processing condition 80-91% of
   the training clips are attacks, so a model can learn "looks like a CRF-20 encode -> manipulated"
   (Phase 4 accepts only 38-45% of those genuine re-encodes, but 83-91% of the CRF-28 ones). Here
   the genuine clips of every condition (dataset, CRF, platform copy) that also holds attacks are
   weighted up to the attacks' count in that condition, before the usual class balancing. Class
   totals stay at 1/3 each as in Phase 4, so only the mix inside the genuine class changes, not the
   class prior.
2. Reliability-aware fusion. The Phase 4 stacking layer weights every model the same for every clip.
   `stack_rel` also sees three reliability indicators measured on the clip's frames (noise residual
   energy, 8x8 blockiness, high-frequency share; all already model inputs, so no new information),
   and the interactions model log-probability x indicator, so the weight of each model can change
   with how strongly the clip was compressed.
3. Inconclusive output (three zones on the genuine probability p_organic, forensic-report style):
   accept as genuine if p_organic >= tau_hi, accuse (manipulated or synthetic, whichever is more
   probable) if p_organic < tau_lo, otherwise "inconclusive" (refer to an examiner). Both thresholds
   come from the validation cameras: tau_lo lets at most ERROR_CAP of the genuine clips be accused,
   tau_hi lets at most ERROR_CAP of the attacked and at most ERROR_CAP of the AI clips be accepted.
   (A single threshold on the top class probability needed 44% inconclusive for the same false-
   accusation rate, so it was dropped.)
4. Camera-calibrated PRNU evidence (camera_calibration): the clip's PRNU score relative to the claimed
   camera's usual genuine score. Tried first and dropped: a compression-aware correlation predictor
   (Chen et al., TIFS 2008) on 120x120 blocks of single I-frames - after video compression the block
   correlation is mostly noise (expected vs observed r = 0.15 on genuine test blocks) and, at clip level,
   compression and content explain little of a genuine clip's PRNU strength (r = 0.22-0.33): the camera
   does. Quick check (XGBoost, 5-fold, 5 frames): F1 0.758 -> 0.773 with 10 calibration videos per camera.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "data" / "phase4"
PHASE5 = ROOT / "data" / "phase5"
ATTACK_SOURCES = ("VISION-attack", "ACID-attack")
RELIABILITY = ["res_std_mean", "fft_block8_mean", "fft_hf_ratio_mean"]
ERROR_CAP = 0.05


def load_step(filename):
    """Import a numbered pipeline script (e.g. 08_models.py) as a module without running it."""
    spec = importlib.util.spec_from_file_location(Path(filename).stem.replace("-", "_"), ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def processing_condition(per_video, attack_crf):
    """'<dataset>|crf<q>|<platform>' for re-encoded and attacked clips (Step 5), None for the others."""
    src = per_video.source.astype(str)
    base = per_video.variant.astype(str).str.split("+").str[0]
    crf = base.str.extract(r"crf(\d+)")[0].fillna(str(attack_crf))
    key = src.str.split("-").str[0] + "|crf" + crf + "|" + per_video.social.astype(str)
    return np.where(src.isin(ATTACK_SOURCES), key, None)


def matched_raw_weights(y, condition):
    """1 per clip; genuine clips (y = 0) of a condition that also holds attacks (y = 1) get
    n_attacks / n_genuine of that condition."""
    r = np.ones(len(y))
    for k in {c for c in condition if c is not None}:
        in_k = condition == k
        n_gen, n_att = (in_k & (y == 0)).sum(), (in_k & (y == 1)).sum()
        if n_gen and n_att:
            r[in_k & (y == 0)] = n_att / n_gen
    return r


def matched_weights(y, condition):
    """Processing-matched raw weights, then class balancing (each class 1/3 of the total). Mean 1."""
    r = matched_raw_weights(y, condition)
    w = np.zeros(len(y))
    classes = np.unique(y)
    for c in classes:
        w[y == c] = r[y == c] / (len(classes) * r[y == c].sum())
    return w * len(y)


def attack_share(y, condition, w):
    """Weighted share of attacks per matched condition (for the log)."""
    out = {}
    for k in sorted({c for c in condition if c is not None}):
        m = condition == k
        if (y[m] == 1).any():
            out[k] = w[m & (y == 1)].sum() / w[m].sum()
    return out


def rel_design(logp, rel):
    """Stacking inputs of stack_rel: log-probabilities, their interactions with the reliability
    indicators, and the indicators themselves."""
    inter = (logp[:, :, None] * rel[:, None, :]).reshape(len(logp), -1)
    return np.concatenate([logp, inter, rel], 1)


def fit_rel_stack(z, y, groups, cs=(0.01, 0.03, 0.1, 0.3, 1.0)):
    """Multinomial logistic regression on rel_design inputs, fit on validation predictions; C by 3-fold
    cross-validation over claimed cameras inside the validation set (best macro-F1)."""
    def model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000))

    best = None
    for c in cs:
        f1 = []
        for a, b in GroupKFold(3).split(z, y, groups):
            f1.append(f1_score(y[b], model(c).fit(z[a], y[a]).predict(z[b]), average="macro"))
        if best is None or np.mean(f1) > best[0]:
            best = (np.mean(f1), c)
    return model(best[1]).fit(z, y), best[1]


CALIBRATION = ["cal_pce0", "cal_ncc", "dev_pce0", "dev_ncc"]


def camera_calibration(per_video, n_videos=10, seed=0):
    """Change 4: camera-calibrated PRNU evidence. How strong a genuine fingerprint match is depends
    mostly on the camera (sensor, stabilisation), so the clip's PRNU score is compared with the claimed
    camera's usual score: the median w_pce0 / w_ncc of `n_videos` genuine native videos of that camera
    (reference material an examiner records with the camera), drawn at random but never the clip's own
    source video (leave-one-video-out). Adds CALIBRATION columns; per_video comes from model_data.load
    (w_* = the window's clip scores, PCE already signed-log)."""
    content = per_video.sample_id.str.split("__").str[0].to_numpy()
    native = ((per_video.label == "organic") & per_video.source.isin(["VISION", "ACID", "FloreView"])
              & (per_video.variant.astype(str) == "native")).to_numpy()
    rng = np.random.default_rng(seed)
    pools = {}
    for d in np.unique(per_video.device[native]):
        idx = np.flatnonzero(native & (per_video.device == d).to_numpy())
        pools[d] = idx[rng.permutation(len(idx))]
    pce0, ncc = per_video.w_pce0_claimed.to_numpy(), per_video.w_ncc_claimed.to_numpy()
    cal = np.full((len(per_video), 2), np.nan)
    for i, (d, c) in enumerate(zip(per_video.claimed_device, content)):
        pool = pools.get(d)
        if pool is None:
            continue
        pick = pool[content[pool] != c][:n_videos]
        if len(pick):
            cal[i] = np.median(pce0[pick]), np.median(ncc[pick])
    return pd.DataFrame({"cal_pce0": cal[:, 0], "cal_ncc": cal[:, 1], "dev_pce0": pce0 - cal[:, 0],
                         "dev_ncc": ncc / (np.abs(cal[:, 1]) + 1e-3)}, index=per_video.index)


def three_zone_thresholds(p_val, y_val, cap=ERROR_CAP):
    """(tau_lo, tau_hi) on p_organic from validation predictions (see 3. above)."""
    s = p_val[:, 0]
    lo = np.quantile(s[y_val == 0], cap)
    hi = max(np.quantile(s[y_val == c], 1 - cap) for c in (1, 2) if (y_val == c).any())
    if hi < lo:                 # no overlap at this cap: one threshold, nothing inconclusive
        lo = hi = (lo + hi) / 2
    return float(lo), float(hi)


def three_zone_decision(p, tau_lo, tau_hi):
    """0 organic / 1 manipulated / 2 synthetic per clip, -1 = inconclusive."""
    s = p[:, 0]
    accuse = 1 + p[:, 1:].argmax(1)
    return np.where(s >= tau_hi, 0, np.where(s < tau_lo, accuse, -1))
