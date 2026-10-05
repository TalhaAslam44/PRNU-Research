"""Step 7 features: PRNU, FFT and temporal (TSNCS) statistics from one pass over the frames.

Each frame is denoised once (prnu.noise_extract); the raw residual feeds the
clip's MLE fingerprint and the whitened residual W (as in prnu.extract_single)
feeds the per-frame statistics.
"""
import numpy as np
from numpy.fft import fft2
from scipy.stats import kurtosis, skew

from .prnu_utils import pce_at_zero_shift, prnu


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def _pce0(k, w):
    """Zero-shift PCE between a reference fingerprint (or I*K) and a residual, as crosscorr_2d + pce."""
    k = k - k.mean()
    w = w - w.mean()
    cc = np.real(np.fft.ifft2(fft2(k) * fft2(np.rot90(w, 2)))).astype(np.float32)
    return float(pce_at_zero_shift(cc))


class _Radial:
    """Cached frequency grids for one frame size."""

    def __init__(self, n):
        f = np.fft.fftfreq(n)
        fy, fx = np.meshgrid(f, f, indexing="ij")
        self.r = np.sqrt(fx ** 2 + fy ** 2) / 0.5            # 1 = Nyquist along an axis
        self.not_dc = self.r > 0
        self.hf = self.r > 0.5
        on8 = lambda g: np.isclose((np.abs(g) * 8) % 1, 0) & (np.abs(g) > 0)
        self.block8 = (on8(fx) | on8(fy)) & self.not_dc        # 8-px periodic lines (block artefacts)


_GRIDS = {}


def fft_stats(w):
    """Spectral statistics of a residual: injected or processed noise tends to leave periodic peaks."""
    g = _GRIDS.setdefault(w.shape[0], _Radial(w.shape[0]))
    p = np.abs(fft2(w)) ** 2
    pn = p[g.not_dc]
    med = np.median(pn) + 1e-12
    return {
        "fft_flatness": float(np.exp(np.mean(np.log(pn + 1e-12))) / (pn.mean() + 1e-12)),
        "fft_hf_ratio": float(p[g.hf].sum() / (pn.sum() + 1e-12)),
        "fft_peak_ratio": float(pn.max() / med),
        "fft_n_peaks": int((pn > 20 * med).sum()),
        "fft_block8": float(p[g.block8].mean() / pn.mean()),
    }


def clip_features(frames, k_claimed, window):
    """Per-frame feature dicts and the clip's raw fingerprint sums (all frames and first `window`).

    Returns (rows, (rp, nn), (rp_w, nn_w)): rows hold PRNU, residual, FFT and TSNCS values per frame.
    """
    h, w_ = frames.shape[1:]
    rp = np.zeros((h, w_), np.float32)
    nn = np.zeros((h, w_), np.float32)
    rows, prev, win = [], None, None
    for i, f in enumerate(frames):
        f3 = np.ascontiguousarray(f)[..., None]
        w_raw = prnu.noise_extract(f3)
        rp += w_raw * (f / 255.0)
        nn += ((prnu.inten_scale(f3) * prnu.saturation(f3)) ** 2)[..., 0]
        w = prnu.zero_mean_total(w_raw.copy())
        w = prnu.wiener_dft(w, w.std(ddof=1)).astype(np.float32)
        ik = f.astype(np.float32) * k_claimed
        row = {
            "frame": i,
            "prnu_ncc": _ncc(w, ik),
            "prnu_pce0": _pce0(ik, w),
            "res_std": float(w_raw.std()),
            "res_skew": float(skew(w_raw, axis=None)),
            "res_kurt": float(kurtosis(w_raw, axis=None)),
            "tsncs": _ncc(w, prev) if prev is not None else np.nan,
            **fft_stats(w),
        }
        rows.append(row)
        prev = w
        if i + 1 == window:
            win = (rp.copy(), nn.copy())
    if win is None:
        win = (rp.copy(), nn.copy())
    return rows, (rp, nn), win


def finish_fingerprint(rp, nn):
    """MLE fingerprint from accumulated sums (same post-processing as prnu_utils.fingerprint)."""
    k = prnu.zero_mean_total(rp / (nn + 1))
    return prnu.wiener_dft(k, k.std(ddof=1)).astype(np.float32)
