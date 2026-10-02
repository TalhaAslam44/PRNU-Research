"""Thin wrapper around the Politecnico di Milano PRNU implementation (third_party/prnu-python)."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "prnu-python"))
import prnu  # noqa: E402


def _accumulate(frames, levels=4, sigma=5):
    """Numerator sum(W_i * I_i) and denominator sum(I_i^2) of the MLE fingerprint."""
    h, w = frames.shape[1:]
    rp = np.zeros((h, w), np.float32)
    nn = np.zeros((h, w), np.float32)
    for im in frames:
        im3 = np.ascontiguousarray(im)[..., None]
        rp += prnu.noise_extract(im3, levels, sigma) * (im / 255.0)
        nn += ((prnu.inten_scale(im3) * prnu.saturation(im3)) ** 2)[..., 0]
    return rp, nn


def fingerprint(frames, processes=1):
    """PRNU fingerprint K from aligned uint8 frames of shape (n, h, w).

    Same estimator and post-processing as prnu.extract_multiple_aligned, which only
    accepts 3-channel images (a single channel broadcasts to (h, w, h) there).
    processes > 1 splits the frames over a worker pool; keep 1 inside other pools.
    """
    if processes > 1 and len(frames) > 1:
        with Pool(processes) as pool:
            parts = pool.map(_accumulate, np.array_split(frames, min(processes, len(frames))))
    else:
        parts = [_accumulate(frames)]
    rp = sum(p[0] for p in parts)
    nn = sum(p[1] for p in parts)
    k = prnu.zero_mean_total(rp / (nn + 1))
    return prnu.wiener_dft(k, k.std(ddof=1)).astype(np.float32)


def residual(frame):
    """Noise residual W of one uint8 frame (h, w)."""
    return prnu.extract_single(np.ascontiguousarray(frame))


class FingerprintBank:
    """A set of reference fingerprints with cached FFTs, for scoring many queries.

    Matches prnu.crosscorr_2d + prnu.pce (same-size inputs) without recomputing
    the reference FFTs for every query.
    """

    def __init__(self, names, ks):
        self.names = list(names)
        ks = np.stack([k - k.mean() for k in ks]).astype(np.float32)
        self.norms = np.linalg.norm(ks.reshape(len(ks), -1), axis=1)
        self.ks = ks
        self.fft = fft2(ks)

    def score(self, w, radius=2):
        """Score a query residual/fingerprint against every reference.

        pce   peak anywhere (blind search, prnu.pce) - the standard detector
        pce0  PCE at zero shift - frames are already aligned, so the peak should be there
        ncc   normalized correlation at zero shift
        """
        w = (w - w.mean()).astype(np.float32)
        cc = np.real(ifft2(self.fft * fft2(np.rot90(w, 2))[None])).astype(np.float32)
        pce = np.array([prnu.pce(c, radius)["pce"] for c in cc])
        pce0 = pce_at_zero_shift(cc, radius)
        ncc = (self.ks.reshape(len(self.ks), -1) @ w.ravel()) / (self.norms * np.linalg.norm(w) + 1e-12)
        return pce, pce0, ncc


def pce_at_zero_shift(cc, radius=2):
    """PCE of crosscorr_2d output(s) (..., h, w) evaluated at zero shift.

    crosscorr_2d correlates with the query rotated by 180 degrees, so zero shift
    sits at (h-1, w-1). The floor energy excludes a (2r+1)^2 window around it (with wrap-around).
    """
    h, w = cc.shape[-2:]
    peak = cc[..., h - 1, w - 1]
    rows = (h - 1 + np.arange(-radius, radius + 1)) % h
    cols = (w - 1 + np.arange(-radius, radius + 1)) % w
    window = cc[..., rows[:, None], cols[None, :]]
    energy = ((cc ** 2).sum(axis=(-2, -1)) - (window ** 2).sum(axis=(-2, -1))) / (h * w - (2 * radius + 1) ** 2)
    return np.sign(peak) * peak ** 2 / energy
