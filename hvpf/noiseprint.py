"""Noiseprint++ camera-model noise maps (GRIP-UNINA, TruFor; nonprofit licence).

Fetch the network and weights first: third_party/get_noiseprintpp.sh
Noiseprint++ expects RGB in [0, 1]; our analysis frames are luma only (the attacks
also touch only luma), so the gray frame is replicated to three channels - closer
to the original, luminance-based Noiseprint than to Noiseprint++'s RGB training data.
"""
import sys
from pathlib import Path

import numpy as np
import torch

NP_DIR = Path(__file__).resolve().parents[1] / "third_party" / "noiseprintpp"


def load_model(device=None):
    """Noiseprint++ DnCNN exactly as TruFor builds it (17 layers, 64 features, 1 output plane)."""
    if not (NP_DIR / "noiseprint++.th").exists():
        raise FileNotFoundError("run third_party/get_noiseprintpp.sh first")
    sys.path.insert(0, str(NP_DIR))
    from DnCNN import make_net
    levels = 17
    net = make_net(3, kernels=[3] * levels, features=[64] * (levels - 1) + [1],
                   bns=[False] + [True] * (levels - 2) + [False], acts=["relu"] * (levels - 1) + ["linear"],
                   dilats=[1] * levels, bn_momentum=0.1, padding=1)
    net.load_state_dict(torch.load(NP_DIR / "noiseprint++.th", map_location="cpu")["network"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(device).eval(), device


@torch.no_grad()
def noiseprint_maps(net, device, frames, batch=8):
    """Noiseprint++ maps (n, h, w) float32 for gray uint8 frames (n, h, w)."""
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch])).to(device, torch.float32).div_(255)
        x = x[:, None].expand(-1, 3, -1, -1).contiguous()
        out.append(net(x)[:, 0].float().cpu().numpy())
    return np.concatenate(out)
