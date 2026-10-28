"""Intrinsic explanation extraction for X-UAD.

Three complementary read-outs are produced, all from a *single* forward pass:

1. **Localisation map** -- the decoder output, supervised by the pixel-level
   attack masks, answering "where is the evidence?".
2. **Quantum evidence spectrum** -- the Born probabilities of every spatial token
   against the learnable orthonormal basis, answering "what kind of evidence?".
   Each basis state is named post-hoc from the class it aligns with, so the names
   are *derived from the trained model*, not hand-assigned.
3. **Attention rollout over the AQA stack** -- answering "what did the model
   actually integrate?", including the per-head classical/Born mixing gates.

The faithfulness of (1) is measured by the standard deletion / insertion curves,
which is what separates an explanation that reflects the decision from one that
merely looks plausible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..data.attacks import ATTACK_NAMES

# Facial regions in normalised crop coordinates (x0, y0, x1, y1)
REGIONS = {
    "forehead":       (0.20, 0.05, 0.80, 0.30),
    "eye region":     (0.13, 0.28, 0.87, 0.50),
    "nose / midface": (0.30, 0.42, 0.70, 0.68),
    "mouth and chin": (0.25, 0.65, 0.75, 0.93),
    "left cheek":     (0.03, 0.42, 0.32, 0.78),
    "right cheek":    (0.68, 0.42, 0.97, 0.78),
    "face boundary":  (0.00, 0.00, 1.00, 1.00),   # handled specially
}


def region_profile(m: np.ndarray) -> Dict[str, float]:
    """Mean activation of the localisation map inside each facial region."""
    H, W = m.shape
    out = {}
    for name, (x0, y0, x1, y1) in REGIONS.items():
        if name == "face boundary":
            inner = np.zeros_like(m, bool)
            b = max(2, int(0.14 * H))
            inner[b:H - b, b:W - b] = True
            out[name] = float(m[~inner].mean())
        else:
            out[name] = float(m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)].mean())
    return out


def spatial_stats(m: np.ndarray) -> Dict[str, float]:
    a = float((m > 0.5).mean())
    p = m / max(m.sum(), 1e-8)
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(m.size))
    return {"area": a, "dispersion": ent, "peak": float(m.max())}


def spectral_descriptors(prof: np.ndarray, n_rad=16, n_ang=12) -> Dict[str, float]:
    """Radial-peak and azimuthal-anisotropy descriptors of the polar log-DFT."""
    P = prof.reshape(n_rad, n_ang)
    rad = P.mean(1)
    rad_n = (rad - rad.mean()) / (rad.std() + 1e-8)
    hi = rad_n[n_rad // 2:]
    ang = P.mean(0)
    return {
        "radial_peak_norm_freq": float((n_rad // 2 + int(np.argmax(hi))) / n_rad),
        "radial_peak_prominence": float(hi.max()),
        "azimuthal_anisotropy": float(ang.std() / (ang.mean() + 1e-8)),
    }


# --------------------------------------------------------------------------- #
@torch.no_grad()
def name_basis_states(model, loader, n_batches: int = 12, device="cpu") -> List[dict]:
    """Assign each quantum basis state the class it most selectively responds to."""
    model.eval()
    B = model.n_basis
    K = model.n_classes
    acc = np.zeros((K, B)); cnt = np.zeros(K)
    conc = np.zeros(B); nb = 0
    for i, (x, _m, y) in enumerate(loader):
        if i >= n_batches:
            break
        out = model(x.to(device))
        born = out["born"].cpu().numpy()               # (N,T,B)
        pooled = born.mean(1)
        for c in range(K):
            s = (y.numpy() == c)
            if s.any():
                acc[c] += pooled[s].sum(0); cnt[c] += s.sum()
        p = born / np.clip(born.sum(1, keepdims=True), 1e-9, None)
        conc += (-(p * np.log(p + 1e-12)).sum(1) / np.log(born.shape[1])).mean(0)
        nb += 1
    mean = acc / np.clip(cnt[:, None], 1, None)
    z = (mean - mean.mean(0, keepdims=True)) / (mean.std(0, keepdims=True) + 1e-8)
    conc /= max(nb, 1)
    info = []
    for b in range(B):
        c = int(np.argmax(z[:, b]))
        info.append({"basis": b, "class": ATTACK_NAMES[c], "class_id": c,
                     "selectivity": float(z[c, b]),
                     "dispersion": float(conc[b]),
                     "class_response": {ATTACK_NAMES[k]: float(mean[k, b]) for k in range(K)}})
    return info


# --------------------------------------------------------------------------- #
@torch.no_grad()
def attention_rollout(attns: List[torch.Tensor], n_spatial: int, grid: int) -> np.ndarray:
    """Abnar & Zuidema rollout, restricted to the CLS row and the spatial tokens."""
    if not attns:
        return None
    A = None
    for a in attns:
        a = a.mean(1)                                  # average heads: (N,T,T)
        a = a + torch.eye(a.shape[-1], device=a.device)[None]
        a = a / a.sum(-1, keepdim=True)
        A = a if A is None else torch.bmm(a, A)
    r = A[:, 0, 1:1 + n_spatial]
    r = r / r.sum(-1, keepdim=True).clamp_min(1e-9)
    return r.reshape(-1, grid, grid).cpu().numpy()


@torch.no_grad()
def deletion_insertion(model, x: torch.Tensor, saliency: np.ndarray,
                       n_steps: int = 12, device="cpu") -> Tuple[float, float]:
    """Faithfulness of ``saliency`` w.r.t. the model's own attack score.

    Deletion: progressively blur the most salient regions; a faithful map makes
    the attack score fall fast (low AUC).  Insertion: reveal the most salient
    regions of an otherwise blurred image; a faithful map makes the score rise
    fast (high AUC).
    """
    model.eval()
    N = x.shape[0]
    sal = torch.from_numpy(saliency).float()
    sal = F.interpolate(sal[:, None], size=x.shape[-2:], mode="bilinear",
                        align_corners=False)[:, 0]
    blur = F.avg_pool2d(x, 17, 1, 8)
    order = sal.reshape(N, -1).argsort(dim=1, descending=True)
    npx = order.shape[1]
    del_s, ins_s = [], []
    for i in range(n_steps + 1):
        k = int(npx * i / n_steps)
        mask = torch.zeros(N, npx)
        if k:
            mask.scatter_(1, order[:, :k], 1.0)
        mask = mask.reshape(N, 1, *x.shape[-2:])
        xd = x * (1 - mask) + blur * mask
        xi = blur * (1 - mask) + x * mask
        del_s.append(float((1 - model(xd.to(device))["prob"][:, 0]).mean()))
        ins_s.append(float((1 - model(xi.to(device))["prob"][:, 0]).mean()))
    return float(np.trapezoid(del_s, dx=1 / n_steps)), float(np.trapezoid(ins_s, dx=1 / n_steps))
