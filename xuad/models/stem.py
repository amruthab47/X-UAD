"""Low-level evidence extractors: central-difference convolutions and a compact
spectral branch.

Central Difference Convolution (CDC) generalises a vanilla convolution by mixing
in the *gradient* response around the kernel centre,

    y = (1 - theta) * sum_p w_p x(p)  +  theta * sum_p w_p (x(p) - x(p_0))
      = sum_p w_p x(p)  -  theta * (sum_p w_p) * x(p_0),

which makes the first layers sensitive to fine, illumination-invariant texture
detail -- the regime where print halftone, silicone smoothing and generator
over-smoothing live.  The identity above means CDC costs one extra 1x1
convolution, not a second full convolution.

The spectral branch supplies the evidence that is *invisible* in the spatial
domain at this resolution: the periodic sub-pixel lattice of a replayed frame
and the azimuthal peaks of an up-sampling generator.  A log-magnitude DFT is
pooled onto a polar (radius x angle) grid, which is translation invariant and
rotation-equivariant, and encoded by a tiny MLP into a small set of *spectral
tokens* that are concatenated to the spatial tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDConv2d(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, padding=1, theta=0.7, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding, bias=bias)
        self.theta = theta

    def forward(self, x):
        y = self.conv(x)
        if self.theta == 0:
            return y
        w = self.conv.weight.sum(dim=(2, 3), keepdim=True)          # (cout,cin,1,1)
        yd = F.conv2d(x, w, None, self.conv.stride, 0)
        return y - self.theta * yd


class CDCBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, theta=0.7):
        super().__init__()
        self.c1 = CDConv2d(cin, cout, 3, stride, 1, theta)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = CDConv2d(cout, cout, 3, 1, 1, theta)
        self.b2 = nn.BatchNorm2d(cout)
        self.act = nn.GELU()
        self.skip = (nn.Identity() if (cin == cout and stride == 1)
                     else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                        nn.BatchNorm2d(cout)))

    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.act(x + r)


class CDCStem(nn.Module):
    """128 -> 64 -> 32 -> 16 -> 8, returning the 32x32, 16x16 and 8x8 maps.

    ``deep=True`` adds a stride-1 CDC block after the two high-resolution stages
    (``e0`` at 64x64, ``e1`` at 32x32), the two-blocks-per-stage layout that the
    ResNet-18 baseline has there.  Fine periodic evidence (moire, halftone,
    checkerboard) is resolved at these resolutions before the token grid coarsens
    it; the low-resolution stages are left alone, so the parameter cost stays
    small.  Off by default; existing checkpoints load unchanged.
    """

    def __init__(self, dims=(24, 48, 96, 128), theta=0.7, in_ch=3, deep=False):
        super().__init__()
        d0, d1, d2, d3 = dims
        self.s0 = nn.Sequential(CDConv2d(in_ch, d0, 3, 2, 1, theta),
                                nn.BatchNorm2d(d0), nn.GELU())      # 64
        self.s1 = CDCBlock(d0, d1, 2, theta)                        # 32
        self.s2 = CDCBlock(d1, d2, 2, theta)                        # 16
        self.s3 = CDCBlock(d2, d3, 2, theta)                        # 8
        self.out_dims = (d1, d2, d3)
        self.deep = deep
        if deep:
            self.e0 = CDCBlock(d0, d0, 1, theta)
            self.e1 = CDCBlock(d1, d1, 1, theta)

    def forward(self, x):
        x = self.s0(x)
        if self.deep:
            x = self.e0(x)
            f32 = self.e1(self.s1(x))
            f16 = self.s2(f32)
            return f32, f16, self.s3(f16)
        f32 = self.s1(x)
        f16 = self.s2(f32)
        f8 = self.s3(f16)
        return f32, f16, f8


class SpectralBranch(nn.Module):
    """Polar-pooled log-DFT -> ``n_tokens`` spectral tokens of width ``dim``."""

    def __init__(self, dim, n_rad=16, n_ang=12, n_tokens=4):
        super().__init__()
        self.n_rad, self.n_ang, self.n_tokens = n_rad, n_ang, n_tokens
        self.enc = nn.Sequential(
            nn.LayerNorm(n_rad * n_ang),
            nn.Linear(n_rad * n_ang, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, n_tokens * dim),
        )
        self.dim = dim
        self._grid = None

    def _polar_index(self, H, W, device):
        if self._grid is not None and self._grid[0] == (H, W):
            return self._grid[1]
        yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing="ij")
        cy, cx = H // 2, W // 2
        r = torch.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2).float()) / (0.5 * min(H, W))
        a = torch.atan2((yy - cy).float(), (xx - cx).float()) % torch.pi   # 0..pi
        ri = (r.clamp(0, 0.999) * self.n_rad).long()
        ai = (a / torch.pi * self.n_ang).long().clamp(0, self.n_ang - 1)
        idx = (ri.clamp(0, self.n_rad - 1) * self.n_ang + ai).reshape(-1)
        self._grid = ((H, W), idx)
        return idx

    def forward(self, x):
        N = x.shape[0]
        g = x.mean(1)                                            # (N,H,W) luminance
        H, W = g.shape[-2:]
        mag = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs()
        mag = torch.log1p(mag).reshape(N, -1)
        idx = self._polar_index(H, W, x.device)
        acc = torch.zeros(N, self.n_rad * self.n_ang, device=x.device, dtype=mag.dtype)
        cnt = torch.zeros_like(acc)
        acc.index_add_(1, idx, mag)
        cnt.index_add_(1, idx, torch.ones_like(mag))
        prof = acc / cnt.clamp_min(1.0)
        return self.enc(prof).reshape(N, self.n_tokens, self.dim), prof
