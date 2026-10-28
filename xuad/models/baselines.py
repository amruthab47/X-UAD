"""Reference detectors, all trained under an identical budget for fairness."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem import CDCStem, CDConv2d


class PlainAttention(nn.Module):
    def __init__(self, dim, heads=4, drop=0.0):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.last_attn = None
        self.last_gate = None

    def forward(self, x, need_weights=False):
        N, T, D = x.shape
        qkv = self.qkv(x).reshape(N, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, -1)
        self.last_attn = a.detach() if need_weights else None
        o = self.drop(self.proj((a @ v).transpose(1, 2).reshape(N, T, D)))
        ent = -(a.clamp_min(1e-9) * a.clamp_min(1e-9).log()).sum(-1).mean()
        return o, ent


class PlainBlock(nn.Module):
    """Classical-attention counterpart of AQABlock, with the same LayerScale so
    that the AQA ablation differs *only* in the attention scoring rule."""

    def __init__(self, dim, heads=4, mlp_ratio=2.0, drop=0.0, drop_path=0.0,
                 gate_bias=None, layer_scale=1e-2):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = PlainAttention(dim, heads, drop)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, dim), nn.Dropout(drop))
        self.dp = drop_path
        self.ls1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.ls2 = nn.Parameter(layer_scale * torch.ones(dim))

    def _dp(self, x):
        if self.dp <= 0 or not self.training:
            return x
        keep = 1 - self.dp
        return x * x.new_empty(x.shape[0], 1, 1).bernoulli_(keep) / keep

    def forward(self, x, need_weights=False):
        a, e = self.attn(self.n1(x), need_weights)
        x = x + self._dp(self.ls1 * a)
        return x + self._dp(self.ls2 * self.mlp(self.n2(x))), e


# --------------------------------------------------------------------------- #
class ResNet18Lite(nn.Module):
    """A width-matched residual CNN (the classical binary-CNN baseline)."""

    class Blk(nn.Module):
        def __init__(self, cin, cout, s):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
            self.sk = (nn.Identity() if s == 1 and cin == cout else
                       nn.Sequential(nn.Conv2d(cin, cout, 1, s, bias=False), nn.BatchNorm2d(cout)))

        def forward(self, x):
            r = self.sk(x)
            x = F.relu(self.b1(self.c1(x)), True)
            return F.relu(self.b2(self.c2(x)) + r, True)

    def __init__(self, n_classes=8, w=(32, 64, 128, 160)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU(True))
        self.l1 = nn.Sequential(self.Blk(w[0], w[0], 1), self.Blk(w[0], w[0], 1))
        self.l2 = nn.Sequential(self.Blk(w[0], w[1], 2), self.Blk(w[1], w[1], 1))
        self.l3 = nn.Sequential(self.Blk(w[1], w[2], 2), self.Blk(w[2], w[2], 1))
        self.l4 = nn.Sequential(self.Blk(w[2], w[3], 2), self.Blk(w[3], w[3], 1))
        self.fc = nn.Linear(w[3], n_classes)

    def forward(self, x, need_attn=False):
        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        z = x.mean((2, 3))
        logits = self.fc(z)
        p = logits.softmax(-1)
        return {"logits": logits, "prob": p, "feature": z,
                "vacuity": 1 - p.max(-1).values, "risk": None}


class ViTTiny(nn.Module):
    """Patch-embedding ViT with classical attention (the plain-transformer baseline)."""

    def __init__(self, n_classes=8, dim=128, depth=4, heads=4, patch=16, img=112, drop=0.05):
        super().__init__()
        self.pe = nn.Conv2d(3, dim, patch, patch)
        g = img // patch
        self.pos = nn.Parameter(torch.zeros(1, g * g + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(PlainBlock(dim, heads, 2.0, drop, 0.05 * i / max(1, depth - 1))
                                    for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, n_classes)
        self.mc_dropout = nn.Dropout(drop)

    def forward(self, x, need_attn=False):
        t = self.pe(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(x.shape[0], -1, -1), t], 1) + self.pos
        for b in self.blocks:
            t, _ = b(t, need_attn)
        t = self.norm(t)
        z = self.mc_dropout(t[:, 0])
        logits = self.fc(z)
        p = logits.softmax(-1)
        return {"logits": logits, "prob": p, "feature": z,
                "vacuity": 1 - p.max(-1).values, "risk": None}


class CDCNpp(nn.Module):
    """A CDCN-style detector: central-difference trunk with pixel-wise supervision."""

    def __init__(self, n_classes=8, theta=0.7, img_size=112):
        super().__init__()
        self.stem = CDCStem((24, 48, 96, 128), theta=theta)
        self.map = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64),
                                 nn.GELU(), nn.Conv2d(64, 1, 3, padding=1))
        self.fc = nn.Sequential(nn.Linear(128 + 1, 128), nn.GELU(), nn.Linear(128, n_classes))
        self.out_size = img_size // 2

    def forward(self, x, need_attn=False):
        f32, f16, f8 = self.stem(x)
        m = self.map(f8)
        ml = F.interpolate(m, size=(self.out_size, self.out_size), mode="bilinear",
                           align_corners=False).squeeze(1)
        loc = torch.sigmoid(ml)
        z = torch.cat([f8.mean((2, 3)), loc.flatten(1).mean(1, keepdim=True)], -1)
        logits = self.fc(z)
        p = logits.softmax(-1)
        return {"logits": logits, "prob": p, "feature": z, "loc": loc, "loc_logit": ml,
                "vacuity": 1 - p.max(-1).values, "risk": None}
