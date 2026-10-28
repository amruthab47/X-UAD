"""X-UAD: eXplainable Unified Attack Detector."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aqa import AQABlock, QuantumEvidenceBasis, amplitude_encode
from .heads import (EvidentialHead, HierarchicalEvidentialHead, LocalisationHead,
                     RegionPool, RiskCalibrator)
from .stem import CDCStem, SpectralBranch


class XUAD(nn.Module):
    def __init__(self,
                 n_classes: int = 8,
                 dim: int = 128,
                 depth: int = 4,
                 heads: int = 4,
                 circuit_depth: int = 4,
                 n_basis: int = 8,
                 mlp_ratio: float = 2.0,
                 drop: float = 0.05,
                 drop_path: float = 0.05,
                 theta: float = 0.7,
                 severity: Optional[torch.Tensor] = None,
                 use_aqa: bool = True,
                 use_spectral: bool = True,
                 use_evidential: bool = True,
                 img_size: int = 128,
                 gate_bias: float = -2.0,
                 hierarchical: bool = False,
                 region_pool: bool = False,
                 region_dim: int = 16,
                 deep_stem: bool = False):
        super().__init__()
        self.n_classes, self.dim = n_classes, dim
        self.use_aqa, self.use_spectral, self.use_evidential = use_aqa, use_spectral, use_evidential
        self.hierarchical = hierarchical
        self.use_region_pool = region_pool
        self.region_dim = region_dim if region_pool else 0

        self.stem = CDCStem((24, 48, 96, dim), theta=theta, deep=deep_stem)
        d32, d16, d8 = self.stem.out_dims
        g = img_size // 16
        self.grid = g
        self.pos = nn.Parameter(torch.zeros(1, g * g, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.spectral = SpectralBranch(dim, n_tokens=4) if use_spectral else None
        self.spec_pos = nn.Parameter(torch.zeros(1, 4, dim)) if use_spectral else None

        if use_aqa:
            self.blocks = nn.ModuleList(
                AQABlock(dim, heads, mlp_ratio, circuit_depth, drop,
                         drop_path * i / max(1, depth - 1), gate_bias)
                for i in range(depth))
        else:                                     # classical-attention ablation
            from .baselines import PlainBlock
            self.blocks = nn.ModuleList(
                PlainBlock(dim, heads, mlp_ratio, drop,
                           drop_path * i / max(1, depth - 1), gate_bias)
                for i in range(depth))
        self.norm = nn.LayerNorm(dim)

        self.basis = QuantumEvidenceBasis(dim, n_basis, depth=6)
        self.n_basis = n_basis
        self.basis_to_class = nn.Sequential(nn.LayerNorm(n_basis),
                                            nn.Linear(n_basis, n_classes))

        # the pooled convolutional descriptor is concatenated alongside the
        # transformer read-out: X-UAD is a *hybrid*, and keeping the direct
        # CNN pathway open means the central-difference evidence reaches the
        # classifier even before the attention stack has converged.
        feat = dim * 3 + n_basis + self.region_dim
        if use_evidential and hierarchical:
            self.head = HierarchicalEvidentialHead(feat, n_classes, hidden=dim)
        elif use_evidential:
            self.head = EvidentialHead(feat, n_classes, hidden=dim)
        else:
            self.head = nn.Sequential(nn.Linear(feat, dim), nn.GELU(), nn.Linear(dim, n_classes))
        self.loc = LocalisationHead(dim, d16, d32, out_size=img_size // 2)
        if region_pool:
            self.regionpool = RegionPool(img_size // 2)
            self.region_proj = nn.Sequential(nn.LayerNorm(self.regionpool.n_regions),
                                             nn.Linear(self.regionpool.n_regions, region_dim),
                                             nn.GELU())
        else:
            self.regionpool = None
            self.region_proj = None
        if severity is None:
            severity = torch.linspace(0, 1, n_classes)
        self.risk = RiskCalibrator(severity.clone().float())

    # ------------------------------------------------------------------ #
    def forward(self, x, need_attn: bool = False) -> Dict[str, torch.Tensor]:
        N = x.shape[0]
        f32, f16, f8 = self.stem(x)
        tok = f8.flatten(2).transpose(1, 2) + self.pos              # (N, g*g, D)
        n_spatial = tok.shape[1]

        spec_prof = None
        if self.spectral is not None:
            st, spec_prof = self.spectral(x)
            tok = torch.cat([tok, st + self.spec_pos], 1)

        tok = torch.cat([self.cls.expand(N, -1, -1), tok], 1)

        ent = tok.new_zeros(())
        attns: List[torch.Tensor] = []
        for blk in self.blocks:
            tok, e = blk(tok, need_weights=need_attn)
            ent = ent + e
            if need_attn and getattr(blk.attn, "last_attn", None) is not None:
                attns.append(blk.attn.last_attn)
        ent = ent / len(self.blocks)
        tok = self.norm(tok)

        cls_tok, spat = tok[:, 0], tok[:, 1:1 + n_spatial]

        # ---- spoof-evidence localisation ---------------------------------- #
        # computed before the classification feature below (rather than after,
        # as in the non-region-pooled version of this model) so that, when
        # ``region_pool`` is enabled, its region-pooled summary can be folded
        # into ``feat`` -- otherwise this block is unchanged and produces
        # bit-identical values regardless of where it runs, since nothing here
        # depends on the classification head having already been evaluated.
        g = self.grid
        tok8 = spat.transpose(1, 2).reshape(N, self.dim, g, g)
        loc_logit = self.loc(tok8, f16, f32)
        loc = torch.sigmoid(loc_logit)
        area = loc.flatten(1).mean(1)

        # ---- quantum evidence measurement (intrinsic explanation) --------- #
        born = self.basis(spat)                                     # (N, T, B)
        born_pool = born.mean(1)                                    # (N, B)

        feat_parts = [cls_tok, spat.mean(1), f8.mean((2, 3)), born_pool * float(self.n_basis)]
        if self.use_region_pool:
            # stop-gradient: the classification objective must not reshape the
            # localisation head, only read from it -- same rationale as
            # EvidentialHead.ev's ``z.detach()`` (heads.py). Without ``.detach()``
            # here, l_cls's gradient flows back through region_proj -> regionpool
            # -> loc -> loc_logit -> self.loc's own conv weights, competing with
            # the localisation loss for those same parameters; this measurably
            # degraded loc_IoU/loc_Dice in the non-detached run
            # (results_region_pooled_refinement/) and made classification worse,
            # not better. ``out["loc"]``/``out["loc_logit"]`` below remain the
            # original, non-detached tensors, so the localisation loss term in
            # XUADLoss (computed from ``out["loc_logit"]``) is completely
            # unaffected and still trains self.loc exactly as before.
            feat_parts.append(self.region_proj(self.regionpool(loc.detach())))
        feat = torch.cat(feat_parts, -1)
        if self.use_evidential:
            out = self.head(feat)
        else:
            logits = self.head(feat)
            p = logits.softmax(-1)
            out = {"prob": p, "logits": logits,
                   "vacuity": 1.0 - p.max(-1).values,
                   "alpha": None, "evidence": None, "strength": None}

        risk = self.risk(out["prob"], out["vacuity"], area)
        gates = torch.stack([b.attn.last_gate for b in self.blocks], 1) \
            if self.use_aqa else None

        out.update({
            "feature": feat, "born": born, "born_pool": born_pool,
            "basis_logits": self.basis_to_class(born_pool),
            "loc_logit": loc_logit, "loc": loc, "area": area,
            "risk": risk, "entropy": ent, "gates": gates,
            "spec_prof": spec_prof, "attn": attns if need_attn else None,
            "grid": g,
        })
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
