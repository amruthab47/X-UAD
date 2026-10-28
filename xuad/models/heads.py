"""Prediction heads: Dirichlet evidential classification, spoof-evidence
localisation and the calibrated risk estimator."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..xai.evidence import REGIONS


class EvidentialHead(nn.Module):
    """Decoupled direction--magnitude subjective-logic head.

    A Dirichlet is fully described by a *direction* on the simplex and a
    *concentration*.  Rather than predicting the ``K`` concentrations jointly --
    the usual evidential parameterisation -- we predict them separately,

        pi    = softmax(W_c z)                      (direction: which class)
        E     = exp(clip(w_e z))                    (magnitude: how much evidence)
        alpha = 1 + E * pi ,   S = K + E ,
        phat  = alpha / S ,    u = K / S .

    Three consequences make this the right factorisation for a detector that has
    to be both accurate and honest:

    1.  ``argmax phat = argmax pi``, so the class decision is trained by an
        ordinary cross-entropy on ``W_c z`` with its full, unbounded log-loss
        gradient.  The joint parameterisation instead routes every gradient
        through ``alpha`` -- whose fit term saturates and whose KL regulariser
        drives the pre-activation negative -- and we measured a large accuracy
        loss from exactly that "evidence starvation".
    2.  The uncertainty is a single scalar with a closed form, ``u = K/(K+E)``,
        so suppressing evidence on out-of-taxonomy samples is a one-dimensional
        constraint on ``E`` and cannot distort the class direction.
    3.  ``u`` and ``phat`` remain a consistent Dirichlet posterior, so subjective
        logic still applies: nothing about the semantics is given up.
    """

    def __init__(self, dim: int, n_classes: int, hidden: int = 0,
                 lo: float = -6.0, hi: float = 9.0):
        super().__init__()
        self.n_classes = n_classes
        self.lo, self.hi = lo, hi
        d = dim if hidden == 0 else hidden
        self.trunk = (nn.Identity() if hidden == 0 else
                      nn.Sequential(nn.Linear(dim, hidden), nn.GELU()))
        self.cls = nn.Linear(d, n_classes)                       # direction
        # The magnitude branch reads a *stop-gradient* copy of the shared
        # representation.  Evidence is a statement about how well the observed
        # features match the learned taxonomy; letting its objective -- and in
        # particular the pseudo-unseen suppression term -- reshape those features
        # is both conceptually circular and, as we measured, costly: it removed
        # several points of family accuracy.  Detaching keeps the uncertainty
        # channel expressive (it sees the full representation) while guaranteeing
        # that adding it cannot degrade detection.
        self.ev = nn.Sequential(nn.Linear(dim, max(32, hidden // 2)), nn.GELU(),
                                nn.Linear(max(32, hidden // 2), 1))
        # start from a moderate amount of evidence (u ~ 0.5) rather than none:
        # the magnitude has to travel several orders of magnitude in log space,
        # and starting at zero evidence wastes a large part of a short schedule.
        nn.init.zeros_(self.ev[-1].weight)
        nn.init.constant_(self.ev[-1].bias, 2.0)

    def forward(self, z):
        logits = self.cls(self.trunk(z))
        pi = torch.softmax(logits, -1)
        E = torch.exp(self.ev(z.detach()).clamp(self.lo, self.hi))  # (N,1)
        # the direction is detached inside alpha so the Dirichlet terms shape the
        # *magnitude* only and can never distort the class decision
        alpha = 1.0 + E * pi.detach()
        S = alpha.sum(-1, keepdim=True)
        return {"evidence": E * pi.detach(), "alpha": alpha,
                # subjective logic reports a belief direction *and* an
                # uncertainty mass; ``prob`` is the belief direction (what the
                # model believes) and ``p_hat`` the projected probability under a
                # uniform base rate, ``p_hat = (1 + E pi) / (K + E)``.
                "prob": pi, "p_hat": alpha / S,
                "logits": logits, "pi": pi,
                "vacuity": (self.n_classes / S).squeeze(-1),
                "strength": S.squeeze(-1), "total_evidence": E.squeeze(-1)}


class HierarchicalEvidentialHead(nn.Module):
    """Evidential head with a hierarchical bonafide/attack -> physical/digital/
    adversarial -> specific-class softmax factorisation, motivated directly by
    the P1 error analysis: 56.1% of the (flat-softmax) improved model's errors
    crossed the physical/digital superclass boundary. The final per-class
    probability is

        p(bonafide)        = p(bonafide | binary head)
        p(class k, attack) = p(attack | binary) * p(superclass(k) | superclass head)
                              * p(k | within-superclass head)

    which cannot place high probability on class k without also being
    confident about k's superclass and about being an attack at all --
    structurally discouraging the cross-superclass leaks a flat softmax made,
    rather than only discouraging them via an auxiliary loss term.

    Drop-in compatible with ``EvidentialHead``: returns the identical set of
    dict keys (plus ``bin_logits``/``sup_logits`` for the new auxiliary
    losses), so every downstream consumer (XUADLoss, metrics, XAI) is
    unchanged. The uncertainty/evidence machinery is untouched from
    ``EvidentialHead`` -- only the *direction* (``pi``) is computed
    hierarchically instead of by one flat linear layer.
    """
    GROUPS = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7]}   # physical, digital, adversarial

    def __init__(self, dim: int, n_classes: int = 8, hidden: int = 0,
                 lo: float = -6.0, hi: float = 9.0):
        super().__init__()
        assert n_classes == 8, "hierarchical grouping is hardcoded to the 8-class taxonomy"
        self.n_classes = n_classes
        self.lo, self.hi = lo, hi
        d = dim if hidden == 0 else hidden
        self.trunk = (nn.Identity() if hidden == 0 else
                      nn.Sequential(nn.Linear(dim, hidden), nn.GELU()))
        self.bin = nn.Linear(d, 2)                    # [bonafide, attack]
        self.sup = nn.Linear(d, 3)                    # [physical, digital, adversarial]
        self.cls = nn.Linear(d, 8)                    # raw per-class scores (used within-group only)
        self.ev = nn.Sequential(nn.Linear(dim, max(32, hidden // 2)), nn.GELU(),
                                nn.Linear(max(32, hidden // 2), 1))
        nn.init.zeros_(self.ev[-1].weight)
        nn.init.constant_(self.ev[-1].bias, 2.0)

    def forward(self, z):
        h = self.trunk(z)
        bin_logits = self.bin(h)
        sup_logits = self.sup(h)
        cls_logits = self.cls(h)
        p_bin = torch.softmax(bin_logits, -1)                 # (N,2)
        p_sup = torch.softmax(sup_logits, -1)                  # (N,3)

        pi = torch.zeros(z.shape[0], self.n_classes, device=z.device, dtype=p_bin.dtype)
        pi[:, 0] = p_bin[:, 0]
        for s, idxs in self.GROUPS.items():
            grp_p = torch.softmax(cls_logits[:, idxs], -1)
            pi[:, idxs] = p_bin[:, 1:2] * p_sup[:, s:s + 1] * grp_p

        # log(pi) makes F.cross_entropy(logits, y) recover exactly -log(pi[y]),
        # so evidential_loss's existing CE/focal code works unmodified on the
        # hierarchical probability without needing to know about the factorisation.
        logits = torch.log(pi.clamp_min(1e-8))

        E = torch.exp(self.ev(z.detach()).clamp(self.lo, self.hi))
        alpha = 1.0 + E * pi.detach()
        S = alpha.sum(-1, keepdim=True)
        return {"evidence": E * pi.detach(), "alpha": alpha,
                "prob": pi, "p_hat": alpha / S,
                "logits": logits, "pi": pi,
                "vacuity": (self.n_classes / S).squeeze(-1),
                "strength": S.squeeze(-1), "total_evidence": E.squeeze(-1),
                "bin_logits": bin_logits, "sup_logits": sup_logits}


class LocalisationHead(nn.Module):
    """Light FPN decoder: 8x8 token map + 16x16 and 32x32 stem maps -> 64x64 map."""

    def __init__(self, dim_tok, d16, d32, width=48, out_size=56):
        super().__init__()
        self.p8 = nn.Conv2d(dim_tok, width, 1)
        self.p16 = nn.Conv2d(d16, width, 1)
        self.p32 = nn.Conv2d(d32, width, 1)
        self.f16 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                 nn.BatchNorm2d(width), nn.GELU())
        self.f32 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                 nn.BatchNorm2d(width), nn.GELU())
        self.out = nn.Conv2d(width, 1, 3, padding=1)
        self.out_size = out_size

    def forward(self, tok8, f16, f32):
        x = self.p8(tok8)
        x = F.interpolate(x, size=f16.shape[-2:], mode="bilinear", align_corners=False)
        x = self.f16(x + self.p16(f16))
        x = F.interpolate(x, size=f32.shape[-2:], mode="bilinear", align_corners=False)
        x = self.f32(x + self.p32(f32))
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear",
                          align_corners=False)
        return self.out(x).squeeze(1)


class RegionPool(nn.Module):
    """Differentiable mean-pool of a localisation map over the named facial
    regions already defined for XAI in ``xai.evidence.REGIONS`` (forehead, eye
    region, nose/midface, mouth and chin, left/right cheek, face-boundary
    ring).  This reuses those same fixed, normalised-coordinate regions as a
    *classification* feature instead of only as a post-hoc explanation, so it
    introduces no new face-geometry assumption beyond what ``xai/evidence.py``
    already relies on for its region-based reasoning output.

    The region boundaries are static (they don't depend on the input), so the
    per-region boolean masks are built once at construction time, at the
    localisation map's fixed output resolution, and stored as a non-trainable
    buffer -- pooling at call time is a single matrix multiply.
    """

    def __init__(self, out_size: int, regions: dict = REGIONS):
        super().__init__()
        H = W = out_size
        b = max(2, int(0.14 * H))
        names, masks = [], []
        for name, (x0, y0, x1, y1) in regions.items():
            m = torch.zeros(H, W, dtype=torch.bool)
            if name == "face boundary":
                inner = torch.zeros(H, W, dtype=torch.bool)
                inner[b:H - b, b:W - b] = True
                m = ~inner
            else:
                m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
            names.append(name)
            masks.append(m)
        self.names = names
        self.n_regions = len(names)
        flat_masks = torch.stack(masks).reshape(self.n_regions, -1).float()
        self.register_buffer("masks", flat_masks)
        self.register_buffer("counts", flat_masks.sum(-1).clamp_min(1.0))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """``m``: (N, H, W) localisation map -> (N, n_regions) region means."""
        flat = m.reshape(m.shape[0], -1)
        return (flat @ self.masks.t()) / self.counts


class RiskCalibrator(nn.Module):
    """Maps (attack posterior, vacuity, evidence area) to a bounded risk score.

    ``risk = (1 - u) * sum_k p_k * severity_k  +  lambda_u * u * u_severity
             + lambda_a * area``

    The first term is the *expected harm* under the Dirichlet posterior, damped
    by how much the model actually knows.  The second term charges an
    unknown-attack premium when the model is vacuous, so a confidently-unknown
    sample is escalated rather than silently accepted -- the behaviour an
    operator needs.  The third rewards spatially corroborated evidence.

    The three coefficients are deployment *policy*, not learned quantities: no
    training signal supervises risk, and pretending otherwise would make the one
    part of the pipeline an operator must be able to audit the least
    inspectable.  They are stored as parameters only so that a site that does
    have risk labels can fine-tune them.
    """

    def __init__(self, severity: torch.Tensor, u_severity: float = 0.75):
        super().__init__()
        self.register_buffer("severity", severity)
        self.u_sev = u_severity
        self.lam_u = nn.Parameter(torch.tensor(1.0))
        self.lam_a = nn.Parameter(torch.tensor(0.15))

    def forward(self, prob, vacuity, area):
        exp_harm = (prob * self.severity).sum(-1)
        r = (1 - vacuity) * exp_harm + self.lam_u.abs() * vacuity * self.u_sev \
            + self.lam_a.abs() * area
        return r.clamp(0, 1.5)
