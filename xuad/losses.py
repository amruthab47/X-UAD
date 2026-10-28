"""Training objectives for X-UAD."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Evidential (subjective-logic) classification
# --------------------------------------------------------------------------- #
def dirichlet_ce(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """Bayes risk of the cross-entropy under Dir(alpha) (Sensoy et al., 2018)."""
    S = alpha.sum(-1, keepdim=True)
    return (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)


def kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) ) -- penalises evidence placed on wrong classes."""
    K = alpha.shape[-1]
    ones = torch.ones_like(alpha)
    S = alpha.sum(-1, keepdim=True)
    t1 = torch.lgamma(S).squeeze(-1) - torch.lgamma(alpha).sum(-1) \
        - (torch.lgamma(ones).sum(-1) - torch.lgamma(ones.sum(-1)))
    t2 = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S))).sum(-1)
    return t1 + t2


def dirichlet_brier(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """Expected sum-of-squares (Brier) loss under Dir(alpha), Eq. (5) of
    Sensoy et al.  It decomposes into a fit term and a variance term,

        E[||y - p||^2] = sum_k (y_k - phat_k)^2 + phat_k (1 - phat_k) / (S + 1),

    and is markedly better conditioned than the digamma (type-II ML) form: its
    gradient does not vanish while the Dirichlet is still close to uniform, which
    is exactly the regime a freshly initialised evidential head starts in."""
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    return ((y_onehot - p) ** 2 + p * (1 - p) / (S + 1.0)).sum(-1)


def evidential_loss(alpha, y, n_classes, anneal: float = 1.0, mode: str = "brier",
                    kl_weight: float = 0.0, logits=None, dirichlet_weight: float = 0.5,
                    focal_gamma: float = 0.0):
    """Cross-entropy on the Dirichlet *direction* plus a Dirichlet fit on the
    concentration.

    With the decoupled head of ``EvidentialHead`` the class decision depends only
    on ``pi = softmax(logits)``, so the representation is trained by an ordinary
    cross-entropy with its full log-loss gradient.  The Dirichlet terms then act
    on the *magnitude*: the Brier fit rewards concentration when the direction is
    right, and the annealed KL removes concentration that sits on wrong classes.
    We use the Brier (expected sum-of-squares) form rather than the digamma
    type-II maximum likelihood because the latter's gradient vanishes while the
    Dirichlet is still near-uniform, which is where a fresh head starts.

    ``kl_weight`` defaults to zero, and that is a *consequence of the
    factorisation* rather than a tuning choice.  In the joint parameterisation the
    KL term is what stops the model putting evidence on wrong classes.  Here the
    wrong-class evidence is ``E * pi_k``, which the cross-entropy already drives to
    zero by sharpening ``pi``; the KL therefore has nothing left to do except pull
    down the shared magnitude ``E``.  We measured the damage: with the KL term
    active, models trained at a reduced budget -- where ``pi`` is still soft --
    saturate at vacuity ~1.0 on most classes, which destroys the open-set channel
    entirely.  Removing it leaves the vacuity informative at every budget.
    """
    y1 = F.one_hot(y, n_classes).float()
    fit = dirichlet_brier(alpha, y1) if mode == "brier" else dirichlet_ce(alpha, y1)
    alpha_t = y1 + (1 - y1) * alpha            # keep only the wrong-class evidence
    dir_term = fit + anneal * kl_weight * kl_to_uniform(alpha_t)
    if logits is None:                          # joint parameterisation fallback
        p = alpha / alpha.sum(-1, keepdim=True)
        ce = F.nll_loss(p.clamp_min(1e-8).log(), y, reduction="none")
    else:
        ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=0.02)
        if focal_gamma > 0:
            # Focal modulation (Lin et al. 2017): down-weight samples the model
            # already classifies confidently and correctly, concentrating
            # gradient on the ambiguous ones -- the taxonomy is perfectly
            # class-balanced (4,200 samples/class), so there is no frequency
            # imbalance to correct here; this instead targets the *hard-pair*
            # confusions (e.g. mask3d/synth, faceswap/reenact) that a frequency
            # reweighting would not touch.
            with torch.no_grad():
                pt = F.softmax(logits, -1).gather(1, y.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
            ce = ((1.0 - pt) ** focal_gamma) * ce
    return (ce + dirichlet_weight * dir_term).mean()


def evidence_correctness_loss(vacuity: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Tie the evidence magnitude to the model's own correctness.

    Certainty should predict correctness: ``1 - u`` is the total belief mass, and
    we train it as a probability that this sample's class decision is right.  The
    Dirichlet fit alone also rewards concentration, but only indirectly and
    through a bounded term, so on short schedules the magnitude head barely leaves
    its initialisation and the vacuity channel stays saturated -- we measured
    exactly that.  This term makes the channel informative within a couple of
    epochs, and together with PUES (which pushes the magnitude *down* on
    out-of-taxonomy inputs) it defines the two ends of the scale.
    """
    return F.binary_cross_entropy((1.0 - vacuity).clamp(1e-5, 1 - 1e-5), correct)


# --------------------------------------------------------------------------- #
# Pseudo-Unseen Evidence Suppression (PUES)
# --------------------------------------------------------------------------- #
def pues_loss(alpha_mix: torch.Tensor, alpha_real: torch.Tensor,
              ratio: float = 0.25) -> torch.Tensor:
    """Suppress the evidence of *synthetic out-of-taxonomy* samples.

    Convex mixtures of two different attack images are, by construction, not a
    member of any training class: they carry a superposition of two forensic
    signatures that no single physical or generative process can produce.
    Constraining the model to emit little evidence for them teaches the vacuity
    channel what "I have not seen this" looks like, without ever showing it a
    real unseen attack -- which is what turns the Dirichlet vacuity from a
    calibration statistic into a usable open-set score.

    The constraint is deliberately *relative*.  An absolute target (drive the
    mixture evidence to zero) is trivially satisfied by the degenerate solution
    that emits zero evidence everywhere, leaving a uniform classifier; we
    observed exactly that collapse in an early version of this work.  Requiring
    instead that the out-of-taxonomy evidence stay below a fraction of the
    in-taxonomy evidence of the same batch makes the degenerate solution
    worthless, because the fit term of the evidential loss still demands peaked
    posteriors on real samples.
    """
    K = alpha_mix.shape[-1]
    e_mix = (alpha_mix.sum(-1) - K).clamp_min(0).mean()
    e_real = (alpha_real.sum(-1) - K).clamp_min(0).mean().detach()
    return F.relu(e_mix - ratio * e_real) / (e_real + 1.0)


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #
def loc_loss(logit: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    if logit.shape[-2:] != target.shape[-2:]:
        logit = F.interpolate(logit.unsqueeze(1), size=target.shape[-2:],
                              mode="bilinear", align_corners=False).squeeze(1)
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    num = 2 * (p * target).flatten(1).sum(1) + eps
    den = (p + target).flatten(1).sum(1) + eps
    return bce + (1 - num / den).mean()


# --------------------------------------------------------------------------- #
# Binary-preserving refinement: reinforce the SAME probability already
# responsible for good ACER, plus a small targeted penalty on the two dominant
# confusion pairs identified by the P1 error analysis. Both operate directly on
# ``prob`` (no separate head/parameters), so there is no second decision
# surface that can diverge from the classifier's own boundary -- unlike the
# hierarchical ablation's dedicated binary head, which regressed ACER.
# --------------------------------------------------------------------------- #
def binary_reinforcement_loss(prob: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p_attack = (1.0 - prob[:, 0]).clamp(eps, 1 - eps)
    y_bin = (y != 0).float()
    return F.binary_cross_entropy(p_attack, y_bin)


# Two-pair map used by the binary-preserving refinement experiment
# (results_binary_preserving_refinement/): faceswap<->reenact, mask3d<->synth.
PAIR_PENALTY_MAP = {4: [5], 5: [4], 3: [6], 6: [3]}

# Four-pair extension (results_four_pair_refinement/): adds replay<->print and
# faceswap<->print, the two additional dominant clusters identified by the
# post-refinement error diagnosis. print and faceswap each list two partners
# since they are each confusable with two different classes.
PAIR_PENALTY_MAP_4WAY = {
    1: [2, 4],   # print: confusable with replay and faceswap
    2: [1],      # replay: confusable with print
    3: [6],      # mask3d: confusable with synth
    4: [5, 1],   # faceswap: confusable with reenact and print
    5: [4],      # reenact: confusable with faceswap
    6: [3],      # synth: confusable with mask3d
}


def pair_confusion_penalty(prob: torch.Tensor, y: torch.Tensor,
                           pairs: dict = PAIR_PENALTY_MAP) -> torch.Tensor:
    """Mean probability mass placed on each confusable partner, for samples
    whose true label is a key in ``pairs`` (a class may list more than one
    partner, e.g. print is confusable with both replay and faceswap in the
    4-way map). Classes not named in ``pairs`` are untouched. Averaging
    (not summing) over however many (sample, partner) terms are collected
    keeps the aggregate loss magnitude comparable whether 2 or 4 pairs are
    active -- broadening scope, not strength."""
    terms = []
    for c, partners in pairs.items():
        sel = y == c
        if not sel.any():
            continue
        for p in partners:
            terms.append(prob[sel, p])
    if not terms:
        return prob.new_zeros(())
    return torch.cat(terms).mean()


# Per-pair-weighted variant (results_pair_weighted_refinement/): the shared-mean
# pair_confusion_penalty above dilutes every pair's gradient by 1/N as more
# pairs are added (N = total pooled terms across all active pairs) -- this is
# what regressed faceswap<->reenact and mask3d<->synth when the 4-pair
# experiment reused the same mechanism. Computing each pair's mean
# independently and giving it its own weight removes that dilution mechanism
# by construction.
PAIR_WEIGHTS_PERPAIR = {
    (4, 5): 0.15, (5, 4): 0.15,   # faceswap<->reenact: same weight as the successful 2-pair reference
    (3, 6): 0.15, (6, 3): 0.15,   # mask3d<->synth: same weight as the successful 2-pair reference
    (2, 1): 0.05, (1, 2): 0.05,   # replay<->print: lighter -- real spectral gap already exists
    (4, 1): 0.05, (1, 4): 0.05,   # faceswap<->print: lighter -- real spectral gap already exists
}


def weighted_pair_confusion_penalty(prob: torch.Tensor, y: torch.Tensor,
                                    pair_weights: dict) -> torch.Tensor:
    """Unlike ``pair_confusion_penalty`` (one pooled mean, one shared weight),
    each ``(true_class, partner_class)`` pair here gets its own mean -- divided
    only by its own sample count, never diluted by how many other pairs are
    also active -- and its own scalar weight, already baked into the returned
    value (no outer multiplier needed)."""
    total = prob.new_zeros(())
    for (c, p), w in pair_weights.items():
        if w <= 0:
            continue
        sel = y == c
        if not sel.any():
            continue
        total = total + w * prob[sel, p].mean()
    return total


# --------------------------------------------------------------------------- #
# Attention-coherence (measurement-entropy) regulariser
# --------------------------------------------------------------------------- #
def coherence_loss(entropy: torch.Tensor, n_tokens: int,
                   lo: float = 0.30, hi: float = 0.80) -> torch.Tensor:
    """Keep the mean row entropy of the attention inside ``[lo, hi] * log T``.

    Too low means the Born channel has collapsed onto a single token (the
    attention has decohered into a delta and stops integrating evidence); too
    high means it has diffused into a uniform mixture and carries no information.
    A two-sided hinge is enough and adds no parameters.
    """
    m = torch.log(torch.tensor(float(n_tokens), device=entropy.device))
    return F.relu(lo * m - entropy) + F.relu(entropy - hi * m)


# --------------------------------------------------------------------------- #
class XUADLoss(nn.Module):
    def __init__(self, n_classes=8, superclass=None, w=None, use_evidential=True,
                 focal_gamma=0.0, pcp_pairs=None, pair_weights=None):
        super().__init__()
        self.K = n_classes
        self.use_evidential = use_evidential
        self.focal_gamma = focal_gamma
        self.pcp_pairs = pcp_pairs if pcp_pairs is not None else PAIR_PENALTY_MAP
        self.pair_weights = pair_weights
        self.register_buffer("superclass",
                             torch.as_tensor(superclass if superclass is not None
                                             else list(range(n_classes))).long())
        from .data.attacks import HIER_SUPERCLASS
        self.register_buffer("hier_superclass", torch.as_tensor(HIER_SUPERCLASS).long())
        d = dict(cls=1.0, loc=0.6, basis=0.25, sup=0.3, pues=0.15, coh=0.02,
                 conf=0.4, hbin=0.0, hsup=0.0, bce=0.0, pcp=0.0)
        if w:
            d.update(w)
        self.w = d

    def forward(self, out, y, mask, n_tokens, anneal=1.0, alpha_mix=None):
        logs = {}
        if self.use_evidential:
            l_cls = evidential_loss(out["alpha"], y, self.K, anneal,
                                    logits=out.get("logits"), focal_gamma=self.focal_gamma)
            correct = (out["prob"].argmax(-1) == y).float().detach()
            l_conf = evidence_correctness_loss(out["vacuity"], correct)
            l_cls = l_cls + self.w.get("conf", 0.4) * l_conf
            logs["conf"] = float(l_conf.detach())
        else:
            ce = F.cross_entropy(out["logits"], y, reduction="none", label_smoothing=0.02)
            if self.focal_gamma > 0:
                with torch.no_grad():
                    pt = F.softmax(out["logits"], -1).gather(1, y.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
                ce = ((1.0 - pt) ** self.focal_gamma) * ce
            l_cls = ce.mean()
        total = self.w["cls"] * l_cls
        logs["cls"] = float(l_cls.detach())

        if out.get("loc_logit") is not None and self.w["loc"] > 0:
            l_loc = loc_loss(out["loc_logit"], mask)
            total = total + self.w["loc"] * l_loc
            logs["loc"] = float(l_loc.detach())

        if out.get("basis_logits") is not None and self.w["basis"] > 0:
            l_b = F.cross_entropy(out["basis_logits"], y)
            total = total + self.w["basis"] * l_b
            logs["basis"] = float(l_b.detach())

        if out.get("bin_logits") is not None and self.w.get("hbin", 0.0) > 0:
            l_hbin = F.cross_entropy(out["bin_logits"], (y != 0).long())
            total = total + self.w["hbin"] * l_hbin
            logs["hbin"] = float(l_hbin.detach())

        if out.get("sup_logits") is not None and self.w.get("hsup", 0.0) > 0:
            atk = y != 0
            if atk.any():
                sup_y = self.hier_superclass[y[atk]]
                l_hsup = F.cross_entropy(out["sup_logits"][atk], sup_y)
                total = total + self.w["hsup"] * l_hsup
                logs["hsup"] = float(l_hsup.detach())

        if self.w.get("bce", 0.0) > 0:
            l_bce = binary_reinforcement_loss(out["prob"], y)
            total = total + self.w["bce"] * l_bce
            logs["bce"] = float(l_bce.detach())

        if self.w.get("pcp", 0.0) > 0:
            l_pcp = pair_confusion_penalty(out["prob"], y, pairs=self.pcp_pairs)
            total = total + self.w["pcp"] * l_pcp
            logs["pcp"] = float(l_pcp.detach())

        if self.pair_weights:
            l_wpcp = weighted_pair_confusion_penalty(out["prob"], y, self.pair_weights)
            total = total + l_wpcp   # per-pair weights already baked in
            logs["wpcp"] = float(l_wpcp.detach())

        if self.w["sup"] > 0:
            sup_y = self.superclass[y]
            n_sup = int(self.superclass.max()) + 1
            p_sup = torch.zeros(out["prob"].shape[0], n_sup, device=out["prob"].device)
            p_sup.index_add_(1, self.superclass, out["prob"])
            l_s = F.nll_loss(p_sup.clamp_min(1e-8).log(), sup_y)
            total = total + self.w["sup"] * l_s
            logs["sup"] = float(l_s.detach())

        if alpha_mix is not None and self.w["pues"] > 0:
            l_p = pues_loss(alpha_mix, out["alpha"])
            total = total + self.w["pues"] * anneal * l_p
            logs["pues"] = float(l_p.detach())

        if out.get("entropy") is not None and self.w["coh"] > 0:
            l_c = coherence_loss(out["entropy"], n_tokens)
            total = total + self.w["coh"] * l_c
            logs["coh"] = float(l_c.detach())

        logs["total"] = float(total.detach())
        return total, logs
