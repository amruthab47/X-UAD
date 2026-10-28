"""Adaptive Quantum Attention (AQA).

Motivation
----------
Classical scaled dot-product attention scores a query/key pair by an *unbounded
linear* functional of the two embeddings.  Forensic evidence for face attacks is
not linear in this sense: a moire lattice, a blending seam and a spectral
generator fingerprint are all *interference* phenomena, i.e. they are visible in
the squared modulus of a projection rather than in the projection itself.

AQA therefore adds a second, physically-motivated scoring channel derived from
the Born rule of quantum measurement.  Each token is amplitude-encoded into a
normalised state vector, evolved by a *parameterised quantum circuit* realised
as a depth-L cascade of Givens (two-level) rotations, and pairs of tokens are
scored by their transition probability

        S_ij = |<psi^Q_i | psi^K_j>|^2  in [0, 1].

Because the Givens cascade is orthogonal by construction it is norm preserving
(a valid real unitary), and it is parameterised by only ``L * d_h / 2`` angles
instead of the ``d_h^2`` weights of a dense projection -- an O(d) rather than
O(d^2) parameterisation, matching the parameter-efficiency argument made for
quantum self-attention in the QViT literature.

The *adaptive* part is a per-head, input-conditioned gate that decides how much
of the Born channel to mix into the classical channel, so the network can fall
back to ordinary attention on inputs whose evidence is not interference-like.

Finally, a learnable **quantum evidence basis** is measured against every token
state.  The resulting Born probabilities form a (token x basis) map that is (a)
pooled into the classification features and (b) exported unchanged as the
localisation / reasoning evidence, which is what makes the explanation
*intrinsic* rather than a post-hoc saliency approximation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GivensCircuit(nn.Module):
    """Depth-``L`` cascade of Givens rotations on a ring of amplitudes.

    Layer ``l`` rotates the disjoint pairs ``(2k+o, 2k+1+o) mod d`` with
    ``o = l mod 2``.  This is the real-amplitude analogue of a hardware-efficient
    ansatz built from nearest-neighbour two-qubit gates with periodic boundary
    conditions; alternating the two offsets makes the circuit connected, so the
    reachable set grows towards the full orthogonal group as ``L`` increases.

    The optional ``groups`` dimension holds one independent circuit per attention
    head and is applied in a single fused op, which is what keeps the module
    cheap on CPU.

    Parameters: ``groups * L * (d/2)`` angles -- O(dL) rather than the O(d^2) of a
    dense projection, and orthogonality holds exactly by construction, so no
    re-orthogonalisation or penalty is ever needed.
    """

    def __init__(self, dim: int, depth: int = 4, groups: int = 1):
        super().__init__()
        assert dim % 2 == 0, "amplitude dimension must be even"
        self.dim, self.depth, self.groups = dim, depth, groups
        self.theta = nn.Parameter(torch.randn(groups, depth, dim // 2) * 0.05)

    def _rot(self, x: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: (..., G, ..., d) with the pair structure on the last axis
        a = x.unflatten(-1, (self.dim // 2, 2))
        x0, x1 = a[..., 0], a[..., 1]
        y = torch.stack([c * x0 - s * x1, s * x0 + c * x1], -1)
        return y.flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (N, G, T, d) when ``groups > 1`` else (..., d)."""
        grouped = self.groups > 1
        cos, sin = torch.cos(self.theta), torch.sin(self.theta)
        for l in range(self.depth):
            c = cos[:, l], sin[:, l]
            cc, ss = cos[:, l], sin[:, l]
            if grouped:                      # broadcast over (N, G, T, d/2)
                cc = cc.view(1, self.groups, 1, self.dim // 2)
                ss = ss.view(1, self.groups, 1, self.dim // 2)
            else:
                cc, ss = cc[0], ss[0]
            if l % 2 == 0:
                x = self._rot(x, cc, ss)
            else:
                x = torch.roll(x, shifts=-1, dims=-1)
                x = self._rot(x, cc, ss)
                x = torch.roll(x, shifts=1, dims=-1)
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, depth={self.depth}, groups={self.groups}, "
                f"params={self.groups * self.depth * (self.dim // 2)}")


def amplitude_encode(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise the last dimension: a real amplitude encoding |psi> = x/||x||."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class QuantumEvidenceBasis(nn.Module):
    """A learnable orthonormal measurement basis of ``n_basis`` states.

    The basis vectors are the images of the first ``n_basis`` computational-basis
    states under a Givens circuit, so orthonormality holds *exactly* at every
    step of training without any Gram-Schmidt or penalty term.  Measuring a token
    state against this basis gives Born probabilities that sum to <= 1 and behave
    like a soft, mutually-exclusive assignment of the token to evidence modes.
    """

    def __init__(self, dim: int, n_basis: int = 8, depth: int = 6):
        super().__init__()
        self.dim, self.n_basis = dim, n_basis
        self.circuit = GivensCircuit(dim, depth)

    def states(self) -> torch.Tensor:
        e = torch.eye(self.dim, device=self.circuit.theta.device,
                      dtype=self.circuit.theta.dtype)[: self.n_basis]
        return self.circuit(e)                                   # (B, d), orthonormal

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: (N, T, d) -> Born probabilities (N, T, n_basis)."""
        psi = amplitude_encode(tokens)
        phi = self.states()
        return (psi @ phi.t()) ** 2


class AdaptiveQuantumAttention(nn.Module):
    """Multi-head attention mixing a classical and a Born-rule channel."""

    def __init__(self, dim: int, heads: int = 4, circuit_depth: int = 4,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 gate_bias: float = -2.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads = dim, heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # one circuit per head for queries and for keys (different circuits are
        # essential: a shared orthogonal circuit would leave <q,k> invariant)
        self.circ_q = GivensCircuit(self.dh, circuit_depth, groups=heads)
        self.circ_k = GivensCircuit(self.dh, circuit_depth, groups=heads)

        # input-conditioned per-head mixing gate  g = sigmoid(W [mu; sigma] + b)
        self.gate = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, heads))
        nn.init.zeros_(self.gate[1].weight)
        nn.init.constant_(self.gate[1].bias, gate_bias)     # start almost classical
        self.beta = nn.Parameter(torch.tensor(4.0))          # Born-channel temperature

        self.last_attn: Optional[torch.Tensor] = None
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        N, T, D = x.shape
        H, dh = self.heads, self.dh
        qkv = self.qkv(x).reshape(N, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # (N,H,T,dh)

        # ---- classical channel -------------------------------------------- #
        logits_c = (q @ k.transpose(-2, -1)) * self.scale

        # ---- Born channel -------------------------------------------------- #
        qe = self.circ_q(amplitude_encode(q))
        ke = self.circ_k(amplitude_encode(k))
        overlap = qe @ ke.transpose(-2, -1)                   # (N,H,T,T) in [-1,1]
        logits_q = self.beta * overlap.pow(2)                 # Born transition prob.

        # ---- adaptive mixing ----------------------------------------------- #
        stats = torch.cat([x.mean(1), x.std(1)], -1)          # (N, 2D)
        g = torch.sigmoid(self.gate(stats)).view(N, H, 1, 1)  # (N,H,1,1)
        attn = torch.softmax((1.0 - g) * logits_c + g * logits_q, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(N, T, D)
        out = self.proj_drop(self.proj(out))

        self.last_gate = g.detach().view(N, H)
        self.last_attn = attn.detach() if need_weights else None
        # row-wise measurement entropy, used by the coherence regulariser
        ent = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(-1).mean()
        return out, ent


class AQABlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0,
                 circuit_depth: int = 4, drop: float = 0.0, drop_path: float = 0.0,
                 gate_bias: float = -2.0, layer_scale: float = 1e-2):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = AdaptiveQuantumAttention(dim, heads, circuit_depth, drop, drop,
                                            gate_bias)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, dim), nn.Dropout(drop))
        self.dp = drop_path
        # LayerScale: the block starts as a near-identity, so the network begins
        # life as "central-difference stem + classifier" -- i.e. as strong as the
        # convolutional detector it contains -- and the attention stack is
        # recruited only as its residual earns gradient.  Without this the
        # freshly initialised attention injects noise into the stem and the
        # hybrid converges more slowly than either of its parts.
        self.ls1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.ls2 = nn.Parameter(layer_scale * torch.ones(dim))

    def _drop_path(self, x):
        if self.dp <= 0.0 or not self.training:
            return x
        keep = 1 - self.dp
        m = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep) / keep
        return x * m

    def forward(self, x, need_weights: bool = False):
        a, ent = self.attn(self.n1(x), need_weights)
        x = x + self._drop_path(self.ls1 * a)
        x = x + self._drop_path(self.ls2 * self.mlp(self.n2(x)))
        return x, ent
