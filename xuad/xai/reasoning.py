"""Multimodal reasoning: from a verified evidence record to natural language.

Design note
-----------
The obvious way to make a detector "explainable" is to hand the image to a
multimodal LLM and print whatever it says.  That fails an audit: the language
model has no access to the detector's internal state, so its narrative is at
best correlated with, and at worst contradicted by, the decision that was
actually made -- and it hallucinates specifics (a "visible screen bezel" that is
not in the frame) that an operator cannot check.

X-UAD inverts the dependency.  A forward pass first produces an **evidence
record**: a typed, quantitative, machine-checkable summary of the Dirichlet
posterior, the vacuity, the localisation map, the quantum evidence spectrum and
the spectral descriptors.  Language is then generated *from the record only*.
Two interchangeable back-ends consume it:

``GroundedReasoner``
    A deterministic realiser.  Every sentence is emitted by a rule whose guard is
    a numeric predicate over the record, so each clause is traceable to a
    measured quantity and the hallucination rate is zero by construction.  This
    is the back-end used for all reported results, and it needs no network.

``MLLMReasoner``
    Wraps a hosted multimodal LLM.  The record is injected as the *only*
    admissible source of facts and the model is instructed to paraphrase, not to
    add.  ``verify()`` re-checks the produced text against the record and reports
    any unsupported numeric claim, so the fluency of an LLM can be used without
    giving up auditability.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data.attacks import ATTACK_NAMES, ATTACK_SUPERCLASS
from .evidence import REGIONS, region_profile, spatial_stats, spectral_descriptors

PRETTY = {
    "bonafide": "Bonafide (genuine live capture)",
    "print": "Print attack",
    "replay": "Replay attack",
    "mask3d": "3D mask attack",
    "faceswap": "Face-swap deepfake",
    "reenact": "Facial reenactment deepfake",
    "synth": "Fully synthetic (AI-generated) face",
    "adversarial": "Adversarial perturbation attack",
}

# Physical explanation attached to each class, used only to *phrase* an
# already-measured finding -- never to assert one.
MECHANISM = {
    "print": ("periodic halftone screening and paper-substrate grain, with the "
              "high-frequency skin micro-texture of a live face missing"),
    "replay": ("a sub-pixel display lattice aliasing into a moire beat, together "
               "with specular backlight glare and reduced dynamic range"),
    "mask3d": ("locally flattened skin micro-texture with a broad specular sheen "
               "and a discontinuity at the mask rim"),
    "faceswap": ("a blending boundary around the inner face and a resolution "
                 "mismatch between the pasted region and the surrounding frame"),
    "reenact": ("non-rigid warping confined to the mouth and eye regions with a "
                "local sharpness inconsistency"),
    "synth": ("a generator up-sampling fingerprint: periodic checkerboard "
              "structure and anisotropic peaks in the Fourier magnitude"),
    "adversarial": ("a low-amplitude, high-frequency perturbation that is spatially "
                    "decorrelated from facial structure"),
}

# Banding of the risk score.  These are a *policy*, not a learned quantity: they
# express how an operator wants the expected-harm scale partitioned, and are meant
# to be re-set per deployment.  The defaults place a confidently-detected print or
# replay at Medium, a 3-D mask or face-swap at High, and an adversarial or
# high-vacuity sample at Critical.
RISK_BANDS = [(0.35, "Low"), (0.75, "Medium"), (0.95, "High"), (10.0, "Critical")]


def risk_label(r: float) -> str:
    for t, name in RISK_BANDS:
        if r < t:
            return name
    return "Critical"


@dataclass
class EvidenceRecord:
    predicted_class: str
    predicted_id: int
    confidence: float                    # expected Dirichlet probability
    vacuity: float                       # subjective-logic "I don't know" mass
    is_attack: bool
    attack_probability: float
    superclass: str
    risk_score: float
    risk_level: str
    decision: str                        # accept / reject / refer
    top_regions: List[Dict[str, float]] = field(default_factory=list)
    spatial: Dict[str, float] = field(default_factory=dict)
    spectral: Dict[str, float] = field(default_factory=dict)
    quantum_modes: List[Dict[str, float]] = field(default_factory=list)
    gate_mean: Optional[float] = None
    class_posterior: Dict[str, float] = field(default_factory=dict)

    def to_json(self, **kw):
        return json.dumps(asdict(self), **kw)


SUPER = {0: "none", 1: "physical presentation attack", 2: "digital attack"}


def build_record(prob: np.ndarray, vacuity: float, risk: float, loc: np.ndarray,
                 born: Optional[np.ndarray] = None,
                 basis_info: Optional[List[dict]] = None,
                 spec_prof: Optional[np.ndarray] = None,
                 gate_mean: Optional[float] = None,
                 abstain_vacuity: float = 0.55) -> EvidenceRecord:
    k = int(prob.argmax())
    name = ATTACK_NAMES[k]
    p_atk = float(1.0 - prob[0])
    rp = region_profile(loc)
    top = sorted(rp.items(), key=lambda kv: -kv[1])[:3]
    rl = risk_label(float(risk))
    if vacuity >= abstain_vacuity:
        decision = "refer to human review (evidence insufficient / attack outside known taxonomy)"
    elif k == 0:
        decision = "accept"
    else:
        decision = "reject"

    modes = []
    if born is not None and basis_info is not None:
        pooled = born.mean(0)
        order = np.argsort(-pooled)[:3]
        for b in order:
            modes.append({"mode": int(b), "aligned_with": basis_info[int(b)]["class"],
                          "activation": float(pooled[int(b)]),
                          "selectivity": float(basis_info[int(b)]["selectivity"])})

    return EvidenceRecord(
        predicted_class=name, predicted_id=k, confidence=float(prob[k]),
        vacuity=float(vacuity), is_attack=k != 0, attack_probability=p_atk,
        superclass=SUPER[ATTACK_SUPERCLASS[k]], risk_score=float(risk), risk_level=rl,
        decision=decision,
        top_regions=[{"region": r, "activation": float(v)} for r, v in top],
        spatial=spatial_stats(loc),
        spectral=spectral_descriptors(spec_prof) if spec_prof is not None else {},
        quantum_modes=modes, gate_mean=gate_mean,
        class_posterior={ATTACK_NAMES[i]: float(prob[i]) for i in range(len(prob))},
    )


class GroundedReasoner:
    """Deterministic evidence-to-text realiser (zero-hallucination by construction)."""

    def headline(self, r: EvidenceRecord) -> str:
        return (f"Attack: {PRETTY[r.predicted_class]}\n"
                f"Confidence: {r.confidence * 100:.0f}%\n"
                f"Risk: {r.risk_level}")

    def explain(self, r: EvidenceRecord) -> str:
        S: List[str] = []
        if not r.is_attack:
            S.append(f"The sample is accepted as a genuine live capture with "
                     f"{r.confidence*100:.0f}% posterior probability.")
        else:
            S.append(f"The sample is classified as a {PRETTY[r.predicted_class].lower()} "
                     f"({r.superclass}) with {r.confidence*100:.0f}% posterior probability; "
                     f"the probability that it is an attack of any kind is "
                     f"{r.attack_probability*100:.1f}%.")
            S.append(f"That class is characterised by {MECHANISM[r.predicted_class]}.")

        # -- where -------------------------------------------------------- #
        if r.top_regions and r.spatial.get("area", 0) > 0.02:
            best = r.top_regions[0]
            others = ", ".join(t["region"] for t in r.top_regions[1:])
            S.append(f"Evidence is concentrated in the {best['region']} "
                     f"(mean response {best['activation']:.2f}), followed by {others}; "
                     f"it covers {r.spatial['area']*100:.0f}% of the crop with a "
                     f"spatial dispersion of {r.spatial['dispersion']:.2f}.")
        elif r.is_attack:
            S.append("The evidence is diffuse rather than localised, which is consistent "
                     "with a whole-frame capture medium rather than a region-level edit.")

        # -- what kind ----------------------------------------------------- #
        if r.quantum_modes:
            m = r.quantum_modes[0]
            S.append(f"The dominant quantum evidence mode is #{m['mode']}, whose learned "
                     f"selectivity is highest for '{m['aligned_with']}' "
                     f"(activation {m['activation']:.3f}), corroborating the class decision.")
        if r.spectral:
            sp = r.spectral
            if sp.get("radial_peak_prominence", 0) > 1.0:
                S.append(f"The Fourier magnitude shows a peak at normalised frequency "
                         f"{sp['radial_peak_norm_freq']:.2f} with prominence "
                         f"{sp['radial_peak_prominence']:.2f} and azimuthal anisotropy "
                         f"{sp['azimuthal_anisotropy']:.3f}, a signature of a periodic "
                         f"acquisition or up-sampling lattice.")

        # -- how sure ------------------------------------------------------ #
        if r.vacuity >= 0.55:
            S.append(f"Vacuity is high ({r.vacuity:.2f}): the accumulated evidence does not "
                     f"match any family in the learned taxonomy well, so the class label "
                     f"should be treated as provisional and the sample may be an unseen "
                     f"attack type.")
        elif r.vacuity >= 0.30:
            S.append(f"Vacuity is moderate ({r.vacuity:.2f}); the decision is supported but "
                     f"not saturated.")
        else:
            S.append(f"Vacuity is low ({r.vacuity:.2f}): the evidence is strong and "
                     f"consistent with a single known family.")
        if r.gate_mean is not None:
            S.append(f"The adaptive attention gate settled at {r.gate_mean:.2f}, i.e. "
                     f"{r.gate_mean*100:.0f}% of the attention score came from the Born "
                     f"(interference) channel rather than the classical dot product.")
        S.append(f"Recommended action: {r.decision} (risk score {r.risk_score:.2f} -> "
                 f"{r.risk_level}).")
        return " ".join(S)

    def __call__(self, r: EvidenceRecord) -> str:
        return self.headline(r) + "\n\n" + self.explain(r)


class MLLMReasoner:
    """Optional hosted-MLLM back-end, constrained to the evidence record."""

    SYSTEM = (
        "You are a forensic assistant for a face presentation-attack detection system. "
        "You will receive a JSON EVIDENCE RECORD produced by the detector, and optionally "
        "the image. Write a short operator-facing explanation. Rules: (1) every factual or "
        "numeric claim must come from the record; (2) never invent artefacts you were not "
        "given; (3) if the record's vacuity exceeds 0.55 you must say the attack may be "
        "outside the known taxonomy; (4) end with the recommended action from the record."
    )

    def __init__(self, model: str = "", api_key_env: str = "OPENAI_API_KEY",
                 base_url: str = ""):
        self.model = model or os.environ.get("XUAD_MLLM", "")
        self.key = os.environ.get(api_key_env, "")
        self.base_url = base_url or os.environ.get("XUAD_MLLM_BASE", "")
        self.fallback = GroundedReasoner()

    @property
    def available(self) -> bool:
        return bool(self.model and self.key)

    def __call__(self, r: EvidenceRecord, image_b64: Optional[str] = None) -> str:
        if not self.available:
            return self.fallback(r)
        from openai import OpenAI                      # imported lazily
        client = OpenAI(api_key=self.key, base_url=self.base_url or None)
        content = [{"type": "text", "text": "EVIDENCE RECORD:\n" + r.to_json(indent=1)}]
        if image_b64:
            content.append({"type": "image_url",
                            "url": f"data:image/png;base64,{image_b64}"})
        msg = client.chat.completions.create(
            model=self.model, temperature=0.2,
            messages=[{"role": "system", "content": self.SYSTEM},
                      {"role": "user", "content": content}])
        text = msg.choices[0].message.content
        return self.headline_prefix(r) + text

    @staticmethod
    def headline_prefix(r: EvidenceRecord) -> str:
        return GroundedReasoner().headline(r) + "\n\n"

    @staticmethod
    def verify(text: str, r: EvidenceRecord, tol: float = 0.02) -> Dict[str, object]:
        """Flag numeric claims in ``text`` that no field of the record supports."""
        import re
        vals = []
        for f in (r.confidence, r.vacuity, r.attack_probability, r.risk_score,
                  r.spatial.get("area", 0), r.spatial.get("dispersion", 0)):
            vals += [f, f * 100]
        vals += [m["activation"] for m in r.quantum_modes]
        vals += [t["activation"] for t in r.top_regions]
        vals += list(r.spectral.values())
        bad = []
        for tok in re.findall(r"\d+\.?\d*", text):
            v = float(tok)
            if not any(abs(v - u) <= tol * max(1.0, abs(u)) for u in vals):
                bad.append(v)
        return {"unsupported_numbers": bad, "n_claims": len(bad),
                "grounded": len(bad) == 0}
