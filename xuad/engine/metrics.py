"""ISO/IEC 30107-3 presentation-attack metrics plus calibration, open-set and
localisation measures."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney U) identity; ties handled by mid-ranks."""
    pos, neg = labels == 1, labels == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def det_curve(scores: np.ndarray, labels: np.ndarray, n: int = 512):
    """Return (thresholds, APCER-like FAR, BPCER-like FRR).  ``scores`` is the
    attack score: higher = more likely an attack; label 1 = attack."""
    lo, hi = float(scores.min()), float(scores.max())
    th = np.linspace(lo, hi, n)
    atk, bon = scores[labels == 1], scores[labels == 0]
    far = np.array([(atk <= t).mean() for t in th])     # attacks accepted as bonafide
    frr = np.array([(bon > t).mean() for t in th])      # bonafide rejected
    return th, far, frr


def eer(scores: np.ndarray, labels: np.ndarray):
    th, far, frr = det_curve(scores, labels, 2048)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(th[i])


def pad_metrics(scores: np.ndarray, y_true: np.ndarray, thr: float,
                attack_types: Optional[np.ndarray] = None) -> Dict[str, float]:
    """``y_true``: 1 = attack, 0 = bonafide.  ``attack_types``: per-sample attack
    class id, used for the ISO worst-case APCER."""
    atk, bon = scores[y_true == 1], scores[y_true == 0]
    bpcer = float((bon > thr).mean()) if len(bon) else float("nan")
    if attack_types is not None and len(atk):
        t = attack_types[y_true == 1]
        per = {int(c): float((atk[t == c] <= thr).mean()) for c in np.unique(t)}
        apcer = float(max(per.values()))
    else:
        per = {}
        apcer = float((atk <= thr).mean()) if len(atk) else float("nan")
    return {"APCER": apcer * 100, "BPCER": bpcer * 100,
            "ACER": (apcer + bpcer) / 2 * 100, "per_type_apcer": per}


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier(prob: np.ndarray, y: np.ndarray) -> float:
    oh = np.zeros_like(prob)
    oh[np.arange(len(y)), y] = 1
    return float(((prob - oh) ** 2).sum(1).mean())


def macro_f1(pred: np.ndarray, y: np.ndarray, k: int) -> float:
    fs = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        p = tp / max(tp + fp, 1e-9)
        r = tp / max(tp + fn, 1e-9)
        fs.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(fs)) if fs else float("nan")


def localisation_metrics(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5):
    """``pred``/``gt``: (N, H, W) in [0,1].  Samples with an empty GT map (i.e.
    bonafide) are excluded from IoU but kept for the false-positive area."""
    keep = gt.reshape(len(gt), -1).max(1) > 0.2
    if keep.sum() == 0:
        return {"IoU": float("nan"), "pixel_AUC": float("nan"), "Dice": float("nan")}
    p, g = pred[keep] > thr, gt[keep] > 0.5
    inter = (p & g).reshape(keep.sum(), -1).sum(1)
    union = (p | g).reshape(keep.sum(), -1).sum(1)
    dice = 2 * inter / np.clip(p.reshape(keep.sum(), -1).sum(1) + g.reshape(keep.sum(), -1).sum(1), 1, None)
    sub = np.random.default_rng(0).permutation(int(keep.sum()))[:200]
    pa = roc_auc(pred[keep][sub].reshape(-1), (gt[keep][sub] > 0.5).reshape(-1).astype(int))
    return {"IoU": float(np.mean(inter / np.clip(union, 1, None))),
            "Dice": float(np.mean(dice)), "pixel_AUC": float(pa)}


LOCALISABLE = (3, 4, 5, 7)   # mask3d, faceswap, reenact, adversarial


def localisation_by_class(pred, gt, y, names, thr=0.5):
    """Per-class IoU.  Only the classes whose ground-truth evidence is a genuine
    *sub-region* of the frame are informative: print / replay / synth alter the
    whole capture, so their masks are near-constant and their IoU is trivially
    high.  ``mean_localisable`` therefore averages only over the region-level
    attacks, which is the number that actually measures localisation ability."""
    out = {}
    for c in np.unique(y):
        m = y == c
        if m.sum() == 0 or gt[m].reshape(m.sum(), -1).max(1).mean() < 0.2:
            continue
        out[names[int(c)]] = localisation_metrics(pred[m], gt[m], thr)["IoU"]
    loc = [v for k, v in out.items() if names.index(k) in LOCALISABLE]
    out["mean_localisable"] = float(np.mean(loc)) if loc else float("nan")
    return out


def summarise(prob: np.ndarray, y: np.ndarray, vacuity: np.ndarray,
              thr: Optional[float] = None, k: int = 8) -> Dict[str, float]:
    attack_score = 1.0 - prob[:, 0]
    y_bin = (y != 0).astype(int)
    auc = roc_auc(attack_score, y_bin)
    e, th_eer = eer(attack_score, y_bin)
    thr = th_eer if thr is None else thr
    m = pad_metrics(attack_score, y_bin, thr, attack_types=y)
    pred = prob.argmax(1)
    conf = prob.max(1)
    m.update({
        "AUC": auc * 100, "EER": e * 100,
        "ACC": float((pred == y).mean()) * 100,
        "ACC_bin": float(((attack_score > thr).astype(int) == y_bin).mean()) * 100,
        "macroF1": macro_f1(pred, y, k) * 100,
        "ECE": expected_calibration_error(conf, (pred == y).astype(float)) * 100,
        "Brier": brier(prob, y),
        "mean_vacuity": float(vacuity.mean()),
        "threshold": float(thr),
    })
    return m
