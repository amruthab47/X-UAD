"""Build every figure and LaTeX table from the saved experiment artefacts."""
from __future__ import annotations

import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES
from xuad.engine.metrics import det_curve, roc_auc

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = {"xuad": "#1b6ca8", "cdcn": "#d1495b", "resnet18": "#7a9e3f",
     "vit_tiny": "#e08214", "xuad_no_aqa": "#8d6e9e", "xuad_no_evid": "#5c5c5c"}
LBL = {"xuad": "X-UAD (ours)", "cdcn": "CDCN++", "resnet18": "ResNet-18",
       "vit_tiny": "ViT-Tiny", "xuad_no_aqa": "X-UAD w/o AQA",
       "xuad_no_evid": "X-UAD w/o evidential"}


def load(tag):
    f = os.path.join(RES, f"{tag}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def loadp(tag):
    f = os.path.join(RES, f"{tag}_pred.npz")
    return np.load(f) if os.path.exists(f) else None


# --------------------------------------------------------------------------- #
def fig_roc_det(models=("xuad", "cdcn", "resnet18", "vit_tiny"), proto="P1"):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for m in models:
        d = loadp(f"{m}_{proto}")
        if d is None:
            continue
        s = 1 - d["prob"][:, 0]; y = (d["y"] != 0).astype(int)
        th = np.unique(np.quantile(s, np.linspace(0, 1, 400)))
        tpr = [ (s[y == 1] > t).mean() for t in th ]
        fpr = [ (s[y == 0] > t).mean() for t in th ]
        ax[0].plot(fpr, tpr, color=C.get(m), lw=1.2,
                   label=f"{LBL.get(m, m)} (AUC {roc_auc(s, y)*100:.2f})")
        _, far, frr = det_curve(s, y, 400)
        ax[1].plot(far * 100, frr * 100, color=C.get(m), lw=1.2, label=LBL.get(m, m))
    ax[0].set_xlabel("BPCER (false alarm rate)"); ax[0].set_ylabel("Attack detection rate")
    ax[0].set_xlim(0, 0.10); ax[0].set_ylim(0.85, 1.001); ax[0].legend(loc="lower right", fontsize=6.5)
    ax[0].set_title("(a) ROC, low-false-alarm regime", fontsize=8)
    ax[1].set_xlabel("APCER (%)"); ax[1].set_ylabel("BPCER (%)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_title("(b) DET", fontsize=8); ax[1].legend(fontsize=6.5)
    fig.savefig(f"{FIG}/roc_det.pdf"); fig.savefig(f"{FIG}/roc_det.png"); plt.close(fig)


def fig_reliability(models=("xuad", "cdcn", "resnet18", "vit_tiny")):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    bins = np.linspace(0, 1, 13)
    for m in models:
        d = loadp(f"{m}_P1")
        if d is None:
            continue
        p = d["prob"]; y = d["y"]
        conf = p.max(1); corr = (p.argmax(1) == y).astype(float)
        xs, ys = [], []
        for i in range(len(bins) - 1):
            s = (conf > bins[i]) & (conf <= bins[i + 1])
            if s.sum() < 20:
                continue
            xs.append(conf[s].mean()); ys.append(corr[s].mean())
        ax[0].plot(xs, ys, "o-", ms=2.5, lw=1.1, color=C.get(m), label=LBL.get(m, m))
        v = d["vacuity"]
        ax[1].hist(v, bins=40, histtype="step", lw=1.1, density=True, color=C.get(m),
                   label=LBL.get(m, m))
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.7, label="perfect calibration")
    ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("(a) Reliability diagram", fontsize=8); ax[0].legend(fontsize=6.5)
    ax[1].set_xlabel("uncertainty (vacuity / 1-max softmax)"); ax[1].set_ylabel("density")
    ax[1].set_title("(b) Uncertainty distribution", fontsize=8); ax[1].legend(fontsize=6.5)
    fig.savefig(f"{FIG}/calibration.pdf"); fig.savefig(f"{FIG}/calibration.png"); plt.close(fig)


def fig_loao():
    folds = [c for c in ATTACK_NAMES if os.path.exists(os.path.join(RES, f"xuad_P2.{c}.json"))]
    if not folds:
        return
    models = [m for m in ("xuad", "cdcn", "resnet18")
              if load(f"{m}_P2.{folds[0]}") is not None]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    w = 0.8 / max(len(models), 1)
    x = np.arange(len(folds))
    for i, m in enumerate(models):
        acer = [load(f"{m}_P2.{c}")["ACER"] for c in folds]
        det = [load(f"{m}_P2.{c}").get("unseen_detect_rate", np.nan) for c in folds]
        ax[0].bar(x + i * w, acer, w, color=C.get(m), label=LBL.get(m, m))
        ax[1].bar(x + i * w, det, w, color=C.get(m), label=LBL.get(m, m))
    for a, t, yl in ((ax[0], "(a) ACER on the held-out family", "ACER (%)"),
                     (ax[1], "(b) Detection rate of the unseen family", "detection rate (%)")):
        a.set_xticks(x + w * (len(models) - 1) / 2)
        a.set_xticklabels(folds, rotation=20, ha="right")
        a.set_title(t, fontsize=8); a.set_ylabel(yl); a.legend(fontsize=6.5)
    fig.savefig(f"{FIG}/loao.pdf"); fig.savefig(f"{FIG}/loao.png"); plt.close(fig)


def fig_confusion(tag="xuad_P1"):
    d = loadp(tag)
    if d is None:
        return
    p, y = d["prob"].argmax(1), d["y"]
    k = len(ATTACK_NAMES)
    M = np.zeros((k, k))
    for a, b in zip(y, p):
        M[a, b] += 1
    keep = M.sum(1) > 0
    Mn = M[keep][:, :] / np.clip(M[keep].sum(1, keepdims=True), 1, None)
    names = [n for n, kk in zip(ATTACK_NAMES, keep) if kk]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(Mn * 100, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(k)); ax.set_xticklabels(ATTACK_NAMES, rotation=55, ha="right", fontsize=6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
    for i in range(len(names)):
        for j in range(k):
            if Mn[i, j] > 0.005:
                ax.text(j, i, f"{Mn[i,j]*100:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if Mn[i, j] > 0.5 else "black")
    ax.grid(False); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, fraction=0.046, label="%")
    fig.savefig(f"{FIG}/confusion.pdf"); fig.savefig(f"{FIG}/confusion.png"); plt.close(fig)


def fig_curves():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for m in ("xuad", "cdcn", "resnet18", "vit_tiny"):
        f = os.path.join(RES, f"{m}_P1_hist.json")
        if not os.path.exists(f):
            continue
        h = json.load(open(f))
        ax[0].plot([e["epoch"] for e in h], [e["val"]["ACER"] for e in h], "o-", ms=2.5,
                   lw=1.1, color=C.get(m), label=LBL.get(m, m))
        ax[1].plot([e["epoch"] for e in h], [e["val"]["ACC"] for e in h], "o-", ms=2.5,
                   lw=1.1, color=C.get(m), label=LBL.get(m, m))
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val ACER (%)"); ax[0].set_yscale("log")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val 8-class accuracy (%)")
    ax[0].legend(fontsize=6.5); ax[1].legend(fontsize=6.5)
    fig.savefig(f"{FIG}/curves.pdf"); fig.savefig(f"{FIG}/curves.png"); plt.close(fig)


if __name__ == "__main__":
    fig_roc_det(); fig_reliability(); fig_loao(); fig_confusion(); fig_curves()
    print("figures written to", FIG)
