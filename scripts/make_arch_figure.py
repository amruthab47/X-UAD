"""Schematic of the X-UAD architecture, drawn to publication quality."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe

plt.rcParams.update({"font.size": 7.2, "font.family": "DejaVu Sans"})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")

C = {"stem": "#dbe9f6", "stem_e": "#1b6ca8",
     "aqa": "#e9dcf2", "aqa_e": "#7a4fa0",
     "head": "#fde4dd", "head_e": "#c0392b",
     "xai": "#dff2e3", "xai_e": "#2e7d4f",
     "out": "#fff4d6", "out_e": "#b8860b"}


def box(ax, x, y, w, h, text, fc, ec, fs=7.2, r=0.03, lw=0.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, weight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, ec="#444", lw=0.9, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8,
                                 lw=lw, color=ec, zorder=1, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(9.6, 3.6))
ax.set_xlim(0, 160); ax.set_ylim(0, 60); ax.axis("off")
FS = 6.3

# ---- input ---------------------------------------------------------------- #
box(ax, 1, 25, 13, 10, "face crop\n$112\\times112\\times3$", "#f2f2f2", "#666", FS)

# ---- stem ----------------------------------------------------------------- #
box(ax, 17, 34, 27, 12,
    "Central-difference stem\n$y=\\sum_p w_p x_p-\\theta(\\sum_p w_p)\\,x_{0}$\n"
    "$112\\to56\\to28\\to14\\to7$", C["stem"], C["stem_e"], FS)
box(ax, 17, 15, 27, 12,
    "Spectral branch\npolar log-DFT $16\\times12$\n$\\to$ 4 spectral tokens",
    C["stem"], C["stem_e"], FS)
arrow(ax, (14, 32), (17, 39), rad=0.16)
arrow(ax, (14, 28), (17, 22), rad=-0.16)

box(ax, 47, 25, 12, 10, "tokens\n$49{+}4{+}1$\n$d=128$", "#f2f2f2", "#666", FS)
arrow(ax, (44, 39), (47, 33), rad=-0.12)
arrow(ax, (44, 22), (47, 28), rad=0.12)

# ---- AQA stack ------------------------------------------------------------ #
ax.add_patch(Rectangle((62, 6), 42, 47, fc="none", ec=C["aqa_e"], lw=0.9,
                       ls=(0, (3, 2)), zorder=1))
ax.text(83, 55.5, "$\\times\\,4$  Adaptive Quantum Attention block",
        ha="center", fontsize=7.0, color=C["aqa_e"], weight="bold")

box(ax, 64, 43, 21, 8, "classical channel\n$\\mathbf{QK}^{\\top}/\\sqrt{d}$",
    "#efefef", "#888", FS)
box(ax, 64, 26, 21, 14,
    "Born channel\n$S_{ij}=|\\langle\\psi^{Q}_{i}|\\psi^{K}_{j}\\rangle|^{2}$\n"
    "Givens circuits $U_Q,U_K$\n$\\mathcal{O}(dL)$ angles\nexactly orthogonal",
    C["aqa"], C["aqa_e"], FS)
box(ax, 64, 9, 21, 9, "adaptive gate $g_h$\n$\\sigma(W_g[\\mu\\Vert\\sigma])$",
    C["aqa"], C["aqa_e"], FS)

box(ax, 88, 24, 14, 16,
    "softmax\n$(1{-}g_h)\\,\\mathrm{cls}$\n$+\\,g_h\\beta S$\n\nvalues\n+ MLP",
    "#f7f2fb", C["aqa_e"], FS)
arrow(ax, (85, 47), (88, 38), rad=-0.12)
arrow(ax, (85, 33), (88, 32))
arrow(ax, (85, 13.5), (88, 26), rad=0.12)
ax.text(95, 20, "entropy band $\\mathcal{L}_{\\mathrm{coh}}$", ha="center",
        fontsize=6.0, color=C["aqa_e"], style="italic")
arrow(ax, (59, 30), (64, 33), rad=-0.05)

# ---- basis + heads -------------------------------------------------------- #
box(ax, 106, 42, 27, 11,
    "Quantum evidence basis\n$\\Pi_{tb}=|\\langle\\phi_b|\\psi_t\\rangle|^{2}$\n"
    "$\\phi_b=V(\\varphi)\\,e_b$  orthonormal", C["xai"], C["xai_e"], FS)
box(ax, 106, 26, 27, 12,
    "Evidential head\n$\\alpha=e+\\mathbf{1}$,   $u=K/S$\n"
    "pseudo-unseen suppression", C["head"], C["head_e"], FS)
box(ax, 106, 10, 27, 11,
    "Localisation decoder\nFPN $7\\to14\\to28\\to56$\nmask-supervised",
    C["head"], C["head_e"], FS)
arrow(ax, (102, 37), (106, 46), rad=-0.14)
arrow(ax, (102, 32), (106, 32))
arrow(ax, (102, 27), (106, 17), rad=0.14)
arrow(ax, (119, 42), (119, 38), style="-|>")
arrow(ax, (44, 36), (106, 13), rad=0.30, ec="#c0c0c0", ls=(0, (2, 2)))
ax.text(78, 3.2, "stem skips (28$^2$, 14$^2$)", fontsize=5.8, color="#999")

# ---- record + output ------------------------------------------------------ #
box(ax, 135, 20, 10, 22, "evidence\nrecord\n\ntyped\nnumeric\nverified",
    C["xai"], C["xai_e"], FS, bold=True)
arrow(ax, (133, 32), (135, 32))
arrow(ax, (133, 15), (135, 26), rad=0.12)

box(ax, 148, 20, 11, 22,
    "Attack\nReplay\n\nConfidence\n97%\n\nRisk\nMedium",
    C["out"], C["out_e"], FS, bold=True)
arrow(ax, (145, 31), (148, 31))
ax.text(153.5, 16, "+ grounded\nrationale", ha="center", fontsize=5.9,
        color=C["out_e"], style="italic")

fig.savefig(f"{FIG}/architecture.png", dpi=280, bbox_inches="tight", pad_inches=0.03)
fig.savefig(f"{FIG}/architecture.pdf", bbox_inches="tight", pad_inches=0.03)
print("ok")
