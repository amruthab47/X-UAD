"""Revision-3 tables and figures for paper/xuad_v2.tex (no number typed by hand).

Protocol P1 now uses the 12-epoch recipe for X-UAD, CDCN++ and ResNet-18 (seed 0,
same 9,600-sample test set).  The X-UAD entry is the variant with the lowest
*validation* ACER among the X-UAD variants trained under that recipe
(results_pair_weighted_refinement/).  ViT-Tiny was not retrained; its row is the
10-epoch original-recipe run, marked as such.  Localisation, faithfulness and the
white-box evaluation come from scripts/eval_extras.py on the same X-UAD checkpoint.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES  # noqa: E402
from xuad.engine.metrics import det_curve, eer, expected_calibration_error, roc_auc  # noqa: E402

TAB, FIG = os.path.join(ROOT, "paper", "tables"), os.path.join(ROOT, "paper", "figures")
EX = os.path.join(ROOT, "results_v3", "extras")

RUNS = {  # key: (label, metrics json, predictions npz with val split, colour)
    "resnet18": (r"ResNet-18~\cite{he2016resnet}", "results_v3/resnet18_P1_recipe_s0.json",
                 "results_v3/resnet18_P1_recipe_s0_pred.npz", "#7a9e3f"),
    "vit_tiny": (r"ViT-Tiny~\cite{dosovitskiy2021vit}$^{\dagger}$", "results/vit_tiny_P1.json",
                 None, "#e08214"),
    "cdcn": (r"CDCN++~\cite{yu2020cdcn}", "results_v3/cdcn_P1_recipe_s0.json",
             "results_v3/cdcn_P1_recipe_s0_pred.npz", "#d1495b"),
    "xuad": (r"\textbf{X-UAD (ours)}", "results_pair_weighted_refinement/xuad_P1_pairweighted.json",
             "results_v3/extras/xuad_P1_pairweighted_pred_full.npz", "#1b6ca8"),
}
PLAIN = {"resnet18": "ResNet-18", "vit_tiny": "ViT-Tiny", "cdcn": "CDCN++", "xuad": "X-UAD (ours)"}
PRETTY = {"print": "Print", "replay": "Replay", "mask3d": "3D mask", "faceswap": "Face-swap",
          "reenact": "Reenact", "synth": "Synthetic", "adversarial": "Adversarial"}
SUMMARY = {}


def J(p):
    return json.load(open(os.path.join(ROOT, p)))


def preds(key):
    """(test prob, test y, test uncertainty, val prob, val y) in a common layout."""
    p = RUNS[key][2]
    if p is None:
        return None
    z = np.load(os.path.join(ROOT, p))
    if "test_prob" in z:                                  # eval_extras layout
        return z["test_prob"], z["test_y"], z["test_vac"], z["val_prob"], z["val_y"]
    return z["prob"], z["y"], z["vacuity"], z["val_prob"], z["val_y"]


def temp_scaled_ece(key):
    """ECE after temperature scaling fitted on the validation split by NLL."""
    d = preds(key)
    if d is None:
        return None
    pt, yt, _, pv, yv = d
    lv, lt = np.log(pv + 1e-12), np.log(pt + 1e-12)

    def soft(l, T):
        z = l / T; z = z - z.max(1, keepdims=True); e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    Ts = np.exp(np.linspace(np.log(0.2), np.log(5), 200))
    nll = [-np.log(soft(lv, T)[np.arange(len(yv)), yv] + 1e-12).mean() for T in Ts]
    T = float(Ts[int(np.argmin(nll))])
    q = soft(lt, T)
    SUMMARY.setdefault("temperature", {})[key] = T
    return expected_calibration_error(q.max(1), (q.argmax(1) == yt).astype(float)) * 100


def mcnemar(a_right, b_right):
    b = int(np.sum(a_right & ~b_right)); c = int(np.sum(~a_right & b_right))
    n, k = b + c, min(b, c)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b, c, p


def emit(name, lines):
    open(os.path.join(TAB, name), "w").write("\n".join(lines))
    print("wrote", name)


def f(v, n=2):
    return "--" if v is None else f"{v:.{n}f}"


# --------------------------------------------------------------------------- tables
def table_main():
    order = ["resnet18", "vit_tiny", "cdcn", "xuad"]
    M = {k: J(RUNS[k][1]) for k in order}
    for k in order:
        M[k]["ECE_TS"] = temp_scaled_ece(k)
    cols = [("ACER", True), ("APCER", True), ("BPCER", True), ("AUC", False), ("EER", True),
            ("ACC", False), ("macroF1", False), ("ECE", True), ("ECE_TS", True)]
    head = {"ACER": r"ACER$\downarrow$", "APCER": r"APCER$\downarrow$", "BPCER": r"BPCER$\downarrow$",
            "AUC": r"AUC$\uparrow$", "EER": r"EER$\downarrow$", "ACC": r"ACC$_8\uparrow$",
            "macroF1": r"F1$_{\text{macro}}\uparrow$", "ECE": r"ECE$\downarrow$",
            "ECE_TS": r"ECE$_{\text{TS}}\downarrow$"}
    # best marks over the three detectors trained under the common recipe
    working = ["resnet18", "cdcn", "xuad"]
    best = {}
    for c, lo in cols:
        vals = [round(M[k][c], 2) for k in working if M[k].get(c) is not None]
        best[c] = min(vals) if lo else max(vals)
    s = [r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule",
         "Method & Params & " + " & ".join(head[c] for c, _ in cols) + r" \\",
         " & (M) & " + " & ".join(["(\\%)"] * len(cols)) + r" \\", r"\midrule"]
    for k in order:
        d = M[k]
        cells = []
        for c, _ in cols:
            v = d.get(c)
            t = f(v)
            if k in working and v is not None and round(v, 2) == best[c]:
                t = r"\textbf{" + t + "}"
            cells.append(t)
        s.append(f"{RUNS[k][0]} & {d['params_M']:.2f} & " + " & ".join(cells) + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("main.tex", s)
    SUMMARY["main"] = {k: {c: M[k].get(c) for c, _ in cols} for k in order}

    # paired tests on the identical test set
    x = preds("xuad")
    out = {}
    for k in ("cdcn", "resnet18"):
        o = preds(k)
        assert np.array_equal(o[1], x[1]), "test sets differ"
        y = x[1]
        b8 = mcnemar(x[0].argmax(1) == y, o[0].argmax(1) == y)
        tx, to = M["xuad"]["threshold"], M[k]["threshold"]
        b2 = mcnemar(((1 - x[0][:, 0]) > tx) == (y != 0), ((1 - o[0][:, 0]) > to) == (y != 0))
        out[k] = {"eight_way": b8, "binary": b2}
    SUMMARY["mcnemar"] = out


def table_per_type():
    order = ["resnet18", "vit_tiny", "cdcn", "xuad"]
    cls = ATTACK_NAMES[1:]
    s = [r"\begin{tabular}{l" + "c" * (len(cls) + 1) + "}", r"\toprule",
         "Method & " + " & ".join(PRETTY[c] for c in cls) + r" & Worst \\", r"\midrule"]
    for k in order:
        d = J(RUNS[k][1])["per_type_apcer"]
        v = [d[c] for c in cls]
        s.append(f"{RUNS[k][0]} & " + " & ".join(f(x) for x in v) + f" & {f(max(v))}" + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("per_type.tex", s)


def table_loc():
    fams = ["print", "replay", "mask3d", "faceswap", "reenact", "synth"]
    s = [r"\begin{tabular}{l" + "c" * (len(fams) + 2) + "}", r"\toprule",
         "Method & " + " & ".join(PRETTY[c] for c in fams) + r" & Region-level mean & Pixel AUC \\",
         r"\midrule"]
    SUMMARY["loc"] = {}
    for key, name in (("cdcn_P1_recipe_s0", RUNS["cdcn"][0]), ("xuad_P1_pairweighted", RUNS["xuad"][0])):
        L = json.load(open(os.path.join(EX, f"{key}_predict.json")))["loc"]
        r = L["fixed_0.5"]
        s.append(f"{name} & " + " & ".join(f(100 * r[c], 1) for c in fams)
                 + f" & {f(100 * r['region_mean'], 1)} & {f(100 * L['pixel_AUC_stratified'], 1)}" + r" \\")
        SUMMARY["loc"][key] = {"fixed_0.5": r["region_mean"], "val_thr": L["val_fitted_thr"],
                               "at_val_thr": L["fitted"]["region_mean"],
                               "pixel_AUC": L["pixel_AUC_stratified"], "n_test": L["n_test"]}
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("localisation.tex", s)


def table_faith():
    d = json.load(open(os.path.join(EX, "xuad_P1_pairweighted_faith.json")))
    ci = lambda a: f"[{a[0]:+.3f}, {a[1]:+.3f}]"
    s = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Families & $n$ & \multicolumn{2}{c}{Deletion AUC$\downarrow$} & "
         r"\multicolumn{2}{c}{Insertion AUC$\uparrow$} & Insertion gain, 95\% CI \\",
         r" & & map & random & map & random & \\", r"\midrule"]
    for grp, name in (("region", "Region-level (3D mask, face-swap, reenact)"),
                      ("whole", "Whole-frame (print, replay, synthetic, adversarial)")):
        g = d[grp]
        s.append(f"{name} & {g['n']} & {g['deletion_loc']:.3f} & {g['deletion_rnd']:.3f} & "
                 f"{g['insertion_loc']:.3f} & {g['insertion_rnd']:.3f} & {ci(g['insertion_gain_CI95'])}" + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("faithfulness.tex", s)
    SUMMARY["faith"] = {g: d[g] for g in ("region", "whole")}


def table_whitebox():
    d = json.load(open(os.path.join(EX, "whitebox_attack_xuad_P1_pairweighted.json")))
    xk = "xuad_P1_pairweighted"
    eps = d["eps_255"]
    s = [r"\begin{tabular}{l" + "c" * (len(eps) + 1) + "}", r"\toprule",
         r"$\varepsilon$ ($\times 1/255$) & clean & " + " & ".join(str(e) for e in eps) + r" \\", r"\midrule"]
    s.append(r"CDCN++, attacks detected (\%) & " + f(d["cdcn_P1_recipe_s0"]["clean"]["attack_detect_rate"], 1)
             + " & " + " & ".join(f(d["cdcn_P1_recipe_s0"][f"evasion_eps{e}"]["attack_detect_rate"], 1) for e in eps) + r" \\")
    s.append(r"X-UAD, attacks detected (\%) & " + f(d[xk]["clean"]["attack_detect_rate"], 1)
             + " & " + " & ".join(f(d[xk][f"evasion_eps{e}"]["attack_detect_rate"], 1) for e in eps) + r" \\")
    s.append(r"\midrule")
    s.append(r"X-UAD, mean vacuity under false-confidence attack & " + f(d[xk]["clean"]["mean_vacuity_attack"], 3)
             + " & " + " & ".join(f(d[xk][f"false_confidence_eps{e}"]["mean_vacuity"], 3) for e in eps) + r" \\")
    s.append(r"X-UAD, predicted class unchanged (\%) & 100.0 & "
             + " & ".join(f(d[xk][f"false_confidence_eps{e}"]["class_kept_%"], 1) for e in eps) + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("whitebox.tex", s)
    SUMMARY["whitebox"] = d


# --------------------------------------------------------------------------- figures
plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.4, "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False})


def fig_roc_det():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for k in ("xuad", "cdcn", "resnet18"):
        p, y, _, _, _ = preds(k)
        s = 1 - p[:, 0]; yb = (y != 0).astype(int)
        th = np.unique(np.quantile(s, np.linspace(0, 1, 400)))
        tpr = [(s[yb == 1] > t).mean() for t in th]
        fpr = [(s[yb == 0] > t).mean() for t in th]
        ax[0].plot(fpr, tpr, color=RUNS[k][3], lw=1.2, label=f"{PLAIN[k]} (AUC {roc_auc(s, yb)*100:.2f})")
        _, far, frr = det_curve(s, yb, 400)
        ax[1].plot(far * 100, frr * 100, color=RUNS[k][3], lw=1.2, label=PLAIN[k])
    ax[0].set_xlabel("BPCER (false alarm rate)"); ax[0].set_ylabel("Attack detection rate")
    ax[0].set_xlim(0, 0.10); ax[0].set_ylim(0.85, 1.001); ax[0].legend(loc="lower right", fontsize=6.5)
    ax[0].set_title("(a) ROC, low-false-alarm regime", fontsize=8)
    ax[1].set_xlabel("APCER (%)"); ax[1].set_ylabel("BPCER (%)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_title("(b) DET", fontsize=8); ax[1].legend(fontsize=6.5)
    fig.savefig(f"{FIG}/roc_det.pdf"); fig.savefig(f"{FIG}/roc_det.png"); plt.close(fig)


def fig_calibration():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    bins = np.linspace(0, 1, 13)
    for k in ("xuad", "cdcn", "resnet18"):
        p, y, u, _, _ = preds(k)
        conf = p.max(1); corr = (p.argmax(1) == y).astype(float)
        xs, ys = [], []
        for i in range(len(bins) - 1):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.sum() >= 20:
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        ax[0].plot(xs, ys, "o-", ms=2.5, lw=1.1, color=RUNS[k][3], label=PLAIN[k])
        ax[1].hist(u, bins=40, histtype="step", lw=1.1, density=True, color=RUNS[k][3], label=PLAIN[k])
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.7, label="perfect calibration")
    ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("(a) Reliability diagram (no temperature scaling)", fontsize=8); ax[0].legend(fontsize=6.5)
    ax[1].set_xlabel("uncertainty (vacuity / 1-max softmax)"); ax[1].set_ylabel("density")
    ax[1].set_title("(b) Uncertainty distribution", fontsize=8); ax[1].legend(fontsize=6.5)
    fig.savefig(f"{FIG}/calibration.pdf"); fig.savefig(f"{FIG}/calibration.png"); plt.close(fig)


def fig_confusion():
    p, y, _, _, _ = preds("xuad")
    k = len(ATTACK_NAMES)
    M = np.zeros((k, k))
    for a, b in zip(y, p.argmax(1)):
        M[a, b] += 1
    Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)
    SUMMARY["confusion_pct"] = {ATTACK_NAMES[i]: {ATTACK_NAMES[j]: round(Mn[i, j] * 100, 1)
                                                  for j in range(k) if i != j and Mn[i, j] >= 0.01}
                                for i in range(k)}
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(Mn * 100, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(k)); ax.set_xticklabels(ATTACK_NAMES, rotation=55, ha="right", fontsize=6)
    ax.set_yticks(range(k)); ax.set_yticklabels(ATTACK_NAMES, fontsize=6)
    for i in range(k):
        for j in range(k):
            if Mn[i, j] > 0.005:
                ax.text(j, i, f"{Mn[i, j]*100:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if Mn[i, j] > 0.5 else "black")
    ax.grid(False); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, fraction=0.046, label="%")
    fig.savefig(f"{FIG}/confusion.pdf"); fig.savefig(f"{FIG}/confusion.png"); plt.close(fig)


if __name__ == "__main__":
    table_main(); table_per_type(); table_loc(); table_faith(); table_whitebox()
    fig_roc_det(); fig_calibration(); fig_confusion()
    json.dump(SUMMARY, open(os.path.join(EX, "rev3_paper_numbers.json"), "w"), indent=1, default=float)
    print(json.dumps({k: SUMMARY[k] for k in ("main", "mcnemar", "loc", "temperature")}, indent=1, default=float))
    print("confusion >=1%:", json.dumps(SUMMARY["confusion_pct"]))
