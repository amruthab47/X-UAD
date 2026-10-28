"""Emit LaTeX tables from the saved experiment JSONs (no number is typed by hand)."""
from __future__ import annotations

import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES

RES, TAB = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper", "tables")
os.makedirs(TAB, exist_ok=True)

NAME = {
    "resnet18": r"ResNet-18~\cite{he2016resnet}",
    "vit_tiny": r"ViT-Tiny~\cite{dosovitskiy2021vit}",
    "cdcn": r"CDCN++~\cite{yu2020cdcn}",
    "xuad": r"\textbf{X-UAD (ours)}",
    "xuad_no_aqa": r"\quad w/o Adaptive Quantum Attention",
    "xuad_no_evid": r"\quad w/o evidential head (softmax)",
    "xuad_no_spec": r"\quad w/o spectral branch",
    "xuad_no_cdc": r"\quad w/o central-difference stem",
    "xuad_no_pues": r"\quad w/o pseudo-unseen suppression",
    "xuad_no_loc": r"\quad w/o localisation supervision",
    "xuad_ab": r"\textbf{X-UAD (full, reduced budget)}",
}
PRETTY_CLS = {"bonafide": "Bonafide", "print": "Print", "replay": "Replay",
              "mask3d": "3D mask", "faceswap": "Face-swap", "reenact": "Reenact",
              "synth": "Synthetic", "adversarial": "Adversarial"}


def L(tag):
    p = os.path.join(RES, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, n=2, dash="--"):
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def best_marks(rows, key, lower_is_better):
    vals = [r[1].get(key) for r in rows if r[1] and r[1].get(key) is not None]
    if not vals:
        return None
    return min(vals) if lower_is_better else max(vals)


def emit(path, s):
    open(os.path.join(TAB, path), "w").write(s)
    print("wrote", path)


# --------------------------------------------------------------------------- #
def table_main(models=("resnet18", "vit_tiny", "cdcn", "xuad"), proto="P1",
               fname="main.tex"):
    rows = [(m, L(f"{m}_{proto}")) for m in models]
    rows = [(m, d) for m, d in rows if d]
    cols = [("ACER", 2, True), ("APCER", 2, True), ("BPCER", 2, True),
            ("AUC", 2, False), ("EER", 2, True), ("ACC", 2, False),
            ("macroF1", 2, False), ("ECE", 2, True)]
    best = {k: best_marks(rows, k, lo) for k, _, lo in cols}
    s = [r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule",
         "Method & Params & " + " & ".join(
             {"ACER": r"ACER$\downarrow$", "APCER": r"APCER$\downarrow$",
              "BPCER": r"BPCER$\downarrow$", "AUC": r"AUC$\uparrow$",
              "EER": r"EER$\downarrow$", "ACC": r"ACC$_8\uparrow$",
              "macroF1": r"F1$_{\text{macro}}\uparrow$", "ECE": r"ECE$\downarrow$"}[k]
             for k, _, _ in cols) + r" \\",
         " & (M) & " + " & ".join(["(\\%)"] * len(cols)) + r" \\", r"\midrule"]
    for m, d in rows:
        cells = []
        for k, n, _ in cols:
            v = d.get(k)
            t = f(v, n)
            if best[k] is not None and v is not None and abs(v - best[k]) < 1e-9:
                t = r"\textbf{" + t + "}"
            cells.append(t)
        s.append(f"{NAME.get(m, m)} & {d['params_M']:.2f} & " + " & ".join(cells) + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit(fname, "\n".join(s))


def table_per_type(models=("resnet18", "vit_tiny", "cdcn", "xuad"), proto="P1"):
    rows = [(m, L(f"{m}_{proto}")) for m in models]
    rows = [(m, d) for m, d in rows if d and d.get("per_type_apcer")]
    if not rows:
        return
    cls = [c for c in ATTACK_NAMES[1:] if c in rows[0][1]["per_type_apcer"]]
    s = [r"\begin{tabular}{l" + "c" * (len(cls) + 1) + "}", r"\toprule",
         "Method & " + " & ".join(PRETTY_CLS[c] for c in cls) + r" & Worst \\", r"\midrule"]
    for m, d in rows:
        v = [d["per_type_apcer"][c] for c in cls]
        s.append(f"{NAME.get(m, m)} & " + " & ".join(f(x) for x in v)
                 + f" & {f(max(v))}" + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("per_type.tex", "\n".join(s))


def table_loao(models=("resnet18", "cdcn", "xuad")):
    folds = [c for c in ATTACK_NAMES
             if os.path.exists(os.path.join(RES, f"xuad_P2.{c}.json"))]
    if not folds:
        return
    s = [r"\begin{tabular}{ll" + "c" * len(folds) + "c}", r"\toprule",
         r"Metric & Method & " + " & ".join(PRETTY_CLS[c] for c in folds) + r" & Mean \\",
         r"\midrule"]
    blocks = [("ACER (\\%)$\\downarrow$", "ACER", True),
              ("ACC$_8$ (\\%)$\\uparrow$", "ACC", False),
              ("Unseen detection rate (\\%)$\\uparrow$", "unseen_detect_rate", False),
              ("Novelty AUROC (\\%)$\\uparrow$", "novelty_AUROC", False)]
    for bi, (title, key, lo) in enumerate(blocks):
        got = False
        for m in models:
            ds = [L(f"{m}_P2.{c}") for c in folds]
            if any(d is None or d.get(key) is None for d in ds):
                continue
            v = [d[key] for d in ds]
            pre = r"\multirow{%d}{*}{%s} & " % (len(models), title) if not got else " & "
            s.append(pre + NAME.get(m, m) + " & " + " & ".join(f(x) for x in v)
                     + f" & {f(float(np.mean(v)))}" + r" \\")
            got = True
        if got and bi < len(blocks) - 1:
            s.append(r"\midrule")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("loao.tex", "\n".join(s))


ABL_TAGS = [("xuad_ab", "xuad_ab"),
            ("xuad_no_aqa", "xuad_no_aqa_ab"),
            ("xuad_no_evid", "xuad_no_evid_ab"),
            ("xuad_no_spec", "xuad_no_spec_ab"),
            ("xuad_no_cdc", "xuad_no_cdc_ab"),
            ("xuad_no_pues", "xuad_no_pues_ab"),
            ("xuad_no_loc", "xuad_no_loc_ab")]


def table_ablation(order=ABL_TAGS):
    rows = [(m, L(tag)) for m, tag in order]
    rows = [(m, d) for m, d in rows if d]
    if not rows:
        return
    keys = [("ACER", True), ("AUC", False), ("ACC", False), ("macroF1", False),
            ("ECE", True), ("loc_mean_localisable", False), ("unseen_mean", False)]
    hdr = {"ACER": r"ACER$\downarrow$", "AUC": r"AUC$\uparrow$", "ACC": r"ACC$_8\uparrow$",
           "macroF1": r"F1$_{\text{macro}}\uparrow$", "ECE": r"ECE$\downarrow$",
           "loc_mean_localisable": r"IoU$_{\text{loc}}\uparrow$",
           "unseen_mean": r"Unseen AUROC$\uparrow$"}
    for m, d in rows:
        if d.get("loc_per_class"):
            d["loc_mean_localisable"] = d["loc_per_class"].get("mean_localisable")
    keys = [k for k in keys if any(r[1].get(k[0]) is not None for r in rows)]
    s = [r"\begin{tabular}{l" + "c" * len(keys) + "}", r"\toprule",
         "Variant & " + " & ".join(hdr[k] for k, _ in keys) + r" \\", r"\midrule"]
    ref = rows[0][1]
    for m, d in rows:
        cells = []
        for k, lo in keys:
            v = d.get(k)
            t = f(v * (100 if k == "loc_mean_localisable" else 1))
            if m != rows[0][0] and v is not None and ref.get(k) is not None:
                dd = (v - ref[k]) * (100 if k == "loc_mean_localisable" else 1)
                t += r"\,\tiny{(" + ("+" if dd >= 0 else "") + f"{dd:.2f})}}"
            cells.append(t)
        s.append(NAME.get(m, m) + " & " + " & ".join(cells) + r" \\")
        if m == rows[0][0]:
            s.append(r"\midrule")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("ablation.tex", "\n".join(s))


def table_loc(models=("cdcn", "xuad")):
    rows = [(m, L(f"{m}_P1")) for m in models]
    rows = [(m, d) for m, d in rows if d and d.get("loc_per_class")]
    if not rows:
        return
    cls = [c for c in ATTACK_NAMES if c in rows[0][1]["loc_per_class"]]
    s = [r"\begin{tabular}{l" + "c" * (len(cls) + 2) + "}", r"\toprule",
         "Method & " + " & ".join(PRETTY_CLS[c] for c in cls)
         + r" & Region-level mean & Pixel AUC \\", r"\midrule"]
    for m, d in rows:
        v = [d["loc_per_class"][c] * 100 for c in cls]
        s.append(NAME.get(m, m) + " & " + " & ".join(f(x, 1) for x in v)
                 + f" & {f(d['loc_per_class']['mean_localisable']*100,1)}"
                 + f" & {f(d.get('loc_pixel_AUC', float('nan'))*100,1)}" + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("localisation.tex", "\n".join(s))


def table_cross():
    rows = [(m, L(f"{m}_P3")) for m in ("resnet18", "vit_tiny", "cdcn", "xuad")]
    rows = [(m, d) for m, d in rows if d]
    if not rows:
        return
    s = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Method & ACER$\downarrow$ & AUC$\uparrow$ & EER$\downarrow$ & ACC$_8\uparrow$ & ECE$\downarrow$ \\",
         r"\midrule"]
    for m, d in rows:
        s.append(NAME.get(m, m) + " & " + " & ".join(
            f(d.get(k)) for k in ("ACER", "AUC", "EER", "ACC", "ECE")) + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("crossdomain.tex", "\n".join(s))


def table_faithfulness():
    p = os.path.join(RES, "faithfulness.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    ctrl = d.get("random", {}).get("deletion")
    s = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Explanation map & Deletion AUC$\downarrow$ & Insertion AUC$\uparrow$ "
         r"& $\Delta$ deletion vs.\ control$\downarrow$ \\",
         r"\midrule"]
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        dd = ("--" if ctrl is None or k == "random"
              else f"{v['deletion'] - ctrl:+.3f}")
        s.append(f"{v.get('label', k)} & {f(v['deletion'],3)} & {f(v['insertion'],3)} & "
                 f"{dd}" + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("faithfulness.tex", "\n".join(s))


def table_efficiency():
    rows = [(m, L(f"{m}_P1")) for m in ("resnet18", "vit_tiny", "cdcn", "xuad")]
    rows = [(m, d) for m, d in rows if d]
    if not rows:
        return
    ex = json.load(open(os.path.join(RES, "efficiency.json"))) \
        if os.path.exists(os.path.join(RES, "efficiency.json")) else {}
    s = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Method & Params (M) & GMACs & Latency (ms) & Explanation outputs \\", r"\midrule"]
    cap = {"resnet18": "none", "vit_tiny": "attention only", "cdcn": "depth-proxy map",
           "xuad": r"map + modes + text"}
    for m, d in rows:
        e = ex.get(m, {})
        s.append(f"{NAME.get(m,m)} & {d['params_M']:.2f} & {f(e.get('gmacs'),3)} & "
                 f"{f(e.get('latency_ms'),1)} & {cap.get(m,'--')}" + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("efficiency.tex", "\n".join(s))


def table_modes():
    p = os.path.join(RES, "basis_states.json")
    if not os.path.exists(p):
        return
    b = json.load(open(p))
    s = [r"\begin{tabular}{ccccc}", r"\toprule",
         r"Mode & Most selective for & Selectivity ($z$) & Mean activation & Spatial dispersion \\",
         r"\midrule"]
    for m in sorted(b, key=lambda d: -d["selectivity"]):
        s.append(f"{m['basis']} & {PRETTY_CLS.get(m['class'], m['class'])} & "
                 f"{m['selectivity']:.2f} & "
                 f"{m['class_response'][m['class']]:.3f} & {m['dispersion']:.3f}" + r" \\")
    s += [r"\bottomrule", r"\end{tabular}"]
    emit("modes.tex", "\n".join(s))


def table_gates():
    p = os.path.join(RES, "gates.json")
    if not os.path.exists(p):
        return
    g = json.load(open(p))
    per = g["per_class"]
    ks = [k for k in ATTACK_NAMES if k in per]
    s = [r"\begin{tabular}{l" + "c" * len(ks) + "}", r"\toprule",
         "Attack family & " + " & ".join(PRETTY_CLS[k] for k in ks) + r" \\", r"\midrule",
         r"Mean Born-channel gate $g$ & " + " & ".join(f"{per[k]:.3f}" for k in ks) + r" \\",
         r"\bottomrule", r"\end{tabular}"]
    emit("gates.tex", "\n".join(s))


def table_oablate():
    rows = [("Full X-UAD", "xuad_P2.faceswap"),
            ("w/o pseudo-unseen suppression", "xuad_no_pues_P2.faceswap"),
            ("w/o Adaptive Quantum Attention", "xuad_no_aqa_P2.faceswap")]
    rows = [(n, L(t)) for n, t in rows]
    if any(d is None for _, d in rows):
        return
    ks = [("ACER", r"ACER$\downarrow$"), ("AUC", r"AUC$\uparrow$"),
          ("ACC", r"ACC$_8\uparrow$"), ("ECE", r"ECE$\downarrow$"),
          ("unseen_detect_rate", r"Unseen detect.$\uparrow$"),
          ("novelty_AUROC", r"Novelty AUROC$\uparrow$")]
    out = [r"\begin{tabular}{l" + "c" * len(ks) + "}", r"\toprule",
           "Variant & " + " & ".join(h for _, h in ks) + r" \\", r"\midrule"]
    for n, d in rows:
        out.append(n + " & " + " & ".join(f(d.get(k)) for k, _ in ks) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    emit("oablate.tex", "\n".join(out))


if __name__ == "__main__":
    table_main(); table_per_type(); table_loao(); table_ablation()
    table_loc(); table_cross(); table_faithfulness(); table_efficiency()
    table_modes(); table_gates(); table_oablate()
