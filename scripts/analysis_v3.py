"""Revision-3 analysis over results_v3/ (and the existing X-UAD 12-epoch seed-0 run).

Writes results_v3/SUMMARY_v3.md and results_v3/summary_v3.json:
  1. reproduction check of Table I with the current code;
  2. 12-epoch recipe, all models, mean +- std over seeds, with McNemar on seed 0;
  3. leave-one-attack-out at the full P1 budget;
  4. referral rates of the u >= 0.55 rule per group;
  5. feature-distance novelty score (Mahalanobis to seen-class means, fitted on the
     validation split only) and its fixed combination with vacuity.
Nothing here is fitted on test labels.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(ROOT, "results_v3")
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES  # noqa: E402

FOLDS = ["replay", "mask3d", "faceswap", "adversarial"]
MODELS = ["xuad", "cdcn", "resnet18"]
NICE = {"xuad": "X-UAD", "cdcn": "CDCN++", "resnet18": "ResNet-18"}
REFER_U = 0.55


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def roc_auc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos, neg = label.sum(), (1 - label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def mcnemar_exact(a_right, b_right):
    b = int(np.sum(a_right & ~b_right)); c = int(np.sum(~a_right & b_right))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return b, c, min(1.0, p)


def ms(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.2f} (1 seed)"
    return f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f} (n={len(v)})"


def mahalanobis_score(val_f, val_y, test_f, classes):
    """Min Mahalanobis distance to seen-class means; shared covariance, val split only."""
    mu = np.stack([val_f[val_y == c].mean(0) for c in classes])
    centred = np.concatenate([val_f[val_y == c] - mu[i] for i, c in enumerate(classes)])
    cov = np.cov(centred, rowvar=False) + 1e-3 * np.eye(val_f.shape[1])
    prec = np.linalg.pinv(cov)
    d = np.stack([np.einsum("ij,jk,ik->i", test_f - m, prec, test_f - m) for m in mu], 1)
    return d.min(1)


def zscore_on_val(val_s, test_s):
    return (test_s - val_s.mean()) / (val_s.std() + 1e-8)


def main():
    out, S = [], {}

    # 1. reproduction
    out.append("# Revision-3 results\n")
    old = load_json(os.path.join(ROOT, "results", "xuad_P1.json"))
    new = load_json(os.path.join(V3, "xuad_P1_repro.json"))
    out.append("## 1. Does the current code reproduce Table I? (X-UAD, P1, 10 epochs, seed 0)\n")
    if new:
        out.append("| Metric | Paper (freeze code) | Current code | Diff |\n|---|---|---|---|")
        for k in ["ACER", "APCER", "BPCER", "AUC", "EER", "ACC", "macroF1", "ECE"]:
            out.append(f"| {k} | {old[k]:.2f} | {new[k]:.2f} | {new[k]-old[k]:+.2f} |")
        lo, ln = old["loc_per_class"]["mean_localisable"], new["loc_per_class"]["mean_localisable"]
        out.append(f"| Region IoU | {100*lo:.2f} | {100*ln:.2f} | {100*(ln-lo):+.2f} |")
        S["repro"] = {k: [old[k], new[k]] for k in ["ACER", "ACC", "ECE", "AUC"]}
    else:
        out.append("_not finished yet_")

    # 2. 12-epoch recipe with seeds
    out.append("\n## 2. 12-epoch recipe, all models (mean ± sample std over seeds)\n")
    rec = {}
    for m in MODELS:
        runs = []
        for s in (0, 1, 2):
            p = os.path.join(V3, f"{m}_P1_recipe_s{s}.json")
            if m == "xuad" and s == 0 and not os.path.exists(p):
                # fall back to the older-campaign run until the current-code rerun finishes
                p = os.path.join(ROOT, "results_region_pooled_detached", "xuad_P1_regionpool_detached.json")
            j = load_json(p)
            if j:
                runs.append((s, j, p))
        rec[m] = runs
    keys = ["ACER", "APCER", "BPCER", "AUC", "ACC", "macroF1", "ECE"]
    out.append("| Model | " + " | ".join(keys) + " | Region IoU |")
    out.append("|---" * (len(keys) + 2) + "|")
    for m in MODELS:
        cells = [ms([j[k] for _, j, _ in rec[m]]) for k in keys]
        iou = [100 * j["loc_per_class"]["mean_localisable"] for _, j, _ in rec[m] if "loc_per_class" in j]
        out.append(f"| {NICE[m]} | " + " | ".join(cells) + f" | {ms(iou) if iou else '—'} |")
    S["recipe"] = {m: {k: [j[k] for _, j, _ in rec[m]] for k in keys} for m in MODELS}

    # McNemar on seed 0 (8-way and binary)
    def pred_path(json_path):
        return json_path.replace(".json", "_pred.npz")
    x0 = [r for r in rec["xuad"] if r[0] == 0]
    if x0:
        zx = np.load(pred_path(x0[0][2]))
        yx = zx["y"]
        out.append("\nMcNemar exact test, seed 0, same 9,600 test samples "
                   "(b = X-UAD right & other wrong, c = the reverse):\n")
        out.append("| vs | 8-way b / c | p | binary b / c | p |\n|---|---|---|---|---|")
        thr_x = x0[0][1]["threshold"]
        for m in ("cdcn", "resnet18"):
            r0 = [r for r in rec[m] if r[0] == 0]
            if not r0:
                continue
            zo = np.load(pred_path(r0[0][2]))
            assert np.array_equal(zo["y"], yx), "test sets differ"
            b8, c8, p8 = mcnemar_exact(zx["prob"].argmax(1) == yx, zo["prob"].argmax(1) == yx)
            bx = (1 - zx["prob"][:, 0] > thr_x) == (yx != 0)
            bo = (1 - zo["prob"][:, 0] > r0[0][1]["threshold"]) == (yx != 0)
            b2, c2, p2 = mcnemar_exact(bx, bo)
            out.append(f"| {NICE[m]} | {b8} / {c8} | {p8:.3g} | {b2} / {c2} | {p2:.3g} |")

    # 3-5. leave-one-attack-out at full budget
    out.append("\n## 3. Leave-one-attack-out at the full P1 budget (10 epochs, 19,200-sample split)\n")
    loao = {m: {} for m in MODELS}
    for m in MODELS:
        for f in FOLDS:
            jp = os.path.join(V3, f"{m}_P2full.{f}.json")
            zp = os.path.join(V3, f"{m}_P2full.{f}_pred.npz")
            j = load_json(jp)
            if not j or not os.path.exists(zp):
                continue
            z = np.load(zp)
            c = ATTACK_NAMES.index(f)
            y, p, u = z["y"], z["prob"], z["vacuity"]
            thr = float(z["threshold"][0])
            seen_atk = (y != 0) & (y != c)
            m1 = (y == c) | seen_atk
            msp = 1.0 - p.max(1)
            d = dict(ACER=j["ACER"], ACC=j["ACC"],
                     detect=float(((1 - p[:, 0]) > thr)[y == c].mean() * 100),
                     nov_u=roc_auc(u[m1], (y[m1] == c).astype(int)) * 100,
                     nov_msp=roc_auc(msp[m1], (y[m1] == c).astype(int)) * 100,
                     refer_unseen=float((u[y == c] >= REFER_U).mean() * 100),
                     refer_bonafide=float((u[y == 0] >= REFER_U).mean() * 100),
                     refer_seen=float((u[seen_atk] >= REFER_U).mean() * 100))
            if z["feature"].ndim == 2 and z["val_feature"].ndim == 2:
                vf, vy, tf = z["val_feature"], z["val_y"], z["feature"]
                seen = [k for k in np.unique(vy) if k != c]
                dist = mahalanobis_score(vf, vy, tf, seen)
                d["nov_dist"] = roc_auc(dist[m1], (y[m1] == c).astype(int)) * 100
                vdist = mahalanobis_score(vf, vy, vf, seen)
                base_val = z["val_vacuity"] if m == "xuad" else 1.0 - z["val_prob"].max(1)
                base_te = u if m == "xuad" else msp
                comb = zscore_on_val(base_val, base_te) + zscore_on_val(vdist, dist)
                d["nov_comb"] = roc_auc(comb[m1], (y[m1] == c).astype(int)) * 100
            loao[m][f] = d
    S["loao_full"] = loao
    blocks = [("ACER", "ACER (%) ↓"), ("ACC", "8-way accuracy (%) ↑"),
              ("detect", "Unseen-family detection rate (%) ↑"),
              ("nov_u", "Novelty AUROC, model's own uncertainty (vacuity for X-UAD, 1-max softmax for baselines) ↑"),
              ("nov_msp", "Novelty AUROC, 1-max softmax (all models) ↑"),
              ("nov_dist", "Novelty AUROC, feature distance (Mahalanobis, fitted on val) ↑"),
              ("nov_comb", "Novelty AUROC, uncertainty + feature distance (fixed equal weights, z-scored on val) ↑")]
    for key, title in blocks:
        out.append(f"\n**{title}**\n")
        out.append("| Model | " + " | ".join(FOLDS) + " | Mean |")
        out.append("|---" * (len(FOLDS) + 2) + "|")
        for m in MODELS:
            vals = [loao[m].get(f, {}).get(key) for f in FOLDS]
            cells = [f"{v:.2f}" if v is not None else "—" for v in vals]
            got = [v for v in vals if v is not None]
            mean = f"{np.mean(got):.2f}" if len(got) == len(FOLDS) else "—"
            out.append(f"| {NICE[m]} | " + " | ".join(cells) + f" | {mean} |")

    out.append(f"\n## 4. Referral rule u ≥ {REFER_U} (X-UAD), % of each group referred to a human\n")
    out.append("| Fold | unseen attack | seen attacks | genuine |\n|---|---|---|---|")
    for f in FOLDS:
        d = loao["xuad"].get(f)
        if d:
            out.append(f"| {f} | {d['refer_unseen']:.1f} | {d['refer_seen']:.1f} | {d['refer_bonafide']:.1f} |")

    open(os.path.join(V3, "SUMMARY_v3.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump(S, open(os.path.join(V3, "summary_v3.json"), "w"), indent=1, default=float)
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
