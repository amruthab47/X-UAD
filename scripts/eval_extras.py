"""Revision-3 evaluation-only analyses on existing checkpoints (no training).

  predict  full val/test predictions for each checkpoint, with and without
           horizontal-flip test-time augmentation (TTA); localisation over the
           *whole* test set per family, at the fixed 0.5 cut and at a cut fitted
           on the validation split.
  faith    deletion/insertion faithfulness per attack family, per-sample, with
           bootstrap confidence intervals and a spatially smoothed random control.
  attack   white-box PGD against X-UAD and CDCN++: (a) evasion, push an attack
           towards "bonafide"; (b) false confidence, push X-UAD's vacuity down
           while keeping its predicted class.

Every threshold is fitted on the validation split; test labels are only used to
score.  Writes to results_v3/extras/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES, ATTACK_SEVERITY  # noqa: E402
from xuad.data.dataset import XUADDataset  # noqa: E402
from xuad.engine.metrics import (LOCALISABLE, eer, localisation_metrics,  # noqa: E402
                                 roc_auc, summarise)
from xuad.engine.train import make_loader  # noqa: E402
from xuad.models.baselines import CDCNpp, ResNet18Lite  # noqa: E402
from xuad.models.xuad import XUAD  # noqa: E402

K = len(ATTACK_NAMES)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results_v3", "extras")

MODELS = {
    # name: (checkpoint, builder)
    "xuad_P1_repro": ("results_v3/xuad_P1_repro.pt",
                      lambda: XUAD(K, severity=torch.tensor(ATTACK_SEVERITY), img_size=112)),
    "xuad_P1_recipe_s0_oldcampaign": (
        "results_region_pooled_detached/xuad_P1_regionpool_detached.pt",
        lambda: XUAD(K, severity=torch.tensor(ATTACK_SEVERITY), img_size=112, region_pool=True)),
    "xuad_P1_pairweighted": ("results_pair_weighted_refinement/xuad_P1_pairweighted.pt",
                             lambda: XUAD(K, severity=torch.tensor(ATTACK_SEVERITY), img_size=112)),
    "cdcn_P1_recipe_s0": ("results_v3/cdcn_P1_recipe_s0.pt", lambda: CDCNpp(K, img_size=112)),
    "resnet18_P1_recipe_s0": ("results_v3/resnet18_P1_recipe_s0.pt", lambda: ResNet18Lite(K)),
}


def extra_store():
    meta = json.load(open(os.path.join(DATA, "adv_meta.json")))
    return {"x": np.load(os.path.join(DATA, "adv_x.npy"), mmap_mode="r"),
            "m": np.load(os.path.join(DATA, "adv_m.npy"), mmap_mode="r"),
            "split": meta["split"], "label": meta["label"], "identity": meta["identity"]}


def load_model(name):
    ckpt, build = MODELS[name]
    m = build()
    m.load_state_dict(torch.load(os.path.join(ROOT, ckpt), map_location="cpu"))
    return m.eval()


def split(name):
    return XUADDataset(DATA, "A", name, img=112, extra=extra_store())


# --------------------------------------------------------------------------- predict
@torch.no_grad()
def run_split(model, ds, bs=96):
    P, Pf, U, Uf, L, G = [], [], [], [], [], []
    for x, m, _ in make_loader(ds, bs, False, 0):
        o = model(x)
        of = model(torch.flip(x, dims=[-1]))
        P.append(o["prob"].numpy()); Pf.append(of["prob"].numpy())
        U.append(o["vacuity"].numpy()); Uf.append(of["vacuity"].numpy())
        if o.get("loc") is not None:
            L.append(o["loc"].numpy().astype(np.float16)); G.append(m.numpy().astype(np.float16))
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    return dict(prob=cat(P), prob_flip=cat(Pf), vac=cat(U), vac_flip=cat(Uf),
                loc=cat(L), loc_gt=cat(G), y=ds.y)


def per_family_iou(loc, gt, y, thr):
    out = {}
    for c in range(1, K):
        sel = y == c
        if sel.any():
            out[ATTACK_NAMES[c]] = localisation_metrics(loc[sel].astype(np.float32),
                                                        gt[sel].astype(np.float32), thr)["IoU"]
    out["region_mean"] = float(np.mean([out[ATTACK_NAMES[c]] for c in LOCALISABLE
                                        if ATTACK_NAMES[c] in out and c != 7]))
    out["region_mean_incl_adv"] = float(np.mean([out[ATTACK_NAMES[c]] for c in LOCALISABLE
                                                 if ATTACK_NAMES[c] in out]))
    return out


def stratified_pixel_auc(loc, gt, y, per=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(1, K):
        cand = np.where(y == c)[0]
        idx += list(rng.choice(cand, min(per, len(cand)), replace=False))
    idx = np.array(idx)
    keep = gt[idx].reshape(len(idx), -1).max(1) > 0.2
    idx = idx[keep]
    return roc_auc(loc[idx].astype(np.float32).reshape(-1),
                   (gt[idx] > 0.5).reshape(-1).astype(int))


def predict(names):
    os.makedirs(OUT, exist_ok=True)
    va, te = split("val"), split("test")
    for name in names:
        t0 = time.time()
        model = load_model(name)
        rv, rt = run_split(model, va), run_split(model, te)
        res = {"model": name, "checkpoint": MODELS[name][0]}
        for mode in ("plain", "tta_flip"):
            pv = rv["prob"] if mode == "plain" else (rv["prob"] + rv["prob_flip"]) / 2
            pt = rt["prob"] if mode == "plain" else (rt["prob"] + rt["prob_flip"]) / 2
            ut = rt["vac"] if mode == "plain" else (rt["vac"] + rt["vac_flip"]) / 2
            _, thr = eer(1 - pv[:, 0], (rv["y"] != 0).astype(int))   # threshold from val
            m = summarise(pt, rt["y"], ut, thr=thr, k=K)
            m["per_type_apcer"] = {ATTACK_NAMES[int(k)]: v * 100
                                   for k, v in m.pop("per_type_apcer").items()}
            res[mode] = m
        if rt["loc"].size:
            # localisation over the full test set; cut either fixed at 0.5 or fitted on val
            grid = np.round(np.arange(0.1, 0.91, 0.05), 2)
            val_scores = {float(t): per_family_iou(rv["loc"], rv["loc_gt"], rv["y"], t)["region_mean"]
                          for t in grid}
            t_fit = max(val_scores, key=val_scores.get)
            res["loc"] = {
                "n_test": int(len(rt["y"])),
                "fixed_0.5": per_family_iou(rt["loc"], rt["loc_gt"], rt["y"], 0.5),
                "val_fitted_thr": t_fit,
                "val_region_mean_by_thr": val_scores,
                "fitted": per_family_iou(rt["loc"], rt["loc_gt"], rt["y"], t_fit),
                "pixel_AUC_stratified": stratified_pixel_auc(rt["loc"], rt["loc_gt"], rt["y"]),
            }
        np.savez_compressed(os.path.join(OUT, f"{name}_pred_full.npz"),
                            **{f"val_{k}": v for k, v in rv.items() if k not in ("loc", "loc_gt")},
                            **{f"test_{k}": v for k, v in rt.items() if k not in ("loc", "loc_gt")})
        res["seconds"] = time.time() - t0
        json.dump(res, open(os.path.join(OUT, f"{name}_predict.json"), "w"), indent=1)
        p, q = res["plain"], res["tta_flip"]
        print(f"{name}: ACER {p['ACER']:.2f} -> TTA {q['ACER']:.2f} | ACC {p['ACC']:.2f} -> "
              f"{q['ACC']:.2f} | ECE {p['ECE']:.2f} -> {q['ECE']:.2f}"
              + (f" | locIoU(0.5) {res['loc']['fixed_0.5']['region_mean']*100:.2f} "
                 f"fitted@{res['loc']['val_fitted_thr']} {res['loc']['fitted']['region_mean']*100:.2f}"
                 if "loc" in res else "") + f"  ({res['seconds']:.0f}s)", flush=True)


# --------------------------------------------------------------------------- faith
@torch.no_grad()
def del_ins_per_sample(model, x, sal, n_steps=10):
    N = x.shape[0]
    s = F.interpolate(torch.from_numpy(sal).float()[:, None], size=x.shape[-2:],
                      mode="bilinear", align_corners=False)[:, 0]
    blur = F.avg_pool2d(x, 17, 1, 8)
    order = s.reshape(N, -1).argsort(1, descending=True)
    npx = order.shape[1]
    D, I = [], []
    for i in range(n_steps + 1):
        k = int(npx * i / n_steps)
        mask = torch.zeros(N, npx)
        if k:
            mask.scatter_(1, order[:, :k], 1.0)
        mask = mask.reshape(N, 1, *x.shape[-2:])
        D.append((1 - model(x * (1 - mask) + blur * mask)["prob"][:, 0]).numpy())
        I.append((1 - model(blur * (1 - mask) + x * mask)["prob"][:, 0]).numpy())
    D, I = np.stack(D, 1), np.stack(I, 1)
    return np.trapezoid(D, dx=1 / n_steps, axis=1), np.trapezoid(I, dx=1 / n_steps, axis=1), D[:, 0], I[:, 0]


def boot_ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def faith(name="xuad_P1_repro", per=48):
    import cv2
    os.makedirs(OUT, exist_ok=True)
    model = load_model(name)
    te = split("test")
    out = {"model": name, "per_family_n": per, "families": {}}
    pooled = {"region": {"loc": [], "rnd": []}, "whole": {"loc": [], "rnd": []}}
    for c in range(1, K):
        cand = np.where(te.y == c)[0]
        idx = cand[np.linspace(0, len(cand) - 1, per).astype(int)]
        x = torch.stack([te[i][0] for i in idx])
        with torch.no_grad():
            loc = model(x)["loc"].numpy()
        g = np.random.default_rng(c).random((len(idx), 8, 8)).astype(np.float32)
        rnd = np.stack([cv2.resize(a, loc.shape[1:][::-1], interpolation=cv2.INTER_CUBIC) for a in g])
        dl, il, base_d, base_i = del_ins_per_sample(model, x, loc)
        dr, ir, _, _ = del_ins_per_sample(model, x, rnd)
        fam = ATTACK_NAMES[c]
        out["families"][fam] = {
            "deletion_loc": float(dl.mean()), "deletion_rnd": float(dr.mean()),
            "insertion_loc": float(il.mean()), "insertion_rnd": float(ir.mean()),
            "deletion_gain": float((dr - dl).mean()), "deletion_gain_CI95": boot_ci(dr - dl),
            "insertion_gain": float((il - ir).mean()), "insertion_gain_CI95": boot_ci(il - ir),
            "score_unperturbed": float(base_d.mean()), "score_fully_blurred": float(base_i.mean()),
        }
        grp = "region" if c in (3, 4, 5) else "whole"
        pooled[grp]["loc"].append(np.stack([dl, il])); pooled[grp]["rnd"].append(np.stack([dr, ir]))
        print(fam, {k: round(v, 3) if isinstance(v, float) else v
                    for k, v in out["families"][fam].items()}, flush=True)
    for grp, d in pooled.items():
        L, R = np.concatenate(d["loc"], 1), np.concatenate(d["rnd"], 1)
        out[grp] = {"n": int(L.shape[1]),
                    "deletion_loc": float(L[0].mean()), "deletion_rnd": float(R[0].mean()),
                    "insertion_loc": float(L[1].mean()), "insertion_rnd": float(R[1].mean()),
                    "deletion_gain_CI95": boot_ci(R[0] - L[0]),
                    "insertion_gain_CI95": boot_ci(L[1] - R[1])}
        print(grp, out[grp], flush=True)
    json.dump(out, open(os.path.join(OUT, f"{name}_faith.json"), "w"), indent=1)


# --------------------------------------------------------------------------- attack
def pgd(model, x, loss_fn, eps, steps=10):
    """L-inf PGD in pixel space [0,1]; inputs are normalised as (p - 0.5) / 0.5."""
    e = 2 * eps                     # the same budget in normalised units
    a = e / 4
    delta = torch.zeros_like(x).uniform_(-e, e)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = loss_fn(model(torch.clamp(x + delta, -1, 1)))
        g, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() - a * g.sign()).clamp(-e, e)    # minimise loss_fn
    return torch.clamp(x + delta, -1, 1).detach()


def attack(xname="xuad_P1_repro", per=24, eps_list=(1, 2, 4, 8)):
    os.makedirs(OUT, exist_ok=True)
    va, te = split("val"), split("test")
    rng = np.random.default_rng(0)
    atk_idx = np.concatenate([rng.choice(np.where(te.y == c)[0], per, replace=False) for c in range(1, K)])
    bon_idx = rng.choice(np.where(te.y == 0)[0], per * 2, replace=False)
    xa = torch.stack([te[i][0] for i in atk_idx]); ya = te.y[atk_idx]
    xb = torch.stack([te[i][0] for i in bon_idx])
    out = {"n_attack": int(len(atk_idx)), "n_bonafide": int(len(bon_idx)), "steps": 10, "eps_255": list(eps_list)}
    for name in (xname, "cdcn_P1_recipe_s0"):
        model = load_model(name)
        for p in model.parameters():
            p.requires_grad_(False)
        thr = json.load(open(os.path.join(OUT, f"{name}_predict.json")))["plain"]["threshold"]
        with torch.no_grad():
            oa, ob = model(xa), model(xb)
        r = {"threshold_from_val": thr,
             "clean": {"attack_detect_rate": float(((1 - oa["prob"][:, 0]) > thr).float().mean() * 100),
                       "mean_vacuity_attack": float(oa["vacuity"].mean()),
                       "mean_vacuity_bonafide": float(ob["vacuity"].mean())}}
        for eps in eps_list:
            # (a) evasion: make attacks look bonafide
            xadv = pgd(model, xa, lambda o: -torch.log(o["prob"][:, 0] + 1e-8).mean(), eps / 255)
            with torch.no_grad():
                o = model(xadv)
            ent = {"attack_detect_rate": float(((1 - o["prob"][:, 0]) > thr).float().mean() * 100),
                   "mean_vacuity": float(o["vacuity"].mean()),
                   # can vacuity catch what the classifier missed?
                   "missed_flagged_by_u>=0.55": float((o["vacuity"][(1 - o["prob"][:, 0]) <= thr] >= 0.55)
                                                      .float().mean() * 100)
                   if ((1 - o["prob"][:, 0]) <= thr).any() else None}
            r[f"evasion_eps{eps}"] = ent
            if name.startswith("xuad"):
                # (b) false confidence: lower vacuity, keep the predicted class
                pred = oa["prob"].argmax(1)
                xfc = pgd(model, xa, lambda o: o["vacuity"].mean()
                          + F.nll_loss(torch.log(o["prob"] + 1e-8), pred), eps / 255)
                with torch.no_grad():
                    o2 = model(xfc)
                r[f"false_confidence_eps{eps}"] = {
                    "mean_vacuity": float(o2["vacuity"].mean()),
                    "class_kept_%": float((o2["prob"].argmax(1) == pred).float().mean() * 100),
                    "attack_detect_rate": float(((1 - o2["prob"][:, 0]) > thr).float().mean() * 100)}
            print(name, eps, r[f"evasion_eps{eps}"],
                  r.get(f"false_confidence_eps{eps}", ""), flush=True)
        out[name] = r
    json.dump(out, open(os.path.join(OUT, f"whitebox_attack_{xname}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["predict", "faith", "attack"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--xuad", default="xuad_P1_repro", help="X-UAD checkpoint for faith/attack")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    if a.what == "predict":
        predict(a.models.split(","))
    elif a.what == "faith":
        faith(a.xuad)
    else:
        attack(a.xuad)
