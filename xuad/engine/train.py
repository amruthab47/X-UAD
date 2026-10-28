"""Training / evaluation loops (CPU-friendly)."""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.attacks import ATTACK_NAMES, ATTACK_SUPERCLASS
from ..losses import XUADLoss
from .metrics import localisation_metrics, summarise


def make_loader(ds, bs, shuffle, workers=2):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                      drop_last=shuffle, persistent_workers=workers > 0,
                      pin_memory=False)


@torch.no_grad()
def evaluate(model, loader, device="cpu", collect_loc=True, max_loc=600):
    model.eval()
    P, Y, U, L, G, R, F = [], [], [], [], [], [], []
    for x, m, y in loader:
        out = model(x.to(device))
        P.append(out["prob"].cpu().numpy())
        if out.get("feature") is not None:
            F.append(out["feature"].float().cpu().numpy())
        Y.append(y.numpy())
        u = out["vacuity"]
        U.append((u if torch.is_tensor(u) else torch.as_tensor(u)).cpu().numpy())
        if out.get("risk") is not None:
            R.append(out["risk"].cpu().numpy())
        if collect_loc and out.get("loc") is not None and sum(len(a) for a in L) < max_loc:
            L.append(out["loc"].cpu().numpy()); G.append(m.numpy())
    P = np.concatenate(P); Y = np.concatenate(Y); U = np.concatenate(U)
    res = {"prob": P, "y": Y, "vacuity": U,
           "risk": np.concatenate(R) if R else None,
           "loc": np.concatenate(L) if L else None,
           "loc_gt": np.concatenate(G) if G else None,
           "feature": np.concatenate(F) if F else None}
    return res


def fit(model, train_ds, val_ds, *, epochs=8, bs=48, lr=2.5e-3, wd=0.02,
        device="cpu", n_classes=8, use_evidential=True, use_pues=True,
        loss_weights=None, log_path=None, workers=2, log_every=40, seed=0,
        pues_k=12, focal_gamma=0.0, select_by="auc_acer",
        checkpoint_path=None, resume_from=None, pcp_pairs=None, pair_weights=None):
    """``checkpoint_path``, if given, is overwritten with a rolling snapshot
    (model, optimiser, scheduler, RNG state, history-so-far) after every epoch,
    so a run killed mid-training loses at most one epoch of progress rather than
    everything -- ``fit`` otherwise only produces a result once it returns, which
    is fine for a healthy run but means an externally-killed process leaves
    nothing recoverable at all.  ``resume_from`` loads exactly that snapshot and
    continues the loop from the next epoch, with the optimiser/scheduler/RNG
    state restored so the continuation is a genuine resumption, not a restart
    with a warm-started model."""
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    bs = max(2, min(bs, max(2, len(train_ds) // 2)))     # tiny corpora: keep >=2 batches
    tl = make_loader(train_ds, bs, True, workers)
    vl = make_loader(val_ds, bs, False, workers)
    crit = XUADLoss(n_classes, ATTACK_SUPERCLASS, loss_weights, use_evidential,
                    focal_gamma=focal_gamma, pcp_pairs=pcp_pairs,
                    pair_weights=pair_weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = max(1, epochs * len(tl))
    if steps >= 8:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                    pct_start=0.25, div_factor=10,
                                                    final_div_factor=50)
    else:
        # one-cycle needs enough steps for its warm-up phase to have non-zero
        # length; tiny runs (a smoke test, or a corpus with a few hundred crops)
        # get a plain cosine decay instead of a crash.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    hist, best, best_state, gstep, start_epoch = [], -1e9, None, 0, 0

    if resume_from and os.path.exists(resume_from):
        # weights_only=False: this checkpoint is our own, self-generated file
        # (never an untrusted third-party artefact), and it embeds the NumPy RNG
        # state tuple alongside the tensors, which PyTorch's default safe
        # unpickler (weights_only=True as of torch 2.6+) rejects.
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        sched.load_state_dict(ck["sched_state"])
        best, best_state, hist, gstep = ck["best"], ck["best_state"], ck["hist"], ck["gstep"]
        torch.set_rng_state(ck["torch_rng_state"])
        np.random.set_state(ck["numpy_rng_state"])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {resume_from}: completed epochs 0-{ck['epoch']}, "
              f"continuing at epoch {start_epoch}", flush=True)

    # for select_by="acc_f1_acer_guard": recovers correctly across a resume,
    # since it's derived from `hist` (which the checkpoint already restores)
    # rather than tracked as separate state.
    best_acer_seen = min((e["val"]["ACER"] for e in hist), default=float("inf"))

    n_tokens = None
    t0 = time.time()
    for ep in range(start_epoch, epochs):
        model.train()
        anneal = min(1.0, (ep + 1) / max(1, epochs * 0.6))
        agg, nb = {}, 0
        for x, m, y in tl:
            x, m, y = x.to(device), m.to(device), y.to(device)
            out = model(x)
            if n_tokens is None:
                n_tokens = out["prob"].shape[0] and (out.get("grid", 8) ** 2 + 5)

            alpha_mix = None
            if use_pues and use_evidential and out.get("alpha") is not None:
                # build pseudo-unseen samples: convex mixtures of two *different*
                # attack classes inside the same batch
                idx = torch.randperm(x.shape[0], device=device)
                ok = (y != 0) & (y[idx] != 0) & (y != y[idx])
                if ok.sum() >= 2:
                    sel = torch.nonzero(ok).squeeze(1)[:pues_k]   # capped for CPU budget
                    lam = torch.distributions.Beta(2.0, 2.0).sample((len(sel), 1, 1, 1)).to(device)
                    xm = lam * x[sel] + (1 - lam) * x[idx][sel]
                    alpha_mix = model(xm)["alpha"]

            loss, logs = crit(out, y, m, n_tokens or 69, anneal, alpha_mix)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step(); sched.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1; gstep += 1
            if gstep % log_every == 0:
                print(f"  ep{ep} step{gstep}/{steps} "
                      + " ".join(f"{k}={v/nb:.3f}" for k, v in agg.items())
                      + f"  {time.time()-t0:.0f}s", flush=True)
        va = evaluate(model, vl, device, collect_loc=False)
        met = summarise(va["prob"], va["y"], va["vacuity"], k=n_classes)
        best_acer_seen = min(best_acer_seen, met["ACER"])
        # "auc_acer": the original rule (binary detection quality).
        # "acc_f1": maximises 8-way accuracy/F1 with no ACER guard -- this is
        #   exactly what let the hierarchical ablation's ACER regress.
        # "acc_f1_acer_guard": only epochs within 0.5pp of the best ACER seen
        #   so far are even eligible; among those, pick the best ACC+macroF1.
        #   Makes "don't trade ACER for accuracy" a hard constraint on
        #   selection, not a post-hoc hope.
        if select_by == "acc_f1_acer_guard":
            qualifies = met["ACER"] <= best_acer_seen + 0.5
            score = (met["ACC"] + met["macroF1"]) if qualifies else -1e9
        elif select_by == "acc_f1":
            score = met["ACC"] + met["macroF1"]
        else:
            score = met["AUC"] - met["ACER"]
        hist.append({"epoch": ep, "train": {k: v / max(nb, 1) for k, v in agg.items()},
                     "val": {k: v for k, v in met.items() if k != "per_type_apcer"}})
        print(f"[ep {ep}] val AUC={met['AUC']:.2f} ACER={met['ACER']:.2f} "
              f"ACC={met['ACC']:.2f} ECE={met['ECE']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log_path:
            with open(log_path, "w") as fh:
                json.dump(hist, fh, indent=1)
        if checkpoint_path:
            tmp = checkpoint_path + ".tmp"
            torch.save({
                "epoch": ep, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(), "best": best, "best_state": best_state,
                "hist": hist, "gstep": gstep,
                "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
            }, tmp)
            os.replace(tmp, checkpoint_path)   # atomic: a kill mid-write can't corrupt the prior checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist
