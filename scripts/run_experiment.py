"""Single experiment runner: build model + data for a protocol, train, evaluate."""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import ConcatDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES, ATTACK_SEVERITY, ATTACK_SUPERCLASS
from xuad.losses import PAIR_PENALTY_MAP_4WAY, PAIR_WEIGHTS_PERPAIR
from xuad.data.dataset import XUADDataset, PROTOCOLS
from xuad.engine.metrics import (localisation_by_class, localisation_metrics,
                                 roc_auc, summarise, eer)
from xuad.engine.train import evaluate, fit, make_loader
from xuad.models.baselines import CDCNpp, ResNet18Lite, ViTTiny
from xuad.models.xuad import XUAD, count_params

K = len(ATTACK_NAMES)


def build_model(name: str, img: int = 112, gate_bias: float = -2.0, hierarchical: bool = False,
                region_pool: bool = False, deep_stem: bool = False):
    sev = torch.tensor(ATTACK_SEVERITY)
    XU = lambda **kw: XUAD(K, severity=sev, img_size=img, gate_bias=gate_bias,
                          hierarchical=hierarchical, region_pool=region_pool,
                          deep_stem=deep_stem, **kw)
    if name == "xuad":
        return XU()
    if name == "xuad_no_aqa":
        return XU(use_aqa=False)
    if name == "xuad_no_evid":
        return XU(use_evidential=False)
    if name == "xuad_no_spec":
        return XU(use_spectral=False)
    if name == "xuad_no_cdc":
        return XU(theta=0.0)
    if name == "xuad_no_loc":
        return XU()
    if name == "resnet18":
        return ResNet18Lite(K)
    if name == "vit_tiny":
        return ViTTiny(K, img=img, patch=16)
    if name == "cdcn":
        return CDCNpp(K, img_size=img)
    raise ValueError(name)


def _flatten_concat(cds):
    """Give a ConcatDataset the ``y`` array the metric code expects."""
    ys = np.concatenate([d.y for d in cds.datasets]) if cds.datasets else np.array([], np.int64)
    cds.y = ys
    return cds


def load_extra(path):
    if not path or not os.path.exists(os.path.join(path, "adv_x.npy")):
        return None
    meta = json.load(open(os.path.join(path, "adv_meta.json")))
    return {"x": np.load(os.path.join(path, "adv_x.npy"), mmap_mode="r"),
            "m": np.load(os.path.join(path, "adv_m.npy"), mmap_mode="r"),
            "split": meta["split"], "label": meta["label"], "identity": meta["identity"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"),
                    help="cache directory (simulated benchmark or a real-corpus cache)")
    ap.add_argument("--test-domain", default="",
                    help="tensor prefix inside --data-test (default A)")
    ap.add_argument("--data-test", default="",
                    help="second cache to evaluate on -- this is the cross-dataset "
                         "protocol: train on --data, test unchanged on --data-test")
    ap.add_argument("--model", default="xuad")
    ap.add_argument("--protocol", default="P1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--img", type=int, default=112)
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-pues", action="store_true")
    ap.add_argument("--no-loc", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-bias", type=float, default=-2.0)
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="focal-loss modulation exponent applied to the classification "
                         "cross-entropy (0 = plain label-smoothed CE, the original "
                         "baseline behaviour; 2.0 is the standard Lin et al. default)")
    ap.add_argument("--select-by", default="auc_acer",
                    choices=["auc_acer", "acc_f1", "acc_f1_acer_guard"],
                    help="validation criterion for best-checkpoint selection: "
                         "'auc_acer' (original default, binary-detection quality), "
                         "'acc_f1' (8-way accuracy + macro-F1, no ACER guard), or "
                         "'acc_f1_acer_guard' (accuracy + macro-F1 among only the epochs "
                         "within 0.5pp of the best validation ACER seen so far)")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="write a rolling resumable checkpoint (model+optimiser+scheduler+RNG "
                         "state) to <out>/<tag>_epoch.pt after every epoch; off by default so "
                         "ordinary runs are unaffected")
    ap.add_argument("--resume-from", default="",
                    help="path to a checkpoint written by --save-every-epoch; resumes training "
                         "from the next epoch with optimiser/scheduler/RNG state restored")
    ap.add_argument("--hierarchical", action="store_true",
                    help="use HierarchicalEvidentialHead (bonafide/attack -> physical/digital/"
                         "adversarial -> specific class) instead of the flat evidential head; "
                         "off by default, no effect on any other run")
    ap.add_argument("--refine", action="store_true",
                    help="add bce (binary-reinforcement, weight 0.5) and pcp (targeted "
                         "faceswap<->reenact / mask3d<->synth confusion penalty, weight 0.15) "
                         "auxiliary losses on top of the existing flat EvidentialHead output -- "
                         "no architecture change; off by default, no effect on any other run")
    ap.add_argument("--refine4", action="store_true",
                    help="same bce/pcp mechanism and weights as --refine, but the pcp penalty "
                         "covers all 4 dominant confusion clusters (faceswap<->reenact, "
                         "mask3d<->synth, replay<->print, faceswap<->print) instead of 2; "
                         "off by default, no effect on any other run")
    ap.add_argument("--refine-weighted", action="store_true",
                    help="bce (0.5, unchanged) plus a per-pair-weighted confusion penalty "
                         "(PAIR_WEIGHTS_PERPAIR): faceswap<->reenact and mask3d<->synth at 0.15 "
                         "each (same as the successful 2-pair reference, computed independently "
                         "so they can't be diluted), replay<->print and faceswap<->print at 0.05 "
                         "each; off by default, no effect on any other run")
    ap.add_argument("--region-pool", action="store_true",
                    help="add a region-pooled summary of the existing localisation map "
                         "(pooled over the named facial regions in xai.evidence.REGIONS: "
                         "forehead, eye region, nose/midface, mouth+chin, left/right cheek, "
                         "face-boundary ring) into the classification feature -- reuses the "
                         "already-computed, already-supervised loc map, adds no new source "
                         "of evidence; off by default, no effect on any other run")
    ap.add_argument("--deep-stem", action="store_true",
                    help="X-UAD only: add a stride-1 CDC block after the 64x64 and 32x32 stem "
                         "stages (two blocks per stage there, as in the ResNet-18 baseline); off by "
                         "default, no effect on any other run")
    ap.add_argument("--hold-out-method", default="",
                    help="leave-one-manipulation-out: withhold this generator "
                         "(e.g. Deepfakes) from train and val while keeping its "
                         "taxonomy class represented by the other generators; "
                         "the test split keeps everything")
    ap.add_argument("--classes", default="",
                    help="comma-separated taxonomy classes to keep; default = all "
                         "classes actually present in the cache")
    ap.add_argument("--aux", default="on", choices=["on", "off"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--eval-checkpoint", default="",
                    help="skip training; load these weights and evaluate only")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    proto = PROTOCOLS[a.protocol]
    tag = a.tag or f"{a.model}_{a.protocol}"
    os.makedirs(a.out, exist_ok=True)
    DATA = a.data
    extra = load_extra(DATA)
    sub = a.subsample or None
    tr_ex = tuple(proto["train_exclude"]) if not extra else tuple(proto["train_exclude"])
    if extra is None:
        tr_ex = tuple(set(tr_ex) | {7})               # adversarial not built yet

    # restrict the label space to what this cache actually contains, so a corpus
    # with three families is not scored against eight
    present = set()
    _td = "A" if a.data_test else proto.get("train_domain", proto["domain"])
    with open(os.path.join(DATA, f"{_td}_manifest.csv")) as fh:
        import csv as _csv
        for r in _csv.DictReader(fh):
            present.add(int(r["label"]))
    if extra is not None:
        present |= set(extra["label"])
    if a.classes:
        want = {ATTACK_NAMES.index(c.strip()) for c in a.classes.split(",")}
        present &= want
    absent = tuple(c for c in range(K) if c not in present)
    tr_ex = tuple(set(tr_ex) | set(absent))

    hold = a.hold_out_method.strip()
    if hold:
        # A generator name that matches nothing withholds nothing: the run would
        # silently become a closed-set run wearing a LOMO label, and its "unseen"
        # numbers would be seen-attack numbers.  Refuse rather than mislead.
        avail = set()
        with open(os.path.join(DATA, f"{proto.get('train_domain', proto['domain'])}"
                               f"_manifest.csv")) as fh:
            import csv as _csv2
            for r in _csv2.DictReader(fh):
                if r.get("method"):
                    avail.add(r["method"])
        if not any(m.lower() == hold.lower() for m in avail):
            names = sorted(avail) or "(none: this cache carries no family metadata)"
            raise SystemExit(
                f"--hold-out-method '{hold}' matches no generator in {DATA}.\n"
                f"  generators present: {names}\n"
                f"  Withholding nothing would turn this into a closed-set run "
                f"reported as leave-one-manipulation-out.")
    if hold and not proto.get("by_method"):
        print(f"   note: --hold-out-method given with protocol {a.protocol}; the "
              f"generator is withheld from training on top of that protocol")
    kw = dict(root=DATA, img=a.img, extra=extra)
    train_domain = proto.get("train_domain", proto["domain"])
    if a.data_test:
        train_domain = "A"
    ex_m = (hold,) if hold else ()
    tr = XUADDataset(domain=train_domain, split="train", exclude=tr_ex, augment=True,
                     subsample=sub, exclude_method=ex_m, **kw)
    va = XUADDataset(domain=train_domain, split="val", exclude=tr_ex,
                     exclude_method=ex_m, **kw)
    te_ex = tuple(set(proto["test_exclude"]) | set(absent) - set(proto["train_exclude"]))
    if extra is None:
        te_ex = tuple(set(te_ex) | ({7} - set(proto["train_exclude"])))
    if a.data_test:
        # cross-dataset: a different corpus entirely.  Its own splits are
        # irrelevant to us, so every sample in it is evaluation data.
        te_kw = dict(root=a.data_test, img=a.img, extra=load_extra(a.data_test))
        td = a.test_domain or "A"
        te = ConcatDataset([XUADDataset(domain=td, split=sp, exclude=te_ex, **te_kw)
                            for sp in ("train", "val", "test")])
        te = _flatten_concat(te)
    else:
        # The adversarial store is a Domain-A artefact: it was crafted from
        # Domain-A images against a Domain-A surrogate.  Appending it to a test
        # set drawn from a *different* domain silently mixes in-domain samples
        # into a cross-domain evaluation -- it made 19.7% of protocol P3's test
        # set in-domain, and inflated three of four AUCs by 6-8 points.  When the
        # evaluation domain differs from the training domain the store is
        # dropped, and the adversarial class is excluded along with it.
        cross_domain = proto["domain"] != train_domain
        te_kw = dict(kw)
        if cross_domain and extra is not None:
            te_kw["extra"] = None
            te_ex = tuple(set(te_ex) | {ATTACK_NAMES.index("adversarial")})
            print(f"   cross-domain evaluation ({train_domain} -> {proto['domain']}): "
                  f"dropping the Domain-{train_domain} adversarial store from the test set",
                  flush=True)
        te = XUADDataset(domain=proto["domain"], split="test", exclude=te_ex, **te_kw)

    model = build_model(a.model, a.img, a.gate_bias, hierarchical=a.hierarchical,
                       region_pool=a.region_pool, deep_stem=a.deep_stem)
    n_par = count_params(model)
    lw = {"loc": 0.0} if (a.no_loc or a.model == "xuad_no_loc") else {}
    if a.aux == "off":
        lw.update({"basis": 0.0, "pues": 0.0, "coh": 0.0, "sup": 0.0})
    if a.hierarchical:
        # the new 3-way physical/digital/adversarial auxiliary supervision
        # (hbin/hsup) supersedes the coarser 2-way physical/digital+adversarial
        # "sup" term -- both active together would be a conflicting, redundant
        # objective on the same pooled probability.
        lw.update({"sup": 0.0, "hbin": 0.3, "hsup": 0.3})
    if a.refine or a.refine4:
        # bce (0.5) > pcp (0.15): binary reinforcement stays dominant over the
        # small, targeted pair penalty, per the "don't increase ACER" requirement.
        # Same weights for --refine4 as --refine -- broadening which pairs are
        # covered, not how strongly any one of them is penalized.
        lw.update({"bce": 0.5, "pcp": 0.15})
    if a.refine_weighted:
        # bce unchanged from the reference; the per-pair-weighted penalty is
        # threaded separately below (pair_weights), not through lw["pcp"].
        lw.update({"bce": 0.5})
    lw = lw or None
    pcp_pairs = PAIR_PENALTY_MAP_4WAY if a.refine4 else None
    pair_weights = PAIR_WEIGHTS_PERPAIR if a.refine_weighted else None
    if hold:
        print(f"   leave-one-manipulation-out: '{hold}' withheld from train/val.\n"
              f"   train generators={tr.method_counts()}  test generators={te.method_counts()}"
              if hasattr(te, "method_counts") else "", flush=True)
    print(f"== {tag}: data={DATA} model={a.model} params={n_par/1e6:.2f}M  "
          f"train={len(tr)} val={len(va)} test={len(te)}\n"
          f"   classes kept={sorted(ATTACK_NAMES[c] for c in present)}  "
          f"train-excluded={[ATTACK_NAMES[c] for c in tr_ex]}", flush=True)

    t0 = time.time()
    if a.eval_checkpoint:
        model.load_state_dict(torch.load(a.eval_checkpoint, map_location="cpu"))
        hist = []
    else:
        ckpt_path = os.path.join(a.out, f"{tag}_epoch.pt") if a.save_every_epoch else None
        model, hist = fit(model, tr, va, epochs=a.epochs, bs=a.bs, lr=a.lr,
                          n_classes=K,
                          use_evidential=("no_evid" not in a.model and "xuad" in a.model),
                          use_pues=not a.no_pues and "xuad" in a.model,
                          loss_weights=lw, workers=a.workers,
                          log_path=os.path.join(a.out, f"{tag}_hist.json"), seed=a.seed,
                          focal_gamma=a.focal_gamma, select_by=a.select_by,
                          checkpoint_path=ckpt_path, resume_from=a.resume_from or None,
                          pcp_pairs=pcp_pairs, pair_weights=pair_weights)
    train_s = time.time() - t0

    # threshold from the *validation* split, applied unchanged to the test split
    vres = evaluate(model, make_loader(va, a.bs, False, a.workers), collect_loc=False)
    _, thr = eer(1 - vres["prob"][:, 0], (vres["y"] != 0).astype(int))

    res = evaluate(model, make_loader(te, a.bs, False, a.workers))
    met = summarise(res["prob"], res["y"], res["vacuity"], thr=thr, k=K)
    if res["loc"] is not None:
        n = len(res["loc"])
        met.update({f"loc_{k}": v for k, v in
                    localisation_metrics(res["loc"], res["loc_gt"][:n]).items()})
        met["loc_per_class"] = localisation_by_class(res["loc"], res["loc_gt"][:n],
                                                     res["y"][:n], ATTACK_NAMES)

    # leave-one-manipulation-out: the held-out generator is the unseen unit, and
    # its taxonomy class is NOT unseen -- other generators of the same class were
    # trained on.  Metrics are therefore computed over samples of that generator,
    # never over the class.
    if hold and hasattr(te, "method_array"):
        marr = te.method_array()
        if len(marr) == len(res["y"]):
            sel = np.array([str(m).lower() == hold.lower() for m in marr])
            seen_atk = (res["y"] != 0) & ~sel
            sc = 1 - res["prob"][:, 0]
            met["held_out_method"] = hold
            met["n_held_out_samples"] = int(sel.sum())
            if sel.any():
                met["unseen_detect_rate"] = float((sc[sel] > thr).mean()) * 100
                met["seen_detect_rate"] = float((sc[seen_atk] > thr).mean()) * 100
                m1 = sel | seen_atk
                met["novelty_AUROC"] = roc_auc(res["vacuity"][m1], sel[m1].astype(int)) * 100
                m2 = sel | (res["y"] == 0)
                met["unseen_vs_bonafide_AUROC"] = roc_auc(res["vacuity"][m2],
                                                          sel[m2].astype(int)) * 100
                if res["loc"] is not None:
                    n = len(res["loc"])
                    ss = sel[:n]
                    if ss.any():
                        met["loc_IoU_held_out"] = localisation_metrics(
                            res["loc"][ss], res["loc_gt"][:n][ss])["IoU"]
        else:
            print("   WARNING: method array length does not match predictions; "
                  "per-generator metrics skipped", flush=True)

    # open-set: is the held-out family flagged as *unknown* by the vacuity channel?
    if "unseen" in proto:
        c = proto["unseen"]
        m = (res["y"] == c) | (res["y"] == 0)
        met["unseen_AUROC_vacuity"] = roc_auc(res["vacuity"][m], (res["y"][m] == c).astype(int)) * 100
        seen_atk = (res["y"] != 0) & (res["y"] != c)
        sc = 1 - res["prob"][:, 0]
        met["unseen_detect_rate"] = float((sc[res["y"] == c] > thr).mean()) * 100
        met["seen_detect_rate"] = float((sc[seen_atk] > thr).mean()) * 100

    met.update({"params_M": n_par / 1e6, "train_seconds": train_s, "data": DATA,
                "eval_only": bool(a.eval_checkpoint),
                "checkpoint": a.eval_checkpoint or "",
                "data_test": a.data_test or DATA,
                "classes": sorted(ATTACK_NAMES[c] for c in present),
                "model": a.model, "protocol": a.protocol, "deep_stem": a.deep_stem,
                "hold_out_method": hold,
                "epochs": 0 if a.eval_checkpoint else a.epochs,
                "n_train": len(tr), "n_test": len(te), "seed": a.seed})
    per = met.pop("per_type_apcer", {})
    met["per_type_apcer"] = {ATTACK_NAMES[int(k)]: v * 100 for k, v in per.items()}
    with open(os.path.join(a.out, f"{tag}.json"), "w") as fh:
        json.dump(met, fh, indent=1)
    empty = np.zeros(1)
    np.savez_compressed(os.path.join(a.out, f"{tag}_pred.npz"),
                        prob=res["prob"], y=res["y"], vacuity=res["vacuity"],
                        risk=res["risk"] if res["risk"] is not None else empty,
                        feature=res["feature"] if res["feature"] is not None else empty,
                        val_prob=vres["prob"], val_y=vres["y"], val_vacuity=vres["vacuity"],
                        val_feature=vres["feature"] if vres["feature"] is not None else empty,
                        threshold=np.array([thr]))
    torch.save(model.state_dict(), os.path.join(a.out, f"{tag}.pt"))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in met.items() if k != "per_type_apcer"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
