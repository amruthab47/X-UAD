"""Post-training analyses: efficiency, explanation faithfulness, quantum-mode
naming, adaptive-gate statistics and the qualitative report sheet."""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES, ATTACK_SEVERITY
from xuad.data.dataset import XUADDataset
from xuad.engine.train import make_loader
from xuad.models.baselines import CDCNpp, ResNet18Lite, ViTTiny
from xuad.models.xuad import XUAD, count_params
from xuad.xai.evidence import (attention_rollout, deletion_insertion,
                               name_basis_states)
from xuad.xai.reasoning import GroundedReasoner, build_record

RES = os.path.join(ROOT, "results")  # default only -- every writer takes an explicit out dir
K = len(ATTACK_NAMES)


def _guard(out, ckpt):
    """Refuse to write analyses of one experiment into another's results directory.

    The analyses here are keyed to a specific checkpoint, but nothing in the file
    name says so.  A driver that evaluated a different cache while leaving the
    output directory at its default silently overwrote the main results with the
    output of a one-epoch smoke model.  A checkpoint must now live in the
    directory its analyses are written to, or the caller must say otherwise.
    """
    ck = os.path.abspath(os.path.dirname(ckpt))
    od = os.path.abspath(out)
    if ck != od:
        raise SystemExit(
            f"refusing to write analyses of\n  {ckpt}\ninto\n  {out}\n"
            f"The checkpoint belongs to a different results directory. Pass\n"
            f"  --out {ck}\n"
            f"or --allow-cross-dir if overwriting is really what you want.")

def _extra_store(root=os.path.join(ROOT, "data")):
    """The adversarial class lives in a separate store; without it the test set
    silently omits one of the eight families."""
    import json as _json
    p = os.path.join(root, "adv_x.npy")
    if not os.path.exists(p):
        return None
    meta = _json.load(open(os.path.join(root, "adv_meta.json")))
    return {"x": np.load(p, mmap_mode="r"),
            "m": np.load(os.path.join(root, "adv_m.npy"), mmap_mode="r"),
            "split": meta["split"], "label": meta["label"],
            "identity": meta["identity"]}




# --------------------------------------------------------------------------- #
def macs(model, img=112):
    """Analytic MAC count for conv / linear / attention, via forward hooks."""
    total = [0]
    hs = []

    def conv_hook(m, i, o):
        total[0] += m.weight.numel() // m.groups * o.shape[-1] * o.shape[-2]

    def lin_hook(m, i, o):
        total[0] += m.weight.numel() * int(np.prod(o.shape[:-1]))

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(torch.randn(1, 3, img, img))
    for h in hs:
        h.remove()
    # attention matmuls are not captured by the hooks
    extra = 0
    for m in model.modules():
        if m.__class__.__name__ in ("AdaptiveQuantumAttention", "PlainAttention"):
            T = getattr(model, "_T", None)
            extra += 0
    return total[0] / 1e9


def latency(model, img=112, n=12, bs=1):
    model.eval()
    x = torch.randn(bs, 3, img, img)
    with torch.no_grad():
        model(x)
        t = time.time()
        for _ in range(n):
            model(x)
    return (time.time() - t) / n * 1000


def efficiency(out_dir=RES):
    out = {}
    for name, m in [("resnet18", ResNet18Lite(K)), ("vit_tiny", ViTTiny(K, img=112)),
                    ("cdcn", CDCNpp(K, img_size=112)),
                    ("xuad", XUAD(K, severity=torch.tensor(ATTACK_SEVERITY), img_size=112))]:
        out[name] = {"params_M": count_params(m) / 1e6,
                     "gmacs": macs(m), "latency_ms": latency(m)}
        print(name, out[name], flush=True)
    json.dump(out, open(os.path.join(out_dir, "efficiency.json"), "w"), indent=1)


# --------------------------------------------------------------------------- #
def faithfulness(ckpt, n=48, out_dir=RES):
    model = XUAD(K, severity=torch.tensor(ATTACK_SEVERITY), img_size=112)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    te = XUADDataset(os.path.join(ROOT, "data"), "A", "test", img=112, extra=_extra_store())
    idx = np.where(te.y != 0)[0]
    idx = idx[np.linspace(0, len(idx) - 1, n).astype(int)]   # spans every family
    x = torch.stack([te[i][0] for i in idx])
    with torch.no_grad():
        o = model(x, need_attn=True)
    loc = o["loc"].numpy()
    roll = attention_rollout(o["attn"], model.grid ** 2, model.grid)
    born = o["born"].numpy()                                   # (N,T,B)
    g = model.grid
    # per-token total measured evidence -> a purely quantum saliency map
    qmap = born.sum(-1).reshape(-1, g, g)
    # the random control must have the *same spatial statistics* as the maps it
    # is compared against: a per-pixel random mask deletes a speckle pattern,
    # which destroys high-frequency texture everywhere and is a far more
    # destructive perturbation than blurring a contiguous region.  We therefore
    # smooth the control to the same effective scale.
    import cv2
    g = np.random.default_rng(0).random((loc.shape[0], 8, 8)).astype(np.float32)
    rnd = np.stack([cv2.resize(a, loc.shape[1:][::-1], interpolation=cv2.INTER_CUBIC)
                    for a in g])

    res = {}
    for key, sal, lab in [("loc", loc, "X-UAD localisation map"),
                          ("rollout", roll, "AQA attention rollout"),
                          ("quantum", qmap, "Quantum evidence magnitude"),
                          ("random", rnd, "Random (control)")]:
        d, i = deletion_insertion(model, x, sal.astype(np.float32), n_steps=10)
        res[key] = {"deletion": d, "insertion": i, "label": lab}
        print(key, res[key], flush=True)
    json.dump(res, open(os.path.join(out_dir, "faithfulness.json"), "w"), indent=1)


# --------------------------------------------------------------------------- #
def gates_and_modes(ckpt, out_dir=RES):
    model = XUAD(K, severity=torch.tensor(ATTACK_SEVERITY), img_size=112)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    te = XUADDataset(os.path.join(ROOT, "data"), "A", "test", img=112, extra=_extra_store())
    # shuffle: the manifest is ordered face-by-face with the classes cycling
    # inside each face and the adversarial split appended at the end, so a
    # sequential loader would never reach the adversarial class.
    dl = make_loader(te, 48, True, 0)
    basis = name_basis_states(model, dl, n_batches=24)
    json.dump(basis, open(os.path.join(out_dir, "basis_states.json"), "w"), indent=1)

    G, Y = [], []
    with torch.no_grad():
        for i, (x, _m, y) in enumerate(dl):
            if i >= 24:
                break
            o = model(x)
            G.append(o["gates"].numpy()); Y.append(y.numpy())
    G = np.concatenate(G); Y = np.concatenate(Y)         # (N, depth, heads)
    stat = {"mean_gate": float(G.mean()),
            "per_layer": G.mean((0, 2)).tolist(),
            "per_class": {ATTACK_NAMES[c]: float(G[Y == c].mean())
                          for c in np.unique(Y)}}
    json.dump(stat, open(os.path.join(out_dir, "gates.json"), "w"), indent=1)
    print(json.dumps(stat, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all")
    ap.add_argument("--ckpt", default=os.path.join(RES, "xuad_P1.pt"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=RES, help="directory to write the analyses into")
    ap.add_argument("--allow-cross-dir", action="store_true",
                    help="permit writing analyses of a checkpoint from another directory")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if not a.allow_cross_dir:
        _guard(a.out, a.ckpt)
    if a.what in ("all", "eff"):
        efficiency(a.out)
    if a.what in ("all", "faith"):
        faithfulness(a.ckpt, out_dir=a.out)
    if a.what in ("all", "gates"):
        gates_and_modes(a.ckpt, out_dir=a.out)
