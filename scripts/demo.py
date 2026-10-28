"""End-to-end inference demo: image in, decision + explanation out."""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES, ATTACK_SEVERITY
from xuad.data.dataset import XUADDataset, MEAN, STD
from xuad.engine.train import make_loader
from xuad.models.xuad import XUAD
from xuad.xai.evidence import attention_rollout, name_basis_states
from xuad.xai.reasoning import GroundedReasoner, MLLMReasoner, build_record


def load_model(ckpt, img=112):
    m = XUAD(len(ATTACK_NAMES), severity=torch.tensor(ATTACK_SEVERITY), img_size=img)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def analyse(model, x, basis_info, reasoner=None):
    """``x``: (N,3,H,W) already normalised.  Returns one report per sample."""
    out = model(x, need_attn=True)
    reasoner = reasoner or GroundedReasoner()
    reports = []
    for i in range(x.shape[0]):
        rec = build_record(
            prob=out["prob"][i].numpy(),
            vacuity=float(out["vacuity"][i]),
            risk=float(out["risk"][i]),
            loc=out["loc"][i].numpy(),
            born=out["born"][i].numpy(),
            basis_info=basis_info,
            spec_prof=out["spec_prof"][i].numpy() if out["spec_prof"] is not None else None,
            gate_mean=float(out["gates"][i].mean()) if out["gates"] is not None else None,
        )
        reports.append({"record": rec, "text": reasoner(rec),
                        "loc": out["loc"][i].numpy()})
    roll = attention_rollout(out["attn"], model.grid ** 2, model.grid)
    return reports, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "results", "xuad_P1.pt"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "demo"))
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    model = load_model(a.ckpt)
    te = XUADDataset(os.path.join(ROOT, "data"), "A", "test", img=112)
    dl = make_loader(te, 32, False, 0)
    print("naming quantum evidence basis states from the test stream ...")
    basis = name_basis_states(model, dl, n_batches=10)
    json.dump(basis, open(os.path.join(a.out, "basis_states.json"), "w"), indent=1)
    for b in basis:
        print(f"  mode {b['basis']}: aligned='{b['class']}' selectivity={b['selectivity']:.2f}")

    # one example of each class
    idx = []
    for c in range(len(ATTACK_NAMES)):
        w = np.where(te.y == c)[0]
        if len(w):
            idx.append(int(w[len(w) // 3]))
    xs = torch.stack([te[i][0] for i in idx])
    ys = [te.y[i] for i in idx]
    reports, _ = analyse(model, xs, basis)

    lines = []
    for i, r in zip(ys, reports):
        head = f"===== ground truth: {ATTACK_NAMES[int(i)]} ====="
        print(head); print(r["text"]); print()
        lines.append(head + "\n" + r["text"] + "\n\nEVIDENCE RECORD\n"
                     + r["record"].to_json(indent=1) + "\n")
    open(os.path.join(a.out, "reports.txt"), "w").write("\n".join(lines))
    print("written to", a.out)


if __name__ == "__main__":
    main()
