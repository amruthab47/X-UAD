"""Qualitative sheet: input / predicted evidence / ground truth / attention rollout,
plus the verbatim operator report used as a figure in the paper."""
from __future__ import annotations

import json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES, ATTACK_SEVERITY
from xuad.data.dataset import MEAN, STD, XUADDataset
from xuad.engine.train import make_loader
from xuad.models.xuad import XUAD
from xuad.xai.evidence import attention_rollout, name_basis_states
from xuad.xai.reasoning import GroundedReasoner, build_record

RES, FIG, TAB = (os.path.join(ROOT, "results"), os.path.join(ROOT, "paper", "figures"),
                 os.path.join(ROOT, "paper", "tables"))
K = len(ATTACK_NAMES)
PRETTY = {"bonafide": "bonafide", "print": "print", "replay": "replay",
          "mask3d": "3D mask", "faceswap": "face-swap", "reenact": "reenact",
          "synth": "synthetic", "adversarial": "adversarial"}

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




def latex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("%", r"\%"), ("_", r"\_"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def main(ckpt=os.path.join(RES, "xuad_P1.pt"), out_dir=None, tab_dir=None,
         allow_cross_dir=False):
    """``out_dir`` defaults to the checkpoint's own directory, so a run against a
    different cache cannot overwrite the main results (see scripts/analysis.py)."""
    out_dir = out_dir or os.path.dirname(os.path.abspath(ckpt))
    tab_dir = tab_dir or TAB
    if not allow_cross_dir and os.path.abspath(out_dir) != os.path.abspath(RES) \
            and os.path.abspath(os.path.dirname(ckpt)) != os.path.abspath(out_dir):
        raise SystemExit(f"checkpoint {ckpt} does not belong to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(2)
    model = XUAD(K, severity=torch.tensor(ATTACK_SEVERITY), img_size=112)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    te = XUADDataset(os.path.join(ROOT, "data"), "A", "test", img=112, extra=_extra_store())
    basis = (json.load(open(os.path.join(out_dir, "basis_states.json")))
             if os.path.exists(os.path.join(out_dir, "basis_states.json"))
             else name_basis_states(model, make_loader(te, 48, False, 0), 10))

    idx, ys = [], []
    for c in range(K):
        w = np.where(te.y == c)[0]
        if len(w):
            idx.append(int(w[len(w) // 3])); ys.append(c)
    xs = torch.stack([te[i][0] for i in idx])
    gts = np.stack([te[i][1].numpy() for i in idx])
    with torch.no_grad():
        o = model(xs, need_attn=True)
    roll = attention_rollout(o["attn"], model.grid ** 2, model.grid)

    n = len(idx)
    fig, ax = plt.subplots(4, n, figsize=(1.15 * n, 4.9))
    rows = ["input", "predicted\nevidence", "ground\ntruth", "AQA attention\nrollout"]
    for j in range(n):
        img = (xs[j].numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1)
        ax[0, j].imshow(img)
        ax[0, j].set_title(PRETTY[ATTACK_NAMES[ys[j]]], fontsize=6.5, pad=2)
        ax[1, j].imshow(o["loc"][j].numpy(), cmap="inferno", vmin=0, vmax=1)
        ax[2, j].imshow(gts[j], cmap="inferno", vmin=0, vmax=1)
        ax[3, j].imshow(roll[j], cmap="viridis")
        for i in range(4):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for s in ax[i, j].spines.values():
                s.set_linewidth(0.3)
    for i, r in enumerate(rows):
        ax[i, 0].set_ylabel(r, fontsize=6.0, rotation=0, ha="right", va="center", labelpad=16)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(f"{FIG}/qualitative.png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{FIG}/qualitative.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- the operator report, for the figure and for the demo listing --------- #
    reasoner = GroundedReasoner()
    reports = []
    for j in range(n):
        rec = build_record(o["prob"][j].numpy(), float(o["vacuity"][j]),
                           float(o["risk"][j]), o["loc"][j].numpy(),
                           o["born"][j].numpy(), basis,
                           o["spec_prof"][j].numpy() if o["spec_prof"] is not None else None,
                           float(o["gates"][j].mean()) if o["gates"] is not None else None)
        reports.append((ATTACK_NAMES[ys[j]], rec, reasoner(rec)))

    pick = next((r for r in reports if r[0] == "replay"), reports[min(2, n - 1)])
    esc = latex_escape(pick[2])          # escape first, then insert line breaks
    esc = esc.replace("\n\n", "\x00").replace("\n", " \\\\\n")
    body = esc.replace("\x00", " \\\\[4pt]\n")
    open(f"{tab_dir}/sample_report.tex", "w").write("\\raggedright\n" + body + "\n")
    with open(os.path.join(out_dir, "reports.txt"), "w") as fh:
        for gt, rec, txt in reports:
            fh.write(f"===== ground truth: {gt} =====\n{txt}\n\nEVIDENCE RECORD\n"
                     f"{rec.to_json(indent=1)}\n\n")
    print("qualitative sheet + reports written")


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "xuad_P1.pt")
    main(ck, out_dir=sys.argv[2] if len(sys.argv) > 2 else None)
