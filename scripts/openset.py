"""Post-hoc open-set metrics for the leave-one-attack-out runs.

Two questions are conflated in the literature and we keep them apart:

  * ``unseen_detect_rate`` -- at the operating threshold, is the held-out family
    rejected as an attack at all?  This is the operational number.
  * ``novelty AUROC``      -- does the model's uncertainty channel rank the
    held-out family *above the attack families it was trained on*?  This is the
    "do I know that this is a kind of attack I have not seen" question, and it is
    the one an analyst uses to decide whether a new attack vector has appeared.

We also report the (weaker, and often misleading) unseen-vs-bonafide AUROC that a
naive reading of open-set recognition would use: a well-generalising detector is
*supposed* to be confident that an unseen attack is an attack, so a low value
there is not by itself a failure.
"""
from __future__ import annotations

import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES
from xuad.engine.metrics import roc_auc

RES = os.path.join(ROOT, "results")


def main():
    for f in sorted(glob.glob(os.path.join(RES, "*_P2.*.json"))):
        if "hist" in f:
            continue
        tag = os.path.basename(f)[:-5]
        npz = os.path.join(RES, f"{tag}_pred.npz")
        if not os.path.exists(npz):
            continue
        d = json.load(open(f))
        c = ATTACK_NAMES.index(d["protocol"].split(".", 1)[1])
        z = np.load(npz)
        v, y, p = z["vacuity"], z["y"], z["prob"]
        seen_atk = (y != 0) & (y != c)
        m1 = (y == c) | seen_atk
        d["novelty_AUROC"] = roc_auc(v[m1], (y[m1] == c).astype(int)) * 100
        m2 = (y == c) | (y == 0)
        d["unseen_vs_bonafide_AUROC"] = roc_auc(v[m2], (y[m2] == c).astype(int)) * 100
        # the same two scores computed on 1 - max posterior, so that the softmax
        # baselines are scored with the best signal they actually possess
        msp = 1.0 - p.max(1)
        d["novelty_AUROC_msp"] = roc_auc(msp[m1], (y[m1] == c).astype(int)) * 100
        d["unseen_mean_vacuity"] = float(v[y == c].mean())
        d["seen_attack_mean_vacuity"] = float(v[seen_atk].mean())
        d["bonafide_mean_vacuity"] = float(v[y == 0].mean())
        json.dump(d, open(f, "w"), indent=1)
        print(f"{tag:26s} novelty={d['novelty_AUROC']:6.2f}  "
              f"vs-bonafide={d['unseen_vs_bonafide_AUROC']:6.2f}  "
              f"detect={d.get('unseen_detect_rate', float('nan')):6.2f}  "
              f"u(unseen)={d['unseen_mean_vacuity']:.3f} "
              f"u(seen)={d['seen_attack_mean_vacuity']:.3f} "
              f"u(bona)={d['bonafide_mean_vacuity']:.3f}")


if __name__ == "__main__":
    main()
