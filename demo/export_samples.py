"""Export test-split face images to PNG files so run_demo.py can be shown on them.

Uses the same selection rule as scripts/demo.py: for each class, the test
sample at position len(class)//3 (ordered by dataset index), chosen by
position only -- not by how well the model does on it.

Usage:
    python demo/export_samples.py            # one image per class
    python demo/export_samples.py --n 3      # three images per class
"""
import argparse
import csv
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="images per class")
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "samples"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    x = np.load(os.path.join(ROOT, "data", "A_x.npy"), mmap_mode="r")
    by_class = {}
    with open(os.path.join(ROOT, "data", "A_manifest.csv")) as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                by_class.setdefault(r["label_name"], []).append(int(r["idx"]))

    sources = {name: (x, idx) for name, idx in by_class.items()}

    # the adversarial class is stored separately (data/adv_x.npy + adv_meta.json)
    adv_meta = os.path.join(ROOT, "data", "adv_meta.json")
    if os.path.exists(adv_meta):
        import json
        meta = json.load(open(adv_meta))
        adv_idx = [i for i, sp in enumerate(meta["split"]) if sp == "test"]
        if adv_idx:
            sources["adversarial"] = (
                np.load(os.path.join(ROOT, "data", "adv_x.npy"), mmap_mode="r"), adv_idx)

    for name, (arr, idx) in sources.items():
        idx.sort()
        for k in range(a.n):
            i = idx[(len(idx) // 3 + k) % len(idx)]
            path = os.path.join(a.out, f"{name}_{k + 1}.png")
            cv2.imwrite(path, np.ascontiguousarray(arr[i]))  # stored as BGR
            print("wrote", path)


if __name__ == "__main__":
    main()
