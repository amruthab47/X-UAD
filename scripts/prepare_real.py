"""Turn a public capture-based corpus into an X-UAD cache.

    # 1. look before you leap -- costs a second, prints the full inventory
    python scripts/prepare_real.py --root /data/CASIA-FASD --corpus auto --dry-run

    # 2. extract
    python scripts/prepare_real.py --root /data/CASIA-FASD --corpus casia-fasd \
           --out data_casia --frames-per-video 4 --workers 8

The output directory is byte-compatible with what ``xuad.data.build_benchmark``
produces, so every training, evaluation, explainability and reasoning script in
this repository runs against it unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES
from xuad.data.real import (REGISTRY, assign_splits, build_cache, discover,
                            inventory)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root directory")
    ap.add_argument("--corpus", default="auto",
                    choices=["auto"] + sorted(REGISTRY), help="adapter to use")
    ap.add_argument("--out", default="", help="cache directory to write")
    ap.add_argument("--name", default="A", help="cache tensor prefix (A = train domain)")
    ap.add_argument("--frames-per-video", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-per-class-split", type=int, default=0,
                    help="cap crops per (split, class); 0 = no cap")
    ap.add_argument("--compression", default="c23",
                    help="FaceForensics++ compression level (raw / c23 / c40)")
    ap.add_argument("--label-json", default="",
                    help="CelebA-Spoof label json, for exact spoof-type mapping")
    ap.add_argument("--ratios", default="0.6,0.15,0.25",
                    help="train,val,test ratios for corpora without a protocol")
    ap.add_argument("--dry-run", action="store_true",
                    help="discover and print the inventory, extract nothing")
    a = ap.parse_args()

    kw = {}
    if a.corpus == "faceforensics":
        kw["compression"] = a.compression
    if a.corpus == "celeba-spoof" and a.label_json:
        kw["label_json"] = a.label_json

    samples, corpus = discover(a.root, a.corpus, **kw)
    if not samples:
        print(f"No recognised files under {a.root}.\n"
              f"Adapters tried: {sorted(REGISTRY)}.\n"
              f"Check that --root points at the folder that *contains* the corpus's "
              f"own top-level directories, and re-run with --dry-run.")
        sys.exit(2)

    ratios = tuple(float(x) for x in a.ratios.split(","))
    assign_splits(samples, ratios)

    print(f"corpus           : {corpus}")
    print(f"root             : {a.root}")
    print(inventory(samples))
    present = sorted({s.label for s in samples})
    print(f"  taxonomy classes : {[ATTACK_NAMES[c] for c in present]}")
    missing = [ATTACK_NAMES[c] for c in range(len(ATTACK_NAMES)) if c not in present]
    if missing:
        print(f"  absent from this corpus : {missing}"
              f"  (train with --classes to restrict the protocol grid)")
    n_masks = sum(1 for s in samples if s.mask_path)
    if n_masks == 0:
        print("  NOTE: this corpus ships no forgery masks. Localisation targets are")
        print("        defined only where they are meaningful (whole-frame for capture")
        print("        media, empty for bonafide); pass --no-loc when training so the")
        print("        localisation head is not fitted to a proxy.")

    if a.dry_run:
        est = sum(a.frames_per_video if s.is_video else 1 for s in samples)
        print(f"\ndry run: would extract about {est} face crops "
              f"({est * 128 * 128 * 3 / 1e9:.2f} GB uint8)")
        return

    out = a.out or f"data_{corpus.replace('-', '_')}"
    stats = build_cache(samples, out, a.name, a.frames_per_video, a.workers,
                        a.max_per_class_split)
    meta = {"corpus": corpus, "root": os.path.abspath(a.root), "name": a.name,
            "frames_per_video": a.frames_per_video,
            "classes_present": [ATTACK_NAMES[c] for c in present],
            "has_masks": n_masks > 0, **stats}
    json.dump(meta, open(os.path.join(out, f"{a.name}_source.json"), "w"), indent=1)
    print(f"\ncache written to {out}\n"
          f"next:\n"
          f"  python scripts/run_experiment.py --data {out} --model xuad "
          f"--protocol P1 --epochs 30" + ("" if n_masks else " --no-loc"))


if __name__ == "__main__":
    main()
