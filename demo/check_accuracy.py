"""Recompute accuracy from the saved test-set predictions (no model, no training).

Usage:
    python demo/check_accuracy.py
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["bonafide", "print", "replay", "mask3d",
           "faceswap", "reenact", "synth", "adversarial"]
RUNS = [
    ("Final 7-model ensemble", "results_final/xuad_final_ensemble_pred.npz"),
    ("Best single model", "results_region_pooled_detached/xuad_P1_regionpool_detached_pred.npz"),
    ("Primary reference", "results_binary_preserving_refinement/xuad_P1_refined_pred.npz"),
    ("Original baseline", "results_full_retry/xuad_P1_full_retry_pred.npz"),
]

for label, rel in RUNS:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f"{label}: {rel} not found, skipped\n")
        continue
    d = np.load(path)
    y, pred = d["y"], d["prob"].argmax(1)
    print(f"{label}  ({rel})")
    print(f"  8-class accuracy: {100 * (pred == y).mean():.3f}%  "
          f"({(pred == y).sum()} / {len(y)} correct, {(pred != y).sum()} errors)")
    for c, name in enumerate(CLASSES):
        m = y == c
        if m.any():
            print(f"    {name:<12} {100 * (pred[m] == c).mean():6.2f}%")
    print()
