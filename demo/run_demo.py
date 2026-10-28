"""X-UAD single-image inference demo.

Reuses the project's existing model class (``xuad.models.xuad.XUAD``), face
cropper (``xuad.data.faces.crop_face``), and explanation pipeline
(``xuad.xai.reasoning``) exactly as-is -- this file adds no second
implementation of the model, the preprocessing, or the reasoning layer. It
only adds the glue to run all three on one arbitrary image file and print/
save a professor-facing report.

Usage:
    python demo/run_demo.py --image <path/to/image.jpg>

No training, no dataset modification, no checkpoint modification. Loads a
trained checkpoint read-only and performs a single forward pass with
``torch.no_grad()``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from xuad.data.attacks import ATTACK_NAMES, ATTACK_SEVERITY  # noqa: E402
from xuad.data.dataset import MEAN, STD  # noqa: E402
from xuad.data.faces import crop_face  # noqa: E402
from xuad.models.xuad import XUAD  # noqa: E402
from xuad.xai.evidence import name_basis_states  # noqa: E402
from xuad.xai.reasoning import GroundedReasoner, build_record  # noqa: E402

# The best SINGLE model produced by this project's experiment series --
# highest single-model accuracy (94.583%) AND the best localisation quality
# (loc_IoU=0.8887, loc_Dice=0.9349) of any model trained this session, with a
# full evidential-uncertainty head. Chosen over the 7-model ensemble
# (results_final/) because the ensemble has no single checkpoint and no
# native localisation output -- unsuitable for a live single-image demo.
DEFAULT_CKPT = os.path.join(ROOT, "results_region_pooled_detached",
                             "xuad_P1_regionpool_detached.pt")
IMG_SIZE = 112
BASIS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "outputs", "_basis_states_cache.json")


def load_model(ckpt: str) -> XUAD:
    """Builds the exact architecture the checkpoint was trained with
    (``region_pool=True`` -- see architecture_feasibility_investigation.md /
    xuad_P1_regionpool_detached_report.md) and loads its weights read-only."""
    m = XUAD(len(ATTACK_NAMES), severity=torch.tensor(ATTACK_SEVERITY),
             img_size=IMG_SIZE, gate_bias=-2.0, region_pool=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


STORE_SIZE = 128  # xuad/data/build_benchmark.py::IMG -- the resolution every
                   # training/test sample was originally cropped to


def preprocess(image_path: str):
    """Exactly reproduces the two-step pipeline that built every training/
    test sample, not an approximation of it:
      1. Haar-cascade face crop to 128px, 12% margin (xuad.data.faces.crop_face,
         same function and same size as xuad/data/build_benchmark.py::IMG=128).
      2. Deterministic centre-crop 128->112 (xuad/data/dataset.py's exact
         non-augment branch: offset = (128-112)//2 = 8, no re-detection).
    A single-step crop straight to 112px was tried first and found, by direct
    comparison against this exact two-step pipeline on real project samples,
    to visibly shift model confidence -- this two-step version is the one
    that matches what every accuracy number in this project was computed on.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR, matches training
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    crop128, bbox, detected = crop_face(img, size=STORE_SIZE, margin=0.12)
    if crop128 is None:
        raise RuntimeError("face crop failed (image too small or unreadable)")
    o = (STORE_SIZE - IMG_SIZE) // 2
    crop = crop128[o:o + IMG_SIZE, o:o + IMG_SIZE]
    x = crop.astype(np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
    x = (x - MEAN) / STD
    return x.unsqueeze(0), crop, detected, bbox


def get_basis_info(model: XUAD, recompute: bool = False):
    """Human-readable naming for the 8 learned quantum-evidence basis modes
    (xai.evidence.name_basis_states) -- a one-time, read-only introspection
    over a slice of the *existing* test set (never the image being demoed),
    cached to disk so a live demo run doesn't reload data/ every time. Purely
    cosmetic for the explanation text; the prediction itself does not depend
    on it. On any failure (e.g. data/ not present), degrades gracefully to
    "no name available" rather than crashing the demo."""
    if not recompute and os.path.exists(BASIS_CACHE):
        try:
            return json.load(open(BASIS_CACHE))
        except Exception:
            pass
    try:
        from xuad.data.dataset import XUADDataset
        from xuad.engine.train import make_loader
        te = XUADDataset(root=os.path.join(ROOT, "data"), domain="A",
                          split="test", img=IMG_SIZE)
        dl = make_loader(te, 48, False, workers=0)
        basis = name_basis_states(model, dl, n_batches=10)
        os.makedirs(os.path.dirname(BASIS_CACHE), exist_ok=True)
        json.dump(basis, open(BASIS_CACHE, "w"), indent=1)
        return basis
    except Exception as e:  # pragma: no cover - defensive, demo must not crash
        warnings.warn(f"quantum-mode naming skipped ({e}); explanation will "
                       f"omit the 'aligned_with' label for evidence modes.")
        return None


def render_localisation(crop_bgr: np.ndarray, loc_map: np.ndarray) -> np.ndarray:
    """Localisation map (56x56, [0,1]) upsampled and overlaid as a heatmap on
    the actual input crop -- no synthetic/placeholder image."""
    h, w = crop_bgr.shape[:2]
    heat = cv2.resize((loc_map * 255).astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop_bgr, 0.55, heat_color, 0.45, 0)
    return overlay


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="path to a single input image")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                     help="path to a trained X-UAD checkpoint (.pt)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
                     help="output directory (a per-image subfolder is created here)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-basis-naming", action="store_true",
                     help="skip the (optional, cosmetic) quantum-mode naming step")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)

    if not os.path.exists(a.ckpt):
        raise SystemExit(f"checkpoint not found: {a.ckpt}")
    if not os.path.exists(a.image):
        raise SystemExit(f"image not found: {a.image}")

    print(f"Loading X-UAD checkpoint: {a.ckpt}")
    model = load_model(a.ckpt)

    print(f"Reading and preprocessing image: {a.image}")
    x, crop_bgr, detected, bbox = preprocess(a.image)
    if not detected:
        print("  WARNING: no face detected by the Haar-cascade face finder; "
              "used a centred fallback crop instead. Results may be less reliable.")

    basis_info = None if a.no_basis_naming else get_basis_info(model)

    print("Running inference (single forward pass, no gradients, no training)...")
    with torch.no_grad():
        out = model(x, need_attn=False)

    prob = out["prob"][0].numpy()
    vacuity = float(out["vacuity"][0])
    risk = float(out["risk"][0])
    loc = out["loc"][0].numpy()
    born = out["born"][0].numpy()
    spec_prof = out["spec_prof"][0].numpy() if out["spec_prof"] is not None else None

    record = build_record(prob=prob, vacuity=vacuity, risk=risk, loc=loc,
                           born=born, basis_info=basis_info, spec_prof=spec_prof,
                           gate_mean=float(out["gates"][0].mean()) if out["gates"] is not None else None)
    reasoner = GroundedReasoner()
    report_text = reasoner(record)

    # ---------------------------------------------------------------- #
    # console output
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60)
    print("\nFull class-probability distribution:")
    for i, name in enumerate(ATTACK_NAMES):
        bar = "#" * int(round(prob[i] * 40))
        print(f"  {name:12s} {prob[i]*100:6.2f}%  {bar}")

    # ---------------------------------------------------------------- #
    # save outputs (demo/outputs/<image-stem>/) -- never outside demo/
    # ---------------------------------------------------------------- #
    stem = os.path.splitext(os.path.basename(a.image))[0]
    out_dir = os.path.join(a.out, stem)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "input_crop.png"), crop_bgr)
    overlay = render_localisation(crop_bgr, loc)
    cv2.imwrite(os.path.join(out_dir, "localization_overlay.png"), overlay)
    heat_only = cv2.applyColorMap((loc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(out_dir, "localization_heatmap.png"),
                cv2.resize(heat_only, (crop_bgr.shape[1], crop_bgr.shape[0])))

    with open(os.path.join(out_dir, "evidence_record.json"), "w") as fh:
        fh.write(record.to_json(indent=2))

    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(report_text + "\n\nFull class-probability distribution:\n")
        for i, name in enumerate(ATTACK_NAMES):
            fh.write(f"  {name:12s} {prob[i]*100:6.2f}%\n")
        fh.write(f"\nFace detected by Haar cascade: {detected}\n")
        fh.write(f"Checkpoint used: {os.path.relpath(a.ckpt, ROOT)}\n")

    print(f"\nSaved: {out_dir}/")
    print("  input_crop.png            -- the 112x112 face crop actually fed to the model")
    print("  localization_overlay.png  -- evidence heatmap over the input")
    print("  localization_heatmap.png  -- raw heatmap")
    print("  evidence_record.json      -- full machine-readable evidence record")
    print("  report.txt                -- this console report, saved to disk")


if __name__ == "__main__":
    main()
