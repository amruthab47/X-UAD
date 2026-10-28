"""X-UAD web demo: upload a face image, get the full X-UAD analysis in a browser.

Reuses ``demo/run_demo.py`` (model loading, the two-step 128->112 face crop,
quantum-mode naming, heatmap rendering) and ``xuad.xai.reasoning`` unchanged --
this file only adds an HTTP layer around them. Standard library only, so it runs
in the project's existing ``.venv`` with nothing extra installed.

Usage:
    .venv/Scripts/python.exe webapp/server.py            # http://127.0.0.1:8000
    .venv/Scripts/python.exe webapp/server.py --port 8080 --ckpt <path.pt>
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))

import cv2  # noqa: E402

from run_demo import (DEFAULT_CKPT, get_basis_info, load_model,  # noqa: E402
                      preprocess, render_localisation)
from xuad.data.attacks import ATTACK_NAMES, ATTACK_SUPERCLASS  # noqa: E402
from xuad.xai.reasoning import (MECHANISM, PRETTY, RISK_BANDS,  # noqa: E402
                                GroundedReasoner, build_record)

STATIC = os.path.join(HERE, "static")
SAMPLES = os.path.join(ROOT, "demo", "samples")
MAX_UPLOAD = 15 * 1024 * 1024
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Classes whose pixels were produced or altered by a generative / digital
# process in the benchmark's own synthesisers (xuad/data/attacks.py).
AI_CLASSES = ("faceswap", "reenact", "synth")


class Detector:
    """Holds the loaded model; one forward pass at a time (CPU, shared with training)."""

    def __init__(self, ckpt: str, basis_naming: bool):
        self.ckpt = ckpt
        self.model = load_model(ckpt)
        self.basis = get_basis_info(self.model) if basis_naming else None
        self.reasoner = GroundedReasoner()
        self.lock = threading.Lock()

    def analyse(self, image_path: str) -> dict:
        t0 = time.perf_counter()
        x, crop_bgr, detected, bbox = preprocess(image_path)
        with self.lock, torch.no_grad():
            out = self.model(x, need_attn=False)
        ms = (time.perf_counter() - t0) * 1000

        prob = out["prob"][0].numpy()
        loc = out["loc"][0].numpy()
        record = build_record(
            prob=prob, vacuity=float(out["vacuity"][0]), risk=float(out["risk"][0]),
            loc=loc, born=out["born"][0].numpy(), basis_info=self.basis,
            spec_prof=out["spec_prof"][0].numpy() if out["spec_prof"] is not None else None,
            gate_mean=float(out["gates"][0].mean()) if out["gates"] is not None else None)

        rec = json.loads(record.to_json())
        report = self.reasoner(record)
        headline, _, explanation = report.partition("\n\n")
        heat = cv2.applyColorMap((loc * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (crop_bgr.shape[1], crop_bgr.shape[0]))

        # Group sums of the model's own posterior (no new quantity is estimated).
        groups = {
            "genuine": float(prob[0]),
            "physical": float(sum(prob[i] for i, s in enumerate(ATTACK_SUPERCLASS) if s == 1)),
            "ai_generated": float(sum(prob[ATTACK_NAMES.index(c)] for c in AI_CLASSES)),
            "adversarial": float(prob[ATTACK_NAMES.index("adversarial")]),
        }
        return {
            "record": rec,
            "pretty_class": PRETTY[rec["predicted_class"]],
            "mechanism": MECHANISM.get(rec["predicted_class"]),
            "headline": headline,
            "explanation": explanation,
            "groups": groups,
            "is_ai_generated": rec["predicted_class"] in AI_CLASSES,
            "face_detected": bool(detected),
            "bbox": [int(v) for v in bbox] if bbox is not None else None,
            "images": {"crop": _png(crop_bgr), "overlay": _png(render_localisation(crop_bgr, loc)),
                       "heatmap": _png(heat)},
            "inference_ms": round(ms, 1),
        }

    def info(self) -> dict:
        return {
            "checkpoint": os.path.relpath(self.ckpt, ROOT).replace("\\", "/"),
            "classes": [{"id": i, "name": n, "pretty": PRETTY[n],
                         "superclass": ["none", "physical", "digital"][ATTACK_SUPERCLASS[i]],
                         "mechanism": MECHANISM.get(n)} for i, n in enumerate(ATTACK_NAMES)],
            "risk_bands": [{"below": t, "label": l} for t, l in RISK_BANDS],
            "basis_naming": self.basis is not None,
            "samples": sorted(f for f in os.listdir(SAMPLES) if f.endswith(".png"))
                       if os.path.isdir(SAMPLES) else [],
        }


def _png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class Handler(BaseHTTPRequestHandler):
    detector: Detector = None  # set in main()

    def log_message(self, fmt, *args):
        sys.stderr.write("[web] " + (fmt % args) + "\n")

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/info":
            return self._json(200, self.detector.info())
        if path.startswith("/samples/"):
            name = os.path.basename(path)
            fp = os.path.join(SAMPLES, name)
            if name.endswith(".png") and os.path.isfile(fp):
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), "image/png")
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/analyze":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self._json(400, {"error": "empty upload"})
        if n > MAX_UPLOAD:
            return self._json(413, {"error": "image larger than 15 MB"})
        data = self.rfile.read(n)
        ext = os.path.splitext(self.headers.get("X-Filename", ""))[1].lower()
        ext = ext if ext in ALLOWED_EXT else ".png"
        # preprocess() takes a path, so hand it a temp file rather than re-implement it.
        fd, tmp = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            if cv2.imread(tmp, cv2.IMREAD_COLOR) is None:
                return self._json(400, {"error": "could not decode that file as an image"})
            return self._json(200, self.detector.analyse(tmp))
        except Exception as e:  # report, don't kill the server
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-basis-naming", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    if not os.path.exists(a.ckpt):
        raise SystemExit(f"checkpoint not found: {a.ckpt}")
    print(f"Loading X-UAD checkpoint: {a.ckpt}")
    Handler.detector = Detector(a.ckpt, basis_naming=not a.no_basis_naming)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"X-UAD web demo running at http://{a.host}:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
