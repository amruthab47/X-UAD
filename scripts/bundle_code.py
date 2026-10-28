"""Flatten the whole codebase into paste-able text bundles.

Produces one complete bundle plus four themed parts, so the source can be handed
to a coding assistant that takes text rather than a repository. Every file is
delimited by an unambiguous marker line and no markdown fences are used, so a
docstring containing backticks cannot corrupt the structure.

    python scripts/bundle_code.py --out bundles

``unpack.py`` (written alongside the bundles) rebuilds the directory tree from
any of them:  ``python unpack.py X-UAD-codebase.txt ./xuad-repo``
"""
from __future__ import annotations

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGIN = "===== BEGIN FILE: {path} ====="
END = "===== END FILE: {path} ====="

PARTS = {
    "part1-model": [
        "xuad/models/aqa.py",
        "xuad/models/stem.py",
        "xuad/models/heads.py",
        "xuad/models/xuad.py",
        "xuad/models/baselines.py",
        "xuad/losses.py",
    ],
    "part2-data": [
        "xuad/data/faces.py",
        "xuad/data/attacks.py",
        "xuad/data/dataset.py",
        "xuad/data/build_benchmark.py",
        "xuad/data/adversarial.py",
        "xuad/data/real.py",
    ],
    "part3-engine-xai": [
        "xuad/engine/train.py",
        "xuad/engine/metrics.py",
        "xuad/xai/evidence.py",
        "xuad/xai/reasoning.py",
    ],
    "part4-scripts-tests": [
        "scripts/run_experiment.py",
        "scripts/prepare_real.py",
        "scripts/run_real.sh",
        "scripts/analysis.py",
        "scripts/openset.py",
        "scripts/make_figures.py",
        "scripts/make_tables.py",
        "scripts/make_qualitative.py",
        "scripts/make_arch_figure.py",
        "scripts/demo.py",
        "tests/make_mock_corpora.py",
        "tests/test_real_adapters.py",
        "configs/xuad_base.yaml",
        "configs/real_datasets.yaml",
        "requirements.txt",
        "README.md",
    ],
}
PACKAGE_INITS = ["xuad/__init__.py", "xuad/data/__init__.py", "xuad/models/__init__.py",
                 "xuad/engine/__init__.py", "xuad/xai/__init__.py", "xuad/utils/__init__.py"]

HEADER = """X-UAD - Explainable Hybrid Vision Transformer with Adaptive Quantum Attention
and Evidence-Grounded Reasoning for Unified Face Attack Detection

{what}

HOW TO USE THIS FILE
  Each source file is delimited by:
      ===== BEGIN FILE: <relative/path> =====
      <contents>
      ===== END FILE: <relative/path> =====
  Recreate the tree with the companion unpack.py:
      python unpack.py <this-file> ./xuad
  or paste the whole thing into a coding assistant and ask it to write out the
  files at the given paths.

  Python 3.10+.  Install:  pip install -r requirements.txt
  Entry points:
      python -m xuad.data.build_benchmark --lfw <faces> --out data   # simulated benchmark
      python scripts/prepare_real.py --root <corpus> --corpus auto --dry-run
      bash   scripts/run_real.sh <corpus-root> <adapter> <epochs> <tag>
      python scripts/run_experiment.py --data data --model xuad --protocol P1

FILES IN THIS BUNDLE ({n})
{listing}
"""

UNPACK = '''"""Rebuild the X-UAD source tree from a code bundle.

    python unpack.py X-UAD-codebase.txt ./xuad
"""
import os, sys

BEGIN, END = "===== BEGIN FILE: ", "===== END FILE: "


def main(bundle, dest):
    path, buf, n = None, [], 0
    for line in open(bundle, encoding="utf-8"):
        s = line.rstrip("\\n")
        if s.startswith(BEGIN) and s.endswith("====="):
            path = s[len(BEGIN):].rsplit(" =====", 1)[0].strip()
            buf = []
        elif s.startswith(END) and path:
            full = os.path.join(dest, path)
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write("\\n".join(buf) + "\\n")
            if full.endswith(".sh"):
                os.chmod(full, 0o755)
            print("wrote", path)
            path, n = None, n + 1
        elif path is not None:
            buf.append(s)
    print(f"\\n{n} files written to {dest}")
    if n == 0:
        print("nothing found - is this a bundle file?")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "xuad_repo")
'''


def emit(paths, out_path, what):
    chunks, listing, kept = [], [], []
    for rel in paths:
        full = os.path.join(ROOT, rel)
        if not os.path.exists(full):
            continue
        body = open(full, encoding="utf-8").read().rstrip("\n")
        chunks.append(BEGIN.format(path=rel) + "\n" + body + "\n" + END.format(path=rel))
        listing.append(f"  {rel}  ({len(body.splitlines())} lines)")
        kept.append(rel)
    text = HEADER.format(what=what, n=len(kept), listing="\n".join(listing)) \
        + "\n\n" + "\n\n".join(chunks) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return len(kept), len(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "bundles"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    all_paths = PACKAGE_INITS + [p for group in PARTS.values() for p in group]
    n, size = emit(all_paths, os.path.join(a.out, "X-UAD-codebase.txt"),
                   "COMPLETE codebase: model, data, engine, explainability, "
                   "reasoning, scripts, tests and configs.")
    print(f"X-UAD-codebase.txt        {n:3d} files  {size/1024:7.1f} KB")

    titles = {
        "part1-model": "PART 1 of 4 - the model: Adaptive Quantum Attention, the "
                       "central-difference and spectral stem, the decoupled evidential "
                       "head, the localisation decoder, the risk calibrator, the "
                       "baselines, and all training objectives.",
        "part2-data": "PART 2 of 4 - data: face indexing and identity-disjoint "
                      "splitting, the eight attack synthesisers with pixel-level "
                      "evidence masks, the torch dataset and protocols, the benchmark "
                      "builder, the transfer-adversarial class, and the adapters for "
                      "nine public capture-based corpora.",
        "part3-engine-xai": "PART 3 of 4 - training engine, ISO/IEC 30107-3 metrics, "
                            "and the explainability stack: quantum evidence basis "
                            "naming, attention rollout, deletion/insertion "
                            "faithfulness, the evidence record and the grounded "
                            "reasoning module.",
        "part4-scripts-tests": "PART 4 of 4 - runnable entry points, the real-dataset "
                               "driver, analyses, figure and LaTeX-table generation, "
                               "the mock-corpus generator, the adapter regression "
                               "test, and configs.",
    }
    for name, paths in PARTS.items():
        extra = PACKAGE_INITS if name == "part1-model" else []
        n, size = emit(extra + paths, os.path.join(a.out, f"X-UAD-{name}.txt"),
                       titles[name])
        print(f"X-UAD-{name}.txt".ljust(26) + f"{n:3d} files  {size/1024:7.1f} KB")

    with open(os.path.join(a.out, "unpack.py"), "w", encoding="utf-8") as fh:
        fh.write(UNPACK)
    print("unpack.py")


if __name__ == "__main__":
    main()
