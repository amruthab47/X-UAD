"""End-to-end regression test for the real-corpus adapters.

Builds miniature corpora with the exact on-disk layout of each public dataset,
then runs discovery -> split assignment -> frame extraction -> face cropping ->
cache write -> training -> evaluation against every one of them.  It asserts the
things that actually go wrong in practice:

  * every adapter recognises its own layout and no other adapter's,
  * the taxonomy mapping is what the corpus documentation says it is,
  * splits are subject-disjoint, including when the corpus ships no validation
    set and one has to be carved out of train,
  * FaceForensics++ mask videos are read as localisation targets and are *not*
    themselves counted as samples,
  * an annotated mask stays *geometrically registered* to the face crop -- the
    defect that a mask which is merely *attached* passes but a mask that is
    resized from the whole frame fails,
  * every manipulation generator (Deepfakes / Face2Face / FaceSwap /
    NeuralTextures) survives discovery, the cache manifest and the dataset,
  * the official FaceForensics++ split files are honoured and not overwritten,
  * neither the target nor the source identity of a forgery crosses a split,
    with a negative control proving that check can actually fail,
  * leave-one-manipulation-out runs for every generator,
  * the cache is byte-compatible with the simulated benchmark, so the unchanged
    training pipeline consumes it,
  * a leave-one-attack-out protocol works on a corpus with only three classes,
  * the cross-dataset protocol trains on one cache and tests on another.

Run:
  python tests/test_real_adapters.py              # full integrity run, fits nothing
  python tests/test_real_adapters.py --quick      # discovery and geometry only
  python tests/test_real_adapters.py --with-training   # adds 1-epoch protocol smoke
                                                       # tests on the mock corpora
  python tests/test_real_adapters.py --real-root /data/FaceForensics++ [--compression=c23]
                                                  # read-only audit of a real corpus

Nothing here may be reported as a benchmark result: the pixels are synthetic.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import ATTACK_NAMES
from xuad.data.dataset import XUADDataset
from xuad.data.faces import crop_face_box, crop_like
from xuad.data.real import (FFPP_CORE_METHODS, REGISTRY, _ffpp_official_splits,
                            assign_splits, discover, discover_faceforensics,
                            inventory, sample_frames)

MOCK = "/tmp/mock"
CACHE = "/tmp/xuad_test_cache"
ROOTS = {
    "casia-fasd": f"{MOCK}/CASIA-FASD",
    "replay-attack": f"{MOCK}/Replay-Attack",
    "oulu-npu": f"{MOCK}/OULU-NPU",
    "siw-mv2": f"{MOCK}/SiW-Mv2",
    "celeba-spoof": f"{MOCK}/CelebA-Spoof",
    "faceforensics": f"{MOCK}/FaceForensics++",
    "celeb-df": f"{MOCK}/Celeb-DF-v2",
    "dfdc": f"{MOCK}/DFDC",
    "uniattackdata": f"{MOCK}/UniAttackData",
}
# what each corpus must map onto, per its own documentation
EXPECTED = {
    "casia-fasd": {"bonafide", "print", "replay"},
    "replay-attack": {"bonafide", "print", "replay"},
    "oulu-npu": {"bonafide", "print", "replay"},
    "siw-mv2": {"bonafide", "print", "replay", "mask3d"},
    "celeba-spoof": {"bonafide", "print", "replay", "mask3d"},
    "faceforensics": {"bonafide", "faceswap", "reenact"},
    "celeb-df": {"bonafide", "faceswap"},
    "dfdc": {"bonafide", "faceswap"},
    "uniattackdata": {"bonafide", "print", "replay", "faceswap"},
}

fails: list = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def run(cmd) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# FaceForensics++: geometry, generators, official splits, identity leakage
# --------------------------------------------------------------------------- #
IMG_T = 128
MASK_THR = 38          # 0.15 * 255, the threshold real._mask_for applies


def _iou(a, b) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0


def _centroid_shift(a, b) -> float:
    """Distance in crop pixels between the centres of mass of two binary maps."""
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    if not len(ya) or not len(yb):
        return 1e9
    return float(np.hypot(xa.mean() - xb.mean(), ya.mean() - yb.mean()))


def _ids_by_split(samples):
    """Every raw FaceForensics++ id a split touches, *including* the source
    identity of each forgery.  Grouping only by the target is the leak."""
    out = {}
    for s in samples:
        bag = out.setdefault(s.split or "unassigned", set())
        bag.add(s.meta.get("target", ""))
        if s.meta.get("source"):
            bag.add(s.meta["source"])
    return out


def _split_overlap(ids_by_split):
    ov = set()
    for a in ids_by_split:
        for b in ids_by_split:
            if a < b:
                ov |= ids_by_split[a] & ids_by_split[b]
    return ov - {""}


def geometry_checks():
    """A mask registered to the full frame must be cut with the *same* square
    that produced the face crop.

    The old code cropped the image with the face square but resized the whole
    mask frame, so the annotation landed somewhere else entirely whenever the
    face did not fill the frame -- the normal case for every video corpus.  The
    suite only ever asserted that a mask was *attached*, never that it was
    *correct*, which is why the defect survived.  The marker below is written
    into the image and into the mask at identical coordinates, so the two crops
    must agree; the whole-frame resize provably does not.
    """
    print("\n== mask / image geometric alignment ==")
    H, W = 360, 640
    frame = np.full((H, W, 3), 40, np.uint8)
    marker = np.zeros((H, W), np.uint8)
    for x, y, w, h in ((40, 30, 70, 60), (250, 150, 120, 90),
                       (500, 260, 90, 70), (410, 40, 60, 55)):
        cv2.rectangle(marker, (x, y), (x + w, y + h), 255, -1)
    frame[:, :, 2] = np.maximum(frame[:, :, 2], marker)

    crop, _bbox, _det, src = crop_face_box(frame, IMG_T)
    check(src is not None and len(src) == 3,
          f"crop_face_box returns a source square {src}")
    x0, y0, side = src
    check(crop.shape == (IMG_T, IMG_T, 3),
          f"image crop has the declared shape {crop.shape}")
    check(0 <= x0 and 0 <= y0 and x0 + side <= W and y0 + side <= H,
          f"the source square lies inside the frame ({src} in {W}x{H})")
    check(crop_like(marker, src, IMG_T).shape == (IMG_T, IMG_T),
          "mask crop has the same spatial shape as the image crop")
    manual = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (IMG_T, IMG_T),
                        interpolation=cv2.INTER_AREA)
    check(np.array_equal(manual, crop),
          "src_box reproduces the crop exactly (contract of crop_face_box)")

    # exact, at the source resolution: no resampling is involved, so the two
    # cut-outs must be identical pixel for pixel
    check(np.array_equal(crop_like(marker, src, side),
                         marker[y0:y0 + side, x0:x0 + side]),
          "crop_like cuts exactly the square crop_face_box cut")

    # through the pipeline, at the crop resolution.  The image is resampled with
    # INTER_AREA and the label map with INTER_NEAREST -- by design, since
    # interpolating a label invents values -- so agreement is bounded by edge
    # resampling, not by registration.  Registration is what the centroid tests.
    in_image = crop[:, :, 2] > 128
    aligned = crop_like(marker, src, IMG_T) > 128
    naive = cv2.resize(marker, (IMG_T, IMG_T)) > 128
    ia, ino = _iou(in_image, aligned), _iou(in_image, naive)
    check(ia > 0.90, f"crop_like keeps the annotation registered (IoU {ia:.3f})")
    check(ino < 0.5, f"whole-frame resize does not (IoU {ino:.3f}) -- "
                     f"the check can fail, so it means something")
    da, dn = _centroid_shift(in_image, aligned), _centroid_shift(in_image, naive)
    check(da < 1.5, f"aligned annotation is centred on the same pixels ({da:.2f} px)")
    check(dn > 5.0, f"resized annotation is displaced ({dn:.2f} px)")

    # a second image cut with the same square must survive a size change too
    big = crop_like(marker, src, 224)
    check(abs(float((big > 128).mean()) - float(aligned.mean())) < 0.02,
          "crop_like is resolution-independent")


def ffpp_frame_checks(root, n=8, compression="c23"):
    """The same invariant on actual videos and their mask videos."""
    print("\n== FaceForensics++ annotated masks land on the face ==")
    samples = [s for s in discover_faceforensics(root, compression=compression)
               if s.mask_path]
    check(len(samples) > 0, f"manipulated samples with masks present ({len(samples)})")
    if not samples:
        return
    step = max(1, len(samples) // n)
    cov_a, cov_n, shapes = [], [], []
    for s in samples[::step][:n]:
        fr = sample_frames(s.path, 1)
        mf = sample_frames(s.mask_path, 1)
        if not fr or not mf:
            continue
        c, _b, _d, src = crop_face_box(fr[0], IMG_T)
        g = cv2.cvtColor(mf[0], cv2.COLOR_BGR2GRAY)
        shapes.append((fr[0].shape[:2], mf[0].shape[:2], c.shape[:2],
                       crop_like(g, src, IMG_T).shape[:2]))
        cov_a.append(float((crop_like(g, src, IMG_T) > MASK_THR).mean()))
        cov_n.append(float((cv2.resize(g, (IMG_T, IMG_T)) > MASK_THR).mean()))
    check(all(f == m for f, m, _c, _mc in shapes),
          "the mask video has the same frame size as the image video "
          + str(sorted({(f, m) for f, m, _c, _mc in shapes})))
    check(all(c == (IMG_T, IMG_T) and mc == (IMG_T, IMG_T)
              for _f, _m, c, mc in shapes),
          f"image and mask crops are both {IMG_T}x{IMG_T} for every sample")
    ma, mn = float(np.mean(cov_a)), float(np.mean(cov_n))
    check(ma > 0.05, f"aligned mask covers a real part of the crop ({ma:.1%})")
    check(ma < 0.9, f"aligned mask is not a degenerate full-frame target ({ma:.1%})")
    check(ma > mn * 1.5,
          f"alignment beats the whole-frame resize ({ma:.1%} vs {mn:.1%})")


def ffpp_adapter_checks(root, strict=True, compression="c23"):
    """``strict`` demands the fixture's exact ids; a real tree is checked for
    consistency with whatever subset of the corpus is actually on disk, since a
    partial download is normal and is not by itself an error."""
    print("\n== FaceForensics++ generators, official splits, identity leakage ==")
    samples = discover_faceforensics(root, compression=compression)
    check(bool(samples), f"discovery finds samples under {root} ({len(samples)})")
    if not samples:
        return

    # -- manipulation-family preservation ----------------------------------- #
    meths = {s.method for s in samples if s.method}
    check(meths >= set(FFPP_CORE_METHODS) if not strict else meths == set(FFPP_CORE_METHODS),
          f"all four core generators are kept as distinct methods: {sorted(meths)}")
    by_cls = {}
    for s in samples:
        if s.method:
            by_cls.setdefault(ATTACK_NAMES[s.label], set()).add(s.method)
    check(all(len(v) >= 2 for v in by_cls.values()),
          "each taxonomy class keeps more than one generator: "
          + str({k: sorted(v) for k, v in sorted(by_cls.items())}))
    check(all(not s.method for s in samples if s.label == 0),
          "bonafide originals carry no generator name")

    # -- official split files ------------------------------------------------ #
    official = _ffpp_official_splits(root)
    check(len(official) > 0, f"splits/*.json read ({len(official)} ids)")
    n_off = sum(1 for s in samples if s.meta.get("official_split"))
    check(n_off == len(samples) if strict else n_off > 0,
          f"{n_off}/{len(samples)} samples placed by the corpus's own split files")
    for name in ("train", "val", "test"):
        f = f"{root}/splits/{name}.json"
        if not os.path.exists(f):
            check(False, f"splits/{name}.json is present")
            continue
        want = {i for pair in json.load(open(f)) for i in pair}
        got = {s.meta["target"] for s in samples if s.split == name}
        ok = (got == want) if strict else (got <= want and bool(got))
        check(ok, f"{name}: {len(got)} ids on disk are the ones splits/{name}.json "
                  f"assigns ({len(want)} listed)"
                  + ("" if ok else f"  extra={sorted(got - want)[:6]}"))
    before = [s.split for s in samples]
    assign_splits(samples)
    check([s.split for s in samples] == before,
          "assign_splits never overwrites a split the corpus defined")

    # -- source-identity leakage --------------------------------------------- #
    orig = {s.meta["target"]: s.subject for s in samples if s.label == 0}
    bad = [s for s in samples if s.meta.get("source")
           and orig.get(s.meta["source"]) != s.subject]
    check(not bad, f"target and source of every forgery share one subject group "
                   f"({len(bad)} violations)")
    ov = _split_overlap(_ids_by_split(samples))
    check(not ov, f"no target or source identity appears in two splits ({sorted(ov)})")

    # -- duplicates ---------------------------------------------------------- #
    paths = [s.path for s in samples]
    check(len(paths) == len(set(paths)),
          f"no video is listed twice as a sample ({len(paths)} samples, "
          f"{len(set(paths))} distinct paths)")
    triples = [(s.meta["target"], s.meta.get("source", ""), s.method) for s in samples]
    check(len(triples) == len(set(triples)),
          f"no (target, source, generator) triple is duplicated ({len(set(triples))} "
          f"distinct of {len(triples)})")
    per_id = {}
    for s in samples:
        per_id.setdefault(s.meta["target"], set()).add(s.split)
    multi = {k: sorted(v) for k, v in per_id.items() if len(v) > 1}
    check(not multi, f"no video id is assigned to more than one split ({multi})")
    subj_split = {}
    for s in samples:
        subj_split.setdefault(s.subject, set()).add(s.split)
    bad_subj = {k: sorted(v) for k, v in subj_split.items() if len(v) > 1}
    check(not bad_subj, f"no subject group straddles two splits ({bad_subj})")

    # negative control: grouping by target alone -- what the adapter used to do --
    # must be caught by the very same check, otherwise the check is vacuous
    leaky = {}
    for s in samples:
        sp = "train" if int(s.meta["target"]) % 2 == 0 else "test"
        leaky.setdefault(sp, set()).add(s.meta["target"])
        if s.meta.get("source"):
            leaky[sp].add(s.meta["source"])
    check(bool(_split_overlap(leaky)),
          "control: a target-only split is reported as leaking")

    # -- the hash split must be safe too, for releases without split files ---- #
    free = discover_faceforensics(root, use_official_splits=False)
    check(all(not s.meta["official_split"] for s in free),
          "official splits can be disabled")
    assign_splits(free)
    ov2 = _split_overlap(_ids_by_split(free))
    check(not ov2, f"pair-aware hash split is also identity-disjoint ({sorted(ov2)})")


def ffpp_cache_checks(cache):
    """Everything above has to survive frame extraction into the memmap cache."""
    print("\n== generators and masks survive into the cache ==")
    rows = list(csv.DictReader(open(f"{cache}/A_manifest.csv")))
    check(rows and "method" in rows[0], "manifest carries a method column")
    meths = {r["method"] for r in rows if r["method"]}
    check(meths == set(FFPP_CORE_METHODS),
          f"all four generators reach the cache: {sorted(meths)}")

    M = np.load(f"{cache}/A_m.npy", mmap_mode="r")
    cov_a = [float((M[int(r["idx"])] > MASK_THR).mean())
             for r in rows if r["has_mask"] == "1"]
    cov_b = [float((M[int(r["idx"])] > MASK_THR).mean())
             for r in rows if int(r["label"]) == 0]
    check(cov_a and 0.02 < float(np.mean(cov_a)) < 0.9,
          f"cached localisation targets are non-trivial ({np.mean(cov_a):.1%} of the crop)")
    check(not cov_b or max(cov_b) == 0.0, "bonafide localisation targets are empty")

    tr = XUADDataset(cache, "A", "train")
    counts = tr.method_counts()
    check(set(counts) == set(FFPP_CORE_METHODS),
          f"XUADDataset exposes the generators: {counts}")
    check(len(tr.method_array()) == len(tr), "method_array aligns with the samples")

    held = XUADDataset(cache, "A", "train", exclude_method=("Deepfakes",))
    check("Deepfakes" not in held.method_counts() and len(held) < len(tr),
          f"exclude_method drops the generator ({len(tr)} -> {len(held)})")
    check(ATTACK_NAMES.index("faceswap") in set(held.y.tolist()),
          "its taxonomy class survives via FaceSwap -- LOMO is not P2 in disguise")
    kept = XUADDataset(cache, "A", "train", keep_method=("Deepfakes",))
    check(set(kept.method_counts()) == {"Deepfakes"},
          f"keep_method restricts to one generator ({kept.method_counts()})")

    # a cache written before the column existed must still load unchanged
    legacy = "/tmp/xuad_legacy_manifest"
    os.makedirs(legacy, exist_ok=True)
    for f in ("A_x.npy", "A_m.npy"):
        if not os.path.exists(f"{legacy}/{f}"):
            os.symlink(f"{cache}/{f}", f"{legacy}/{f}")
    with open(f"{legacy}/A_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "identity", "split", "label", "label_name", "has_mask"])
        for r in rows:
            w.writerow([r["idx"], r["identity"], r["split"], r["label"],
                        r["label_name"], r["has_mask"]])
    old = XUADDataset(legacy, "A", "train")
    check(len(old) == len(tr) and old.method_counts() == {},
          "a manifest without the method column still loads (frozen benchmark)")


def lomo_containment_checks(cache):
    """A held-out generator must not reach the leave-one-manipulation-out
    training set -- proven at the level of dataset construction, with no model
    fitted.

    This rebuilds exactly the datasets ``scripts/run_experiment.py`` builds for
    ``--protocol LOMO --hold-out-method X`` and compares *sample index sets*, not
    counts: a count can coincide, an index set cannot.  It also pins the two ways
    a LOMO run can be silently invalid -- a generator name whose case does not
    match, and a name that matches nothing at all.
    """
    print("\n== a held-out family cannot enter the LOMO training set ==")
    rows = list(csv.DictReader(open(f"{cache}/A_manifest.csv")))
    for meth in FFPP_CORE_METHODS:
        held_idx = {int(r["idx"]) for r in rows if r["method"] == meth}
        tr = XUADDataset(cache, "A", "train", exclude_method=(meth,))
        va = XUADDataset(cache, "A", "val", exclude_method=(meth,))
        te = XUADDataset(cache, "A", "test")
        leaked = (set(tr.idx) | set(va.idx)) & held_idx
        check(not leaked,
              f"{meth}: no held-out sample index reaches train or val "
              f"({len(leaked)} of {len(held_idx)} leaked)")
        check(meth not in tr.method_counts() and meth not in va.method_counts(),
              f"{meth}: absent from the train and val generator inventories")
        check(bool(set(te.idx) & held_idx),
              f"{meth}: still present in the test split, as LOMO requires")

        # LOMO is not leave-one-attack-out: the taxonomy class must survive
        cls = {int(r["label"]) for r in rows if r["method"] == meth}
        check(cls and cls <= set(tr.y.tolist()),
              f"{meth}: its taxonomy class {[ATTACK_NAMES[c] for c in cls]} is still "
              f"represented in training by the other generators")

        # counts must reconcile exactly with the manifest
        full = XUADDataset(cache, "A", "train")
        n_held_train = len([r for r in rows
                            if r["method"] == meth and r["split"] == "train"])
        check(len(full) - len(tr) == n_held_train,
              f"{meth}: exactly the {n_held_train} manifest rows of that generator "
              f"were removed from train ({len(full)} -> {len(tr)})")

    # -- spelling: 'DeepFakes' must behave identically to 'Deepfakes' --------- #
    a = XUADDataset(cache, "A", "train", exclude_method=("Deepfakes",))
    b = XUADDataset(cache, "A", "train", exclude_method=("DeepFakes",))
    check(set(a.idx) == set(b.idx),
          "generator matching is case-insensitive, so a capitalisation variant "
          "cannot silently disable the exclusion")

    # -- and a name that matches nothing must be visible, not silent ---------- #
    typo = XUADDataset(cache, "A", "train", exclude_method=("Deepfake",))
    check(set(typo.idx) == set(XUADDataset(cache, "A", "train").idx),
          "a misspelled generator excludes nothing (which is why run_experiment "
          "refuses it rather than reporting a closed-set run as LOMO)")
    o = run(f"{sys.executable} {ROOT}/scripts/run_experiment.py --data {cache} "
            f"--model xuad --protocol LOMO --hold-out-method Deepfake --epochs 1 "
            f"--out /tmp/xuad_test_res --tag t_guard")
    check("matches no generator" in o and not os.path.exists("/tmp/xuad_test_res/t_guard.json"),
          "run_experiment refuses a hold-out name that matches nothing, before "
          "any model is built")


def frozen_benchmark_check(lfw="/tmp/fe/images", cache=os.path.join(ROOT, "data"), n=8):
    """The simulated LFW benchmark and every published number on it are frozen.

    ``faces.crop_face`` is the single point where this test suite's code touches
    the benchmark builder, so a change there -- such as the ``crop_face_box``
    refactor that made mask alignment possible -- silently invalidates 29,400
    cached samples and every result derived from them.  This rebuilds a random
    sample of entries from the original sources and demands they come back
    byte-for-byte identical.  Skipped when the source corpus or the cache is not
    on this machine.
    """
    if not (os.path.isdir(lfw) and os.path.exists(f"{cache}/A_x.npy")):
        print("\n== frozen benchmark ==\n  SKIP  sources or cache not present here")
        return
    print("\n== the frozen LFW benchmark still rebuilds byte-for-byte ==")
    from xuad.data import build_benchmark as bb
    from xuad.data.faces import identity_disjoint_split, index_corpus

    classes = [c for c in ATTACK_NAMES if c != "adversarial"]
    faces = index_corpus(lfw)
    caps = {"train": 2400, "val": 600, "test": 1200}
    sub = {}
    for k, v in identity_disjoint_split(faces).items():
        r = np.random.default_rng(1234 + len(k))
        sub[k] = [v[i] for i in sorted(r.permutation(len(v))[: caps[k]])]
    jobs = []
    for sp, fs in sub.items():
        paths = [f.path for f in fs]
        for i, f in enumerate(fs):
            donor = paths[(i + 1 + (i * 7919) % max(1, len(paths) - 1)) % len(paths)]
            for ci, cn in enumerate(classes):
                jobs.append((f.path, f.identity, sp, ci, cn,
                             donor if cn == "faceswap" else None, 0.62, i))

    X = np.load(f"{cache}/A_x.npy", mmap_mode="r")
    M = np.load(f"{cache}/A_m.npy", mmap_mode="r")
    rows = list(csv.DictReader(open(f"{cache}/A_manifest.csv")))
    total = int(np.load(f"{cache}/A_n.npy")[0])
    check(len(jobs) == total == len(rows),
          f"the build plan still reproduces ({len(jobs)} jobs, {total} cached rows)")
    if len(jobs) != total:
        return
    same = 0
    for i in np.random.default_rng(0).choice(total, n, replace=False).tolist():
        res = bb._worker(jobs[int(i)])
        if res is None:
            continue
        img, msk, ident, sp, ci = res
        r = rows[int(i)]
        same += int(r["identity"] == ident and r["split"] == sp and int(r["label"]) == ci
                    and np.array_equal(np.asarray(X[int(i)]), img)
                    and np.array_equal(np.asarray(M[int(i)]), msk))
    check(same == n, f"{same}/{n} sampled entries are byte-identical after the "
                     f"crop_face_box refactor")


def audit_real_ffpp(root, compression="c23", frames=8):
    """Run the FaceForensics++ integrity checks against a *real* corpus tree.

    Identical checks to the ones the mock fixture exercises -- family
    preservation, official splits, duplicate and cross-split identities, and
    image/mask geometric alignment -- but read-only: nothing is extracted,
    nothing is cached, no model is built.  This is the audit to run once the
    real c23 videos and official masks are on disk.
    """
    print(f"\n{'=' * 66}\n FaceForensics++ integrity audit: {root}\n"
          f" compression={compression}   (read-only; no extraction, no training)\n"
          f"{'=' * 66}")
    if not os.path.isdir(root):
        check(False, f"corpus root exists: {root}")
        return 1
    samples = discover_faceforensics(root, compression=compression)
    if samples:
        print(inventory(samples))
    ffpp_adapter_checks(root, strict=False, compression=compression)
    ffpp_frame_checks(root, n=frames, compression=compression)
    return 1 if fails else 0


def training_smoke_checks():
    """Optional: 1-epoch fits on the mock corpora, to prove the protocols run.

    These *train*, so they are opt-in (``--with-training``) and are never part of
    a pre-experiment integrity run.  They say nothing whatever about detection
    quality -- the corpora are synthetic fixtures of a few dozen crops -- and no
    number they produce may be reported.
    """
    print("\n== optional training smoke tests (--with-training) ==")
    for corpus in ("faceforensics", "casia-fasd"):
        out = f"{CACHE}/{corpus}"
        noloc = "" if corpus == "faceforensics" else "--no-loc"
        o = run(f"{sys.executable} {ROOT}/scripts/run_experiment.py --data {out} "
                f"--model xuad --protocol P1 --epochs 1 --bs 16 --threads 2 --workers 0 "
                f"{noloc} --tag t_{corpus} --out /tmp/xuad_test_res")
        check('"ACER"' in o, f"{corpus}: P1 trains and evaluates")

    o = run(f"{sys.executable} {ROOT}/scripts/run_experiment.py "
            f"--data {CACHE}/casia-fasd --model xuad --protocol P2.replay --epochs 1 "
            f"--bs 16 --threads 2 --workers 0 --no-loc --tag t_loao --out /tmp/xuad_test_res")
    check('"unseen_detect_rate"' in o, "leave-one-attack-out works on a 3-class corpus")

    for meth in FFPP_CORE_METHODS:
        o = run(f"{sys.executable} {ROOT}/scripts/run_experiment.py "
                f"--data {CACHE}/faceforensics --model xuad --protocol LOMO "
                f"--hold-out-method {meth} --epochs 1 --bs 16 --threads 2 --workers 0 "
                f"--tag t_lomo_{meth} --out /tmp/xuad_test_res")
        j = f"/tmp/xuad_test_res/t_lomo_{meth}.json"
        d = json.load(open(j)) if os.path.exists(j) else {}
        check(d.get("held_out_method") == meth and d.get("n_held_out_samples", 0) > 0,
              f"LOMO/{meth}: runs and scores {d.get('n_held_out_samples', 0)} held-out samples")
        check(all(k in d for k in ("unseen_detect_rate", "seen_detect_rate",
                                   "novelty_AUROC", "unseen_vs_bonafide_AUROC")),
              f"LOMO/{meth}: per-generator open-set metrics are reported")

    o = run(f"{sys.executable} {ROOT}/scripts/run_experiment.py "
            f"--data {CACHE}/faceforensics --data-test {CACHE}/casia-fasd --protocol P3 "
            f"--model xuad --epochs 1 --bs 16 --threads 2 --workers 0 --no-loc "
            f"--tag t_cross --out /tmp/xuad_test_res")
    check('"data_test"' in o, "cross-dataset protocol trains on one cache, tests on another")


def main(quick=False, with_training=False):
    if not os.path.isdir(MOCK):
        print("building mock corpora ...")
        run(f"{sys.executable} {ROOT}/tests/make_mock_corpora.py")

    print("\n== discovery, taxonomy mapping and split discipline ==")
    for corpus, root in ROOTS.items():
        kw = {}
        if corpus == "celeba-spoof":
            kw["label_json"] = f"{root}/metas/intra_test/train_label.json"
        samples, _ = discover(root, corpus, **kw)
        check(len(samples) > 0, f"{corpus}: discovers files ({len(samples)})")
        got = {ATTACK_NAMES[s.label] for s in samples}
        check(got == EXPECTED[corpus],
              f"{corpus}: taxonomy {sorted(got)} == expected {sorted(EXPECTED[corpus])}")
        assign_splits(samples)
        subj = {}
        for s in samples:
            subj.setdefault(s.split, set()).add(s.subject)
        overlap = set()
        for a in subj:
            for b in subj:
                if a < b:
                    overlap |= subj[a] & subj[b]
        check(not overlap, f"{corpus}: splits are subject-disjoint")
        check(set(subj) >= {"train", "val", "test"} or len(subj) >= 2,
              f"{corpus}: has train/val/test ({sorted(subj)})")

    print("\n== auto-detection picks the right adapter ==")
    for corpus, root in ROOTS.items():
        if corpus in ("replay-attack", "uniattackdata", "siw-mv2"):
            continue      # these layouts deliberately overlap on generic keywords
        _s, name = discover(root, "auto")
        check(name == corpus, f"auto({os.path.basename(root)}) -> {name}")

    print("\n== FaceForensics++ masks are targets, not samples ==")
    s, _ = discover(ROOTS["faceforensics"], "faceforensics")
    check(all("/masks/" not in x.path.replace("\\", "/") for x in s),
          "no mask video is listed as a sample")
    check(sum(1 for x in s if x.mask_path) > 0, "manipulated samples carry a mask path")

    frozen_benchmark_check()
    geometry_checks()
    ffpp_frame_checks(ROOTS["faceforensics"])
    ffpp_adapter_checks(ROOTS["faceforensics"])

    if quick:
        return 1 if fails else 0

    print("\n== frame extraction -> cache (no model is fitted) ==")
    os.makedirs(CACHE, exist_ok=True)
    for corpus in ("faceforensics", "casia-fasd"):
        out = f"{CACHE}/{corpus}"
        extra = ("--label-json " + ROOTS[corpus] + "/metas/intra_test/train_label.json") \
            if corpus == "celeba-spoof" else ""
        o = run(f"{sys.executable} {ROOT}/scripts/prepare_real.py --root '{ROOTS[corpus]}' "
                f"--corpus {corpus} --out {out} --frames-per-video 3 --workers 2 {extra}")
        check("wrote" in o and os.path.exists(f"{out}/A_x.npy"), f"{corpus}: cache written")

    ffpp_cache_checks(f"{CACHE}/faceforensics")
    lomo_containment_checks(f"{CACHE}/faceforensics")

    if with_training:
        training_smoke_checks()
    else:
        print("\n  training smoke tests skipped (pass --with-training to run them);\n"
              "  no model was built and nothing was fitted in this run.")
    return 1 if fails else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    real = next((a.split("=", 1)[1] for a in argv if a.startswith("--real-root=")), None)
    if real is None and "--real-root" in argv:
        real = argv[argv.index("--real-root") + 1]
    if real:
        comp = next((a.split("=", 1)[1] for a in argv
                     if a.startswith("--compression=")), "c23")
        code = audit_real_ffpp(real, compression=comp)
    else:
        code = main("--quick" in argv, "--with-training" in argv)
    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES:"))
    for f in fails:
        print("  - " + f)
    sys.exit(code)
