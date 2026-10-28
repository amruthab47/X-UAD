"""Generate miniature corpora that replicate the *exact* on-disk layout and
filename conventions of the public datasets.

These are not a substitute for the real data -- the pixels come from our own
synthesisers. They exist so that every directory parser, protocol reader,
split rule, mask reader and frame decoder in :mod:`xuad.data.real` can be
exercised end-to-end before it is ever pointed at a 500 GB download, which is
the only point at which a layout mistake is cheap to fix.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xuad.data.attacks import SYNTHESISERS
from xuad.data.faces import crop_face, index_corpus, np_rng

SRC = "/tmp/fe/images"
OUT = "/tmp/mock"
SIZE = 160
NFRAMES = 6


def _faces(n, seed=0):
    fs = index_corpus(SRC)
    rng = random.Random(seed)
    picks = rng.sample(range(len(fs)), n)
    out = []
    for i in picks:
        im = cv2.imread(fs[i].path)
        c, b, _ = crop_face(im, SIZE)
        if c is not None:
            out.append((c, b, fs[i].identity))
    return out


def _attack(img, bbox, kind, seed, donor=None, donor_bbox=None):
    if kind == "bonafide":
        return img, np.zeros(img.shape[:2], np.float32)
    fn = SYNTHESISERS[kind]
    kw = {"donor": donor, "donor_bbox": donor_bbox} if kind == "faceswap" else {}
    return fn(img, bbox, np_rng(kind, seed), 0.7, **kw)


def _write_video(path, frames, ext=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = ext or os.path.splitext(path)[1]
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext == ".avi" else "mp4v"))
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, fourcc, 10.0, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return os.path.exists(path) and os.path.getsize(path) > 0


def _clip(img, bbox, kind, seed, donor=None, donor_bbox=None, jitter=True):
    """A short clip: the same attack with mild per-frame variation."""
    out, masks = [], []
    for t in range(NFRAMES):
        base = img
        if jitter:
            M = np.float32([[1, 0, np.sin(t) * 1.5], [0, 1, np.cos(t) * 1.5]])
            base = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                  borderMode=cv2.BORDER_REPLICATE)
        a, m = _attack(base, bbox, kind, seed * 100 + t, donor, donor_bbox)
        out.append(a); masks.append(m)
    return out, masks


# --------------------------------------------------------------------------- #
def mock_casia(faces):
    root = f"{OUT}/CASIA-FASD"
    plan = {"train_release": faces[:4], "test_release": faces[4:8]}
    for split, group in plan.items():
        for si, (img, bbox, _id) in enumerate(group, 1):
            for stem, kind in [("1", "bonafide"), ("2", "bonafide"), ("HR_1", "bonafide"),
                               ("3", "print"), ("5", "print"), ("HR_2", "print"),
                               ("7", "replay"), ("8", "replay"), ("HR_4", "replay")]:
                fr, _ = _clip(img, bbox, kind, si)
                _write_video(f"{root}/{split}/{si}/{stem}.avi", fr)
    return root


def mock_replay(faces):
    root = f"{OUT}/Replay-Attack"
    base = 0
    for split, group in (("train", faces[:3]), ("devel", faces[3:5]), ("test", faces[5:8])):
        for k, (img, bbox, _id) in enumerate(group, 1):
            si = base + k
            fr, _ = _clip(img, bbox, "bonafide", si)
            _write_video(f"{root}/{split}/real/client{si:03d}_session01_webcam_authenticate.mov", fr)
            for hand, kind, tag in (("fixed", "print", "print"), ("hand", "replay", "mobile"),
                                    ("fixed", "replay", "highdef")):
                fr, _ = _clip(img, bbox, kind, si + 7)
                _write_video(f"{root}/{split}/attack/{hand}/"
                             f"attack_{hand}_client{si:03d}_session01_{tag}_video_controlled.mov", fr)
        base += len(group)
    return root


def mock_oulu(faces):
    root = f"{OUT}/OULU-NPU"
    base = 0
    for folder, group in (("Train_files", faces[:4]), ("Dev_files", faces[4:6]),
                          ("Test_files", faces[6:9])):
        for (img, bbox, _id), user in zip(group, range(base + 1, base + len(group) + 1)):
            for acc, kind in ((1, "bonafide"), (2, "print"), (3, "print"),
                              (4, "replay"), (5, "replay")):
                fr, _ = _clip(img, bbox, kind, user * 10 + acc)
                _write_video(f"{root}/{folder}/6_1_{user:02d}_{acc}.avi", fr)
        base += len(group)
    os.makedirs(f"{root}/Protocols/Protocol_1", exist_ok=True)
    open(f"{root}/Protocols/Protocol_1/Train.txt", "w").write("+1,6_1_01_1\n-1,6_1_01_2\n")
    return root


def mock_siwm(faces):
    root = f"{OUT}/SiW-Mv2"
    for i, (img, bbox, _id) in enumerate(faces[:8], 1):
        for folder, kind in (("Live", "bonafide"), ("Spoof/Print", "print"),
                             ("Spoof/Replay", "replay"), ("Spoof/SiliconeMask", "mask3d"),
                             ("Spoof/PaperMask", "mask3d"), ("Spoof/PartialEye", "mask3d")):
            fr, _ = _clip(img, bbox, kind, i)
            _write_video(f"{root}/{folder}/{i:03d}_clip{i}.mp4", fr)
    return root


def mock_celeba_spoof(faces):
    root = f"{OUT}/CelebA-Spoof"
    labels = {}
    for split, group in (("train", faces[:5]), ("test", faces[5:8])):
        for i, (img, bbox, _id) in enumerate(group, 1):
            idd = f"{i:06d}"
            for j, kind in enumerate(["bonafide", "bonafide"]):
                p = f"Data/{split}/{idd}/live/{j}.png"
                a, _ = _attack(img, bbox, kind, i)
                os.makedirs(os.path.dirname(f"{root}/{p}"), exist_ok=True)
                cv2.imwrite(f"{root}/{p}", a)
                open(f"{root}/Data/{split}/{idd}/live/{j}_BB.txt", "w").write(
                    f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} 0.99\n")
                labels[p] = [0] * 40 + [0, 0, 0, 0]
            for j, (kind, sp_idx) in enumerate([("print", 1), ("replay", 7), ("mask3d", 10)]):
                p = f"Data/{split}/{idd}/spoof/{j}.png"
                a, _ = _attack(img, bbox, kind, i + j)
                os.makedirs(os.path.dirname(f"{root}/{p}"), exist_ok=True)
                cv2.imwrite(f"{root}/{p}", a)
                open(f"{root}/Data/{split}/{idd}/spoof/{j}_BB.txt", "w").write(
                    f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} 0.99\n")
                lab = [0] * 40 + [sp_idx, 0, 0, 1]
                labels[p] = lab
    os.makedirs(f"{root}/metas/intra_test", exist_ok=True)
    json.dump(labels, open(f"{root}/metas/intra_test/train_label.json", "w"))
    return root


def _embed(patch, mask, canvas_hw=(360, 640), pos=(0.12, 0.55)):
    """Place a face patch inside a larger frame, FaceForensics++ style.

    Real FF++ frames are full video frames in which the face occupies a modest
    fraction of the image.  The earlier fixture wrote bare 160x160 face crops, so
    a mask that was resized from the whole frame instead of being cropped landed
    in almost the right place by accident -- which is exactly why the alignment
    defect survived the test suite.  Embedding the face makes the geometry
    testable.
    """
    H, W = canvas_hw
    ph, pw = patch.shape[:2]
    y0, x0 = int(pos[0] * H), int(pos[1] * W)
    y0 = max(0, min(y0, H - ph)); x0 = max(0, min(x0, W - pw))
    frame = np.full((H, W, 3), 48, np.uint8)
    frame[:, :, 1] = 60
    frame[y0:y0 + ph, x0:x0 + pw] = patch
    mfull = np.zeros((H, W), np.float32)
    mfull[y0:y0 + ph, x0:x0 + pw] = mask
    return frame, mfull, (x0, y0, pw, ph)


def mock_ffpp(faces):
    """Mirrors the real layout, including splits/{train,val,test}.json and the
    <target>_<source> naming, with the face embedded in a larger frame."""
    root = f"{OUT}/FaceForensics++"
    donors = faces[8:12]
    n_vid = 8
    # deterministic, varied face placement so the mask geometry is exercised
    POS = [(0.05, 0.04), (0.30, 0.62), (0.10, 0.35), (0.42, 0.10),
           (0.20, 0.70), (0.02, 0.50), (0.35, 0.28), (0.15, 0.05)]
    # FaceForensics++ pairs its 1000 originals into disjoint couples and swaps in
    # both directions, so the pairing graph is a set of 2-cycles rather than one
    # big component.  The fixture must match, or a split that is legitimate on the
    # real corpus looks like leakage here.
    PAIR = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
    for i, (img, bbox, _id) in enumerate(faces[:n_vid]):
        frames = []
        for fr in _clip(img, bbox, "bonafide", i)[0]:
            f, _m, _b = _embed(fr, np.zeros(fr.shape[:2], np.float32), pos=POS[i])
            frames.append(f)
        _write_video(f"{root}/original_sequences/youtube/c23/videos/{i:03d}.mp4", frames)

    for meth, kind in (("Deepfakes", "faceswap"), ("FaceSwap", "faceswap"),
                       ("Face2Face", "reenact"), ("NeuralTextures", "reenact")):
        for i, (img, bbox, _id) in enumerate(faces[:n_vid]):
            d, db, _ = donors[i % len(donors)]
            clips, masks = _clip(img, bbox, kind, i + 5, d, db)
            vid, msk = [], []
            for fr, mk in zip(clips, masks):
                f, mfull, _b = _embed(fr, np.clip(mk, 0, 1), pos=POS[i])
                vid.append(f)
                msk.append(cv2.cvtColor((mfull * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            src = PAIR[i]
            base = f"{root}/manipulated_sequences/{meth}/c23"
            _write_video(f"{base}/videos/{i:03d}_{src:03d}.mp4", vid)
            _write_video(f"{base}/masks/videos/{i:03d}_{src:03d}.mp4", msk)

    # official split files: lists of [target, source] pairs, as FF++ ships them
    os.makedirs(f"{root}/splits", exist_ok=True)
    pairs = [[f"{a:03d}", f"{b:03d}"] for a, b in ((0, 1), (2, 3), (4, 5), (6, 7))]
    json.dump(pairs[:2], open(f"{root}/splits/train.json", "w"))
    json.dump(pairs[2:3], open(f"{root}/splits/val.json", "w"))
    json.dump(pairs[3:], open(f"{root}/splits/test.json", "w"))
    return root


def mock_celebdf(faces):
    root = f"{OUT}/Celeb-DF-v2"
    lines = []
    for i, (img, bbox, _id) in enumerate(faces[:6]):
        p = f"Celeb-real/id{i}_000{i}.mp4"
        _write_video(f"{root}/{p}", _clip(img, bbox, "bonafide", i)[0])
        if i >= 4:
            lines.append(f"1 {p}")
    donors = faces[6:10]
    for i, (img, bbox, _id) in enumerate(faces[:6]):
        d, db, _ = donors[i % len(donors)]
        p = f"Celeb-synthesis/id{i}_id{(i+1)%6}_000{i}.mp4"
        _write_video(f"{root}/{p}", _clip(img, bbox, "faceswap", i + 3, d, db)[0])
        if i >= 4:
            lines.append(f"0 {p}")
    for i, (img, bbox, _id) in enumerate(faces[10:13]):
        _write_video(f"{root}/YouTube-real/{i:05d}.mp4", _clip(img, bbox, "bonafide", i + 30)[0])
    open(f"{root}/List_of_testing_videos.txt", "w").write("\n".join(lines) + "\n")
    return root


def mock_dfdc(faces):
    root = f"{OUT}/DFDC/dfdc_train_part_0"
    meta = {}
    donors = faces[6:10]
    for i, (img, bbox, _id) in enumerate(faces[:6]):
        real = f"real_{i:03d}.mp4"
        _write_video(f"{root}/{real}", _clip(img, bbox, "bonafide", i)[0])
        meta[real] = {"label": "REAL", "split": "train"}
        d, db, _ = donors[i % len(donors)]
        fake = f"fake_{i:03d}.mp4"
        _write_video(f"{root}/{fake}", _clip(img, bbox, "faceswap", i + 11, d, db)[0])
        meta[fake] = {"label": "FAKE", "split": "train", "original": real}
    json.dump(meta, open(f"{root}/metadata.json", "w"))
    return f"{OUT}/DFDC"


def mock_uniattack(faces):
    root = f"{OUT}/UniAttackData"
    donors = faces[8:12]
    for split, group in (("train", faces[:4]), ("dev", faces[4:6]), ("test", faces[6:9])):
        for i, (img, bbox, _id) in enumerate(group, 1):
            sid = f"{i:04d}"
            _write_video(f"{root}/{split}/live/{sid}_live.mp4",
                         _clip(img, bbox, "bonafide", i)[0])
            _write_video(f"{root}/{split}/physical/print/{sid}_print.mp4",
                         _clip(img, bbox, "print", i)[0])
            _write_video(f"{root}/{split}/physical/replay/{sid}_replay.mp4",
                         _clip(img, bbox, "replay", i)[0])
            d, db, _ = donors[i % len(donors)]
            _write_video(f"{root}/{split}/digital/deepfake/{sid}_swap.mp4",
                         _clip(img, bbox, "faceswap", i, d, db)[0])
    return root


BUILDERS = {"casia-fasd": mock_casia, "replay-attack": mock_replay,
            "oulu-npu": mock_oulu, "siw-mv2": mock_siwm,
            "celeba-spoof": mock_celeba_spoof, "faceforensics": mock_ffpp,
            "celeb-df": mock_celebdf, "dfdc": mock_dfdc,
            "uniattackdata": mock_uniattack}


if __name__ == "__main__":
    which = sys.argv[1:] or list(BUILDERS)
    shutil.rmtree(OUT, ignore_errors=True)
    faces = _faces(14, seed=7)
    print(f"source faces: {len(faces)}")
    for name in which:
        root = BUILDERS[name](faces)
        n = sum(len(f) for _r, _d, f in os.walk(root))
        print(f"  {name:16s} -> {root}  ({n} files)")
