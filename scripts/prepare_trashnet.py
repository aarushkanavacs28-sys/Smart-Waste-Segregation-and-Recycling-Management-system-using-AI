"""
prepare_trashnet.py - split TrashNet into train / val / test folders.

Default split = 63% train / 7% val / 30% test, which matches the split reported in
Rajakumaran et al. (1593 / 176 / 758 images), so results can be compared fairly.

Expected input: a folder that contains the six class folders somewhere inside it:
    cardboard, glass, metal, paper, plastic, trash
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())      # "Cardboard " -> "cardboard"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder that contains the 6 class folders")
    ap.add_argument("--output", default="data/trashnet_processed")
    ap.add_argument("--val-frac", type=float, default=0.07)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source, out = Path(args.source), Path(args.output)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists. Use --overwrite.")
        shutil.rmtree(out)

    # find the six class folders anywhere under the source folder
    found = {}
    for p in source.rglob("*"):
        if "__MACOSX" in p.parts:          # junk folder created by zips made on a Mac
            continue
        if p.is_dir() and norm(p.name) in CLASSES:
            found[norm(p.name)] = p
    missing = [c for c in CLASSES if c not in found]
    if missing:
        raise SystemExit(f"Class folders not found: {missing}")

    rng = random.Random(args.seed)                    # fixed seed -> same split every run
    for cls in CLASSES:
        imgs = sorted(
            p for p in found[cls].rglob("*")
            if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")   # skip Mac junk files
        )
        rng.shuffle(imgs)
        n = len(imgs)
        n_test = round(n * args.test_frac)
        n_val = round(n * args.val_frac)
        parts = {
            "test": imgs[:n_test],
            "val": imgs[n_test:n_test + n_val],
            "train": imgs[n_test + n_val:],
        }
        if min(len(v) for v in parts.values()) == 0:
            raise SystemExit(f"Class '{cls}' ended up with an empty split ({n} images found). Check the source folder.")
        for split, items in parts.items():
            (out / split / cls).mkdir(parents=True, exist_ok=True)
            for p in items:
                shutil.copy2(p, out / split / cls / p.name)
        print(f"{cls:10s} total={n:4d} train={len(parts['train']):4d} "
              f"val={len(parts['val']):3d} test={len(parts['test']):4d}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
