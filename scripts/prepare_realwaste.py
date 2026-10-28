from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def norm_name(s: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        s.lower(),
    )


SOURCE_MAP = {
    "cardboard": "paper_cardboard",
    "paper": "paper_cardboard",
    "foodorganics": "organic",
    "vegetation": "organic",
    "glass": "glass",
    "metal": "metal",
    "plastic": "plastic",
    "miscellaneoustrash": "miscellaneous_trash",
    "textiletrash": "ood_textile",
}


def find_class_dirs(source: Path):
    matches = {}
    for p in source.rglob("*"):
        if p.is_dir():
            n = norm_name(p.name)
            if n in SOURCE_MAP:
                matches[n] = p
    return matches


def collect_images(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    )


def copy_images(
    paths,
    dest_dir: Path,
    prefix: str,
):
    dest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    for i, src in enumerate(paths):
        ext = src.suffix.lower()
        dst = (
            dest_dir
            / f"{prefix}_{i:05d}{ext}"
        )
        shutil.copy2(
            src,
            dst,
        )


def split_list(
    items,
    train_ratio,
    val_ratio,
    seed,
):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_train = int(
        round(n * train_ratio)
    )
    n_val = int(
        round(n * val_ratio)
    )

    if n >= 3:
        n_train = min(
            n_train,
            n - 2,
        )
        n_val = min(
            n_val,
            n - n_train - 1,
        )

    train = items[:n_train]
    val = items[
        n_train:n_train + n_val
    ]
    test = items[
        n_train + n_val:
    ]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/raw/realwaste",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    if (
        args.train_ratio
        + args.val_ratio
        >= 1.0
    ):
        raise ValueError(
            "train_ratio + val_ratio must be < 1.0"
        )

    source = Path(args.source)
    output = Path(args.output)

    if output.exists() and args.overwrite:
        shutil.rmtree(output)

    if (
        output.exists()
        and any(output.iterdir())
    ):
        raise RuntimeError(
            f"{output} is not empty. "
            "Use --overwrite to rebuild it."
        )

    class_dirs = find_class_dirs(source)
    missing = sorted(
        set(SOURCE_MAP)
        - set(class_dirs)
    )

    if missing:
        raise RuntimeError(
            "Could not locate all expected "
            "RealWaste class folders.\n"
            f"Missing: {missing}\n"
            f"Found: {sorted(class_dirs)}"
        )

    grouped = defaultdict(list)

    for src_norm, folder in class_dirs.items():
        target = SOURCE_MAP[src_norm]
        imgs = collect_images(folder)
        grouped[target].extend(imgs)
        print(
            f"{folder.name:25s} -> "
            f"{target:22s}: "
            f"{len(imgs)} images"
        )

    copy_images(
        grouped["ood_textile"],
        output / "ood_textile",
        "textile",
    )

    id_classes = sorted(
        k for k in grouped
        if k != "ood_textile"
    )

    summary = {}

    for class_idx, cls in enumerate(
        id_classes
    ):
        train, val, test = split_list(
            grouped[cls],
            args.train_ratio,
            args.val_ratio,
            args.seed + class_idx,
        )

        copy_images(
            train,
            output / "train" / cls,
            f"{cls}_train",
        )
        copy_images(
            val,
            output / "val" / cls,
            f"{cls}_val",
        )
        copy_images(
            test,
            output / "test" / cls,
            f"{cls}_test",
        )

        summary[cls] = {
            "total": len(
                grouped[cls]
            ),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

    print("\nPrepared dataset:")
    for cls, s in summary.items():
        print(
            f"{cls:22s} "
            f"total={s['total']:4d} "
            f"train={s['train']:4d} "
            f"val={s['val']:4d} "
            f"test={s['test']:4d}"
        )

    print(
        f"OOD textile images: "
        f"{len(grouped['ood_textile'])}"
    )
    print(
        f"Output: {output.resolve()}"
    )


if __name__ == "__main__":
    main()
