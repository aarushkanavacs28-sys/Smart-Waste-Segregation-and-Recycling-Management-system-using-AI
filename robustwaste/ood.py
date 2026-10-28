from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .data import build_transforms
from .models import load_model_from_checkpoint
from .utils import (
    get_device,
    checkpoint_run_dir,
    load_json,
    save_json,
)


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class ImagePathDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = sorted(
            p for p in Path(root).rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as img:
            x = self.transform(
                img.convert("RGB")
            )
        return x, str(p)


def resolve_threshold(checkpoint, explicit):
    if explicit is not None:
        return explicit

    path = (
        checkpoint_run_dir(checkpoint)
        / "threshold.json"
    )
    if path.exists():
        return float(
            load_json(path)["selected_threshold"]
        )
    return 0.75


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    model, ckpt = load_model_from_checkpoint(
        args.checkpoint,
        device,
    )
    threshold = resolve_threshold(
        args.checkpoint,
        args.threshold,
    )

    ood_dir = Path(args.data_dir) / "ood_textile"
    transform = build_transforms(
        ckpt["image_size"],
        training=False,
    )
    ds = ImagePathDataset(
        ood_dir,
        transform,
    )

    if len(ds) == 0:
        raise RuntimeError(
            f"No OOD images found in {ood_dir}. "
            "Run prepare_realwaste.py first."
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows = []
    with torch.no_grad():
        for x, paths in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            for p, c, idx in zip(
                paths,
                conf.cpu().tolist(),
                pred.cpu().tolist(),
            ):
                rejected = c < threshold
                rows.append({
                    "path": p,
                    "predicted_class": (
                        ckpt["classes"][idx]
                    ),
                    "confidence": c,
                    "threshold": threshold,
                    "decision": (
                        "REJECT"
                        if rejected
                        else "ACCEPT"
                    ),
                })

    df = pd.DataFrame(rows)
    run_dir = checkpoint_run_dir(args.checkpoint)
    df.to_csv(
        run_dir / "ood_textile_predictions.csv",
        index=False,
    )

    rejection_rate = float(
        (df["decision"] == "REJECT").mean()
    )

    summary = {
        "ood_set": "textile_trash (held out, never seen in training)",
        "samples": int(len(df)),
        "threshold": float(threshold),
        "mean_confidence": float(df["confidence"].mean()),
        "reject_rate": rejection_rate,
        "wrongly_accepted": int((df["decision"] == "ACCEPT").sum()),
    }
    save_json(summary, run_dir / "ood_summary.json")

    print(f"OOD/Textile samples: {len(df)}")
    print(f"Threshold: {threshold:.2f}")
    print(
        f"OOD rejection rate: "
        f"{rejection_rate:.4f}"
    )


if __name__ == "__main__":
    main()
