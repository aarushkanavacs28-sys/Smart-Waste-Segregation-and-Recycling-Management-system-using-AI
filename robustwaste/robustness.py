from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .data import IMAGENET_MEAN, IMAGENET_STD
from .degradations import apply_degradation
from .models import load_model_from_checkpoint
from .utils import get_device, checkpoint_run_dir


class FixedDegradation:
    def __init__(self, kind: str, severity: float):
        self.kind = kind
        self.severity = severity

    def __call__(self, img: Image.Image):
        # Seed from the image itself so the SAME image always gets the SAME degradation.
        # Without this, noise/occlusion are random on every run and results move by ~1 point.
        seed = int(np.asarray(img).sum()) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        return apply_degradation(
            img,
            self.kind,
            self.severity,
        )


def make_transform(
    image_size,
    condition,
    severity,
):
    return transforms.Compose([
        FixedDegradation(condition, severity),
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])


@torch.no_grad()
def evaluate_condition(
    model,
    loader,
    device,
    classes,
):
    y_true, y_pred, confs = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        y_true.extend(y.tolist())
        y_pred.extend(pred.cpu().tolist())
        confs.extend(conf.cpu().tolist())

    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    df = pd.DataFrame({
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_class": [classes[i] for i in y_true],
        "pred_class": [classes[i] for i in y_pred],
        "confidence": confs,
    })

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "mean_confidence": float(
            pd.Series(confs).mean()
        ),
    }, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--severity", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    model, ckpt = load_model_from_checkpoint(
        args.checkpoint,
        device,
    )
    run_dir = checkpoint_run_dir(args.checkpoint)

    conditions = [
        "normal",
        "low_light",
        "blur",
        "noise",
        "occlusion",
    ]
    rows = []

    for condition in conditions:
        ds = datasets.ImageFolder(
            Path(args.data_dir) / "test",
            transform=make_transform(
                ckpt["image_size"],
                condition,
                args.severity,
            ),
        )

        if ds.classes != ckpt["classes"]:
            raise RuntimeError(
                "Dataset classes do not match checkpoint classes."
            )

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        metrics, pred_df = evaluate_condition(
            model,
            loader,
            device,
            ckpt["classes"],
        )
        metrics["condition"] = condition
        metrics["severity"] = (
            0.0
            if condition == "normal"
            else args.severity
        )
        rows.append(metrics)

        pred_df.to_csv(
            run_dir
            / f"predictions_robustness_{condition}.csv",
            index=False,
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        run_dir / "robustness_results.csv",
        index=False,
    )
    print(results.to_string(index=False))

    fig = plt.figure(figsize=(8, 5))
    plt.bar(
        results["condition"],
        results["accuracy"],
    )
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Test condition")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(
        run_dir / "robustness_accuracy.png",
        dpi=200,
    )
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.bar(
        results["condition"],
        results["macro_f1"],
    )
    plt.ylim(0, 1)
    plt.ylabel("Macro F1-score")
    plt.xlabel("Test condition")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(
        run_dir / "robustness_f1.png",
        dpi=200,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
