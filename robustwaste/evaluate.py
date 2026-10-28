from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from .data import build_loaders
from .models import load_model_from_checkpoint, count_parameters
from .utils import (
    get_device,
    save_json,
    checkpoint_run_dir,
    model_file_size_mb,
)


@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true, y_pred, confs = [], [], []
    latency_ms = []

    for x, _ in loader:
        x = x.to(device)
        for _ in range(3):
            _ = model(x[:1])
        break

    for x, y in loader:
        x = x.to(device)

        start = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        per_image_ms = (elapsed / x.size(0)) * 1000.0
        latency_ms.extend([per_image_ms] * x.size(0))

        y_true.extend(y.tolist())
        y_pred.extend(pred.cpu().tolist())
        confs.extend(conf.cpu().tolist())

    return y_true, y_pred, confs, latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    model, ckpt = load_model_from_checkpoint(
        args.checkpoint,
        device,
    )

    _, _, _, _, val_loader, test_loader = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=ckpt["image_size"],
        num_workers=args.num_workers,
        degradation_aware=False,
    )
    loader = val_loader if args.split == "val" else test_loader

    y_true, y_pred, confs, latency_ms = collect_predictions(
        model,
        loader,
        device,
    )
    classes = ckpt["classes"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "split": args.split,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "mean_confidence": float(np.mean(confs)),
        "mean_inference_latency_ms_per_image": float(np.mean(latency_ms)),
        "parameters": count_parameters(model),
        "model_state_size_mb": model_file_size_mb(model),
    }

    run_dir = checkpoint_run_dir(args.checkpoint)
    save_json(
        metrics,
        run_dir / f"metrics_{args.split}.json",
    )

    pd.DataFrame({
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_class": [classes[i] for i in y_true],
        "pred_class": [classes[i] for i in y_pred],
        "confidence": confs,
        "latency_ms": latency_ms,
    }).to_csv(
        run_dir / f"predictions_{args.split}.csv",
        index=False,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(
        cm,
        display_labels=classes,
    ).plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(
        run_dir / f"confusion_matrix_{args.split}.png",
        dpi=200,
    )
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
