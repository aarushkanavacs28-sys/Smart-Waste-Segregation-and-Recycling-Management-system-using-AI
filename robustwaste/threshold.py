from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from .data import build_loaders
from .models import load_model_from_checkpoint
from .utils import get_device, save_json, checkpoint_run_dir


@torch.no_grad()
def validation_predictions(model, loader, device):
    y_true, y_pred, confs = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        y_true.extend(y.tolist())
        y_pred.extend(pred.cpu().tolist())
        confs.extend(conf.cpu().tolist())

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(confs),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--target-accepted-accuracy",
        type=float,
        default=0.95,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    model, ckpt = load_model_from_checkpoint(
        args.checkpoint,
        device,
    )

    _, _, _, _, val_loader, _ = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=ckpt["image_size"],
        num_workers=args.num_workers,
        degradation_aware=False,
    )

    y_true, y_pred, confs = validation_predictions(
        model,
        val_loader,
        device,
    )

    rows = []
    for t in np.round(np.arange(0.50, 0.951, 0.01), 2):
        accepted = confs >= t
        coverage = float(accepted.mean())

        if accepted.sum() == 0:
            accepted_acc = np.nan
            wrong_accepted = 0
        else:
            accepted_acc = float(
                (y_true[accepted] == y_pred[accepted]).mean()
            )
            wrong_accepted = int(
                (y_true[accepted] != y_pred[accepted]).sum()
            )

        rows.append({
            "threshold": float(t),
            "coverage": coverage,
            "reject_rate": 1.0 - coverage,
            "accepted_accuracy": accepted_acc,
            "wrong_accepted": wrong_accepted,
            "accepted_count": int(accepted.sum()),
            "total_count": len(y_true),
        })

    df = pd.DataFrame(rows)
    run_dir = checkpoint_run_dir(args.checkpoint)
    df.to_csv(
        run_dir / "threshold_scan.csv",
        index=False,
    )

    feasible = df[
        (df["accepted_accuracy"].notna()) &
        (
            df["accepted_accuracy"]
            >= args.target_accepted_accuracy
        )
    ].copy()

    if len(feasible) > 0:
        selected = feasible.sort_values(
            ["coverage", "accepted_accuracy"],
            ascending=[False, False],
        ).iloc[0]
        target_met = True
    else:
        valid = df[
            df["accepted_accuracy"].notna()
        ].copy()
        selected = valid.sort_values(
            ["accepted_accuracy", "coverage"],
            ascending=[False, False],
        ).iloc[0]
        target_met = False

    out = {
        "selected_threshold": float(
            selected["threshold"]
        ),
        "validation_coverage": float(
            selected["coverage"]
        ),
        "validation_reject_rate": float(
            selected["reject_rate"]
        ),
        "validation_accepted_accuracy": float(
            selected["accepted_accuracy"]
        ),
        "target_accepted_accuracy": (
            args.target_accepted_accuracy
        ),
        "target_met": target_met,
        "selection_rule": (
            "maximum validation coverage while meeting target accepted accuracy"
            if target_met
            else "best available accepted accuracy because target was not reached"
        ),
    }

    save_json(
        out,
        run_dir / "threshold.json",
    )
    print(pd.DataFrame([out]).to_string(index=False))


if __name__ == "__main__":
    main()
