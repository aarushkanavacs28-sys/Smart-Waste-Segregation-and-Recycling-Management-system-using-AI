from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .data import build_loaders
from .models import build_model, count_parameters
from .utils import (
    set_seed,
    get_device,
    ensure_dir,
    save_json,
    model_file_size_mb,
)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_y, all_pred = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())

        pred = logits.argmax(dim=1)
        all_y.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": accuracy_score(all_y, all_pred),
        "macro_f1": f1_score(
            all_y,
            all_pred,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument(
        "--model",
        default="mobilenet_v3_large",
        choices=[
            "resnet50",
            "efficientnet_b0",
            "mobilenet_v3_large",
            "convnext_tiny",
            "efficientnet_v2_s",
        ],
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--degradation-aware", action="store_true")
    parser.add_argument("--patience", type=int, default=6)
    # Useful for offline machines / quick smoke tests (no ImageNet weight download).
    parser.add_argument("--no-pretrained", action="store_true")
    # Label smoothing (e.g. 0.1) stops the model from becoming over-confident and
    # usually gives a small, free accuracy gain. 0.0 = off (the old behaviour).
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    (
        train_ds,
        val_ds,
        _test_ds,
        train_loader,
        val_loader,
        _test_loader,
    ) = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        degradation_aware=args.degradation_aware,
    )

    model = build_model(
        args.model,
        len(train_ds.classes),
        pretrained=not args.no_pretrained,
    ).to(device)

    counts = torch.bincount(
        torch.tensor(train_ds.targets),
        minlength=len(train_ds.classes),
    ).float()
    weights = counts.sum() / (len(counts) * counts.clamp_min(1.0))
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=args.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    run_dir = ensure_dir(Path("outputs") / args.run_name)
    history = []
    best_f1 = -1.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_y, train_pred = [], []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        )

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pred = logits.argmax(dim=1)
            train_y.extend(y.detach().cpu().tolist())
            train_pred.extend(pred.detach().cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_metrics = {
            "loss": sum(train_losses) / max(1, len(train_losses)),
            "accuracy": accuracy_score(train_y, train_pred),
            "macro_f1": f1_score(
                train_y,
                train_pred,
                average="macro",
                zero_division=0,
            ),
        }

        val_metrics = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step(val_metrics["macro_f1"])

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        pd.DataFrame(history).to_csv(
            run_dir / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch}: "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            no_improve = 0

            torch.save({
                "model_name": args.model,
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "image_size": args.image_size,
                "degradation_aware": args.degradation_aware,
                "seed": args.seed,
                "best_val_macro_f1": best_f1,
            }, run_dir / "best.pt")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print("Early stopping.")
            break

    save_json({
        "run_name": args.run_name,
        "model": args.model,
        "device": str(device),
        "classes": train_ds.classes,
        "degradation_aware": args.degradation_aware,
        "best_val_macro_f1": best_f1,
        "parameters": count_parameters(model),
        "model_state_size_mb": model_file_size_mb(model),
    }, run_dir / "train_summary.json")

    print(f"Best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
