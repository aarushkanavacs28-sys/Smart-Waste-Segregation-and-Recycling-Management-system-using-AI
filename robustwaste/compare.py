"""
compare.py  -  builds every comparison table needed for the paper in ONE command.

It loads the four trained runs (ResNet50, EfficientNet-B0, MobileNetV3 baseline,
proposed RobustWaste MobileNetV3) and produces:

    Table 1  clean test performance + model size + latency
    Table 2  robustness (accuracy / macro-F1) under low light, blur, noise, occlusion
    Table 3  confidence-aware ACCEPT/REJECT behaviour under every condition
    Table 4  unknown-waste (held-out textile) rejection
    Ablation A/B/C/D  (normal vs degradation-aware training, with/without reject)

Rules kept from the experiment plan:
    * the confidence threshold of EVERY model is chosen on the VALIDATION set only
    * the test set is used only for the final numbers
    * nothing is hard-coded: all numbers come from the trained checkpoints
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets

from .data import build_transforms
from .models import load_model_from_checkpoint, count_parameters
from .ood import ImagePathDataset
from .robustness import make_transform
from .utils import get_device, ensure_dir, model_file_size_mb

CONDITIONS = ["normal", "low_light", "blur", "noise", "occlusion"]

# label shown in the tables -> run folder name inside outputs/
DEFAULT_RUNS = {
    "ResNet50": "resnet50_baseline",
    "EfficientNet-B0": "efficientnet_b0_baseline",
    "MobileNetV3 (baseline)": "mobilenet_v3_baseline",
    "RobustWaste (proposed)": "robustwaste_proposed",
}


def batch_probs(model, x, tta):
    """Class probabilities for a batch. With TTA the image and its mirror image are
    both classified and the two probability vectors are averaged (2x inference cost)."""
    probs = torch.softmax(model(x), dim=1)
    if tta:
        probs = (probs + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2
    return probs


@torch.no_grad()
def get_predictions(model, loader, device, tta=False):
    """Run the model on a loader and return (true labels, predicted labels, confidences)."""
    y_true, y_pred, conf_all = [], [], []
    for x, y in loader:                                   # one batch at a time
        probs = batch_probs(model, x.to(device), tta)      # logits -> probabilities
        conf, pred = probs.max(dim=1)                      # confidence = highest probability
        y_true += list(y) if not torch.is_tensor(y) else y.tolist()
        y_pred += pred.cpu().tolist()
        conf_all += conf.cpu().tolist()
    return np.array(y_true), np.array(y_pred), np.array(conf_all)


def pick_threshold(y, pred, conf, target):
    """Validation-only threshold: highest coverage that still reaches the target accepted accuracy."""
    best_t, best_cov, fallback_t, fallback_acc = None, -1.0, 0.5, -1.0
    for t in np.round(np.arange(0.50, 0.951, 0.01), 2):    # scan 0.50 ... 0.95
        keep = conf >= t                                    # samples the system would ACCEPT
        if keep.sum() == 0:
            continue
        acc = float((y[keep] == pred[keep]).mean())         # accuracy of accepted samples only
        cov = float(keep.mean())                            # fraction accepted
        if acc >= target and cov > best_cov:                # meets target and accepts more
            best_t, best_cov = float(t), cov
        if acc > fallback_acc:                              # remember best accuracy as a fallback
            fallback_t, fallback_acc = float(t), acc
    return (best_t if best_t is not None else fallback_t), best_t is not None


def reject_stats(y, pred, conf, t):
    """ACCEPT/REJECT statistics for one threshold."""
    keep = conf >= t
    wrong = int((y[keep] != pred[keep]).sum())              # wrong samples that were still accepted
    return {
        "coverage": float(keep.mean()),
        "reject_rate": float(1 - keep.mean()),
        "accepted_accuracy": float((y[keep] == pred[keep]).mean()) if keep.any() else float("nan"),
        "wrong_accepted": wrong,
        "wrong_accepted_rate": wrong / len(y),              # unsafe errors over ALL samples
    }


def cpu_latency_ms(model, image_size, runs=30):
    """Average single-image CPU latency (what an edge device would see)."""
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        for _ in range(5):                                  # warm-up so timing is stable
            model(x)
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
    return (time.perf_counter() - start) / runs * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--severity", type=float, default=0.65)
    ap.add_argument("--target-accepted-accuracy", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--latency-runs", type=int, default=30)
    # e.g. --suffix _trashnet  ->  reads runs named resnet50_baseline_trashnet ...
    #                              and writes outputs/comparison_trashnet
    ap.add_argument("--suffix", default="")
    # Test-time augmentation: average the prediction of each image and its mirror image.
    # Apply to ALL models (it is done in this one script) and say so in the paper.
    ap.add_argument("--tta", action="store_true")
    # Extra model variants to include in every table, written as "label=run_name".
    # Example: --extra "RobustWaste-R50 (accuracy variant)=robustwaste_r50"
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    device = get_device(args.device)
    out = ensure_dir(Path(args.outputs_dir) / f"comparison{args.suffix}")
    data_dir = Path(args.data_dir)

    t1, t2, t3, t4 = [], [], [], []   # rows for tables 1-4
    cache = {}                        # cache[label][condition] = (y, pred, conf) for the ablation

    runs = dict(DEFAULT_RUNS)                       # the four standard models ...
    for item in args.extra:                         # ... plus any extra variants
        label_x, run_x = item.split("=", 1)
        runs[label_x.strip()] = run_x.strip()

    for label, run in runs.items():
        ckpt_path = Path(args.outputs_dir) / f"{run}{args.suffix}" / "best.pt"
        if not ckpt_path.exists():
            print(f"[skip] {label}: {ckpt_path} not found (train it first)")
            continue

        print(f"\n=== {label} ===")
        model, ckpt = load_model_from_checkpoint(str(ckpt_path), device)
        size = ckpt["image_size"]
        classes = ckpt["classes"]
        cache[label] = {}

        def loader_for(folder, transform):
            ds = datasets.ImageFolder(data_dir / folder, transform=transform)
            if ds.classes != classes:
                raise RuntimeError("Dataset classes do not match checkpoint classes.")
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

        # ---- threshold from the VALIDATION set (never from the test set) ----
        yv, pv, cv = get_predictions(model, loader_for("val", build_transforms(size)), device, args.tta)
        thr, met = pick_threshold(yv, pv, cv, args.target_accepted_accuracy)
        print(f"selected threshold = {thr:.2f} (target met on validation: {met})")

        # ---- every test condition ----
        for cond in CONDITIONS:
            sev = 0.0 if cond == "normal" else args.severity
            y, p, c = get_predictions(model, loader_for("test", make_transform(size, cond, sev)), device, args.tta)
            cache[label][cond] = (y, p, c)
            pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
            acc = accuracy_score(y, p)

            if cond == "normal":                            # Table 1 uses clean images only
                t1.append({
                    "model": label, "accuracy": acc, "macro_precision": pr,
                    "macro_recall": rc, "macro_f1": f1,
                    "parameters": count_parameters(model),
                    "model_size_mb": model_file_size_mb(model),
                    "cpu_latency_ms": cpu_latency_ms(model, size, args.latency_runs) * (2 if args.tta else 1),
                })
                model.to(device)                            # cpu_latency_ms moved it to CPU
            t2.append({"model": label, "condition": cond, "accuracy": acc, "macro_f1": f1})

            r = reject_stats(y, p, c, thr)                  # Table 3: confidence-aware decision
            t3.append({"model": label, "condition": cond, "threshold": thr, **r})

        # ---- Table 4: unknown waste (held-out textile) ----
        ood = ImagePathDataset(data_dir / "ood_textile", build_transforms(size))
        if len(ood) > 0:
            oc = []
            with torch.no_grad():
                for x, _ in DataLoader(ood, batch_size=args.batch_size, num_workers=args.num_workers):
                    oc += batch_probs(model, x.to(device), args.tta).max(dim=1)[0].cpu().tolist()
            oc = np.array(oc)
            t4.append({
                "model": label, "ood_samples": len(oc), "threshold": thr,
                "mean_confidence": float(oc.mean()),
                "reject_rate": float((oc < thr).mean()),
                "wrongly_accepted": int((oc >= thr).sum()),
            })

    if not t1:
        raise SystemExit("No trained runs found. Train the models first (see README).")

    # ---- save tables ----
    df1, df2, df3, df4 = map(pd.DataFrame, (t1, t2, t3, t4))
    # robustness drop relative to the clean accuracy of the same model
    clean = df2[df2.condition == "normal"].set_index("model")["accuracy"]
    df2["accuracy_drop_vs_normal"] = df2.apply(lambda r: clean[r["model"]] - r["accuracy"], axis=1)
    for name, df in [("table1_clean", df1), ("table2_robustness", df2),
                     ("table3_confidence_aware", df3), ("table4_ood", df4)]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n{name}\n{df.round(4).to_string(index=False)}")

    # ---- Ablation A/B/C/D (needs the two MobileNetV3 runs) ----
    base, prop = "MobileNetV3 (baseline)", "RobustWaste (proposed)"
    if base in cache and prop in cache:
        thr_of = {m: df3[df3.model == m].threshold.iloc[0] for m in (base, prop)}
        rows = []
        for tag, model_label, use_reject in [("A", base, False), ("B", prop, False),
                                             ("C", base, True), ("D", prop, True)]:
            accs, covs, wrongs = [], [], []
            for cond in CONDITIONS[1:]:                     # the 4 degraded conditions
                y, p, c = cache[model_label][cond]
                if use_reject:
                    r = reject_stats(y, p, c, thr_of[model_label])
                    accs.append(r["accepted_accuracy"]); covs.append(r["coverage"])
                    wrongs.append(r["wrong_accepted_rate"])
                else:                                       # no reject: everything is accepted
                    accs.append(accuracy_score(y, p)); covs.append(1.0)
                    wrongs.append(float((y != p).mean()))
            rows.append({
                "config": tag,
                "training": "degradation-aware" if model_label == prop else "normal",
                "reject": "yes" if use_reject else "no",
                "mean_degraded_accuracy_of_accepted": float(np.nanmean(accs)),
                "mean_coverage": float(np.mean(covs)),
                "mean_wrong_accepted_rate": float(np.mean(wrongs)),
            })
        df5 = pd.DataFrame(rows)
        df5.to_csv(out / "table5_ablation.csv", index=False)
        print(f"\ntable5_ablation\n{df5.round(4).to_string(index=False)}")

    # ---- figures ----
    pivot = df2.pivot(index="condition", columns="model", values="accuracy").reindex(CONDITIONS)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1); ax.set_xlabel("Test condition")
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(out / "fig_robustness_comparison.png", dpi=200); plt.close()

    pivot = df3.pivot(index="condition", columns="model", values="wrong_accepted_rate").reindex(CONDITIONS)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Wrongly accepted / all samples"); ax.set_xlabel("Test condition")
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(out / "fig_wrong_accepted_comparison.png", dpi=200); plt.close()

    print(f"\nAll tables and figures saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
