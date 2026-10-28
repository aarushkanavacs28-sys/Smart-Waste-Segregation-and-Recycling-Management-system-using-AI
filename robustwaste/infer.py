from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .degradations import apply_degradation
from .models import load_model_from_checkpoint
from .utils import (
    get_device,
    checkpoint_run_dir,
    load_json,
)


def resolve_threshold(checkpoint, explicit):
    if explicit is not None:
        return float(explicit)

    path = (
        checkpoint_run_dir(checkpoint)
        / "threshold.json"
    )
    if path.exists():
        return float(
            load_json(path)["selected_threshold"]
        )
    return 0.75


def append_csv(event, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    exists = csv_path.exists()

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(event.keys()),
        )
        if not exists:
            writer.writeheader()
        writer.writerow(event)


def publish_mqtt(
    event,
    host,
    port,
    topic,
):
    import paho.mqtt.publish as publish

    publish.single(
        topic,
        payload=json.dumps(event),
        hostname=host,
        port=port,
        qos=0,
        retain=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--condition",
        default="normal",
        choices=["normal", "low_light", "blur", "noise", "occlusion"],
        help="Optionally degrade the image first (simulates a bad camera view).",
    )
    parser.add_argument("--severity", type=float, default=0.65)
    parser.add_argument(
        "--csv-log",
        default="outputs/iot_events.csv",
    )
    parser.add_argument("--mqtt-host", default=None)
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument(
        "--mqtt-topic",
        default="robustwaste/events",
    )
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

    transform = build_transforms(
        ckpt["image_size"],
        training=False,
    )

    image_path = Path(args.image)
    with Image.open(image_path) as img:
        degraded = apply_degradation(
            img.convert("RGB"),
            args.condition,
            args.severity,
        )
        x = transform(degraded).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(
            logits,
            dim=1,
        )
        conf, pred = probs.max(dim=1)

    confidence = float(conf.item())
    predicted_class = ckpt["classes"][
        int(pred.item())
    ]
    decision = (
        "ACCEPT"
        if confidence >= threshold
        else "REJECT"
    )

    event = {
        "timestamp": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "predicted_class": predicted_class,
        "confidence": round(
            confidence,
            6,
        ),
        "decision": decision,
        "threshold": round(
            threshold,
            6,
        ),
        "image_condition": args.condition,
        "source_image": image_path.name,
    }

    append_csv(
        event,
        args.csv_log,
    )

    if args.mqtt_host:
        publish_mqtt(
            event,
            args.mqtt_host,
            args.mqtt_port,
            args.mqtt_topic,
        )

    print(
        json.dumps(
            event,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
