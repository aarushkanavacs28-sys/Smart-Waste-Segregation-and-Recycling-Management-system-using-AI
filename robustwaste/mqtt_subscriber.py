from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import paho.mqtt.client as mqtt


def append_row(row, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument(
        "--topic",
        default="robustwaste/events",
    )
    parser.add_argument(
        "--csv",
        default="outputs/iot_events.csv",
    )
    args = parser.parse_args()

    def on_connect(
        client,
        userdata,
        flags,
        reason_code,
        properties=None,
    ):
        print(f"Connected: {reason_code}")
        client.subscribe(args.topic)

    def on_message(
        client,
        userdata,
        msg,
    ):
        try:
            row = json.loads(
                msg.payload.decode("utf-8")
            )
            print(
                json.dumps(
                    row,
                    indent=2,
                )
            )
            append_row(
                row,
                args.csv,
            )
        except Exception as exc:
            print(
                f"Failed to process message: "
                f"{exc}"
            )

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2
    )
    client.on_connect = on_connect
    client.on_message = on_message

    print(
        f"Subscribing to "
        f"mqtt://{args.host}:{args.port}/"
        f"{args.topic}"
    )
    client.connect(
        args.host,
        args.port,
        60,
    )
    client.loop_forever()


if __name__ == "__main__":
    main()
