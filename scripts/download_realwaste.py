from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_URL = (
    "https://archive.ics.uci.edu/"
    "static/public/908/realwaste.zip"
)


def download(url: str, output: Path):
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with requests.get(
        url,
        stream=True,
        timeout=60,
    ) as r:
        r.raise_for_status()
        total = int(
            r.headers.get(
                "content-length",
                0,
            )
        )

        with output.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="Downloading RealWaste",
        ) as pbar:
            for chunk in r.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    f.write(chunk)
                    pbar.update(
                        len(chunk)
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/realwaste",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    zip_path = (
        output_dir.parent
        / "realwaste.zip"
    )

    if not zip_path.exists():
        download(
            args.url,
            zip_path,
        )
    else:
        print(
            f"Using existing archive: "
            f"{zip_path}"
        )

    print("Extracting...")
    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:
        zf.extractall(output_dir)

    if not args.keep_zip:
        zip_path.unlink(
            missing_ok=True
        )

    print(
        f"Dataset extracted to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
