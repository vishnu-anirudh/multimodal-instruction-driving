from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

FILES = [
    "doScenesAnnotations - Annotator 1.csv",
    "doScenesAnnotations - Annotator 2.csv",
    "doScenesAnnotations - Annotator 3.csv",
    "doScenesAnnotations - Annotator 4.csv",
    "doScenesAnnotations - Annotator 5.csv",
    "doScenesAnnotations - Annotator 6.csv",
]

DEFAULT_BASE_URL = "https://raw.githubusercontent.com/rossgreer/doScenes/main/Annotations"


def normalize_base_url(base_url: str) -> str:
    if "github.com" in base_url and "/tree/" in base_url:
        # Convert GitHub tree URL to raw content base.
        base_url = base_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        base_url = base_url.replace("/tree/", "/")
    return base_url.rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = normalize_base_url(args.base_url)
    for name in FILES:
        url = f"{base_url}/{name.replace(' ', '%20')}"
        dst = out_dir / name
        print(f"Downloading {url} -> {dst}")
        urlretrieve(url, dst)


if __name__ == "__main__":
    main()
