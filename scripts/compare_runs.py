from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", type=str, required=True)
    parser.add_argument("--instruction_metrics", type=str, required=True)
    args = parser.parse_args()

    baseline = load_metrics(Path(args.baseline_metrics))
    instruction = load_metrics(Path(args.instruction_metrics))
    delta_ade = baseline["ade"] - instruction["ade"]
    delta_q975 = baseline["q975_ade"] - instruction["q975_ade"]

    print("Delta ADE:", round(delta_ade, 4))
    print("Delta Q97.5 ADE:", round(delta_q975, 4))


if __name__ == "__main__":
    main()
