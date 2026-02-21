from __future__ import annotations

import argparse
import json

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"scene_id", "future"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    try:
        json.loads(df["future"].iloc[0])
        print("Detected JSON-encoded future column.")
    except Exception:
        print("Future column is not JSON-encoded; ensure format matches official schema.")

    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
