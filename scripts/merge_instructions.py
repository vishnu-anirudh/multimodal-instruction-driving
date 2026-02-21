from __future__ import annotations

import argparse
from pathlib import Path

from doscenes.data.instructions import load_instruction_csvs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    df = load_instruction_csvs(args.csv_dir)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote merged instructions to {out_path}")


if __name__ == "__main__":
    main()
