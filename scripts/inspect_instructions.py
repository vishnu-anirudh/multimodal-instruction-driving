from __future__ import annotations

import argparse
from collections import Counter

from doscenes.data.instructions import load_instruction_csvs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    args = parser.parse_args()

    df = load_instruction_csvs(args.csv_dir)
    print(f"Total rows: {len(df)}")
    print("Instruction type counts:")
    print(df["instruction_type"].fillna("").value_counts().to_string())

    words = Counter()
    for text in df["instruction"].fillna("").astype(str).tolist():
        words.update(w.strip(".,").lower() for w in text.split())
    print("\nTop words:")
    for word, count in words.most_common(20):
        if word:
            print(f"{word}: {count}")


if __name__ == "__main__":
    main()
