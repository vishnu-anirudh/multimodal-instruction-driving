from __future__ import annotations

import argparse

from doscenes.data.toy_data import make_toy_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--history_steps", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=24)
    parser.add_argument("--map_dim", type=int, default=16)
    args = parser.parse_args()

    out_path = make_toy_dataset(
        out_path=args.out_path,
        num_samples=args.num_samples,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        map_dim=args.map_dim,
    )
    print(f"Wrote toy dataset to {out_path}")


if __name__ == "__main__":
    main()
