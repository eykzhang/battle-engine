"""Build a cached (state-vector, win/loss-label) dataset from fetched replay
files (see scripts/fetch_replay_sample.py to fetch them first).

    .venv/bin/python scripts/build_dataset.py --replay-dir data/replays_raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from battle_engine.dataset import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=Path("data/replays_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (x_train, y_train), (x_val, y_val) = build_dataset(
        args.replay_dir, args.val_fraction, args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "train.npz", X=x_train, y=y_train)
    np.savez(args.out_dir / "val.npz", X=x_val, y=y_val)
    print(
        f"train: {x_train.shape[0]} states ({y_train.mean():.1%} win rate), "
        f"val: {x_val.shape[0]} states ({y_val.mean():.1%} win rate)"
    )


if __name__ == "__main__":
    main()
