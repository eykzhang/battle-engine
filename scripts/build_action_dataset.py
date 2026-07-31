"""Build a cached (state-vector, action-label) dataset from fetched replay
files, for the imitation model (see scripts/fetch_replay_sample.py to fetch
replays first). Uses the same train/val split as scripts/build_dataset.py
(same default seed) - both models are evaluated on the same held-out
battles.

    .venv/bin/python scripts/build_action_dataset.py --replay-dir data/replays_raw
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from battle_engine.dataset import build_action_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=Path("data/replays_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (x_train, y_train), (x_val, y_val) = build_action_dataset(
        args.replay_dir, args.val_fraction, args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "train_actions.npz", X=x_train, y=y_train)
    np.savez(args.out_dir / "val_actions.npz", X=x_val, y=y_val)

    train_dist = Counter(y_train.tolist())
    print(f"train: {x_train.shape[0]} states, val: {x_val.shape[0]} states")
    print(f"train action distribution: {dict(sorted(train_dist.items()))}")


if __name__ == "__main__":
    main()
