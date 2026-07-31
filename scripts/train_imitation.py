"""Train the imitation model (Phase 2's originally-planned second
milestone) on the cached action dataset built by
scripts/build_action_dataset.py.

Check the printed "train: N states" line for the dataset's actual current
size rather than assuming a number. Watch val_loss: if it stops improving
or rises while train_loss keeps dropping, that's overfitting - see
battle_engine/win_prob.py's docstring for why that's expected on this
project's still-small dataset, same reasoning applies here.

    .venv/bin/python scripts/train_imitation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from battle_engine.imitation import ImitationModel, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-sizes", type=str, default="128,64",
        help="comma-separated hidden layer sizes, e.g. '128,64' for two layers",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device", default="cpu",
        help="cpu or mps - cpu is likely faster at this dataset's current scale "
             "(MPS transfer/kernel-launch overhead isn't worth it yet)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/models/imitation.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    hidden_sizes = tuple(int(s) for s in args.hidden_sizes.split(","))

    train_data = np.load(args.dataset_dir / "train_actions.npz")
    val_data = np.load(args.dataset_dir / "val_actions.npz")
    print(f"train: {train_data['X'].shape[0]} states, val: {val_data['X'].shape[0]} states")

    model = ImitationModel(hidden_sizes=hidden_sizes, dropout=args.dropout)
    history, best_state = train(
        model,
        train_data["X"], train_data["y"],
        val_data["X"], val_data["y"],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, device=args.device,
    )

    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "hidden_sizes": hidden_sizes,
        "dropout": args.dropout,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)

    best = min(history, key=lambda m: m.val_loss)
    best_epoch = history.index(best) + 1
    print(f"\nSaved best-epoch model to {args.out}")
    print(
        f"Best val_loss={best.val_loss:.4f} at epoch {best_epoch}/{len(history)} "
        f"(val_top1_acc={best.val_top1_accuracy:.3f})"
    )
    if history[-1].val_loss > best.val_loss * 1.1:
        print("Note: training continued well past the best epoch (val_loss rose "
              "notably afterward) - the saved checkpoint is from the best epoch, "
              "not the final one.")


if __name__ == "__main__":
    main()
