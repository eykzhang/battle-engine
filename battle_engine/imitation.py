"""Imitation model: state vector -> human's action — Phase 2's originally-
planned second milestone (the roadmap's "move-prediction / imitation model
(state -> human's move): doubles as an opponent-modeling prior and later as
an RL policy init"), built 2026-07-31 after the win-probability model
(win_prob.py) and its milestone-E integration, once it became clear the
project couldn't call Phase 2 complete without it.

Action space: battle_engine.dataset.ACTION_SPACE_SIZE (13 classes: 4 move
slots, 5 switch slots, 4 move-slots-while-terastallized) - Metamon's own
replay-label scheme, not poke-env's Gymnasium action space (a real,
deliberately deferred reconciliation - see dataset.py's ACTION_SPACE_SIZE
docstring). This model is a straight multi-class classifier over that
scheme; nothing here masks illegal actions at train or predict time (e.g.
scoring a switch-slot the current bench doesn't actually have 5 of) - the
model has to learn legality from the data's own patterns, same as a human
imitation-learning setup typically starts from before any legality masking
is layered on top.

Same architecture family as WinProbModel (see that module's docstring for
the two-hidden-layer/dropout/Adam reasoning - equally applicable here,
same input vector, same overfitting-prone small-dataset regime), swapped to
a softmax classification head: CrossEntropyLoss (combines log-softmax +
negative-log-likelihood in one numerically stable op, same reasoning as
BCEWithLogitsLoss on the win-prob side) over ACTION_SPACE_SIZE logits
instead of a single win-probability logit.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.encoding import VECTOR_LEN


class ImitationModel(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int] = (128, 64), dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        in_size = VECTOR_LEN
        for size in hidden_sizes:
            layers += [nn.Linear(in_size, size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = size
        layers.append(nn.Linear(in_size, ACTION_SPACE_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits, shape (batch, ACTION_SPACE_SIZE) - not probabilities.
        Use predict_proba for that; kept separate since CrossEntropyLoss
        wants raw logits.
        """
        return self.net(x)

    @staticmethod
    def load(path: Path) -> "ImitationModel":
        """Loads a checkpoint saved by scripts/train_imitation.py: a dict of
        {state_dict, hidden_sizes, dropout} - same convention as
        WinProbModel.load, for the same reason (architecture shouldn't have
        to be remembered separately from the CLI flags used to train it).
        """
        checkpoint = torch.load(path, map_location="cpu")
        model = ImitationModel(
            hidden_sizes=checkpoint["hidden_sizes"], dropout=checkpoint["dropout"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities, shape (batch, ACTION_SPACE_SIZE). Restores
        whatever train()/eval() mode the model was already in before
        returning - same dropout-leak fix as WinProbModel.predict_proba.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.softmax(self.forward(x), dim=-1)
        finally:
            self.train(was_training)


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean().item()


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_top1_accuracy: float
    seconds: float


def _iterate_batches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    n = x.shape[0]
    indices = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y[idx]


def train(
    model: ImitationModel,
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[List[EpochMetrics], Dict[str, torch.Tensor]]:
    """Returns (per-epoch history, best-val-loss state dict) - same
    best-checkpoint-tracked-internally shape as win_prob.train, for the same
    reason (a caller that saves the final epoch instead of the best one is
    exactly the bug an earlier review caught on the win-prob side).
    """
    if x_val.shape[0] == 0:
        raise ValueError(
            "empty validation set - val_loss/accuracy would silently be "
            "NaN rather than erroring, so refusing instead"
        )

    torch_device = torch.device(device)
    model = model.to(torch_device)

    x_train_t = torch.from_numpy(x_train).float().to(torch_device)
    y_train_t = torch.from_numpy(y_train).long().to(torch_device)
    x_val_t = torch.from_numpy(x_val).float().to(torch_device)
    y_val_t = torch.from_numpy(y_val).long().to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history: List[EpochMetrics] = []
    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        start = time.perf_counter()

        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb in _iterate_batches(x_train_t, y_train_t, batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        metrics = EpochMetrics(
            train_loss=train_loss_sum / max(n_batches, 1),
            val_loss=val_loss,
            val_top1_accuracy=top1_accuracy(val_logits, y_val_t),
            seconds=time.perf_counter() - start,
        )
        history.append(metrics)
        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs}  train_loss={metrics.train_loss:.4f}  "
                f"val_loss={metrics.val_loss:.4f}  val_top1_acc={metrics.val_top1_accuracy:.3f}  "
                f"({metrics.seconds:.2f}s)"
            )

    return history, best_state
