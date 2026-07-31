"""Win-probability model: state vector -> P(win) — the Phase-2 "learned eval"
that will (milestone E) replace evaluation.py's hand-crafted evaluate() call
inside search.py's scoring function.

Deliberately small: `WinProbModel` is one hidden layer, sized for a dataset
(scripts/build_dataset.py) that's still tiny relative to the 316-dim input -
check `data/dataset/{train,val}.npz`'s actual current shape rather than
trusting a specific number here, since it changes as more replays get
fetched (a stale "~1.7k states" claim here is exactly what an earlier review
caught). Watch val_loss vs train_loss in the printed history: if train keeps
dropping while val stalls or rises, that's overfitting.

Architecture and training choices, explained once here rather than repeated
in comments below:
- Two hidden layers (ReLU) by default, not one — bumped from a single
  64-unit layer after adding item/moveset features (encoding.py, 2026-07-31)
  took the input from 316 to 656 dims: measured directly, the old single-
  layer architecture didn't just fail to benefit from the new dimensions, it
  got *worse* (val_acc 0.651->0.636, and a head-to-head benchmark dropped
  32.0% from 38.4%) - more input dimensions gave a fixed-capacity net more
  ways to overfit the same ~800k training rows, not more signal to use.
  Nonlinearity (any hidden layer) is what lets the model represent feature
  *interactions* (e.g. "type matchup matters more at low HP") that a linear
  model (equivalent to evaluate()'s weighted sum) structurally can't; a
  second layer lets it compose those interactions further (e.g. "matchup x
  HP x whether the switch-in resists the active hazard").
- Dropout (randomly zeroes hidden units during training only) and Adam's
  weight_decay (an L2 penalty added to the loss) are both regularization —
  ways to discourage the model from memorizing training-set idiosyncrasies,
  chosen specifically because our dataset is small enough to overfit easily.
- BCEWithLogitsLoss: combines the sigmoid squash and binary cross-entropy
  loss into one numerically stable op (computing them as separate steps can
  quietly hit log(0) in edge cases) — the model itself outputs raw logits,
  not probabilities; sigmoid is only applied when we want an actual P(win).
- Adam optimizer: the standard default — adapts its effective step size per
  weight, more forgiving of an imperfectly-tuned learning rate than plain
  gradient descent.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from battle_engine.encoding import VECTOR_LEN, battle_view_from_poke_env, encode


class WinProbModel(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int] = (128, 64), dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        in_size = VECTOR_LEN
        for size in hidden_sizes:
            layers += [nn.Linear(in_size, size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = size
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits, shape (batch,) — not probabilities. Use predict_proba
        for that; kept separate since BCEWithLogitsLoss wants raw logits.
        """
        return self.net(x).squeeze(-1)

    @staticmethod
    def load(path: Path) -> "WinProbModel":
        """Loads a checkpoint saved by scripts/train_win_prob.py: a dict of
        {state_dict, hidden_sizes, dropout}, not a bare state_dict, so the
        architecture doesn't have to be remembered/guessed separately from
        the CLI flags used to train it (a gap review pointed out).
        """
        checkpoint = torch.load(path, map_location="cpu")
        model = WinProbModel(
            hidden_sizes=checkpoint["hidden_sizes"], dropout=checkpoint["dropout"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Restores whatever train()/eval() mode the model was already in
        before returning - review found this previously left the model
        permanently in eval() mode (dropout silently off) if called mid-
        training-loop; harmless today only because train() unconditionally
        re-calls model.train() at the start of every epoch.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.sigmoid(self.forward(x))
        finally:
            self.train(was_training)


def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (probs >= 0.5).float()
    return (predictions == labels).float().mean().item()


def calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Mean, over probability bins with at least one example, of
    |average predicted probability in that bin - actual win rate in that
    bin|. A well-calibrated model's "70% confident" predictions should win
    about 70% of the time; this is the number that would catch it if not.
    Unweighted by bin size (simple, and fine for our small val set) rather
    than the more standard sample-weighted "expected calibration error".
    """
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    errors = []
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        # Real bug caught by review: [lo, hi) for every bin left a
        # prediction of exactly 1.0 in no bin at all (silently dropped from
        # the metric, from the single worst-possible position - a confident
        # wrong prediction). torch.sigmoid returns exactly 1.0 in float32
        # for any logit >= ~17, which a bigger/longer training run reaches.
        # Only the last bin is closed on the right for this reason.
        is_last = i == len(bin_edges) - 2
        mask = (probs_np >= lo) & ((probs_np <= hi) if is_last else (probs_np < hi))
        if not mask.any():
            continue
        errors.append(abs(probs_np[mask].mean() - labels_np[mask].mean()))
    return float(np.mean(errors)) if errors else 0.0


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_calibration_error: float
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
    model: WinProbModel,
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[List[EpochMetrics], Dict[str, torch.Tensor]]:
    """Returns (per-epoch history, best-val-loss state dict).

    Review caught a real bug in the original version of this pipeline:
    scripts/train_win_prob.py was saving the model's *final*-epoch weights
    while printing "best val_loss at epoch k" - since this model reliably
    overfits well before 30 epochs (see module docstring), that meant the
    single worst checkpoint of the run was the one actually persisted to
    disk. train() now tracks and returns the best-val-loss state dict
    itself, so the caller can't repeat that mistake by construction.
    """
    if x_val.shape[0] == 0:
        raise ValueError(
            "empty validation set - val_loss/accuracy/calibration would "
            "silently be NaN rather than erroring, so refusing instead"
        )

    torch_device = torch.device(device)
    model = model.to(torch_device)

    x_train_t = torch.from_numpy(x_train).float().to(torch_device)
    y_train_t = torch.from_numpy(y_train).float().to(torch_device)
    x_val_t = torch.from_numpy(x_val).float().to(torch_device)
    y_val_t = torch.from_numpy(y_val).float().to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

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
            val_probs = torch.sigmoid(val_logits)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        metrics = EpochMetrics(
            train_loss=train_loss_sum / max(n_batches, 1),
            val_loss=val_loss,
            val_accuracy=accuracy(val_probs, y_val_t),
            val_calibration_error=calibration_error(val_probs, y_val_t),
            seconds=time.perf_counter() - start,
        )
        history.append(metrics)
        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs}  train_loss={metrics.train_loss:.4f}  "
                f"val_loss={metrics.val_loss:.4f}  val_acc={metrics.val_accuracy:.3f}  "
                f"val_cal_err={metrics.val_calibration_error:.3f}  ({metrics.seconds:.2f}s)"
            )

    return history, best_state


def make_eval_fn(model: WinProbModel) -> Callable[[object], float]:
    """Adapts a trained WinProbModel into the same shape evaluation.evaluate()
    has (battle-like -> score, higher favors that battle's own perspective),
    so it drops directly into search.TwoPlySearchPlayer's eval_fn parameter
    (milestone E). P(win) in [0, 1] is already a valid score for
    max-ranking candidates within one search call - no rescaling needed,
    only relative order matters.

    Don't reuse switch_urgency_weight's default (tuned for evaluate()'s
    roughly [-4, 4] scale) unchanged with this eval_fn - it would swamp a
    [0, 1] probability output. But don't zero it either: replay-inspection
    diagnosis found the learned eval, with no switch-urgency term at all,
    undervalues switching away from a critically low-HP Pokemon relative to
    attacking with it - the same blind spot the switch-urgency patch exists
    to fix for evaluate(), just silently unfixed for this eval_fn. See
    scripts/benchmark.py's LEARNED_SWITCH_URGENCY_WEIGHT for a swept value
    on this scale.
    """
    model.eval()

    def eval_fn(battle) -> float:
        vector = encode(battle_view_from_poke_env(battle))
        x = torch.from_numpy(vector).float().unsqueeze(0)
        return model.predict_proba(x).item()

    return eval_fn
