import numpy as np
import torch

from battle_engine.encoding import VECTOR_LEN
from battle_engine.win_prob import WinProbModel, accuracy, calibration_error, train


def test_model_forward_pass_returns_one_logit_per_example():
    model = WinProbModel(hidden_sizes=(8,))
    x = torch.zeros(5, VECTOR_LEN)

    logits = model(x)

    assert logits.shape == (5,)


def test_predict_proba_squashes_logits_into_zero_one():
    model = WinProbModel(hidden_sizes=(8,))
    x = torch.randn(10, VECTOR_LEN)

    probs = model.predict_proba(x)

    assert probs.shape == (10,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_accuracy_counts_correct_threshold_predictions():
    probs = torch.tensor([0.9, 0.1, 0.6, 0.4])
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # thresholded at 0.5: 1, 0, 1, 0 vs true 1, 0, 0, 0 -> 3/4 correct
    assert accuracy(probs, labels) == 0.75


def test_calibration_error_is_near_zero_for_well_calibrated_predictions():
    # 10 examples all predicted at 0.7, and empirically 7/10 actually won.
    probs = torch.full((10,), 0.7)
    labels = torch.tensor([1.0] * 7 + [0.0] * 3)

    assert calibration_error(probs, labels, n_bins=10) < 1e-6


def test_calibration_error_is_large_for_overconfident_predictions():
    # Predicted 90% confident, but only actually won half the time.
    probs = torch.full((10,), 0.9)
    labels = torch.tensor([1.0] * 5 + [0.0] * 5)

    assert calibration_error(probs, labels, n_bins=10) > 0.3


def test_calibration_error_ignores_empty_bins():
    # All predictions land in one bin; empty bins shouldn't contribute 0s
    # that would artificially deflate the average error.
    probs = torch.tensor([0.95, 0.96])
    labels = torch.tensor([0.0, 0.0])  # both wrong, badly overconfident

    assert calibration_error(probs, labels, n_bins=10) > 0.9


def test_training_loop_reduces_loss_on_a_trivially_separable_dataset():
    # Standard ML sanity check: can the model overfit an obviously learnable
    # dataset? If not, something in the forward/backward/optimizer wiring is
    # broken. This tests "does the pipeline train at all", not real
    # generalization capability (train and val are the same data on purpose).
    rng = np.random.default_rng(0)
    n = 64
    x = rng.normal(size=(n, VECTOR_LEN)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)  # label = sign of one feature

    model = WinProbModel(hidden_sizes=(16,), dropout=0.0)
    history, _ = train(
        model, x, y, x, y,
        epochs=50, batch_size=16, lr=1e-2, weight_decay=0.0,
        device="cpu", verbose=False,
    )

    assert history[-1].val_loss < history[0].val_loss
    assert history[-1].val_accuracy > 0.95


def test_training_loop_runs_on_real_shaped_tiny_dataset():
    # Guards against shape/dtype mismatches (e.g. y as int vs float, x as
    # float64 vs float32) that a purely-synthetic test above might not
    # exercise the same way real cached .npz data does.
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(20, VECTOR_LEN)).astype(np.float32)
    y_train = rng.integers(0, 2, size=20).astype(np.float32)
    x_val = rng.normal(size=(5, VECTOR_LEN)).astype(np.float32)
    y_val = rng.integers(0, 2, size=5).astype(np.float32)

    model = WinProbModel(hidden_sizes=(8,))
    history, best_state = train(
        model, x_train, y_train, x_val, y_val,
        epochs=2, batch_size=8, device="cpu", verbose=False,
    )

    assert len(history) == 2
    assert all(np.isfinite(m.val_loss) for m in history)
    assert set(best_state.keys()) == set(model.state_dict().keys())


def test_train_returns_the_best_checkpoint_not_the_final_epoch():
    # Real bug caught by review: scripts/train_win_prob.py used to save
    # model.state_dict() (the *final* epoch's weights) after training, right
    # next to a printed "Best val_loss at epoch k" line - since this model
    # reliably overfits well before training ends, that saved the single
    # worst checkpoint of the run while claiming otherwise. train() now
    # tracks and returns the best-val-loss state dict itself. This test
    # deliberately induces overfitting (val labels are pure noise,
    # uncorrelated with anything the model can learn from train) so val_loss
    # is guaranteed to get worse than its best value by the end of a long
    # enough run, then checks the returned checkpoint really does reproduce
    # the best epoch's val_loss - not just "some" checkpoint.
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(64, VECTOR_LEN)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.float32)  # learnable signal
    x_val = rng.normal(size=(32, VECTOR_LEN)).astype(np.float32)
    y_val = rng.integers(0, 2, size=32).astype(np.float32)  # unrelated to x_val

    model = WinProbModel(hidden_sizes=(64,), dropout=0.0)
    history, best_state = train(
        model, x_train, y_train, x_val, y_val,
        epochs=40, batch_size=16, lr=1e-2, weight_decay=0.0,
        device="cpu", verbose=False,
    )

    best_metrics = min(history, key=lambda m: m.val_loss)
    assert history[-1].val_loss > best_metrics.val_loss  # confirms overfitting really happened

    reloaded = WinProbModel(hidden_sizes=(64,), dropout=0.0)
    reloaded.load_state_dict(best_state)
    with torch.no_grad():
        logits = reloaded(torch.from_numpy(x_val).float())
        recomputed_val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(y_val).float()
        ).item()
    assert abs(recomputed_val_loss - best_metrics.val_loss) < 1e-5


def test_calibration_error_includes_predictions_of_exactly_one():
    # Real bug caught by review: [lo, hi) bins for every bin (including the
    # last) left a prediction of exactly 1.0 in no bin at all, silently
    # dropped from the metric - from the single worst-possible position (a
    # maximally confident wrong prediction, if the true label is 0).
    probs = torch.tensor([1.0, 1.0, 1.0, 0.2])
    labels = torch.tensor([0.0, 0.0, 0.0, 0.0])

    error = calibration_error(probs, labels, n_bins=10)

    assert error > 0.5  # would be ~0.2 if the 1.0-predictions were dropped


def test_predict_proba_restores_prior_train_mode():
    # Real bug caught by review: predict_proba used to call self.eval() and
    # never restore the caller's prior mode, so calling it mid-training-loop
    # (e.g. for early-stopping logic) would silently leave dropout off for
    # the rest of training.
    model = WinProbModel(hidden_sizes=(8,), dropout=0.5)
    model.train()

    model.predict_proba(torch.randn(4, VECTOR_LEN))

    assert model.training is True
