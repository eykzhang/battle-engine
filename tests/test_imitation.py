import numpy as np
import torch

from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.encoding import VECTOR_LEN
from battle_engine.imitation import ImitationModel, top1_accuracy, train


def test_model_forward_pass_returns_one_logit_row_per_example():
    model = ImitationModel(hidden_sizes=(8,))
    x = torch.zeros(5, VECTOR_LEN)

    logits = model(x)

    assert logits.shape == (5, ACTION_SPACE_SIZE)


def test_predict_proba_returns_a_valid_probability_distribution():
    model = ImitationModel(hidden_sizes=(8,))
    x = torch.randn(10, VECTOR_LEN)

    probs = model.predict_proba(x)

    assert probs.shape == (10, ACTION_SPACE_SIZE)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_top1_accuracy_counts_argmax_matches():
    logits = torch.tensor([
        [5.0, 0.0, 0.0],  # argmax 0
        [0.0, 5.0, 0.0],  # argmax 1
        [0.0, 0.0, 5.0],  # argmax 2
        [5.0, 0.0, 0.0],  # argmax 0
    ])
    labels = torch.tensor([0, 1, 0, 0])  # 3/4 match
    assert top1_accuracy(logits, labels) == 0.75


def test_training_loop_reduces_loss_on_a_trivially_separable_dataset():
    # Same "does the pipeline train at all" sanity check as
    # win_prob.py's equivalent test - label = which of 3 classes a single
    # feature's sign/magnitude maps to, trivially learnable if the
    # forward/backward/optimizer wiring is correct.
    rng = np.random.default_rng(0)
    n = 90
    x = rng.normal(size=(n, VECTOR_LEN)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[x[:, 0] > 0.5] = 1
    y[x[:, 0] < -0.5] = 2

    model = ImitationModel(hidden_sizes=(16,), dropout=0.0)
    history, _ = train(
        model, x, y, x, y,
        epochs=60, batch_size=16, lr=1e-2, weight_decay=0.0,
        device="cpu", verbose=False,
    )

    assert history[-1].val_loss < history[0].val_loss
    assert history[-1].val_top1_accuracy > 0.9


def test_training_loop_runs_on_real_shaped_tiny_dataset():
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(20, VECTOR_LEN)).astype(np.float32)
    y_train = rng.integers(0, ACTION_SPACE_SIZE, size=20).astype(np.int64)
    x_val = rng.normal(size=(5, VECTOR_LEN)).astype(np.float32)
    y_val = rng.integers(0, ACTION_SPACE_SIZE, size=5).astype(np.int64)

    model = ImitationModel(hidden_sizes=(8,))
    history, best_state = train(
        model, x_train, y_train, x_val, y_val,
        epochs=2, batch_size=8, device="cpu", verbose=False,
    )

    assert len(history) == 2
    assert all(np.isfinite(m.val_loss) for m in history)
    assert set(best_state.keys()) == set(model.state_dict().keys())


def test_train_returns_the_best_checkpoint_not_the_final_epoch():
    # Same guard as win_prob.py's equivalent test, for the same reason - a
    # caller saving model.state_dict() after train() returns would get the
    # final (possibly-overfit) epoch's weights, not the best one, if train()
    # didn't track and hand back the best checkpoint itself.
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(64, VECTOR_LEN)).astype(np.float32)
    y_train = np.zeros(64, dtype=np.int64)
    y_train[x_train[:, 0] > 0] = 1  # learnable signal
    x_val = rng.normal(size=(32, VECTOR_LEN)).astype(np.float32)
    y_val = rng.integers(0, ACTION_SPACE_SIZE, size=32).astype(np.int64)  # unrelated to x_val

    model = ImitationModel(hidden_sizes=(64,), dropout=0.0)
    history, best_state = train(
        model, x_train, y_train, x_val, y_val,
        epochs=40, batch_size=16, lr=1e-2, weight_decay=0.0,
        device="cpu", verbose=False,
    )

    best_metrics = min(history, key=lambda m: m.val_loss)
    assert history[-1].val_loss > best_metrics.val_loss  # confirms overfitting really happened

    reloaded = ImitationModel(hidden_sizes=(64,), dropout=0.0)
    reloaded.load_state_dict(best_state)
    with torch.no_grad():
        logits = reloaded(torch.from_numpy(x_val).float())
        recomputed_val_loss = torch.nn.functional.cross_entropy(
            logits, torch.from_numpy(y_val).long()
        ).item()
    assert abs(recomputed_val_loss - best_metrics.val_loss) < 1e-5


def test_predict_proba_restores_prior_train_mode():
    model = ImitationModel(hidden_sizes=(8,), dropout=0.5)
    model.train()

    model.predict_proba(torch.randn(4, VECTOR_LEN))

    assert model.training is True
