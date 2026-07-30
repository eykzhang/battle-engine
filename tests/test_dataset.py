import json
from pathlib import Path

import lz4.frame
import numpy as np
import pytest

from battle_engine.dataset import build_dataset, encode_replay, split_replays
from battle_engine.encoding import VECTOR_LEN


def _pokemon(**overrides) -> dict:
    base = {
        "name": "garchomp",
        "base_species": "garchomp",
        "hp_pct": 1.0,
        "types": "normal notype",
        "status": "nostatus",
        "ability": "unknownability",
        **{f"{s}_boost": 0 for s in
           ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
        **{f"base_{s}": 80 for s in ("hp", "atk", "def", "spa", "spd", "spe")},
    }
    base.update(overrides)
    return base


def _state(won: bool, lost: bool) -> dict:
    return {
        "player_active_pokemon": _pokemon(),
        "opponent_active_pokemon": _pokemon(),
        "available_switches": [],
        "opponents_remaining": 6,
        "player_conditions": "noconditions",
        "opponent_conditions": "noconditions",
        "weather": "noweather",
        "battle_field": "nofield",
        "battle_won": won,
        "battle_lost": lost,
    }


def _write_replay(path: Path, n_states: int, final_won: bool) -> None:
    # Mirrors real data (verified before writing dataset.py): battle_won/
    # battle_lost are False/False until the last state, which reflects the
    # actual outcome.
    states = [_state(False, False) for _ in range(n_states - 1)]
    states.append(_state(final_won, not final_won))
    payload = json.dumps({"states": states, "actions": [0] * n_states}).encode()
    path.write_bytes(lz4.frame.compress(payload))


def test_encode_replay_labels_every_state_with_final_outcome_not_per_state_flag(tmp_path):
    # The exact bug worth guarding against: a replay's own states are
    # (False, False) - "not decided yet" - for almost the whole game, only
    # flipping on the last state. The label must be the game's FINAL result,
    # not that per-state flag, or nearly every state gets mislabeled 0.
    path = tmp_path / "gen9ou-1_1500_a_vs_b_01-01-2024_WIN.json.lz4"
    _write_replay(path, n_states=5, final_won=True)

    vectors, labels = encode_replay(path)

    assert vectors.shape == (5, VECTOR_LEN)
    assert (labels == 1.0).all()  # every state, not just the last one


def test_encode_replay_loss_labels_every_state_zero(tmp_path):
    path = tmp_path / "gen9ou-2_1500_a_vs_b_01-01-2024_LOSS.json.lz4"
    _write_replay(path, n_states=3, final_won=False)

    _, labels = encode_replay(path)

    assert (labels == 0.0).all()


def test_encode_replay_rejects_filename_data_mismatch(tmp_path):
    # Filename claims WIN, but the data's last state disagrees - should
    # raise rather than silently mislabel.
    path = tmp_path / "gen9ou-3_1500_a_vs_b_01-01-2024_WIN.json.lz4"
    _write_replay(path, n_states=2, final_won=False)

    with pytest.raises(ValueError):
        encode_replay(path)


def _synthetic_path(battle_id: str, suffix: str = "a", result: str = "WIN") -> Path:
    # battle id must contain no underscores - the same convention real
    # Metamon filenames use (see fetch_replay_sample.py) and what
    # _battle_id's regex relies on.
    return Path(f"{battle_id}_1500_{suffix}_vs_b_01-01-2024_{result}.json.lz4")


def test_split_replays_has_no_overlap_and_covers_every_path():
    paths = [_synthetic_path(f"battle{i}") for i in range(20)]

    train, val = split_replays(paths, val_fraction=0.25, seed=0)

    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(paths)
    assert len(val) == 5


def test_split_replays_is_deterministic_for_a_given_seed():
    paths = [_synthetic_path(f"battle{i}") for i in range(10)]

    train1, val1 = split_replays(paths, seed=42)
    train2, val2 = split_replays(paths, seed=42)

    assert train1 == train2
    assert val1 == val2


def test_split_replays_keeps_mirrored_pov_files_of_the_same_battle_together():
    # Real bug caught by review: Metamon's archive stores many battles from
    # both players' perspectives as two separate files (different ELO,
    # swapped pov_vs_opponent) sharing one battle id. Splitting by file
    # alone let 5 of 2060 real battles have one POV in train and the
    # mirrored POV - the same game, inverted label - in val.
    mirrored_a = _synthetic_path("battle0", suffix="alice", result="WIN")
    mirrored_b = _synthetic_path("battle0", suffix="bob", result="LOSS")
    other_battles = [_synthetic_path(f"battle{i}") for i in range(1, 20)]
    paths = [mirrored_a, mirrored_b] + other_battles

    train, val = split_replays(paths, val_fraction=0.3, seed=7)

    both_in_train = mirrored_a in train and mirrored_b in train
    both_in_val = mirrored_a in val and mirrored_b in val
    assert both_in_train or both_in_val


def test_build_dataset_never_splits_a_single_replays_states_across_train_and_val(tmp_path):
    for i in range(6):
        _write_replay(
            tmp_path / f"gen9ou-{i}_1500_a_vs_b_01-01-2024_WIN.json.lz4",
            n_states=3, final_won=True,
        )
    for i in range(6, 10):
        _write_replay(
            tmp_path / f"gen9ou-{i}_1500_a_vs_b_01-01-2024_LOSS.json.lz4",
            n_states=4, final_won=False,
        )

    (x_train, y_train), (x_val, y_val) = build_dataset(tmp_path, val_fraction=0.2, seed=0)

    total_states = 6 * 3 + 4 * 4
    assert x_train.shape[0] + x_val.shape[0] == total_states
    assert x_train.shape[1] == VECTOR_LEN == x_val.shape[1]
    assert set(np.unique(y_train)) <= {0.0, 1.0}


def test_build_dataset_on_real_fetched_sample():
    replay_dir = Path("data/replays_raw")
    if not replay_dir.exists() or not any(replay_dir.glob("*.json.lz4")):
        pytest.skip("no fetched replay sample at data/replays_raw "
                    "(run scripts/fetch_replay_sample.py first)")

    (x_train, y_train), (x_val, y_val) = build_dataset(replay_dir, val_fraction=0.2, seed=0)

    assert x_train.shape[1] == VECTOR_LEN
    assert x_train.shape[0] > 0
    assert set(np.unique(y_train)) <= {0.0, 1.0}
