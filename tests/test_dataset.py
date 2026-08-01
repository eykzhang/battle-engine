import json
from pathlib import Path

import lz4.frame
import numpy as np
import pytest

from battle_engine.dataset import (
    ACTION_SPACE_SIZE,
    build_dataset,
    encode_replay,
    encode_replay_actions,
    split_replays,
)
from battle_engine.encoding import VECTOR_LEN


def _pokemon(**overrides) -> dict:
    base = {
        "name": "garchomp",
        "base_species": "garchomp",
        "hp_pct": 1.0,
        "types": "normal notype",
        "status": "nostatus",
        "ability": "unknownability",
        "item": "unknownitem",
        "moves": [],
        **{f"{s}_boost": 0 for s in
           ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
        **{f"base_{s}": 80 for s in ("hp", "atk", "def", "spa", "spd", "spe")},
    }
    base.update(overrides)
    return base


def _move(name="nomove") -> dict:
    return {
        "name": name, "move_type": "nomove", "category": "nomove",
        "base_power": 0, "accuracy": 1.0, "priority": 0, "current_pp": 0, "max_pp": 0,
    }


def _state(won: bool, lost: bool, active=None, available_switches=None) -> dict:
    return {
        "player_active_pokemon": active or _pokemon(),
        "opponent_active_pokemon": _pokemon(base_species="dragapult", name="dragapult"),
        "available_switches": available_switches or [],
        "opponents_remaining": 6,
        "player_conditions": "noconditions",
        "opponent_conditions": "noconditions",
        "weather": "noweather",
        "battle_field": "nofield",
        "battle_won": won,
        "battle_lost": lost,
        "player_prev_move": _move(),
        "opponent_prev_move": _move(),
    }


def _write_replay_with_actions(path: Path, states: list, actions: list) -> None:
    # encode_replay_actions doesn't read battle_won/battle_lost or the
    # filename's WIN/LOSS suffix at all (unlike encode_replay) - only
    # states/actions matter here.
    payload = json.dumps({"states": states, "actions": actions}).encode()
    path.write_bytes(lz4.frame.compress(payload))


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


def test_encode_replay_actions_drops_missing_action_states(tmp_path):
    # action == -1 means "no ground truth recorded" (see ACTION_SPACE_SIZE's
    # docstring) - those states must be excluded, not given a fabricated
    # label like 0.
    path = tmp_path / "gen9ou-4_1500_a_vs_b_01-01-2024_WIN.json.lz4"
    states = [_state(False, False) for _ in range(4)]
    _write_replay_with_actions(path, states, actions=[0, -1, 2, -1])

    vectors, labels = encode_replay_actions(path)

    assert vectors.shape == (2, VECTOR_LEN)
    assert list(labels) == [0, 2]


def test_encode_replay_actions_passes_move_and_tera_move_actions_through_unchanged(tmp_path):
    # Move slots (0-3) and move-while-terastallized slots (9-12) don't need
    # remapping - a Pokemon's own move order is stable turn to turn, unlike
    # switch targets (see the remapping test below).
    path = tmp_path / "gen9ou-5_1500_a_vs_b_01-01-2024_WIN.json.lz4"
    states = [_state(False, False) for _ in range(4)]
    _write_replay_with_actions(path, states, actions=[0, 3, 9, 12])

    _, labels = encode_replay_actions(path)

    assert list(labels) == [0, 3, 9, 12]


def test_encode_replay_actions_remaps_switch_target_to_stable_species_order(tmp_path):
    # The exact bug worth guarding against (measured on real data: 331/424,
    # 78%, of switch actions would be mislabeled without this): a switch
    # action's raw index is a position within *that turn's* live
    # available_switches list, whose order isn't stable turn to turn. Two
    # states here offer the same two switch targets in opposite raw order;
    # the remapped label must point at the same species (aggron, which
    # sorts before beedrill) in both, tracking *species* not *list position*.
    aggron = _pokemon(base_species="aggron", name="aggron")
    beedrill = _pokemon(base_species="beedrill", name="beedrill")
    path = tmp_path / "gen9ou-6_1500_a_vs_b_01-01-2024_WIN.json.lz4"

    state_aggron_first = _state(False, False, available_switches=[aggron, beedrill])
    state_beedrill_first = _state(False, False, available_switches=[beedrill, aggron])
    # raw index 0 in both states - but that's aggron in the first state and
    # beedrill in the second, since the raw list order flipped.
    _write_replay_with_actions(
        path, [state_aggron_first, state_beedrill_first], actions=[4, 4]
    )

    _, labels = encode_replay_actions(path)

    # aggron sorts before beedrill, so "switch to aggron" is always stable
    # slot 4 (the first switch slot) regardless of raw list position -
    # both states' raw action=4 (raw index 0) should remap to the species
    # actually at raw index 0 in each: aggron -> stays 4, beedrill -> becomes 5.
    assert list(labels) == [4, 5]


def test_build_action_dataset_on_real_fetched_sample():
    from battle_engine.dataset import build_action_dataset

    replay_dir = Path("data/replays_raw")
    if not replay_dir.exists() or not any(replay_dir.glob("*.json.lz4")):
        pytest.skip("no fetched replay sample at data/replays_raw "
                    "(run scripts/fetch_replay_sample.py first)")

    (x_train, y_train), (x_val, y_val) = build_action_dataset(
        replay_dir, val_fraction=0.2, seed=0
    )

    assert x_train.shape[1] == VECTOR_LEN
    assert x_train.shape[0] > 0
    assert y_train.min() >= 0
    assert y_train.max() < ACTION_SPACE_SIZE
