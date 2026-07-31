"""Cached datasets built from fetched replay files (see
scripts/fetch_replay_sample.py to fetch them). Two dataset types, two label
schemes, same underlying state vectors and the same train/val split:

- build_dataset / encode_replay: (state-vector, win/loss-label) for the
  win-probability model (win_prob.py) - see below for why the label isn't
  the state's own battle_won/battle_lost flag.
- build_action_dataset / encode_replay_actions: (state-vector, action-label)
  for the imitation model (imitation.py) - Phase 2's originally-planned
  second milestone (move-prediction, "state -> human's move"), built
  2026-07-31 alongside a code-review pass on the win-prob work. Each row's
  label is that specific turn's action (not one label per replay, unlike
  the win-prob case) - see ACTION_SPACE_SIZE's docstring for the label
  scheme and _remap_action for why raw switch-action indices can't be used
  directly (measured: 78% would be mislabeled without remapping to a
  stable species order).

Label semantics for the win-prob dataset were verified against real
downloaded data before writing
this, and it's worth spelling out why: a replay state's own `battle_won`/
`battle_lost` flags are *not* the label to use directly. Checked across
every currently downloaded sample file, both are `False` for almost the
entire game — the battle genuinely hasn't been won or lost yet — and only
flip on the last state (confirmed to agree with the filename's WIN/LOSS
suffix on every sample checked). Naively using the per-state flag as the
label would train a model to predict "is the game already over and won *this
turn*", which is nearly always 0 regardless of how the game actually ends —
not a win-probability signal. The label used here is the replay's *final*
outcome (does this player go on to win the whole game), applied to every
state in that replay — the standard "Monte Carlo return" framing for this
kind of value-function target. Sourced from the filename's WIN/LOSS suffix
(the same field scripts/fetch_replay_sample.py already parses for ELO), and
cross-checked against the last state's battle_won/battle_lost as a sanity
check rather than a second source of truth.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import lz4.frame
import numpy as np

from battle_engine.encoding import VECTOR_LEN, _species_key, battle_views_from_replay, encode

_RESULT_RE = re.compile(r"_(WIN|LOSS)\.json\.lz4$")
_BATTLE_ID_RE = re.compile(r"^([^_]+)_")  # battle id has no underscores (see fetch_replay_sample.py)

# Metamon's per-state action label (verified against real data, not
# documented anywhere): an integer 0-12, or -1 for "no action recorded"
# (excluded from training below). Confirmed empirically by correlating
# action value against each state's own can_tera/forced_switch/
# available_switches fields (2026-07-31):
#   0-3   move slot 0-3
#   4-8   switch (see _replay_switch_slot_order below - NOT usable as a raw
#         index, needs remapping)
#   9-12  move slot 0-3, terastallized (100% of these states have
#         can_tera=True, 0% for plain move actions 0-3 that happen to also
#         have can_tera=True but weren't used - consistent)
# This is Metamon's own action-space shape, not poke-env's: poke-env's
# Gymnasium SinglesEnv.action_to_order uses a different 26-way scheme (0-5
# switch by absolute team index, 6-9 move, 10-25 mega/z-move/dynamax/tera
# variants gen9ou doesn't use). Reconciling the two is a real, deliberately
# deferred step for whenever this model feeds Phase 3's PPO policy init -
# not attempted here.
ACTION_SPACE_SIZE = 13
_MOVE_ACTIONS = set(range(0, 4))
_SWITCH_ACTIONS = set(range(4, 9))
_TERA_MOVE_ACTIONS = set(range(9, 13))
_MISSING_ACTION = -1


def _replay_switch_slot_order(states: list) -> List[List[str]]:
    """For each state, the species-sorted bench order a switch action's raw
    index needs remapping into - same ordering battle_views_from_replay
    builds for its encoded bench slots (encoding.py), so a remapped action
    label actually points at the same slot position the model's input
    vector represents that species in.

    Deliberately reimplemented here rather than sharing code with
    battle_views_from_replay: that function returns PokemonView objects
    with no species identity attached (by design - see encoding.py), so it
    can't answer "which slot is species X in" directly. This needn't build
    full views, only track which species have been seen - a strict subset
    of that function's work. Provably equivalent by construction rather
    than just asserted: battle_views_from_replay's per-state bench key set
    is (switches ∪ (seen - {active} - switches)), and since switches ⊆ seen
    always (both add every switch to `seen` unconditionally), that
    simplifies to exactly `seen - {active}` - which is what this computes
    directly. Keep both in sync if encoding.py's bench algorithm changes.
    """
    seen: set = set()
    orders = []
    for state in states:
        active_species = _species_key(state["player_active_pokemon"])
        seen.add(active_species)
        for mon in state["available_switches"]:
            seen.add(_species_key(mon))
        orders.append(sorted(seen - {active_species}))
    return orders


def _remap_action(action: int, switch_order: List[str], available_switches: list) -> int:
    """Maps a raw Metamon action label to this project's stable action
    space: moves/tera-moves pass through unchanged (a Pokemon's own move
    slots don't reorder turn to turn), but a switch action's raw index
    (position within that turn's live available_switches list) does NOT
    correspond to a stable species - measured on real data: 331/424 (78%)
    of switch actions in a 50-replay sample would be mislabeled if used
    directly, since available_switches' order isn't stable (same root cause
    documented in encoding.py's bench-slot-ordering fix). Remapped to the
    target's position in switch_order (the species-sorted bench order used
    by encoding.py for the *input* vector's bench slots), so the label
    means "switch to whichever species is in encoded bench slot N", not
    "switch to whichever species Showdown happened to list Nth that turn".
    """
    if action not in _SWITCH_ACTIONS:
        return action
    raw_target = available_switches[action - 4]
    target_species = _species_key(raw_target)
    return 4 + switch_order.index(target_species)


def _battle_id(path: Path) -> str:
    match = _BATTLE_ID_RE.match(path.name)
    if match is None:
        raise ValueError(f"can't parse battle id from filename: {path.name}")
    return match.group(1)


def _replay_outcome(path: Path) -> bool:
    """True if the POV player in this replay went on to win the game."""
    match = _RESULT_RE.search(path.name)
    if match is None:
        raise ValueError(f"can't parse WIN/LOSS from filename: {path.name}")
    return match.group(1) == "WIN"


def _load_replay(path: Path) -> dict:
    with path.open("rb") as f:
        return json.loads(lz4.frame.decompress(f.read()))


def encode_replay(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """One replay file -> (vectors, labels), one row per turn. Every row from
    a single replay gets the *same* label (see module docstring): the game's
    final outcome, not that turn's own not-yet-decided battle_won flag.

    Uses battle_views_from_replay (not battle_view_from_replay_state) since
    it needs the whole replay's turn sequence to reconstruct fainted
    teammates correctly - see battle_engine.encoding's module docstring.
    """
    data = _load_replay(path)
    won = _replay_outcome(path)

    states = data["states"]
    if not states:
        raise ValueError(f"replay has no states: {path.name}")
    last = states[-1]
    if won and not last["battle_won"]:
        raise ValueError(f"filename says WIN but last state disagrees: {path.name}")
    if not won and not last["battle_lost"]:
        raise ValueError(f"filename says LOSS but last state disagrees: {path.name}")

    vectors = np.stack([encode(view) for view in battle_views_from_replay(states)])
    labels = np.full(len(states), 1.0 if won else 0.0, dtype=np.float32)
    return vectors, labels


def encode_replay_actions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """One replay file -> (vectors, action_labels) for the imitation model -
    unlike encode_replay, each row gets its *own* state's action (what the
    human actually chose that turn), not one label repeated for the whole
    replay. States whose action is -1 ("missing", see ACTION_SPACE_SIZE's
    docstring - no ground truth recorded) are dropped entirely rather than
    given a fabricated label; a replay with zero non-missing actions
    returns empty arrays (the caller's concat step already handles that,
    same as an empty-states replay would).
    """
    data = _load_replay(path)
    states = data["states"]
    actions = data["actions"]
    if len(states) != len(actions):
        raise ValueError(
            f"states/actions length mismatch ({len(states)} vs {len(actions)}): {path.name}"
        )

    switch_orders = _replay_switch_slot_order(states)
    views = battle_views_from_replay(states)

    kept_vectors, kept_labels = [], []
    for state, action, order, view in zip(states, actions, switch_orders, views):
        if action == _MISSING_ACTION:
            continue
        remapped = _remap_action(action, order, state["available_switches"])
        kept_vectors.append(encode(view))
        kept_labels.append(remapped)

    if not kept_vectors:
        return (
            np.zeros((0, VECTOR_LEN), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return np.stack(kept_vectors), np.array(kept_labels, dtype=np.int64)


def _list_replay_files(replay_dir: Path) -> List[Path]:
    return sorted(replay_dir.glob("*.json.lz4"))


def split_replays(
    paths: List[Path], val_fraction: float = 0.1, seed: int = 0
) -> Tuple[List[Path], List[Path]]:
    """Splits by battle, not by file or by turn: every state from one battle
    lands in exactly one of train/val, so validation never sees turns from a
    game the model also trained on other turns of.

    Grouping by *battle id*, not just by file, matters: review found real
    train/val leakage here. Metamon's archive stores many battles from both
    players' perspectives as two separate files (different ELO field,
    swapped pov_vs_opponent) sharing the same battle id prefix. Splitting by
    file alone let a real battle land with one player's POV in train and the
    mirrored POV - the same game, inverted label - in val (measured: 5 of
    2060 files affected on the dataset this fix responds to).
    """
    battle_ids = sorted({_battle_id(p) for p in paths})
    random.Random(seed).shuffle(battle_ids)
    n_val = round(len(battle_ids) * val_fraction)
    val_ids = set(battle_ids[:n_val])

    train_paths = [p for p in paths if _battle_id(p) not in val_ids]
    val_paths = [p for p in paths if _battle_id(p) in val_ids]
    return train_paths, val_paths


def _encode_many(paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    vector_chunks, label_chunks = [], []
    for path in paths:
        vectors, labels = encode_replay(path)
        vector_chunks.append(vectors)
        label_chunks.append(labels)
    if not vector_chunks:
        return np.zeros((0, VECTOR_LEN), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(vector_chunks), np.concatenate(label_chunks)


def build_dataset(
    replay_dir: Path, val_fraction: float = 0.1, seed: int = 0
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    paths = _list_replay_files(replay_dir)
    train_paths, val_paths = split_replays(paths, val_fraction, seed)
    return _encode_many(train_paths), _encode_many(val_paths)


def _encode_many_actions(paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    vector_chunks, label_chunks = [], []
    for path in paths:
        vectors, labels = encode_replay_actions(path)
        vector_chunks.append(vectors)
        label_chunks.append(labels)
    if not vector_chunks:
        return np.zeros((0, VECTOR_LEN), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(vector_chunks), np.concatenate(label_chunks)


def build_action_dataset(
    replay_dir: Path, val_fraction: float = 0.1, seed: int = 0
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Same train/val split as build_dataset (same default seed, same
    split_replays call) - the win-prob and imitation models are evaluated
    on the same held-out battles, not independently-sampled ones.
    """
    paths = _list_replay_files(replay_dir)
    train_paths, val_paths = split_replays(paths, val_fraction, seed)
    return _encode_many_actions(train_paths), _encode_many_actions(val_paths)
