"""Cached (state-vector, win/loss-label) dataset built from fetched replay
files (see scripts/fetch_replay_sample.py to fetch them, scripts/
build_dataset.py to build the cache).

Label semantics were verified against real downloaded data before writing
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

from battle_engine.encoding import VECTOR_LEN, battle_views_from_replay, encode

_RESULT_RE = re.compile(r"_(WIN|LOSS)\.json\.lz4$")
_BATTLE_ID_RE = re.compile(r"^([^_]+)_")  # battle id has no underscores (see fetch_replay_sample.py)


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
