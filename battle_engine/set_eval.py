"""Phase 6 / M4: scoring a set-prediction filler against what battles reveal.

M4's done-when is that a filler's opponent teams are "measurably better than
the revealed-only baseline at predicting the opponent's actual revealed sets
later in the same battle". This module is that measurement, and it is one half
of M4's evidence. The other half is `scripts/fidelity_harness.py --prior
usage-stats`, which asks the different and harder question of whether the
prediction moves the *forward model's* accuracy.

Both halves are needed because they fail in different directions. A prior can
name the right species and still mis-model every turn (wrong spread), and it
can improve the model's damage numbers while naming the wrong Pokemon (a
metagame-average spread is closer than a 0-EV one either way). Neither number
alone would say the prior works.

## Method

One pass over a replay does two things at once. It accumulates the **eventual
truth** for both sides - every species that appeared, and per species the
ability, item, Tera type and moves the log ever showed - and it snapshots each
side's `SideObservation` at every turn boundary. Afterwards the filler is asked
what it thinks at each snapshot, and its answer is scored against the truth.

Only *hidden-at-the-time* facts are scored. An attribute the battle had already
revealed by turn N is an observation, the filler is structurally forbidden from
overwriting it, and counting it would inflate every condition equally. The
question is always: of what was still hidden at turn N and would later be
shown, how much did the prior get right?

## What this cannot measure, stated up front

- **Spreads.** No replay reveals EVs, IVs or a nature. There is no truth to
  score against, so the accuracy of the single most valuable thing this prior
  supplies is not in this file at all. It shows up in the fidelity harness's HP
  numbers instead, which is why that run is not optional.
- **Sets that never came out.** A species held in the back all game is never
  revealed, so a correct prediction of it scores as a miss. Species precision
  here is therefore a *lower bound*, and the gap is real: a gen9 OU battle
  reveals only 40.5% of abilities and 26.8% of items of the Pokemon that do
  appear, let alone of those that do not.
- **Both sides are scored**, not just the opponent's. In a replay both teams are
  observed through the same protocol log and are equally hidden, so scoring both
  doubles the sample without changing the question.

## The denominators are turn-weighted, and that is deliberate

The same slot is re-scored at every turn it stays hidden, so a 30-turn battle
counts one wrong Tera-type guess thirty times. That is the right weighting for
the decision this feeds - the search re-asks the prior every turn, so an error
that persists for thirty turns costs thirty turns of search - but it is the
wrong number to quote as a sample size. `distinct cases` is reported alongside
every rate for exactly that reason: 111 asks across 8 distinct Pokemon is one
observation about eight Pokemon, not 111 independent trials.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from poke_env.data import GenData, to_id_str

from battle_engine.fidelity import ReplayDriver
from battle_engine.poke_engine_state import (
    SideObservation,
    UnknownFiller,
    observe_side,
    require_tera_type,
)

# Turn buckets for the "does it get better as the battle goes on" breakdown.
# Open-ended at the top because gen9 OU games run long and the tail is thin.
TURN_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("turns 1-5", 1, 5),
    ("turns 6-10", 6, 10),
    ("turns 11-20", 11, 20),
    ("turns 21+", 21, 10_000),
)


@dataclass
class RevealedSet:
    """Everything one battle ever showed about one Pokemon.

    Accumulated across the drive rather than read off the final state, because
    the final state is lossy in one specific way: poke-env sets `item` to `""`
    when Knock Off removes it, so a battle that revealed Heavy-Duty Boots on
    turn 3 and knocked them off on turn 9 ends with no item at all. For a
    *prediction* score the item the Pokemon was holding is the right target -
    which is the opposite of `fidelity.Hindsight`'s reason for excluding items,
    where the question was what to put in a turn-N state and a non-monotone
    attribute could not be back-filled honestly.
    """

    species: str
    ability: Optional[str] = None
    item: Optional[str] = None
    tera_type: Optional[str] = None
    moves: FrozenSet[str] = frozenset()


def _tera_id(mon) -> Optional[str]:
    return require_tera_type(mon.tera_type.name) if mon.tera_type is not None else None


def _observed_item(mon) -> Optional[str]:
    """The item, or None for "nothing shown". `""` (Knock Off) is not a reveal
    of a *set*, so it is not recorded as one - the real item, if it was ever
    named, was recorded on an earlier turn."""
    if mon.item is None or mon.item == GenData.UNKNOWN_ITEM or mon.item == "":
        return None
    return to_id_str(mon.item)


def _accumulate(truth: Dict[str, RevealedSet], battle, ours: bool) -> None:
    team = battle.team if ours else battle.opponent_team
    for mon in team.values():
        species = to_id_str(mon.species)
        current = truth.get(species) or RevealedSet(species=species)
        truth[species] = RevealedSet(
            species=species,
            ability=to_id_str(mon.ability) if mon.ability else current.ability,
            item=_observed_item(mon) or current.item,
            tera_type=_tera_id(mon) or current.tera_type,
            moves=current.moves | {to_id_str(m) for m in mon.moves},
        )


@dataclass
class Counts:
    """Numerator/denominator pairs, one per predictable attribute.

    Every denominator is "hidden at the time and revealed later", so a
    condition is never credited or charged for something it could not have
    been asked.
    """

    species_asked: int = 0
    species_hit: int = 0
    species_guesses: int = 0
    species_guesses_correct: int = 0
    ability_asked: int = 0
    ability_hit: int = 0
    item_asked: int = 0
    item_hit: int = 0
    tera_asked: int = 0
    tera_hit: int = 0
    move_asked: int = 0
    move_hit: int = 0
    snapshots: int = 0

    def add(self, other: "Counts") -> None:
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def rates(self) -> Dict[str, Tuple[int, int]]:
        return {
            "species recall": (self.species_hit, self.species_asked),
            "species precision": (self.species_guesses_correct, self.species_guesses),
            "ability": (self.ability_hit, self.ability_asked),
            "item": (self.item_hit, self.item_asked),
            "tera type": (self.tera_hit, self.tera_asked),
            "moves recall": (self.move_hit, self.move_asked),
        }


def score_snapshot(
    observation: SideObservation,
    truth: Dict[str, RevealedSet],
    filler: UnknownFiller,
    asked: Optional[Dict[str, set]] = None,
) -> Counts:
    """Ask `filler` what it thinks of one side at one turn, and score it.

    `asked`, when given, collects `(species, attribute)` keys so the caller can
    count how many *distinct* cases the turn-weighted totals cover.
    """
    counts = Counts(snapshots=1)
    asked = asked if asked is not None else defaultdict(set)
    fills = list(filler.fill_side(observation))
    revealed_now = {slot.species for slot in observation.slots if slot.species}

    hidden_species = {s for s in truth if s not in revealed_now}
    guessed = {to_id_str(f.species) for f in fills if f.species}
    counts.species_asked += len(hidden_species)
    asked["species recall"].update(hidden_species)
    asked["species precision"].update(guessed)
    counts.species_hit += len(hidden_species & guessed)
    counts.species_guesses += len(guessed)
    counts.species_guesses_correct += len(guessed & set(truth))

    for slot, fill in zip(observation.slots, fills):
        if slot.species is None:
            continue
        known = truth.get(slot.species)
        if known is None:
            continue

        if slot.ability is None and known.ability is not None:
            counts.ability_asked += 1
            asked["ability"].add(slot.species)
            counts.ability_hit += int(
                fill.ability is not None and to_id_str(fill.ability) == known.ability
            )
        # `item == ""` is an observation ("holding nothing"), not a gap.
        if slot.item is None and known.item is not None:
            counts.item_asked += 1
            asked["item"].add(slot.species)
            counts.item_hit += int(fill.item is not None and to_id_str(fill.item) == known.item)
        if slot.tera_type is None and known.tera_type is not None:
            counts.tera_asked += 1
            asked["tera type"].add(slot.species)
            counts.tera_hit += int(
                fill.tera_type is not None and require_tera_type(fill.tera_type) == known.tera_type
            )

        seen = {to_id_str(m) for m in slot.moves}
        hidden_moves = known.moves - seen
        # A fill can only offer `4 - len(seen)` moves, so recall is capped by
        # the four-move limit rather than by the filler's judgment.
        # Left uncapped on purpose: the capped number would flatter every
        # condition equally and hide that late-battle prediction has less room.
        predicted = {to_id_str(m) for m in fill.moves}
        counts.move_asked += len(hidden_moves)
        counts.move_hit += len(hidden_moves & predicted)
        asked["moves recall"].update((slot.species, m) for m in hidden_moves)

    return counts


@dataclass
class SetEvalReport:
    filler: str
    replays: int = 0
    snapshots: int = 0
    overall: Counts = field(default_factory=Counts)
    by_bucket: Dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    # attribute -> number of distinct (battle, side, subject) cases behind the
    # turn-weighted totals. See the module docstring.
    distinct: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    seconds: float = 0.0

    def render(self) -> str:
        def rate(hit: int, asked: int) -> str:
            return f"{hit}/{asked} = {100.0 * hit / asked:.1f}%" if asked else f"{hit}/0 = n/a"

        lines = [
            f"Set prediction - filler: {self.filler}",
            f"  replays: {self.replays}   side-snapshots scored: {self.overall.snapshots}",
            "",
            "Of what was still hidden at the turn and revealed later:",
        ]
        for label, (hit, asked) in self.overall.rates().items():
            lines.append(
                f"  {label:<20}{rate(hit, asked):<22}"
                f"({self.distinct.get(label, 0)} distinct cases)"
            )
        lines.append("")
        lines.append("By point in the battle (species recall / ability / item / moves recall):")
        for label, _, _ in TURN_BUCKETS:
            counts = self.by_bucket.get(label)
            if counts is None or not counts.snapshots:
                continue
            r = counts.rates()
            lines.append(
                f"  {label:<14}"
                f"{rate(*r['species recall']):>18}"
                f"{rate(*r['ability']):>18}"
                f"{rate(*r['item']):>18}"
                f"{rate(*r['moves recall']):>18}"
            )
        if self.skipped:
            lines.append("")
            lines.append("Replays skipped:")
            for reason, n in self.skipped.most_common():
                lines.append(f"  {reason:<34}{n:>6}")
        lines.append("")
        lines.append(f"  {self.seconds:.1f}s")
        return "\n".join(lines)


def _bucket(turn: int) -> Optional[str]:
    for label, low, high in TURN_BUCKETS:
        if low <= turn <= high:
            return label
    return None


def score_replay(
    path: Path | str, filler: UnknownFiller, report: Optional[SetEvalReport] = None
) -> SetEvalReport:
    report = report or SetEvalReport(filler=getattr(filler, "name", "unnamed"))
    path = Path(path)
    payload = json.loads(path.read_text())
    driver = ReplayDriver(payload["log"], payload.get("id") or path.stem)

    truth: Dict[bool, Dict[str, RevealedSet]] = {True: {}, False: {}}
    snapshots: List[Tuple[int, bool, SideObservation]] = []
    for marker, turn, battle in driver:
        for ours in (True, False):
            _accumulate(truth[ours], battle, ours)
        if marker != "turn":
            continue
        for ours in (True, False):
            try:
                snapshots.append((turn, ours, observe_side(battle, ours=ours)))
            except Exception:  # noqa: BLE001 - a side with no team yet
                continue

    distinct: Dict[str, set] = defaultdict(set)
    for turn, ours, observation in snapshots:
        asked: Dict[str, set] = defaultdict(set)
        counts = score_snapshot(observation, truth[ours], filler, asked)
        for attribute, keys in asked.items():
            distinct[attribute].update((ours, key) for key in keys)
        report.overall.add(counts)
        bucket = _bucket(turn)
        if bucket is not None:
            report.by_bucket[bucket].add(counts)
    for attribute, keys in distinct.items():
        report.distinct[attribute] += len(keys)
    report.replays += 1
    return report


def score_corpus(
    paths: Sequence[Path | str],
    filler: UnknownFiller,
    on_replay: Optional[Callable[[int, Path | str, SetEvalReport], None]] = None,
) -> SetEvalReport:
    report = SetEvalReport(filler=getattr(filler, "name", "unnamed"))
    start = time.perf_counter()
    for index, path in enumerate(paths):
        try:
            score_replay(path, filler, report)
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the run
            report.skipped[f"replay_failed:{type(exc).__name__}"] += 1
        if on_replay is not None:
            on_replay(index, path, report)
    report.seconds = time.perf_counter() - start
    return report
