"""Phase 6 / M4: Smogon usage statistics as a prior over opponent sets.

M3 turned this milestone's motivation into numbers. Over 5,846 scored turns of
real gen9ou, both players' actions were addressable from revealed information
alone on only 30.4% of them, and supplying every ability and move the battle
would *eventually* reveal moved strict fidelity by just 2.5 points - because a
battle reveals only 40.5% of abilities and 26.8% of items, and never reveals an
EV spread at all. So the prior has to come from outside the battle, and it has to
carry spreads, not only species and items.

This module is the data layer: it loads a cached chaos file
(`scripts/fetch_usage_stats.py`), normalizes it into validated distributions, and
answers the questions a filler asks. It knows nothing about poke-engine `State`s
and nothing about poke-env - `battle_engine/set_prediction.py` is what joins it
to the `UnknownFiller` seam.

## What the chaos file actually contains, verified on 2026-08-30

`data[<Display Name>]` per species:

- `usage` - weighted fraction of teams carrying it (0-1). Summed over every
  species this comes to 5.995, i.e. one unit per team slot, which is the check
  that it means "fraction of teams containing" rather than anything else.
- `Abilities`, `Items`, `Tera Types`, `Spreads` - weighted counts, one draw per
  set, so each of the four sums to the same number: the species' **weighted set
  count**. That number is the denominator for everything here.
- `Raw count` - **not that denominator, and not safe as one.** It is the
  *unweighted* count, and it is byte-identical across the 1500 and 1695 cutoff
  files (389,490 for Great Tusk in both) because the cutoff acts on the weights,
  not on the tally. Dividing weighted counts by it silently rescales every
  probability by the cutoff's weighting factor: it made P(Head Smash | Great
  Tusk) read 0.54 at the 1500 cut and 0.09 at 1695 for the same metagame. Kept
  on the dataclass for reference and deliberately used as a denominator nowhere.
- `Moves` - weighted counts, up to *four* draws per set, summing to exactly
  `4 x` the set count. `count / set_count` is therefore the marginal probability
  that a set carries that move, not a probability of drawing it. An `""` key
  counts sets with fewer than four moves.
- `Spreads` keys are `"<Nature>:<hp>/<atk>/<def>/<spa>/<spd>/<spe>"`.
- `Teammates` - **not co-occurrence counts.** For Great Tusk the Gholdengo entry
  is 11,622 against a set count of 38,067, which cannot be a count of anything
  when Gholdengo's own usage is 0.2608. The stored value is the *deviation from
  independence*:

      n(A, B) = Teammates_A[B] + set_count(A) * usage(B)
      P(B | A) = usage(B) + Teammates_A[B] / set_count(A)

  Established by symmetry rather than by reading anyone's source: co-occurrence
  must satisfy `n(A,B) == n(B,A)`, and over the 299 top-25 pairs that identity
  holds to a median relative error of 0.06% with the set count and 15% with the
  raw count. Smogon clamps the deviation at zero (12,992 of 113,097 entries are
  exactly 0.0, none negative), so this estimator can raise a species'
  probability above its base usage but never lower it. Anti-correlation is
  information the file does not carry, and no arithmetic here recovers it.

## The approximation this module makes, stated once

Chaos gives **marginals, not joint sets**. There is no record that a particular
set paired Bulk Up with Ice Spinner, only that each appeared. Sampling four
distinct moves weighted by their marginals is therefore an approximation, and it
is the same one Foul Play's usage-stats path makes. It is wrong in a specific,
predictable direction: it will occasionally assemble a moveset no real player
runs (two setup moves that are alternatives, say). Correcting it needs joint
data - scraped Smogon sample sets or teams parsed out of replays - which is a
worthwhile later addition, not a prerequisite. Same caveat for pairing a spread
with a moveset: the two are drawn independently.

## Validation policy

A species, move or ability the built poke-engine cannot represent must never be
sampled: an unknown species serializes as `NONE` (no stats, no typing) and an
unknown move deals 0 damage, both silently - footguns 1 and 2 in
`poke_engine_state`'s module docstring. Those are dropped at load time and
counted in `LoadReport` so the drop is visible rather than assumed to be empty.

**Items are the deliberate exception**, for the same reason `poke_engine_state`
makes it: 59.5% of the item dex has no poke-engine mechanics, including Heat Rock
and Safety Goggles. Dropping those would renormalize the item distribution toward
the modelled ones and quietly overstate how often the opponent holds Leftovers.
They are kept, predicted honestly, and downgraded to `unknownitem` by the
translator, which records the downgrade in the provenance ledger.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Generic, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

from poke_env.data import to_id_str

from battle_engine.poke_engine_state import (
    is_known_ability,
    is_known_move,
    is_known_species,
)

DEFAULT_STATS_DIR = Path("data/usage_stats")

# The six stats in the order Smogon writes a spread, which is also the order
# Showdown packs EVs and the order poke-engine's Pokemon fields appear in.
STAT_ORDER: Tuple[str, ...] = ("hp", "atk", "def", "spa", "spd", "spe")

MOVESET_SIZE = 4
TEAM_SIZE = 6

# Every nature poke-engine accepts, lowercased. A spread naming anything else is
# dropped rather than passed on: PokemonNature is one of the four enums with no
# default arm, so a bad value panics through pyo3 rather than raising (footgun 4).
NATURES = frozenset(
    """hardy lonely brave adamant naughty bold docile relaxed impish lax timid
    hasty serious jolly naive modest mild quiet bashful rash calm gentle sassy
    careful quirky""".split()
)

_STAT_MODIFIED_BY: Mapping[str, Tuple[Optional[str], Optional[str]]] = {
    # nature -> (boosted stat, hindered stat); None for a neutral nature.
    "hardy": (None, None),
    "lonely": ("atk", "def"),
    "brave": ("atk", "spe"),
    "adamant": ("atk", "spa"),
    "naughty": ("atk", "spd"),
    "bold": ("def", "atk"),
    "docile": (None, None),
    "relaxed": ("def", "spe"),
    "impish": ("def", "spa"),
    "lax": ("def", "spd"),
    "timid": ("spe", "atk"),
    "hasty": ("spe", "def"),
    "serious": (None, None),
    "jolly": ("spe", "spa"),
    "naive": ("spe", "spd"),
    "modest": ("spa", "atk"),
    "mild": ("spa", "def"),
    "quiet": ("spa", "spe"),
    "bashful": (None, None),
    "rash": ("spa", "spd"),
    "calm": ("spd", "atk"),
    "gentle": ("spd", "def"),
    "sassy": ("spd", "spe"),
    "careful": ("spd", "spa"),
    "quirky": (None, None),
}


# ---------------------------------------------------------------------------
# Spreads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spread:
    """A nature plus six EVs, the one thing no battle log can ever reveal.

    IVs are not in the chaos file and are assumed to be 31 across the board,
    which is what a competitive team runs except for the deliberate 0-Attack
    confusion/Foul-Play dodge that Smogon's own spread strings do not record
    either.
    """

    nature: str
    evs: Tuple[int, int, int, int, int, int]

    @property
    def ev_total(self) -> int:
        return sum(self.evs)

    def ev(self, stat: str) -> int:
        return self.evs[STAT_ORDER.index(stat)]

    def nature_multiplier(self, stat: str) -> float:
        boosted, hindered = _STAT_MODIFIED_BY.get(self.nature, (None, None))
        if stat == boosted:
            return 1.1
        if stat == hindered:
            return 0.9
        return 1.0

    def stat(self, base: int, stat: str, level: int = 100, iv: int = 31) -> int:
        """The gen-3+ stat formula. HP has its own arm; everything else takes
        the nature multiplier, floored after it."""
        if stat == "hp":
            return int((2 * base + iv + self.ev(stat) // 4) * level / 100) + level + 10
        raw = int((2 * base + iv + self.ev(stat) // 4) * level / 100) + 5
        return int(raw * self.nature_multiplier(stat))

    def __str__(self) -> str:
        return f"{self.nature}:{'/'.join(str(e) for e in self.evs)}"

    @classmethod
    def parse(cls, key: str) -> Optional["Spread"]:
        """`"Jolly:0/252/0/0/4/252"` -> a Spread, or None if unusable.

        Returns None rather than raising: the chaos file is scraped from real
        ladder teams and carries whatever those teams packed, including EV
        totals above the legal 508 and natures that no longer exist. A caller
        wants those skipped, not a crash mid-load.
        """
        nature, _, evs = key.partition(":")
        nature = nature.strip().lower()
        if nature not in NATURES:
            return None
        parts = evs.split("/")
        if len(parts) != len(STAT_ORDER):
            return None
        try:
            values = tuple(int(p) for p in parts)
        except ValueError:
            return None
        if any(v < 0 or v > 252 for v in values):
            return None
        return cls(nature=nature, evs=values)  # type: ignore[arg-type]


# The convention `poke_engine_state` already applies to any Pokemon whose real
# spread is unknown: 0 EVs, 31 IVs, neutral nature. Named here so "no prediction"
# and "predicted the neutral spread" are the same object and can be compared.
NEUTRAL_SPREAD = Spread(nature="serious", evs=(0, 0, 0, 0, 0, 0))


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

K = TypeVar("K")


class Distribution(Generic[K]):
    """A weighted vocabulary over one attribute, normalized on demand.

    Weights stay in the file's own units (weighted counts) rather than being
    normalized at construction, because the `Moves` counts have to be divided by
    `Raw count` rather than by their own sum to mean anything.
    """

    __slots__ = ("_weights", "_total")

    def __init__(self, weights: Mapping[K, float]):
        self._weights: Dict[K, float] = {k: float(v) for k, v in weights.items() if v > 0}
        self._total = sum(self._weights.values())

    def __len__(self) -> int:
        return len(self._weights)

    def __bool__(self) -> bool:
        return bool(self._weights)

    def __contains__(self, key: object) -> bool:
        return key in self._weights

    @property
    def total(self) -> float:
        return self._total

    def weight(self, key: K) -> float:
        return self._weights.get(key, 0.0)

    def items(self) -> Iterable[Tuple[K, float]]:
        return self._weights.items()

    def probability(self, key: K) -> float:
        """Share of this distribution's own mass. Not the right question for
        `Moves` - use `SpeciesUsage.move_probability`, which divides by the
        species' set count instead."""
        return self._weights.get(key, 0.0) / self._total if self._total else 0.0

    def most_likely(self, default: Optional[K] = None) -> Optional[K]:
        if not self._weights:
            return default
        return max(self._weights.items(), key=lambda kv: kv[1])[0]

    def top(self, n: int) -> Tuple[Tuple[K, float], ...]:
        ranked = sorted(self._weights.items(), key=lambda kv: -kv[1])[:n]
        return tuple((k, w / self._total if self._total else 0.0) for k, w in ranked)

    def sample(self, rng: random.Random, default: Optional[K] = None) -> Optional[K]:
        if not self._weights:
            return default
        keys = list(self._weights)
        return rng.choices(keys, weights=[self._weights[k] for k in keys], k=1)[0]

    def sample_without_replacement(self, rng: random.Random, n: int) -> Tuple[K, ...]:
        """`n` distinct keys, each drawn proportional to its remaining weight.

        This is how a moveset gets assembled from per-move marginals, and it is
        the joint-vs-marginal approximation the module docstring names. Returns
        fewer than `n` keys when the vocabulary is smaller.
        """
        remaining = dict(self._weights)
        drawn: list[K] = []
        for _ in range(min(n, len(remaining))):
            keys = list(remaining)
            choice = rng.choices(keys, weights=[remaining[k] for k in keys], k=1)[0]
            drawn.append(choice)
            del remaining[choice]
        return tuple(drawn)

    def without(self, excluded: Iterable[K]) -> "Distribution[K]":
        blocked = set(excluded)
        return Distribution({k: w for k, w in self._weights.items() if k not in blocked})

    def reweighted(self, factor) -> "Distribution[K]":
        """A new distribution with each weight scaled by `factor(key)`."""
        return Distribution({k: w * factor(k) for k, w in self._weights.items()})


# ---------------------------------------------------------------------------
# Per-species statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeciesUsage:
    species: str
    usage: float
    # Weighted count of sets of this species, i.e. the sum of any of the
    # one-draw-per-set distributions. THE denominator - see the module docstring.
    set_count: float
    # Unweighted, cutoff-invariant, and never a denominator. Kept for reference.
    raw_count: float
    abilities: Distribution[str]
    items: Distribution[str]
    moves: Distribution[str]
    spreads: Distribution[Spread]
    tera_types: Distribution[str]
    # Raw chaos deviations, id-keyed. `UsageStats.conditional_usage` is what
    # turns these into probabilities; kept raw here so the arithmetic lives in
    # exactly one place.
    teammate_deltas: Mapping[str, float] = field(default_factory=dict)

    def move_probability(self, move_id: str) -> float:
        """P(a set of this species carries this move).

        Divides by the species' set count - not by the move distribution's own
        sum (which is 4x too big, one draw per move slot) and not by `Raw count`
        (which is on the unweighted scale). See the module docstring.
        """
        if self.set_count <= 0:
            return 0.0
        return min(1.0, self.moves.weight(move_id) / self.set_count)


@dataclass(frozen=True)
class LoadReport:
    """What the loader dropped, so a silent empty vocabulary is impossible."""

    month: str
    format_id: str
    cutoff: int
    battles: int
    species_kept: int
    species_dropped: Tuple[str, ...]
    moves_dropped: Tuple[str, ...]
    abilities_dropped: Tuple[str, ...]
    spreads_dropped: int
    empty_move_slots: float

    def summary(self) -> str:
        return (
            f"{self.format_id}-{self.cutoff} {self.month}: {self.species_kept} species, "
            f"{self.battles} battles; dropped {len(self.species_dropped)} species, "
            f"{len(self.moves_dropped)} moves, {len(self.abilities_dropped)} abilities, "
            f"{self.spreads_dropped} spreads"
        )


class UsageStats:
    """One month of one format's chaos data, validated and id-keyed."""

    def __init__(self, species: Mapping[str, SpeciesUsage], report: LoadReport):
        self._species = dict(species)
        self.report = report
        self.usage = Distribution({s: e.usage for s, e in self._species.items()})

    def __len__(self) -> int:
        return len(self._species)

    def __contains__(self, species_id: object) -> bool:
        return species_id in self._species

    def __iter__(self):
        return iter(self._species)

    def get(self, species_id: str) -> Optional[SpeciesUsage]:
        return self._species.get(species_id)

    def entry(self, species_id: str) -> SpeciesUsage:
        entry = self._species.get(species_id)
        if entry is None:
            raise KeyError(f"{species_id} is not in {self.report.format_id}-{self.report.cutoff}")
        return entry

    def conditional_probability(self, given: str, candidate: str) -> float:
        """P(candidate on the team | `given` on the team), per the chaos
        deviation formula. Falls back to the candidate's base usage when
        `given` is unknown or has no recorded count."""
        base = self._species[candidate].usage if candidate in self._species else 0.0
        source = self._species.get(given)
        if source is None or source.set_count <= 0:
            return base
        return min(1.0, base + source.teammate_deltas.get(candidate, 0.0) / source.set_count)

    def conditional_usage(self, given: Sequence[str]) -> Distribution[str]:
        """A distribution over species to put in an unrevealed slot, given the
        team members already known.

        Naive-Bayes lift: each known teammate multiplies a candidate's base
        usage by `P(candidate | teammate) / usage(candidate)`. Because Smogon
        clamps the teammate deviations at zero that lift is always >= 1, so this
        can only sharpen toward well-known cores, never push a candidate below
        its base rate. Known members are excluded - gen9 OU has Species Clause.
        """
        known = [g for g in given if g]
        available = self.usage.without(known)
        if not known:
            return available

        def lift(candidate: str) -> float:
            base = self._species[candidate].usage
            if base <= 0:
                return 0.0
            factor = 1.0
            for teammate in known:
                factor *= self.conditional_probability(teammate, candidate) / base
            return factor

        return available.reweighted(lift)

    def sample_moves(
        self,
        species_id: str,
        rng: random.Random,
        known: Sequence[str] = (),
        n: int = MOVESET_SIZE,
    ) -> Tuple[str, ...]:
        """`n` distinct move ids for this species, keeping `known` and drawing
        the remainder from the marginals. Returns only the *drawn* moves."""
        entry = self._species.get(species_id)
        if entry is None:
            return ()
        pool = entry.moves.without(known)
        return pool.sample_without_replacement(rng, max(0, n - len(known)))

    def likeliest_moves(
        self, species_id: str, known: Sequence[str] = (), n: int = MOVESET_SIZE
    ) -> Tuple[str, ...]:
        entry = self._species.get(species_id)
        if entry is None:
            return ()
        pool = entry.moves.without(known)
        return tuple(k for k, _ in pool.top(max(0, n - len(known))))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def parse_chaos(payload: Mapping) -> UsageStats:
    """Chaos JSON -> `UsageStats`, dropping anything poke-engine cannot name."""
    info = payload.get("info", {})
    data = payload.get("data", {})

    display_to_id = {name: to_id_str(name) for name in data}
    species_dropped: list[str] = []
    moves_dropped: set[str] = set()
    abilities_dropped: set[str] = set()
    spreads_dropped = 0
    empty_move_slots = 0.0

    entries: Dict[str, SpeciesUsage] = {}
    for name, raw in data.items():
        species_id = display_to_id[name]
        if not is_known_species(species_id):
            species_dropped.append(species_id)
            continue

        abilities: Dict[str, float] = {}
        for ability, weight in raw.get("Abilities", {}).items():
            ability_id = to_id_str(ability)
            if is_known_ability(ability_id):
                abilities[ability_id] = abilities.get(ability_id, 0.0) + weight
            else:
                abilities_dropped.add(ability_id)

        moves: Dict[str, float] = {}
        for move, weight in raw.get("Moves", {}).items():
            if not move:
                empty_move_slots += weight
                continue
            move_id = to_id_str(move)
            if is_known_move(move_id):
                moves[move_id] = moves.get(move_id, 0.0) + weight
            else:
                moves_dropped.add(move_id)

        spreads: Dict[Spread, float] = {}
        for key, weight in raw.get("Spreads", {}).items():
            spread = Spread.parse(key)
            if spread is None:
                spreads_dropped += 1
                continue
            spreads[spread] = spreads.get(spread, 0.0) + weight

        # Every one-draw-per-set distribution sums to the same set count; the
        # ability tally is used because a species always has an ability, while
        # Items can be thinned by an id that normalizes away.
        set_count = sum(raw.get("Abilities", {}).values())

        entries[species_id] = SpeciesUsage(
            species=species_id,
            usage=float(raw.get("usage", 0.0)),
            set_count=float(set_count),
            raw_count=float(raw.get("Raw count", 0.0)),
            abilities=Distribution(abilities),
            # Items are kept unvalidated on purpose - see the module docstring.
            items=Distribution({to_id_str(i): w for i, w in raw.get("Items", {}).items() if i}),
            moves=Distribution(moves),
            spreads=Distribution(spreads),
            tera_types=Distribution(
                {to_id_str(t): w for t, w in raw.get("Tera Types", {}).items() if t}
            ),
            teammate_deltas={},
        )

    # Teammates are resolved in a second pass: a delta pointing at a species
    # that was itself dropped has to go, and the first pass does not yet know
    # which species survived.
    resolved: Dict[str, SpeciesUsage] = {}
    for name, raw in data.items():
        species_id = display_to_id[name]
        if species_id not in entries:
            continue
        deltas = {}
        for teammate, weight in raw.get("Teammates", {}).items():
            teammate_id = display_to_id.get(teammate, to_id_str(teammate))
            if teammate_id in entries and weight > 0:
                deltas[teammate_id] = float(weight)
        resolved[species_id] = SpeciesUsage(
            **{**entries[species_id].__dict__, "teammate_deltas": deltas}
        )

    report = LoadReport(
        month=str(info.get("month", "")),
        format_id=str(info.get("metagame", "")),
        cutoff=int(info.get("cutoff", 0)),
        battles=int(info.get("number of battles", 0)),
        species_kept=len(resolved),
        species_dropped=tuple(sorted(species_dropped)),
        moves_dropped=tuple(sorted(moves_dropped)),
        abilities_dropped=tuple(sorted(abilities_dropped)),
        spreads_dropped=spreads_dropped,
        empty_move_slots=empty_move_slots,
    )
    return UsageStats(resolved, report)


def load_usage_stats(path: Path) -> UsageStats:
    payload = json.loads(Path(path).read_text())
    stats = parse_chaos(payload)
    # The chaos payload's `info` has no month field; the filename does.
    if not stats.report.month:
        stats.report = LoadReport(**{**stats.report.__dict__, "month": Path(path).stem.split("_")[0]})
    return stats


def find_cached(
    format_id: str = "gen9ou",
    cutoff: int = 1500,
    stats_dir: Path = DEFAULT_STATS_DIR,
) -> Optional[Path]:
    """Newest cached chaos file for a format/cutoff, by the month in its name."""
    matches = sorted(Path(stats_dir).glob(f"*_{format_id}-{cutoff}.json"), reverse=True)
    return matches[0] if matches else None


@lru_cache(maxsize=4)
def default_usage_stats(
    format_id: str = "gen9ou",
    cutoff: int = 1500,
    stats_dir: Path = DEFAULT_STATS_DIR,
) -> UsageStats:
    """The cached stats a filler uses when it is not handed any.

    Cached across calls because parsing ~14 MB of JSON is not something a
    per-battle `Player` should redo, and because M5 runs several searches per
    turn over the same prior.
    """
    path = find_cached(format_id, cutoff, stats_dir)
    if path is None:
        raise FileNotFoundError(
            f"no cached usage stats for {format_id}-{cutoff} in {stats_dir}. "
            f"Run: .venv/bin/python scripts/fetch_usage_stats.py "
            f"--format {format_id} --cutoff {cutoff}"
        )
    return load_usage_stats(path)
