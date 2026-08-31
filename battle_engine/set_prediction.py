"""Phase 6 / M4: predicting the opponent's sets from Smogon usage statistics.

This is the `UnknownFiller` M3 built its seam for. `RevealedOnlyFiller` assumes
nothing beyond what the battle showed; this one answers the same questions from
the metagame's own distributions, and the M4 gate is the measured gap between
the two (`scripts/set_prediction_eval.py`).

## Why the prior has to carry the weight

M3's corpus run settled what set prediction is for, against 5,846 scored turns
of real gen9ou:

- On **69.6%** of turns at least one player's actual action was not addressable
  from revealed information alone. Not mis-evaluated - unrepresentable, because
  poke-engine resolves an action by name against the active Pokemon's current
  moveset and an unrevealed move has no name to resolve.
- Supplying every ability, move and Tera type the battle would *eventually*
  reveal moved strict per-turn fidelity by 2.5 points and barely moved the HP
  error at all. The residual is EV spreads and items.
- A battle eventually reveals only **40.5%** of abilities and **26.8%** of
  items, and never reveals a spread. So mid-battle inference cannot be the
  primary source. The prior is.

That last point is why this module exists before any damage-roll or
move-ordering refinement: the refinement narrows a distribution that has to be
right in the first place, and on most slots there is nothing to narrow from.

## What it fills, and the one rule it cannot break

An observation always wins. This filler never sees a `State` and cannot
overwrite anything - the translator merges observation over fill structurally,
so a slot whose item the battle revealed keeps that item no matter what the
metagame says. What is left for the filler is exactly what was never shown:

- **an unrevealed slot's species**, drawn from usage conditioned on the
  teammates already revealed (Species Clause excludes what is on the team);
- **ability, item and Tera type**, from that species' own distributions;
- **the moves not yet used**, filling the moveset out to four;
- **the spread**, converted to final stats before it crosses the seam.

## Two modes, because they answer different questions

`rng=None` is deterministic: every choice is the distribution's mode. That is
one maximum-likelihood opponent, it is reproducible, and it is what a test or a
single-state diagnostic wants.

With an `rng` it samples instead. That is the mode M5 needs: root-parallel
search over K sampled opponent states, aggregated by summed visits, is how Foul
Play converts a distribution over opponents into a move choice. A search over
one modal opponent is confidently wrong whenever the mode is wrong; a search
over K samples is the point of having a distribution at all.

## Known approximations, named rather than buried

1. **Marginals, not joint sets.** `usage_stats` explains this at length: the
   chaos file records that Great Tusk carried Ice Spinner and that it carried
   Bulk Up, never that one set carried both. Moves, item, ability, Tera type and
   spread are therefore drawn independently, and the assembled set is
   occasionally one no real player runs.
2. **No mid-battle refinement yet.** Damage rolls, move ordering and item tells
   would narrow these distributions using what the battle has shown. That is the
   second half of M4's plan text and it is not in this module.
3. **Level is not predicted.** gen9 OU is level 100 throughout, so the
   translator's own default is already right and a fill would add nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from poke_env.data import GenData, to_id_str

from battle_engine.poke_engine_state import (
    MOVESET_SIZE,
    SideObservation,
    SlotFill,
    SlotObservation,
)
from battle_engine.usage_stats import (
    STAT_ORDER,
    Spread,
    UsageStats,
    default_usage_stats,
)

GEN = 9
_DEX = GenData.from_gen(GEN).pokedex

DEFAULT_LEVEL = 100


def spread_to_stats(species_id: str, spread: Spread, level: int = DEFAULT_LEVEL) -> Dict[str, int]:
    """Final stats in real points, from base stats and a spread.

    Crossing the seam as stats rather than as a nature and six EVs is
    deliberate - see `SlotFill.stats`. It also means the caller of this
    function, not the translator, owns the IV assumption (31 across the board,
    which the chaos file does not record either way).
    """
    base = _DEX[species_id]["baseStats"]
    return {stat: spread.stat(int(base[stat]), stat, level=level) for stat in STAT_ORDER}


@dataclass
class UsageStatsFiller:
    """Fills a side's unknowns from a metagame's usage statistics.

    `rng=None` takes the mode of every distribution; an `rng` samples. The
    filler is stateless across calls apart from that generator, so the same
    instance can serve every turn of a battle and, with a seeded generator, K
    independent opponent samples per turn.
    """

    stats: UsageStats
    rng: Optional[random.Random] = None
    fill_species: bool = True
    fill_spreads: bool = True
    # Off, an unrevealed slot is drawn from raw usage instead of from usage
    # conditioned on the teammates already seen. Kept as a switch because it is
    # the one modelling choice here with a cheap ablation, and "the teammate
    # matrix earns its keep" should be a measurement rather than an assumption.
    condition_on_teammates: bool = True
    level: int = DEFAULT_LEVEL

    @classmethod
    def from_cache(
        cls,
        format_id: str = "gen9ou",
        cutoff: int = 1500,
        rng: Optional[random.Random] = None,
        **kwargs,
    ) -> "UsageStatsFiller":
        return cls(stats=default_usage_stats(format_id=format_id, cutoff=cutoff), rng=rng, **kwargs)

    @property
    def name(self) -> str:
        report = self.stats.report
        mode = "sampled" if self.rng is not None else "modal"
        tags = [f"{report.format_id}-{report.cutoff}@{report.month}", mode]
        if not self.condition_on_teammates:
            tags.append("no-teammates")
        if not self.fill_spreads:
            tags.append("no-spreads")
        if not self.fill_species:
            tags.append("no-species")
        return f"usage-stats({','.join(tags)})"

    # -- distribution access -------------------------------------------------

    def _pick(self, distribution, taken: Sequence[str] = ()) -> Optional[str]:
        pool = distribution.without(taken) if taken else distribution
        return pool.sample(self.rng) if self.rng is not None else pool.most_likely()

    def _pick_moves(self, species_id: str, known: Sequence[str], n: int) -> Tuple[str, ...]:
        if self.rng is not None:
            return self.stats.sample_moves(species_id, self.rng, known=known, n=n)
        return self.stats.likeliest_moves(species_id, known=known, n=n)

    def _pick_spread(self, species_id: str) -> Optional[Spread]:
        entry = self.stats.get(species_id)
        if entry is None or not entry.spreads:
            return None
        if self.rng is not None:
            return entry.spreads.sample(self.rng)
        return entry.spreads.most_likely()

    # -- the seam ------------------------------------------------------------

    def fill_side(self, observation: SideObservation) -> Sequence[SlotFill]:
        """Both sides go through the same path.

        Our own side is fully known from the request JSON in a live battle, so
        every fill there is a no-op the translator discards. In a replay-driven
        battle it is not - both sides are observed through the same protocol log
        - and treating them alike is what makes the fidelity harness measure the
        thing the player will actually run.
        """
        revealed = [slot.species for slot in observation.slots if slot.species]
        # Species Clause: a sampled slot may not repeat anything already on the
        # team, including species this same call has already invented.
        team: List[str] = list(revealed)

        fills: List[SlotFill] = []
        for slot in observation.slots:
            fill = self._fill_slot(slot, team)
            if fill.species:
                team.append(to_id_str(fill.species))
            fills.append(fill)
        return fills

    def _fill_slot(self, slot: SlotObservation, team: Sequence[str]) -> SlotFill:
        species = slot.species
        if species is None:
            if not self.fill_species:
                return SlotFill()
            pool = (
                self.stats.conditional_usage(team)
                if self.condition_on_teammates
                else self.stats.usage.without(team)
            )
            species = self._pick(pool)
            if species is None:
                return SlotFill()
        elif species not in self.stats:
            # A revealed species the metagame file has never seen - too rare to
            # make the usage cut, or a format mismatch. Nothing to predict from,
            # so the translator's own conventions stand.
            return SlotFill()

        entry = self.stats.get(species)
        if entry is None:
            return SlotFill()

        invented_species = slot.species is None

        ability = None if slot.ability else self._pick(entry.abilities)
        # `item == ""` means "revealed to be holding nothing" (Knock Off), which
        # is an observation, not a gap. Only `None` is a gap.
        item = None if slot.item is not None else self._pick(entry.items)
        tera_type = None if slot.tera_type else self._pick(entry.tera_types)

        known_moves = tuple(to_id_str(m) for m in slot.moves)
        moves = self._pick_moves(species, known_moves, MOVESET_SIZE)

        stats: Optional[Mapping[str, int]] = None
        if self.fill_spreads:
            spread = self._pick_spread(species)
            if spread is not None:
                stats = spread_to_stats(species, spread, self.level)

        return SlotFill(
            species=species if invented_species else None,
            item=item,
            ability=ability,
            moves=moves,
            tera_type=tera_type,
            stats=stats,
        )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class LayeredFiller:
    """Several fillers stacked, earlier layers winning field by field.

    This exists so the M3 fidelity harness can run its action-oracle *on top
    of* this module's prior: the oracle keeps supplying exactly the action the
    turn needs, and everything it does not speak to falls through to usage
    statistics instead of to the translator's neutral defaults. That makes the
    harness answer M4's real question - does the prior move the forward model's
    accuracy - through the same seam, with the same guarantee that no fill can
    overwrite an observation.

    **Species coherence is the trap here.** For an unrevealed slot two layers
    can name two different Pokemon, and merging field by field would then bolt
    a Great Tusk spread and moveset onto a Gholdengo. So the winning species is
    resolved first, and a layer that named a *different* species contributes
    nothing else to that slot - its item, ability, moves, Tera type and stats
    were all predictions about a Pokemon that is not there.
    """

    layers: Tuple[object, ...]

    def __init__(self, *layers: object) -> None:
        if not layers:
            raise ValueError("LayeredFiller needs at least one layer")
        self.layers = tuple(layers)

    @property
    def name(self) -> str:
        return "+".join(getattr(layer, "name", type(layer).__name__) for layer in self.layers)

    def fill_side(self, observation: SideObservation) -> Sequence[SlotFill]:
        per_layer = [list(layer.fill_side(observation)) for layer in self.layers]  # type: ignore[attr-defined]
        for index, fills in enumerate(per_layer):
            if len(fills) != observation.team_size:
                raise ValueError(
                    f"{getattr(self.layers[index], 'name', index)} returned {len(fills)} fills "
                    f"for a {observation.team_size}-slot side"
                )
        return [
            _merge_fills(tuple(fills[slot] for fills in per_layer))
            for slot in range(observation.team_size)
        ]


def _merge_fills(candidates: Sequence[SlotFill]) -> SlotFill:
    species = next((c.species for c in candidates if c.species), None)
    if species is not None:
        species_id = to_id_str(species)
        contributing = [
            c for c in candidates if c.species is None or to_id_str(c.species) == species_id
        ]
    else:
        contributing = list(candidates)

    def first(attribute: str):
        for candidate in contributing:
            value = getattr(candidate, attribute)
            if value is not None:
                return value
        return None

    moves: List[str] = []
    for candidate in contributing:
        for move in candidate.moves:
            move_id = to_id_str(move)
            if move_id not in moves:
                moves.append(move_id)

    return SlotFill(
        species=species,
        item=first("item"),
        ability=first("ability"),
        moves=tuple(moves[:MOVESET_SIZE]),
        level=first("level"),
        tera_type=first("tera_type"),
        stats=first("stats"),
    )
