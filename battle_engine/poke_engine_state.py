"""Phase 6 / M3: a live poke-env `Battle` translated into a poke-engine `State`.

Everything else in Phase 6 sits on this file, and a forward-model swap goes
wrong *silently*, so this module is written defensively on purpose. Three
already-documented footguns
(notes/gotcha-poke-engine-pypi-wheel-is-gen4-not-gen9.md), plus points 4-7
and the volatile-status half of point 3, found while writing this module and
written up in notes/gotcha-poke-env-poke-engine-name-mismatches.md. All
verified against the installed gen9 build, not inferred:

1. An unknown species id is accepted and serializes as `NONE` - a Pokemon
   with no base stats and no typing. No exception.
2. An unknown move id serializes as `NONE` and `calculate_damage` returns
   [0, 0] for it. `generate_instructions` *does* raise for it. Neither
   validates for the other.
3. An unknown item silently becomes `UNKNOWNITEM`, an unknown ability
   silently becomes `NONE`, and an unknown volatile status is silently
   inserted into the side's volatile set *as* `NONE`.
4. **The opposite polarity, and new here**: `PokemonStatus`, `PokemonNature`,
   `Weather` and `Terrain` are the four enums poke-engine declares with no
   `default =` arm (src/lib.rs's `define_enum_with_from_str!`), so their
   `from_str` calls `panic!` instead of falling back. A bad value there
   crosses pyo3 as `pyo3_runtime.PanicException`, which is not a
   `ValueError` and is not caught by `except Exception`. Measured:
   `weather="raindance"` panics; only `"rain"` is accepted.
5. **poke-env and poke-engine do not spell these the same way.** Verified
   mismatches, all of which would have hit footgun 4:
   - status: poke-env `BRN/FRZ/PAR/PSN/SLP/TOX` vs poke-engine
     `BURN/FREEZE/PARALYZE/POISON/SLEEP/TOXIC`. poke-env's `FNT` has no
     counterpart - poke-engine represents fainted as `hp == 0`.
   - weather: poke-env `SUNNYDAY/RAINDANCE/SANDSTORM/SNOWSCAPE` vs
     poke-engine `SUN/RAIN/SAND/SNOW`. poke-env's `DELTASTREAM` has no
     poke-engine counterpart at all.
   - terrain: these DO agree once normalized (`electricterrain`, ...).
   - types: poke-env's `THREE_QUESTION_MARKS` is poke-engine's `TYPELESS`,
     which is also poke-engine's "this Pokemon has only one type" filler.
6. **A Terastallized Pokemon's typing is the trap in the other direction.**
   poke-env's `mon.types` already *returns* the Tera type once a Pokemon has
   terastallized (`type_1` returns `_terastallized_type`, `type_2` returns
   None). poke-engine expects the opposite: `Pokemon.types` stays the
   original typing and `damage_calc.rs::type_effectiveness_modifier`
   substitutes `(tera_type, TYPELESS)` itself when `terastallized` is set.
   Passing `mon.types` through would double-apply the Tera and destroy the
   original typing poke-engine still needs for STAB. Original typing is
   therefore read from the gen-9 dex, not from the live Pokemon.
7. **`to_string()` / `from_string()` is not an identity round trip once any
   volatile status is set.** `Side::serialize` writes each volatile with a
   trailing ":", and `Side::deserialize` feeds the resulting empty final
   element to `PokemonVolatileStatus::from_str`, which defaults to NONE. So
   a state with volatiles gains a spurious `NONE` on the first round trip
   and is stable only from the second. Anything comparing serialized states
   (M3's fidelity harness) has to normalize for it. Pinned by a test rather
   than worked around here, so a future poke-engine fix shows up as a
   failure.

So: every species, move, ability and item id is checked against
poke-engine's own accepted vocabulary before a state is built, by
round-tripping a probe state through `to_string()` and looking for the
fallback value. The vocabulary is discovered from the engine at runtime
rather than transcribed from the Rust enums, because a transcription would
be a second source of truth that can silently drift from the built wheel.

**Items are the one deliberate exception to "unknown means raise."**
poke-engine's `Items` enum only covers items whose mechanics it implements:
measured against `pokemon-showdown/data/items.ts`, 347 of 583 item ids
(59.5%) are absent, including Heat Rock, Damp Rock, Icy Rock, Smooth Rock,
Red Card, Sticky Barb, Safety Goggles, Utility Umbrella, Mirror Herb,
Ability Shield, Luminous Moss and Room Service - all legal and none rare in
gen9 OU. Raising on those would make this translator unusable on real
battles. poke-engine ships a first-class sentinel for exactly this case
(`UNKNOWNITEM`), so an unmodelled item is downgraded to `unknownitem` and
recorded as an assumption in the provenance ledger. `strict_items=True`
turns it back into an error for callers that own their item values (e.g.
building a state from one of `teams.py`'s own packed teams). Species,
moves and abilities get no such exception: measured the same way, the only
species poke-engine lacks are CAP/fakemon and the only moves are Z-moves
and G-Max moves, none of them legal in gen9 OU, so an unknown one there is
a naming bug and is raised.

**Unknowns are pluggable and always traceable.** A real battle hides the
opponent's unrevealed Pokemon, spreads, items, abilities and unrevealed
moves. poke-engine needs concrete values anyway. `UnknownFiller` is the
seam M4 (Smogon usage statistics) plugs into; `RevealedOnlyFiller` is the
deliberately simple baseline M4 will be measured against. The translator
never lets a fill overwrite an observation - an observed attribute wins
structurally, not by convention - and every attribute that could have been
unknown lands in `TranslationResult.attributions` marked observed or
assumed with its source. A placeholder that looks like an observation is
the failure this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import poke_engine
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data import GenData, to_id_str

from battle_engine.damage import estimate_stat

GEN = 9
_DEX = GenData.from_gen(GEN).pokedex

TEAM_SIZE = 6
MOVESET_SIZE = 4


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UnknownToPokeEngine(ValueError):
    """A value poke-engine's own vocabulary does not contain.

    Raised instead of letting the value through, because every unknown id
    poke-engine accepts fails silently (footguns 1-3 in the module
    docstring) - a `NONE` species has no stats and no typing, and a `NONE`
    move deals no damage, so a state built from one is quietly wrong rather
    than obviously broken.
    """

    def __init__(self, kind: str, value: Any, hint: str = ""):
        self.kind = kind
        self.value = value
        suffix = f" {hint}" if hint else ""
        super().__init__(
            f"poke-engine does not know the {kind} {value!r}. It would be "
            f"silently accepted and become poke-engine's fallback value, not "
            f"an error.{suffix}"
        )


_ID_HINT = (
    "poke-engine ids are lowercase and space-free; poke-env display names "
    "like 'Great Tusk' must go through poke_env.data.to_id_str first."
)


# ---------------------------------------------------------------------------
# Vocabulary probes
#
# Each probe builds a one-slot state, serializes it, and reads back the field
# it set. Field offsets are the argument order of `Pokemon::serialize` in
# poke-engine/src/state.rs - id first, ability 8, item 10, move slots 22-25.
# ~0.05 ms per probe (measured), and every probe is cached, so a whole battle
# costs a few dozen round trips once per process.
# ---------------------------------------------------------------------------

_SERIALIZED_ID = 0
_SERIALIZED_ABILITY = 8
_SERIALIZED_ITEM = 10
_SERIALIZED_FIRST_MOVE = 22


def _probe_fields(**pokemon_kwargs: Any) -> Sequence[str]:
    """The comma-split serialization of a single probe Pokemon."""
    kwargs = {"id": "pikachu", "level": 100, "hp": 100, "maxhp": 100, **pokemon_kwargs}
    probe = poke_engine.Pokemon(**kwargs)
    filler = [poke_engine.Pokemon.create_fainted() for _ in range(TEAM_SIZE - 1)]
    side = poke_engine.Side(pokemon=[probe] + filler, active_index="0")
    state = poke_engine.State(side_one=side, side_two=side)
    return state.to_string().split("=")[0].split(",")


@lru_cache(maxsize=None)
def is_known_species(species_id: str) -> bool:
    return _probe_fields(id=species_id)[_SERIALIZED_ID] != "NONE"


@lru_cache(maxsize=None)
def is_known_move(move_id: str) -> bool:
    move = poke_engine.Move(id=move_id, pp=16, disabled=False)
    serialized = _probe_fields(moves=[move])[_SERIALIZED_FIRST_MOVE]
    return serialized.split(";")[0] != "NONE"


@lru_cache(maxsize=None)
def is_known_ability(ability_id: str) -> bool:
    return _probe_fields(ability=ability_id)[_SERIALIZED_ABILITY] != "NONE"


@lru_cache(maxsize=None)
def is_known_item(item_id: str) -> bool:
    """False for both a typo and an item poke-engine simply does not model.

    Those two cases are indistinguishable from the outside - both land on
    `UNKNOWNITEM` - which is why unmodelled items are downgraded rather
    than raised by default. See the module docstring's item paragraph.
    """
    return _probe_fields(item=item_id)[_SERIALIZED_ITEM] != "UNKNOWNITEM"


@lru_cache(maxsize=None)
def is_known_volatile_status(volatile_id: str) -> bool:
    """An unknown volatile is inserted into the side's set *as* `NONE`,
    rather than dropped, so this checks that the id survives a round trip
    under its own name.
    """
    side = poke_engine.Side(
        pokemon=[poke_engine.Pokemon.create_fainted() for _ in range(TEAM_SIZE)],
        active_index="0",
        volatile_statuses={volatile_id},
    )
    state = poke_engine.State(side_one=side, side_two=side)
    return volatile_id.upper() in poke_engine.State.from_string(state.to_string()).side_one.volatile_statuses


def require_species(species_id: str) -> str:
    if not is_known_species(species_id):
        raise UnknownToPokeEngine("species", species_id, _ID_HINT)
    return species_id


def require_move(move_id: str) -> str:
    if not is_known_move(move_id):
        raise UnknownToPokeEngine("move", move_id, _ID_HINT)
    return move_id


def require_ability(ability_id: str) -> str:
    if not is_known_ability(ability_id):
        raise UnknownToPokeEngine("ability", ability_id, _ID_HINT)
    return ability_id


def require_tera_type(type_name: str) -> str:
    """Normalize a poke-env type name or a poke-engine type id to the latter.

    poke-engine's `PokemonType` is the one enum with a default arm
    (module docstring, point 4), so a bad Tera type here would not panic -
    it would silently become `typeless`, which reads as "no Tera type" and
    is exactly the kind of assumption that looks like an observation. So it
    is checked.
    """
    candidate = to_id_str(type_name)
    if candidate not in _ENGINE_TYPES:
        raise UnknownToPokeEngine("type", type_name, "poke-engine PokemonType")
    return candidate


def require_item(item_id: str) -> str:
    if not is_known_item(item_id):
        raise UnknownToPokeEngine(
            "item",
            item_id,
            "poke-engine's Items enum covers only items whose mechanics it "
            "implements, so this may be a real gen9 item rather than a typo; "
            "the default strict_items=False downgrades it to 'unknownitem'.",
        )
    return item_id


# ---------------------------------------------------------------------------
# poke-env -> poke-engine enum vocabularies
#
# Every one of these four crosses an enum poke-engine declares WITHOUT a
# default arm, so a wrong spelling panics through pyo3 rather than raising.
# Written as explicit maps, not as `name.lower()`, because for status and
# weather the two libraries genuinely disagree (module docstring, point 5).
# ---------------------------------------------------------------------------

NO_STATUS = "none"
NO_WEATHER = "none"
NO_TERRAIN = "none"
NO_ITEM = "none"
UNKNOWN_ITEM = "unknownitem"
NO_ABILITY = "none"
TYPELESS = "typeless"

_STATUS_TO_ENGINE: Dict[Status, str] = {
    Status.BRN: "burn",
    Status.FRZ: "freeze",
    Status.PAR: "paralyze",
    Status.PSN: "poison",
    Status.SLP: "sleep",
    Status.TOX: "toxic",
    # Status.FNT is deliberately absent: poke-engine has no FAINTED status
    # variant and represents a fainted Pokemon as hp == 0.
}

_WEATHER_TO_ENGINE: Dict[Weather, str] = {
    Weather.SUNNYDAY: "sun",
    Weather.RAINDANCE: "rain",
    Weather.SANDSTORM: "sand",
    Weather.SNOWSCAPE: "snow",
    Weather.HAIL: "hail",
    Weather.DESOLATELAND: "harshsun",
    Weather.PRIMORDIALSEA: "heavyrain",
    # Weather.DELTASTREAM has no poke-engine counterpart (its Weather enum
    # stops at HEAVYRAIN) and Weather.UNKNOWN is poke-env's own placeholder;
    # both fall through to "none" with an assumption recorded.
}

_TERRAIN_TO_ENGINE: Dict[Field, str] = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
}

# poke-engine's PokemonType is the one enum here that DOES have a default
# arm (TYPELESS), so a bad type is silent rather than a panic - but it is
# still wrong, so the map is exhaustive over poke-env's own enum.
_TYPE_TO_ENGINE: Dict[PokemonType, str] = {
    t: t.name.lower() for t in PokemonType if t is not PokemonType.THREE_QUESTION_MARKS
}
_TYPE_TO_ENGINE[PokemonType.THREE_QUESTION_MARKS] = TYPELESS
_ENGINE_TYPES = frozenset(_TYPE_TO_ENGINE.values())

# poke-engine keeps every side condition as one i8 field on `SideConditions`.
# Stealth Rock and Sticky Web are 0/1 flags there; Spikes and Toxic Spikes
# are layer counts, which poke-env also stores as counts (its own
# STACKABLE_CONDITIONS). The screens and Tailwind are turns REMAINING in
# poke-engine but turn-STARTED in poke-env, so they are converted below.
# {poke-env condition: (poke-engine SideConditions field, base duration)}.
# Light Clay (8 turns for Reflect/Light Screen/Aurora Veil) is not
# detectable from a battle log until the item is revealed, so the base
# duration is assumed and recorded as such.
_SCREENS: Dict[SideCondition, Tuple[str, int]] = {
    SideCondition.REFLECT: ("reflect", 5),
    SideCondition.LIGHT_SCREEN: ("light_screen", 5),
    SideCondition.AURORA_VEIL: ("aurora_veil", 5),
    SideCondition.TAILWIND: ("tailwind", 4),
}

_WEATHER_DURATION = 5  # 8 with the matching rock, which is not observable.
_TERRAIN_DURATION = 5  # 8 with Terrain Extender, same caveat.
_TRICK_ROOM_DURATION = 5


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attribution:
    """One attribute of the built state, and where its value came from.

    `observed` is the whole point: a caller can ask what the engine was
    actually told versus what was invented for it. `source` names the
    origin - "poke-env" for an observation, the filler's own `name` for a
    fill, or a short literal for a translator-level convention (screen
    durations, placeholder slots).
    """

    slot: str
    attribute: str
    value: Any
    observed: bool
    source: str


@dataclass(frozen=True)
class TranslationResult:
    state: Any  # poke_engine.State
    attributions: Tuple[Attribution, ...]

    def assumed(self) -> Tuple[Attribution, ...]:
        return tuple(a for a in self.attributions if not a.observed)

    def observed(self) -> Tuple[Attribution, ...]:
        return tuple(a for a in self.attributions if a.observed)

    def for_slot(self, slot: str) -> Tuple[Attribution, ...]:
        return tuple(a for a in self.attributions if a.slot == slot)


# ---------------------------------------------------------------------------
# The unknown-filler seam
#
# The filler sees only what a battle revealed and returns what to assume. It
# never touches poke_engine, never sees a `State`, and cannot overwrite an
# observation - the translator merges observation over fill, not the other
# way round. M4's usage-statistics filler implements the same protocol.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotObservation:
    """What a battle has actually revealed about one team slot.

    A field is None when nothing has been revealed. `species is None` means
    the slot itself is unrevealed - the opponent has a Pokemon here that has
    never been sent out.
    """

    index: int
    species: Optional[str] = None
    level: Optional[int] = None
    # None = never revealed; "" = revealed to be holding nothing.
    item: Optional[str] = None
    ability: Optional[str] = None
    moves: Tuple[str, ...] = ()
    tera_type: Optional[str] = None
    fainted: bool = False
    hp_fraction: float = 1.0


@dataclass(frozen=True)
class SideObservation:
    is_ours: bool
    format_id: Optional[str]
    team_size: int
    slots: Tuple[SlotObservation, ...]


@dataclass(frozen=True)
class SlotFill:
    """What to assume for one slot. Every field is optional; anything left
    None falls through to the translator's own documented placeholder.

    Values that the matching `SlotObservation` already carries are ignored -
    a fill can only supply what was never observed.
    """

    species: Optional[str] = None
    item: Optional[str] = None
    ability: Optional[str] = None
    moves: Tuple[str, ...] = ()
    level: Optional[int] = None
    tera_type: Optional[str] = None
    """A poke-engine type id, or a poke-env type name - `require_tera_type`
    normalizes either. Only meaningful before the Pokemon has actually
    terastallized: once it has, poke-env reveals the real Tera type and the
    observation wins, like every other field here."""

    stats: Optional[Mapping[str, int]] = None
    """Final stats keyed `hp/atk/def/spa/spd/spe`, in real points.

    This is the seam M3's measurement said M4 actually needed, and it is
    deliberately final stats rather than a nature and an EV spread. Two
    reasons. Nature and EVs are inert on a poke-engine `Pokemon` once explicit
    stats are set - it only recomputes from base stats on a form change - so
    passing them through would be passing through something the engine ignores.
    And keeping the spread arithmetic on the filler's side of the seam means
    this module never has to know what a nature is, which is what stops
    `usage_stats` (which imports this one for its vocabulary checks) from
    becoming a circular import.

    Applied only where nothing was observed. Our own team's real stats come
    from the request JSON and always win; an opponent's never do, because
    poke-env is guessing there too - `estimate_stat` returns a 0-EV,
    neutral-nature, 31-IV estimate, which is exactly the systematic error M3
    measured over 5,846 real turns: a mean signed HP error of -2.9%, with 26.9%
    of HP divergences past the 20% mark where a KO judgment flips. Filling this
    field from usage statistics removes the bias (+0.5%) and shrinks the tail.
    """


class UnknownFiller(Protocol):
    """The M4 seam. `name` is what shows up in the provenance ledger."""

    name: str

    def fill_side(self, observation: SideObservation) -> Sequence[SlotFill]:
        """Exactly `observation.team_size` fills, index-aligned with
        `observation.slots` for the slots that exist and appended for the
        unrevealed remainder.
        """
        ...


class RevealedOnlyFiller:
    """The baseline: assume nothing beyond what the battle revealed.

    This is deliberately the weakest possible filler, and it is the one M4
    has to beat. Two documented consequences:

    - An unrevealed opponent slot becomes a placeholder Pokemon, not a
      fainted one. A fainted slot would tell the search the opponent is
      down to N Pokemon and let it evaluate an even position as nearly won,
      which is a worse distortion than a neutral unknown. The placeholder's
      stats are stated constants (see PLACEHOLDER_*), not a guess dressed
      up as a species.
    - An unrevealed item stays `unknownitem` and an unrevealed ability
      stays `none`, which is what poke-engine's own defaults mean. The
      search will simply not model a Choice item or a Levitate it has not
      been shown.
    """

    name = "revealed-only"

    def fill_side(self, observation: SideObservation) -> Sequence[SlotFill]:
        return [SlotFill() for _ in range(observation.team_size)]


# A level-100 Pokemon whose every base stat is 100, with 0 EVs, 31 IVs and a
# neutral nature - the same formula estimate_stat() uses for an opponent's
# unrevealed stats, applied to a deliberately median base line. Normal type
# so it has no resistances and no immunities to bias the search either way.
PLACEHOLDER_LEVEL = 100
PLACEHOLDER_MAX_HP = 341
PLACEHOLDER_STAT = 236
PLACEHOLDER_TYPES = ("normal", TYPELESS)
PLACEHOLDER_WEIGHT_KG = 50.0


# ---------------------------------------------------------------------------
# poke-env readers
# ---------------------------------------------------------------------------


def _species_id(mon: Pokemon) -> str:
    """`species`, NOT `base_species`.

    poke-env's `base_species` collapses forms - urshifurapidstrike ->
    urshifu, ogerponwellspring -> ogerpon, landorustherian -> landorus -
    and poke-engine has separate, correctly-statted entries for every one of
    those (verified against the installed build). Handing it the base form
    would give the search the wrong typing and the wrong base stats while
    still resolving to a real id, so nothing would raise.

    `battle_engine/mcts_player.py` deliberately uses `base_species` for the
    opposite reason: there the species is an opaque identity key, and
    `base_species` is stable across an in-battle form change that renames
    `species`. Both are right for their own consumer; they are not
    interchangeable.
    """
    return to_id_str(mon.species)


def _dex_entry(species_id: str) -> Mapping:
    entry = _DEX.get(species_id)
    if entry is None:
        raise UnknownToPokeEngine("species", species_id, "not in poke-env's gen-9 dex either.")
    return entry


def _base_types(species_id: str) -> Tuple[str, str]:
    """Original typing, read from the dex rather than from the live Pokemon.

    See footgun 6 in the module docstring: `mon.types` returns the Tera type
    once a Pokemon has terastallized, and poke-engine applies Tera itself.
    """
    types = [to_id_str(t) for t in _dex_entry(species_id)["types"]]
    return (types[0], types[1] if len(types) > 1 else TYPELESS)


def _dex_profile(species_id: str, level: int) -> Tuple[Tuple[str, str], Dict[str, int], int, float]:
    """Typing, stats, max HP and weight for a species nobody has seen yet.

    Uses the same 0 EVs / 31 IVs / neutral-nature convention `estimate_stat`
    applies to an opponent's revealed-but-unmeasured Pokemon, so a filled
    slot and a revealed one are on the same footing. A filler that knows a
    real spread (M4) can still override the moves and item it returns; the
    spread itself stays this convention until there is a reason to widen the
    seam.
    """
    entry = _dex_entry(species_id)
    base = entry["baseStats"]
    stats = {s: int(int(2 * base[s] + 31) * level / 100) + 5 for s in ("atk", "def", "spa", "spd", "spe")}
    maxhp = int(int(2 * base["hp"] + 31) * level / 100) + level + 10
    return _base_types(species_id), stats, maxhp, float(entry["weightkg"])


def _hp_pair(mon: Pokemon) -> Tuple[int, int]:
    """(hp, maxhp) in real HP points on both sides of the field.

    THE trap this project has already paid for once
    (notes/gotcha-opponent-max-hp-is-on-a-percent-scale.md): poke-env
    reports the opponent's `max_hp` on a 0-100 percent scale, because their
    real HP pool is no more known than their EVs. Routing through
    `estimate_stat(mon, "hp")` - which returns the real stat when poke-env
    knows it and a base-stat estimate when it does not - gives real points
    for both sides, and `current_hp_fraction` is correct on both sides
    regardless of scale, so multiplying the two is scale-safe by
    construction rather than by remembering.
    """
    maxhp = int(estimate_stat(mon, "hp"))
    if mon.fainted:
        return 0, maxhp
    hp = max(1, round(mon.current_hp_fraction * maxhp))
    return min(hp, maxhp), maxhp


def _known_stat(mon: Pokemon, stat: str) -> bool:
    return (mon.stats or {}).get(stat) is not None


def _status_id(mon: Pokemon) -> str:
    # FNT has no poke-engine counterpart; hp == 0 already carries it.
    if mon.status is None or mon.status is Status.FNT:
        return NO_STATUS
    return _STATUS_TO_ENGINE[mon.status]


def _move_ids(mon: Pokemon) -> Tuple[str, ...]:
    return tuple(to_id_str(move_id) for move_id in mon.moves)


def _turns_remaining(started_on: int, current_turn: int, duration: int) -> int:
    """poke-env stores the turn a condition started; poke-engine wants turns
    remaining. Never below 1 for a condition poke-env still lists as active -
    poke-env removes it on expiry, so a computed 0 means the assumed base
    duration was too short (Light Clay, Damp Rock, ...) rather than that the
    condition is gone.
    """
    return max(1, duration - (current_turn - started_on))


# ---------------------------------------------------------------------------
# Building one Pokemon
# ---------------------------------------------------------------------------


def _observed_item(mon: Pokemon) -> Optional[str]:
    """None for "never revealed", "" for "revealed to be holding nothing".

    poke-env keeps these two apart and they mean different things to
    poke-engine (`unknownitem` vs `none`), but the distinction is easy to
    lose: an unrevealed item is the *string* `GenData.UNKNOWN_ITEM`
    ("unknown_item"), which is truthy, while an item that was knocked off or
    consumed is a real `None` (`Pokemon.end_item` sets `self._item = None`,
    verified in poke-env's source). Reading `mon.item` as "falsy means
    unknown" gets both backwards.
    """
    if mon.item == GenData.UNKNOWN_ITEM:
        return None
    return mon.item if mon.item else ""


def _observe_slot(index: int, mon: Pokemon) -> SlotObservation:
    return SlotObservation(
        index=index,
        species=_species_id(mon),
        level=mon.level,
        item=_observed_item(mon),
        ability=to_id_str(mon.ability) if mon.ability else None,
        moves=_move_ids(mon),
        tera_type=_TYPE_TO_ENGINE[mon.tera_type] if mon.tera_type is not None else None,
        fainted=mon.fainted,
        hp_fraction=mon.current_hp_fraction,
    )


def _placeholder_pokemon(slot: str, source: str, log: list) -> Any:
    log.append(Attribution(slot, "species", None, False, source))
    log.append(Attribution(slot, "stats", "placeholder", False, source))
    return poke_engine.Pokemon(
        id="none",
        level=PLACEHOLDER_LEVEL,
        types=PLACEHOLDER_TYPES,
        base_types=PLACEHOLDER_TYPES,
        hp=PLACEHOLDER_MAX_HP,
        maxhp=PLACEHOLDER_MAX_HP,
        attack=PLACEHOLDER_STAT,
        defense=PLACEHOLDER_STAT,
        special_attack=PLACEHOLDER_STAT,
        special_defense=PLACEHOLDER_STAT,
        speed=PLACEHOLDER_STAT,
        weight_kg=PLACEHOLDER_WEIGHT_KG,
        moves=[],
    )


def _resolve_item(
    observed: Optional[str], filled: Optional[str], slot: str, source: str, strict: bool, log: list
) -> str:
    """Observation wins over fill, and both are validated.

    poke-env's three item states are distinct and stay distinct:
    None = not revealed, "" = revealed to be holding nothing (Knock Off),
    anything else = the real item.
    """
    if observed == "":
        log.append(Attribution(slot, "item", None, True, "poke-env"))
        return NO_ITEM
    candidate = observed if observed else filled
    if not candidate:
        log.append(Attribution(slot, "item", UNKNOWN_ITEM, False, source))
        return UNKNOWN_ITEM
    candidate = to_id_str(candidate)
    if strict:
        require_item(candidate)
    elif not is_known_item(candidate):
        # A real, common case, not a typo guard: 59.5% of the Showdown item
        # dex has no poke-engine mechanics. Recorded so a caller can see the
        # search is not modelling this item.
        log.append(Attribution(slot, "item", UNKNOWN_ITEM, False, f"{source}:unmodelled({candidate})"))
        return UNKNOWN_ITEM
    log.append(Attribution(slot, "item", candidate, observed is not None, "poke-env" if observed else source))
    return candidate


def _resolve_ability(
    observed: Optional[str], filled: Optional[str], slot: str, source: str, log: list
) -> str:
    candidate = observed or filled
    if not candidate:
        log.append(Attribution(slot, "ability", NO_ABILITY, False, source))
        return NO_ABILITY
    candidate = require_ability(to_id_str(candidate))
    log.append(Attribution(slot, "ability", candidate, observed is not None, "poke-env" if observed else source))
    return candidate


def _resolve_tera_type(
    observed: Optional[str], filled: Optional[str], slot: str, source: str, log: list
) -> str:
    """Observation wins over fill, same as everywhere else here.

    A Pokemon that has not terastallized yet has no *observable* Tera type,
    but it still has one, and `-tera` move choices are resolved against it -
    so a fill that names it is the difference between a simulated Tera
    getting real STAB and getting `typeless`. Left alone, the value is
    `typeless`, which is also what poke-engine uses for "no second type",
    so an assumed one is always recorded rather than left to look like a
    reading.
    """
    if observed:
        log.append(Attribution(slot, "tera_type", observed, True, "poke-env"))
        return observed
    if not filled:
        return TYPELESS
    candidate = require_tera_type(filled)
    log.append(Attribution(slot, "tera_type", candidate, False, source))
    return candidate


def _resolve_moves(
    observed: Tuple[str, ...], filled: Tuple[str, ...], slot: str, source: str, log: list
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(observed_moves, assumed_moves), both validated, capped at 4 total."""
    kept = tuple(require_move(m) for m in observed)[:MOVESET_SIZE]
    room = MOVESET_SIZE - len(kept)
    added = tuple(require_move(to_id_str(m)) for m in filled if to_id_str(m) not in kept)[:room]
    if kept:
        log.append(Attribution(slot, "moves", kept, True, "poke-env"))
    if added:
        log.append(Attribution(slot, "moves", added, False, source))
    if not kept and not added:
        log.append(Attribution(slot, "moves", (), False, source))
    return kept, added


def _pokemon_from_observation(
    observation: SlotObservation,
    fill: SlotFill,
    mon: Optional[Pokemon],
    slot: str,
    source: str,
    strict_items: bool,
    disabled_moves: frozenset,
    log: list,
) -> Any:
    species = observation.species or (to_id_str(fill.species) if fill.species else None)
    if species is None:
        return _placeholder_pokemon(slot, source, log)
    require_species(species)
    observed_species = observation.species is not None
    log.append(
        Attribution(slot, "species", species, observed_species, "poke-env" if observed_species else source)
    )

    item = _resolve_item(observation.item, fill.item, slot, source, strict_items, log)
    ability = _resolve_ability(observation.ability, fill.ability, slot, source, log)
    kept, added = _resolve_moves(observation.moves, tuple(fill.moves), slot, source, log)

    if mon is not None:
        hp, maxhp = _hp_pair(mon)
        stats = {s: int(estimate_stat(mon, s)) for s in ("atk", "def", "spa", "spd", "spe")}
        base_types = _base_types(_species_id(mon))
        level = mon.level
        weight = mon.weight
        status = _status_id(mon)
        # poke-env only counts turns for TOX and SLP (Pokemon.status_counter's
        # own docstring); poke-engine's sleep_turns means the same thing.
        sleep_turns = mon.status_counter if mon.status is Status.SLP else 0
        terastallized = mon.is_terastallized
        tera_type = _resolve_tera_type(observation.tera_type, fill.tera_type, slot, source, log)
        pp = {to_id_str(mid): move.current_pp for mid, move in mon.moves.items()}
        stats_observed = _known_stat(mon, "atk")
        hp_observed = _known_stat(mon, "hp")
        hp_fraction = 0.0 if mon.fainted else mon.current_hp_fraction
    else:
        # A slot the battle never revealed, for which the filler named a
        # species anyway - the M4 case. Nothing about it was observed, so
        # everything comes from the dex at the level the format fixes.
        level = fill.level if fill.level is not None else PLACEHOLDER_LEVEL
        base_types, stats, maxhp, weight = _dex_profile(species, level)
        hp = maxhp
        status = NO_STATUS
        sleep_turns = 0
        terastallized = False
        tera_type = _resolve_tera_type(None, fill.tera_type, slot, source, log)
        pp = {}
        stats_observed = hp_observed = False
        hp_fraction = 1.0

    # A fill can only supply what was never observed, same rule as everywhere
    # else here. `estimate_stat`'s 0-EV neutral guess counts as unobserved, so a
    # filler that knows the metagame's real spreads replaces it; our own team's
    # stats come from the request JSON and are untouchable.
    stats_source = "poke-env" if stats_observed else source
    hp_source = "poke-env" if hp_observed else source
    if fill.stats:
        supplied = {s: int(v) for s, v in fill.stats.items() if v}
        if not stats_observed:
            replaced = {s: supplied[s] for s in ("atk", "def", "spa", "spd", "spe") if s in supplied}
            if replaced:
                stats = {**stats, **replaced}
                stats_source = f"{source}:spread"
        if not hp_observed and "hp" in supplied:
            maxhp = supplied["hp"]
            # `hp_fraction` survives the rescale by construction - poke-env
            # reports an opponent's HP as a percentage, so the fraction is the
            # only part of it that was ever real.
            hp = 0 if hp_fraction <= 0 else max(1, min(maxhp, round(hp_fraction * maxhp)))
            hp_source = f"{source}:spread"

    log.append(Attribution(slot, "stats", stats, stats_observed, stats_source))
    log.append(Attribution(slot, "max_hp", maxhp, hp_observed, hp_source))

    moves = [
        poke_engine.Move(id=move_id, pp=pp.get(move_id, 16), disabled=move_id in disabled_moves)
        for move_id in kept
    ] + [poke_engine.Move(id=move_id, pp=16, disabled=False) for move_id in added]

    return poke_engine.Pokemon(
        id=species,
        level=level,
        # `types` and `base_types` both carry the ORIGINAL typing; poke-engine
        # substitutes the Tera type itself when `terastallized` is set.
        types=base_types,
        base_types=base_types,
        hp=hp,
        maxhp=maxhp,
        ability=ability,
        base_ability=ability,
        item=item,
        # Nature and EVs are never observable from a battle log, on either
        # side. They are inert here because explicit stats are supplied
        # above, and poke-engine only recalculates from base stats on a
        # form change - recorded once per side rather than per slot.
        nature="serious",
        attack=stats["atk"],
        defense=stats["def"],
        special_attack=stats["spa"],
        special_defense=stats["spd"],
        speed=stats["spe"],
        status=status,
        rest_turns=0,
        sleep_turns=sleep_turns,
        weight_kg=weight,
        moves=moves,
        terastallized=terastallized,
        tera_type=tera_type,
    )


# ---------------------------------------------------------------------------
# Building one Side
# ---------------------------------------------------------------------------


def _volatile_statuses(mon: Optional[Pokemon], slot: str, log: list) -> set:
    """poke-env `Effect`s that poke-engine also models, by normalized name.

    Derived by probing rather than transcribed: 85 of poke-env's 224
    `Effect` members round-trip through poke-engine's own
    `PokemonVolatileStatus`. The rest are dropped and listed in the ledger -
    silently keeping them would insert `NONE` into the side's volatile set
    (footgun 3), which is worse than dropping.
    """
    if mon is None:
        return set()
    kept, dropped = set(), []
    for effect in mon.effects:
        name = to_id_str(effect.name)
        if is_known_volatile_status(name):
            kept.add(name)
        else:
            dropped.append(name)
    if kept:
        log.append(Attribution(slot, "volatile_statuses", tuple(sorted(kept)), True, "poke-env"))
    if dropped:
        log.append(Attribution(slot, "volatile_statuses_dropped", tuple(sorted(dropped)), False, "not-modelled"))
    return kept


def _side_conditions(conditions: Mapping, turn: int, side: str, toxic_count: int, log: list) -> Any:
    """poke-engine's `SideConditions` fields are read-only from Python
    (`#[pyclass(get_all)]`, no setters), so this builds the kwargs and
    constructs once rather than assigning field by field.
    """
    fields = {
        # poke-env stores these two as real layer counts (its own
        # STACKABLE_CONDITIONS), which is exactly what poke-engine wants.
        "spikes": conditions.get(SideCondition.SPIKES, 0),
        "toxic_spikes": conditions.get(SideCondition.TOXIC_SPIKES, 0),
        # These two are 0/1 flags in poke-engine but turn numbers in
        # poke-env, so presence is the only thing carried across.
        "stealth_rock": 1 if SideCondition.STEALTH_ROCK in conditions else 0,
        "sticky_web": 1 if SideCondition.STICKY_WEB in conditions else 0,
        "toxic_count": toxic_count,
    }
    for condition, (attribute, duration) in _SCREENS.items():
        if condition not in conditions:
            continue
        remaining = _turns_remaining(conditions[condition], turn, duration)
        fields[attribute] = remaining
        # Turns remaining is derived, not observed: poke-env records only the
        # turn a screen started, and Light Clay (which extends it to 8) is
        # not visible until the item is revealed.
        log.append(Attribution(side, attribute, remaining, False, "assumed-base-duration"))
    return poke_engine.SideConditions(**fields)


def _last_used_move(
    active: Optional[Pokemon], built_moves: Sequence, volatiles: set, side: str, log: list
) -> str:
    """poke-engine's `Side.last_used_move`, as `move:<index>` or `move:none`.

    This field looks inert and is not. `generate_instructions` **panics**
    (not raises - see this module's footgun 4) with "Encore should not be
    active when last used move is not a move" whenever a side carries the
    `encore` volatile and this is anything but `Move(_)`: encore.rs looks up
    `side.get_active_immutable().moves[&last_used_move]` unconditionally.
    Leaving the field at its `move:none` default therefore turns every
    Encored position into a hard crash of the whole process, which on the
    real ladder is a forfeited game. Found by
    battle_engine/fidelity.py's corpus run, not by reading the code.

    poke-env tracks the move via `Move.is_last_used`, which is exactly the
    move Encore locks, so the observation is available. When it is not -
    the Pokemon has not moved since it came in - `encore` is *dropped* from
    the volatile set rather than pointed at an arbitrary move index: a wrong
    index would silently force the wrong move every turn, which is worse
    than not modelling Encore at all.
    """
    last = active.last_move if active is not None else None
    if last is not None:
        last_id = to_id_str(last.id)
        for index, move in enumerate(built_moves):
            if move.id == last_id:
                log.append(Attribution(side, "last_used_move", last_id, True, "poke-env"))
                return f"move:{index}"
    if "encore" in volatiles:
        volatiles.discard("encore")
        log.append(Attribution(side, "volatile_statuses_dropped", ("encore",), False, "no-last-used-move"))
    return "move:none"


def side_observation_from_team(
    mons: Sequence[Pokemon], *, is_ours: bool, format_id: Optional[str] = None
) -> SideObservation:
    """What a battle has revealed about one side, as a filler sees it.

    Public because the fill is worth evaluating on its own, separately from
    the state it ends up in: M4's set-prediction evaluation asks a filler what
    it thinks at turn N and scores that against what the battle eventually
    reveals, with no poke-engine `State` in the loop at all.
    """
    revealed = tuple(_observe_slot(i, mon) for i, mon in enumerate(mons))
    unrevealed = tuple(SlotObservation(index=i) for i in range(len(mons), TEAM_SIZE))
    return SideObservation(
        is_ours=is_ours, format_id=format_id, team_size=TEAM_SIZE, slots=revealed + unrevealed
    )


def observe_side(battle: AbstractBattle, *, ours: bool) -> SideObservation:
    """`side_observation_from_team` straight off a live or replayed battle."""
    team = battle.team if ours else battle.opponent_team
    return side_observation_from_team(list(team.values()), is_ours=ours, format_id=battle.format)


def _build_side(
    team: Mapping[str, Pokemon],
    active: Optional[Pokemon],
    conditions: Mapping,
    turn: int,
    filler: UnknownFiller,
    is_ours: bool,
    format_id: Optional[str],
    force_switch: bool,
    disabled_moves: frozenset,
    side: str,
    strict_items: bool,
    log: list,
) -> Any:
    mons = list(team.values())
    if len(mons) > TEAM_SIZE:
        raise ValueError(f"{side} has {len(mons)} Pokemon, expected at most {TEAM_SIZE}")
    if active is None:
        # Every real mid-battle state has an active Pokemon on both sides;
        # team preview does not, and this shape has nowhere to put "nobody
        # is out yet" (same gap encoding.py's own adapter documents).
        raise ValueError(
            f"{side} has no active Pokemon - a poke-engine State always has one active "
            "slot per side, so team preview needs its own handling, not this translator"
        )

    side_observation = side_observation_from_team(mons, is_ours=is_ours, format_id=format_id)
    fills = list(filler.fill_side(side_observation))
    if len(fills) != TEAM_SIZE:
        raise ValueError(
            f"{filler.name} returned {len(fills)} fills for a {TEAM_SIZE}-slot side; "
            "fill_side must return exactly team_size fills, index-aligned with observation.slots"
        )

    pokemon = []
    for observation, fill in zip(side_observation.slots, fills):
        slot = f"{side}:{observation.index}"
        mon = mons[observation.index] if observation.index < len(mons) else None
        pokemon.append(
            _pokemon_from_observation(
                observation,
                fill,
                mon,
                slot,
                filler.name,
                strict_items,
                disabled_moves if mon is not None and mon is active else frozenset(),
                log,
            )
        )

    # poke-engine keeps boosts, the substitute and the toxic counter on the
    # Side, not on the Pokemon - they belong to the slot that is active.
    boosts = active.boosts
    active_index = mons.index(active)
    # Nature and EVs are inert on a poke-engine Pokemon once explicit stats are
    # set (it recomputes from base stats only on a form change), so the spread
    # question is answered by the per-slot "stats" attribution above and its
    # source, not by these two fields. They serialize as SERIOUS and 85 EVs
    # across the board - poke-engine's own defaults - and are recorded once
    # here so a reader of the ledger knows they were left there on purpose.
    log.append(Attribution(side, "nature_and_evs", "serious/85s (inert)", False, "not-observable"))

    # poke-engine tracks the badly-poisoned counter per SIDE, not per
    # Pokemon, so it is the active Pokemon's counter that goes here.
    toxic_count = active.status_counter if active.status is Status.TOX else 0
    side_conditions = _side_conditions(conditions, turn, side, toxic_count, log)

    substitute_health = 0
    if Effect.SUBSTITUTE in active.effects:
        # Showdown builds a Substitute at exactly 1/4 of max HP; the amount
        # left after it has taken a hit is not in the protocol.
        substitute_health = _hp_pair(active)[1] // 4
        log.append(Attribution(side, "substitute_health", substitute_health, False, "assumed-full"))

    volatiles = _volatile_statuses(active, f"{side}:{active_index}", log)
    last_used_move = _last_used_move(active, pokemon[active_index].moves, volatiles, side, log)

    return poke_engine.Side(
        pokemon=pokemon,
        active_index=str(active_index),
        side_conditions=side_conditions,
        volatile_statuses=volatiles,
        last_used_move=last_used_move,
        substitute_health=substitute_health,
        attack_boost=boosts.get("atk", 0),
        defense_boost=boosts.get("def", 0),
        special_attack_boost=boosts.get("spa", 0),
        special_defense_boost=boosts.get("spd", 0),
        speed_boost=boosts.get("spe", 0),
        accuracy_boost=boosts.get("accuracy", 0),
        evasion_boost=boosts.get("evasion", 0),
        force_switch=force_switch,
    )


# ---------------------------------------------------------------------------
# The translator
# ---------------------------------------------------------------------------


def _terrain(fields: Mapping, turn: int, log: list) -> Tuple[str, int]:
    """The single active terrain, as (id, turns_remaining).

    `battle.fields` also carries non-terrain effects (Trick Room, Gravity,
    Magic Room), so this filters rather than taking the first entry - the
    same correction `encoding.py::_poke_env_terrain` documents.
    """
    for field, engine_id in _TERRAIN_TO_ENGINE.items():
        if field in fields:
            remaining = _turns_remaining(fields[field], turn, _TERRAIN_DURATION)
            log.append(Attribution("field", "terrain_turns_remaining", remaining, False, "assumed-base-duration"))
            return engine_id, remaining
    return NO_TERRAIN, 0


def _weather(weather: Mapping, turn: int, log: list) -> Tuple[str, int]:
    for poke_env_weather, started_on in weather.items():
        engine_id = _WEATHER_TO_ENGINE.get(poke_env_weather)
        if engine_id is None:
            # Delta Stream and poke-env's own UNKNOWN placeholder. Dropping
            # is the only option - passing the name through would panic
            # (footgun 4), and there is no neutral stand-in for it.
            log.append(Attribution("field", "weather", NO_WEATHER, False, f"not-modelled({poke_env_weather.name})"))
            continue
        remaining = _turns_remaining(started_on, turn, _WEATHER_DURATION)
        log.append(Attribution("field", "weather", engine_id, True, "poke-env"))
        log.append(Attribution("field", "weather_turns_remaining", remaining, False, "assumed-base-duration"))
        return engine_id, remaining
    return NO_WEATHER, 0


def _disabled_moves(battle: AbstractBattle) -> frozenset:
    """Move ids the active Pokemon cannot pick this turn.

    `battle.available_moves` is the server's own answer, so it covers
    Choice lock, Taunt, Encore, Disable, Torment and 0 PP without this
    module reimplementing any of them. It is empty on a forced-switch turn
    and at team preview, where "nothing is available" does not mean "every
    move is disabled" - hence the emptiness check rather than a plain
    difference.
    """
    available = battle.available_moves
    if not available or battle.active_pokemon is None:
        return frozenset()
    allowed = {to_id_str(move.id) for move in available}
    return frozenset(_move_ids(battle.active_pokemon)) - allowed


def state_from_poke_env(
    battle: AbstractBattle,
    *,
    filler: Optional[UnknownFiller] = None,
    strict_items: bool = False,
) -> TranslationResult:
    """Translate `battle`'s current turn into a poke-engine `State`.

    Our side is `side_one` and the opponent is `side_two`, matching what
    poke-engine's own `mcts`/`id` entry points return results for.

    Raises `UnknownToPokeEngine` for any species, move or ability
    poke-engine does not know - and for an item too when `strict_items` is
    set, which is right when the caller owns the values (a packed team from
    `teams.py`) and wrong against a live battle, where 59.5% of the real
    item dex has no poke-engine mechanics. Raises `ValueError` at team
    preview, where neither side has an active Pokemon and this shape has
    nowhere to put that.

    Every attribute that could have been unknown is in the returned
    `TranslationResult.attributions`, marked observed or assumed. Callers
    that care about fidelity should read `.assumed()` rather than trusting
    the state to be a faithful picture of the battle.
    """
    filler = filler if filler is not None else RevealedOnlyFiller()
    log: list = []
    turn = battle.turn
    format_id = battle.format

    side_one = _build_side(
        team=battle.team,
        active=battle.active_pokemon,
        conditions=battle.side_conditions,
        turn=turn,
        filler=filler,
        is_ours=True,
        format_id=format_id,
        force_switch=bool(battle.force_switch),
        disabled_moves=_disabled_moves(battle),
        side="p1",
        strict_items=strict_items,
        log=log,
    )
    side_two = _build_side(
        team=battle.opponent_team,
        active=battle.opponent_active_pokemon,
        conditions=battle.opponent_side_conditions,
        turn=turn,
        filler=filler,
        is_ours=False,
        format_id=format_id,
        # The opponent's forced switches are not in our view of the battle.
        force_switch=False,
        disabled_moves=frozenset(),
        side="p2",
        strict_items=strict_items,
        log=log,
    )

    weather_id, weather_turns = _weather(battle.weather, turn, log)
    terrain_id, terrain_turns = _terrain(battle.fields, turn, log)
    trick_room = Field.TRICK_ROOM in battle.fields
    trick_room_turns = (
        _turns_remaining(battle.fields[Field.TRICK_ROOM], turn, _TRICK_ROOM_DURATION) if trick_room else 0
    )

    state = poke_engine.State(
        side_one=side_one,
        side_two=side_two,
        weather=weather_id,
        weather_turns_remaining=weather_turns,
        terrain=terrain_id,
        terrain_turns_remaining=terrain_turns,
        trick_room=trick_room,
        trick_room_turns_remaining=trick_room_turns,
        team_preview=False,
    )
    return TranslationResult(state=state, attributions=tuple(log))
