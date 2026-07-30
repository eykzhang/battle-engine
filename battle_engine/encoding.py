"""Fixed-size battle-state encoding — the core Phase-2 lesson (see project
roadmap): design a vector representation of a battle state, unit-test it hard.

Two adapters map either a live poke-env `AbstractBattle` or one turn's
`state` dict from a Metamon parsed replay into a common intermediate
`BattleView` (this project's own small dataclasses, not poke-env's or
metamon's), then `encode()` walks that one shape into a fixed-length numpy
vector. Same "one function, two adapters" shape `search.py`'s
`_project_after_action` already uses for live-vs-simulated states.

The replay schema below was verified against a real downloaded file (see
scripts/fetch_replay_sample.py), not assumed from metamon's documented
UniversalState/ReplayState field names — those turned out not to match the
raw .json.lz4 layout 1:1 (e.g. no top-level `turnlist`/`winner`; a replay
file is actually `{"states": [...], "actions": [...]}`, each state already
POV-relative and pre-flattened: `hp_pct` not raw HP, `base_atk`/`atk_boost`
as separate scalars, space-separated type strings with a literal "notype"
placeholder for single-typed Pokemon).

Corrections from an independent review (2026-07-30), after this module had
already been built and unit-tested once — both caught by checking real
downloaded replay data and poke-env's actual source, not by inspection alone:

- **My-side bench, fainted teammates**: Metamon's per-state
  `available_switches` silently drops fainted teammates entirely (verified:
  0 of 3247 real switch-list entries are fainted) — Showdown just stops
  offering them as a switch target. A single replay `state` therefore *looks*
  identical whether a teammate fainted or was simply never brought, which is
  wrong: `battle_view_from_replay_state` (single-state, kept for
  ad-hoc/debugging use) still has this gap, but `battle_views_from_replay`
  (state**s**, plural — use this for real work, e.g. `dataset.py`) tracks a
  running per-replay roster and reconstructs fainted teammates as
  known-but-fainted slots, matching what `battle_view_from_poke_env` already
  gets for free from poke-env's `battle.team` (which keeps fainted mons).
- **Bench slot ordering**: both adapters now sort bench slots by species name.
  Replay data's `available_switches` order isn't stable — measured 10.1% of
  consecutive same-replay states reordering a teammate with no faint
  involved — so a fixed slot index carried no consistent identity before
  this; poke-env's own team-dict order isn't guaranteed stable either.
  Sorting by species name (available on both sides, doesn't change turn to
  turn) fixes both.

Named simplifications (same convention as damage.py/evaluation.py):
- Opponent bench detail: only the opponent's current active Pokemon is
  encoded in full; the rest of their team is summarized as a single "fraction
  still alive" scalar (`opponents_remaining` in the replay data), not
  per-mon slots. Unlike the my-side fainted-teammate gap above, this one
  isn't reconstructable even with the full turn sequence: a replay state
  doesn't carry species/stats for opponent Pokemon that aren't currently
  active, only a remaining-count and a teampreview species list. A live
  poke-env battle actually knows more than this (accumulates revealed
  opponent-team detail turn over turn), so this is a real, deliberate
  asymmetry, not a wash.
- Hazards report only the single most-recently-changed condition per side,
  not every condition simultaneously active, and not stack count. This was
  originally meant as "presence-only, for parity between the two adapters" —
  review found that claim was actually false: real replay data's
  `player_conditions`/`opponent_conditions` field is single-valued and
  overwritten on every new hazard/screen event (verified: 267 states show
  `stealthrock`, 86 show `spikes`, zero show both, despite that being a
  completely ordinary simultaneous board state). `_poke_env_hazards` now
  deliberately narrows the live side to match — real fidelity a live battle
  could otherwise provide, given up on purpose so training data and
  live-inference data mean the same thing. A second review pass fixed a
  real (not deliberate) gap in this: Aurora Veil and Tailwind weren't in
  `_HAZARD_TOKENS` at all, so both adapters silently ignored them even
  though poke-env tracks their turn number exactly like Stealth Rock/
  Reflect/etc. That same pass also found the *original* justification for
  narrowing at all — "no removal signal for Defog/Rapid Spin/screen-expiry"
  — is factually shaky: the field does revert to `noconditions` on a clean
  sweep (e.g. `stealthrock→noconditions` 277×, `auroraveil→noconditions`
  25× across a 600-replay sample), so an add-on-new-token /
  clear-on-noconditions reconstruction may be more tractable than assumed
  when this simplification was first chosen. Not attempted here — revisit
  if it turns out to matter for the trained model, since it's a real
  question, not something to quietly redecide unilaterally.
- Terrain has the same single-valued masking as hazards, for the same
  reason (`battle_field` is one token, and its real vocabulary includes
  non-terrain entries: `trickroom`, `gravity`). `_poke_env_terrain` ranks
  all of `battle.fields` by turn and only reports the winner if it's
  actually a terrain, matching what a replay-derived state can show.
  Trick Room/Gravity themselves aren't modeled as their own feature -
  same "not attempted, revisit if it matters" status as the hazard question
  above, not a design decision made either way yet.
- Level is omitted: every replay here is a standard, fully-leveled (100)
  format, so it carries no signal.
- Abilities aren't a dimension of their own. Ability identity is asymmetric
  info in real play (always known for your own Pokemon, `None` until
  revealed for the opponent's — poke-env already models this via
  `mon.ability`), and gen 9 has a few hundred distinct abilities, too many
  for a one-hot in a hand-built vector. Rather than an identity feature, the
  known subset of *type-immunity* abilities (Levitate, Water Absorb, Flash
  Fire, Wonder Guard, ...) is folded directly into the active-vs-active
  matchup-score dimension via `_TYPE_IMMUNITY_ABILITIES` — the same
  "hand-engineered prior over raw features" reasoning already used for the
  type chart itself. Abilities with non-type-effect consequences (Intimidate,
  Speed Boost, Protean, Multiscale, Unaware, ...) aren't modeled at all;
  representing those well is a job for a learned ability embedding once an
  actual model exists, not something to hand-derive here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import STACKABLE_CONDITIONS, SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data import GenData

_TYPE_CHART = GenData.from_gen(9).type_chart
_ALL_TYPES = list(PokemonType)

_STATUSES = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]
_STAT_NAMES = ["hp", "atk", "def", "spa", "spd", "spe"]
_BOOST_NAMES = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
_BASE_STAT_SCALE = 255.0  # Blissey's base HP, the real max base stat in the dex

MAX_BENCH = 5

# Verified against the real replay vocabulary across all currently
# downloaded replays (review, 2026-07-30): these 8 are every non-empty
# value player_conditions/opponent_conditions ever takes. An earlier version
# of this list was missing auroraveil/tailwind entirely, a real gap (not a
# documented tradeoff) - they were silently invisible to the live adapter
# too, since _poke_env_hazards only ever looked at the other 6.
_HAZARD_TOKENS = [
    "stealthrock", "spikes", "toxicspikes", "stickyweb",
    "reflect", "lightscreen", "auroraveil", "tailwind",
]
_HAZARD_SIDE_CONDITIONS: Dict[str, SideCondition] = {
    "stealthrock": SideCondition.STEALTH_ROCK,
    "spikes": SideCondition.SPIKES,
    "toxicspikes": SideCondition.TOXIC_SPIKES,
    "stickyweb": SideCondition.STICKY_WEB,
    "reflect": SideCondition.REFLECT,
    "lightscreen": SideCondition.LIGHT_SCREEN,
    "auroraveil": SideCondition.AURORA_VEIL,
    "tailwind": SideCondition.TAILWIND,
}

_WEATHER_NAMES = ["sandstorm", "raindance", "sunnyday", "snow"]
_WEATHER_FROM_POKE_ENV: Dict[Weather, str] = {
    Weather.SANDSTORM: "sandstorm",
    Weather.RAINDANCE: "raindance",
    Weather.SUNNYDAY: "sunnyday",
    Weather.SNOWSCAPE: "snow",
}

_TERRAIN_NAMES = ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
_TERRAIN_FROM_POKE_ENV: Dict[Field, str] = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
}

# Abilities that grant a hard immunity to one attacking type (multiplier -> 0),
# keyed the same way poke-env/Metamon both already normalize ability names
# (lowercase, no spaces/punctuation - poke-env via to_id_str, Metamon's replay
# data the same way natively). Not exhaustive of every ability that touches
# type effectiveness (partial resistances like Thick Fat, move-flag immunities
# like Soundproof/Bulletproof are out of scope) - see module docstring.
_TYPE_IMMUNITY_ABILITIES: Dict[str, PokemonType] = {
    "levitate": PokemonType.GROUND,
    "waterabsorb": PokemonType.WATER,
    "stormdrain": PokemonType.WATER,
    "dryskin": PokemonType.WATER,
    "voltabsorb": PokemonType.ELECTRIC,
    "lightningrod": PokemonType.ELECTRIC,
    "motordrive": PokemonType.ELECTRIC,
    "sapsipper": PokemonType.GRASS,
    "flashfire": PokemonType.FIRE,
    "eartheater": PokemonType.GROUND,
    "wellbakedbody": PokemonType.FIRE,
}
_UNKNOWN_ABILITY_TOKEN = "unknownability"  # Metamon's placeholder for "not yet revealed"

_POKEMON_VEC_LEN = (
    1  # known
    + 1  # hp_fraction
    + 1  # fainted
    + len(_STATUSES)
    + len(_ALL_TYPES)
    + len(_BOOST_NAMES)
    + len(_STAT_NAMES)
)
VECTOR_LEN = (
    _POKEMON_VEC_LEN * (1 + MAX_BENCH + 1)  # my active, my bench, opponent active
    + 1  # opponent fraction remaining
    + 2 * len(_HAZARD_TOKENS)  # hazards, both sides
    + len(_WEATHER_NAMES)
    + len(_TERRAIN_NAMES)
    + 1  # active-vs-active type matchup score
)


@dataclass
class PokemonView:
    known: bool
    hp_fraction: float
    fainted: bool
    status: Optional[Status]
    types: Tuple[PokemonType, ...]
    boosts: Dict[str, int]
    base_stats: Dict[str, int]
    # Not itself an encoded dimension (see module docstring) - consumed only by
    # _type_multiplier to correct the matchup-score dimension for known
    # type-immunity abilities. None means "no ability" or "not yet revealed";
    # those are indistinguishable from the outside, same as in a real battle.
    ability: Optional[str] = None

    @staticmethod
    def unknown() -> "PokemonView":
        return PokemonView(
            known=False,
            hp_fraction=0.0,
            fainted=False,
            status=None,
            types=(),
            boosts={name: 0 for name in _BOOST_NAMES},
            base_stats={name: 0 for name in _STAT_NAMES},
            ability=None,
        )


@dataclass
class BattleView:
    my_active: PokemonView
    my_bench: Sequence[PokemonView]
    opp_active: PokemonView
    opp_remaining_fraction: float
    my_hazards: set
    opp_hazards: set
    weather: Optional[str]
    terrain: Optional[str]


def _pad_bench(bench: list) -> list:
    assert len(bench) <= MAX_BENCH, (
        f"{len(bench)} bench slots exceeds MAX_BENCH={MAX_BENCH} - silently "
        "truncating would drop a real Pokemon rather than surface a bug"
    )
    return bench + [PokemonView.unknown() for _ in range(MAX_BENCH - len(bench))]


# --- poke-env (live battle) adapter -----------------------------------------


def _poke_env_pokemon_view(mon: Optional[Pokemon]) -> PokemonView:
    if mon is None:
        return PokemonView.unknown()
    return PokemonView(
        known=True,
        hp_fraction=mon.current_hp_fraction,
        fainted=mon.fainted,
        status=mon.status if mon.status != Status.FNT else None,
        types=tuple(t for t in mon.types if t is not None),
        boosts=dict(mon.boosts),
        base_stats=dict(mon.base_stats),
        ability=mon.ability,  # None if not yet revealed, already normalized (to_id_str)
    )


def _poke_env_hazards(side_conditions: dict) -> set:
    """The single most-recently-set hazard/screen, matching the real replay
    data's single-valued field (see module docstring) rather than every
    condition simultaneously active - a deliberate fidelity trade for
    train/inference parity.

    poke-env's own STACKABLE_CONDITIONS (side_condition.py) is the
    authoritative list of which conditions it tracks as a stack *count*
    (Spikes, Toxic Spikes) rather than the *turn number* it stores for
    everything else (verified against abstract_battle.py's _side_start) -
    used directly here rather than a hand-maintained guess at the split, so
    this stays correct if poke-env's own classification ever changes.
    True recency across a stack-counted and a turn-tracked condition can't
    be compared directly, so turn-tracked conditions are ranked by turn
    when any are active; a stackable condition is only reported when none
    are, tie-broken by _HAZARD_SIDE_CONDITIONS' fixed order. An
    approximation, not exact reconstruction - documented rather than
    silently assumed.
    """
    turn_tracked = {
        token: side_conditions[condition]
        for token, condition in _HAZARD_SIDE_CONDITIONS.items()
        if condition in side_conditions and condition not in STACKABLE_CONDITIONS
    }
    if turn_tracked:
        return {max(turn_tracked, key=turn_tracked.get)}

    for token, condition in _HAZARD_SIDE_CONDITIONS.items():
        if condition in side_conditions and condition in STACKABLE_CONDITIONS:
            return {token}
    return set()


def _poke_env_weather(weather: dict) -> Optional[str]:
    for w in weather:
        name = _WEATHER_FROM_POKE_ENV.get(w)
        if name is not None:
            return name
    return None


def _poke_env_terrain(fields: dict) -> Optional[str]:
    """Real bug caught by review: the previous version returned the first
    terrain-type entry found in `fields`, ignoring non-terrain field effects
    entirely (Trick Room, Gravity) - so if Trick Room was set *more
    recently* than an active terrain, this still reported the terrain,
    while the replay adapter's single-valued battle_field would correctly
    show "trickroom" and map to no terrain. Matches that now: rank the
    single most-recently-set field effect by turn (battle.fields stores a
    real turn number for every entry - verified in abstract_battle.py,
    `self._fields[field] = self.turn` unconditionally, no stackable-count
    exception like side_conditions has), and only report it as a terrain if
    it actually is one.
    """
    if not fields:
        return None
    most_recent = max(fields, key=fields.get)
    return _TERRAIN_FROM_POKE_ENV.get(most_recent)


def battle_view_from_poke_env(battle: AbstractBattle) -> BattleView:
    if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
        # Real gap caught by review: at team preview, poke-env's
        # battle.team already has all 6 mons (populated from the
        # teampreview request - verified in poke-env's battle.py) but
        # battle.active_pokemon is still None (nothing has switched in
        # yet). Without this guard, "bench = everything except the active
        # one" keeps all 6, and _pad_bench's overflow assert fires with a
        # confusing "6 bench slots exceeds MAX_BENCH" message that has
        # nothing to do with the actual problem (no active Pokemon yet).
        # This BattleView shape has no representation for "no active
        # Pokemon chosen yet" - callers (e.g. a future team-preview-aware
        # search) need to handle that decision separately, not via this
        # function. Nothing in this codebase calls this during team
        # preview today - search.py already guards on active_pokemon is
        # None before doing anything encoding-related - but milestone E
        # wiring a model into a Player could plausibly do so.
        raise ValueError(
            "battle_view_from_poke_env requires both active Pokemon to be "
            "chosen (not team preview) - no active_pokemon means there's no "
            "well-defined 'bench' to encode yet"
        )
    bench_mons = sorted(
        (mon for mon in battle.team.values() if mon is not battle.active_pokemon),
        key=lambda mon: mon.base_species,
    )
    bench = [_poke_env_pokemon_view(mon) for mon in bench_mons]
    opp_fainted = sum(1 for mon in battle.opponent_team.values() if mon.fainted)
    return BattleView(
        my_active=_poke_env_pokemon_view(battle.active_pokemon),
        my_bench=_pad_bench(bench),
        opp_active=_poke_env_pokemon_view(battle.opponent_active_pokemon),
        opp_remaining_fraction=(6 - opp_fainted) / 6,
        my_hazards=_poke_env_hazards(battle.side_conditions),
        opp_hazards=_poke_env_hazards(battle.opponent_side_conditions),
        weather=_poke_env_weather(battle.weather),
        terrain=_poke_env_terrain(battle.fields),
    )


# --- Metamon replay-state adapter -------------------------------------------


def _species_key(mon: dict) -> str:
    # base_species, not name: a real bug caught against actual data - Metamon
    # renames a Pokemon's "name" field on in-battle form changes (observed:
    # Terapagos -> "terapagosterastal" on Tera, Minior -> "miniormeteor" on
    # its shield-break trigger), so name-based identity treated one physical
    # teammate as two, overflowing a real replay's bench past MAX_BENCH.
    # base_species is stable across these (confirmed against the same data).
    return mon["base_species"]


def _replay_types(mon: dict) -> Tuple[PokemonType, ...]:
    return tuple(PokemonType.from_name(t) for t in mon["types"].split() if t != "notype")


def _replay_ability(mon: dict) -> Optional[str]:
    return None if mon["ability"] == _UNKNOWN_ABILITY_TOKEN else mon["ability"]


def _replay_pokemon_view(mon: Optional[dict]) -> PokemonView:
    if mon is None:
        return PokemonView.unknown()
    status_token = mon["status"]
    status = None
    if status_token not in ("nostatus", "fnt"):
        status = Status[status_token.upper()]
    return PokemonView(
        known=True,
        hp_fraction=mon["hp_pct"],
        fainted=status_token == "fnt",
        status=status,
        types=_replay_types(mon),
        boosts={name: mon[f"{name}_boost"] for name in _BOOST_NAMES},
        base_stats={name: mon[f"base_{name}"] for name in _STAT_NAMES},
        ability=_replay_ability(mon),
    )


def _replay_pokemon_view_fainted(mon: dict) -> PokemonView:
    """A previously-seen teammate no longer active or in available_switches -
    i.e. it fainted since it was last seen. Real per-state replay data
    doesn't record fainted teammates in available_switches at all (verified:
    0 of 3247 real switch-list entries are fainted; Showdown just stops
    offering them), unlike poke-env's live battle.team, which keeps them.
    Reconstructed here from the last snapshot this replay had of the mon,
    with hp/status/boosts overwritten to reflect having fainted - its exact
    HP right before fainting isn't recoverable from this data, only that
    it's now 0, and boosts don't survive a faint anyway.
    """
    return PokemonView(
        known=True,
        hp_fraction=0.0,
        fainted=True,
        status=None,
        types=_replay_types(mon),
        boosts={name: 0 for name in _BOOST_NAMES},
        base_stats={name: mon[f"base_{name}"] for name in _STAT_NAMES},
        ability=_replay_ability(mon),
    )


def _replay_hazards(conditions: str) -> set:
    return set(conditions.split()) & set(_HAZARD_TOKENS)


def _replay_weather(token: str) -> Optional[str]:
    return token if token in _WEATHER_NAMES else None


def _replay_terrain(token: str) -> Optional[str]:
    return token if token in _TERRAIN_NAMES else None


def battle_view_from_replay_state(state: dict) -> BattleView:
    """Single-state mapper - kept for ad-hoc/debugging use on one turn in
    isolation. Cannot reconstruct fainted teammates (see module docstring):
    a fainted mon is just absent from `available_switches` with no way to
    tell "fainted" from "never on this team" without the rest of the replay.
    Use battle_views_from_replay for real work (e.g. dataset building).
    """
    bench_mons = sorted(state["available_switches"], key=_species_key)
    bench = [_replay_pokemon_view(mon) for mon in bench_mons]
    return BattleView(
        my_active=_replay_pokemon_view(state["player_active_pokemon"]),
        my_bench=_pad_bench(bench),
        opp_active=_replay_pokemon_view(state["opponent_active_pokemon"]),
        opp_remaining_fraction=state["opponents_remaining"] / 6,
        my_hazards=_replay_hazards(state["player_conditions"]),
        opp_hazards=_replay_hazards(state["opponent_conditions"]),
        weather=_replay_weather(state["weather"]),
        terrain=_replay_terrain(state["battle_field"]),
    )


def battle_views_from_replay(states: list) -> list:
    """Maps a whole replay's state sequence to one BattleView per state,
    reconstructing fainted teammates that a single isolated state can't
    (see module docstring and battle_view_from_replay_state). Needs the full
    sequence: "this teammate fainted" can only be inferred by noticing it's
    no longer active or switchable after having been seen earlier in the
    same replay.

    Unverified edge case, not seen in the downloaded sample and not chased
    down: if Metamon's replay parser records Illusion Zoroark under its
    disguised species rather than its true one, the roster here would see
    two "names" for one physical teammate, which could push a reconstructed
    bench past 5 slots and trip _pad_bench's assert. Left as a loud failure
    rather than silently handled, consistent with why that assert exists.
    """
    seen: Dict[str, dict] = {}
    views = []
    for state in states:
        active = state["player_active_pokemon"]
        switches = state["available_switches"]
        seen[_species_key(active)] = active
        for mon in switches:
            seen[_species_key(mon)] = mon

        alive_names = {_species_key(active)} | {_species_key(m) for m in switches}
        bench_entries = {_species_key(m): (m, False) for m in switches}
        for name, mon in seen.items():
            if name not in alive_names:
                bench_entries[name] = (mon, True)

        bench = [
            _replay_pokemon_view_fainted(mon) if fainted else _replay_pokemon_view(mon)
            for name, (mon, fainted) in sorted(bench_entries.items())
        ]

        views.append(BattleView(
            my_active=_replay_pokemon_view(active),
            my_bench=_pad_bench(bench),
            opp_active=_replay_pokemon_view(state["opponent_active_pokemon"]),
            opp_remaining_fraction=state["opponents_remaining"] / 6,
            my_hazards=_replay_hazards(state["player_conditions"]),
            opp_hazards=_replay_hazards(state["opponent_conditions"]),
            weather=_replay_weather(state["weather"]),
            terrain=_replay_terrain(state["battle_field"]),
        ))
    return views


# --- encoding ----------------------------------------------------------------


def _one_hot_status(status: Optional[Status]) -> np.ndarray:
    vec = np.zeros(len(_STATUSES), dtype=np.float32)
    if status in _STATUSES:
        vec[_STATUSES.index(status)] = 1.0
    return vec


def _multi_hot_types(types: Tuple[PokemonType, ...]) -> np.ndarray:
    vec = np.zeros(len(_ALL_TYPES), dtype=np.float32)
    for t in types:
        vec[_ALL_TYPES.index(t)] = 1.0
    return vec


def _boost_vector(boosts: Dict[str, int]) -> np.ndarray:
    return np.array([boosts[name] / 6.0 for name in _BOOST_NAMES], dtype=np.float32)


def _base_stat_vector(base_stats: Dict[str, int]) -> np.ndarray:
    return np.array(
        [base_stats[name] / _BASE_STAT_SCALE for name in _STAT_NAMES], dtype=np.float32
    )


def _encode_pokemon(view: PokemonView) -> np.ndarray:
    return np.concatenate(
        [
            np.array([1.0 if view.known else 0.0], dtype=np.float32),
            np.array([view.hp_fraction], dtype=np.float32),
            np.array([1.0 if view.fainted else 0.0], dtype=np.float32),
            _one_hot_status(view.status),
            _multi_hot_types(view.types),
            _boost_vector(view.boosts),
            _base_stat_vector(view.base_stats),
        ]
    )


def _hazard_vector(hazards: set) -> np.ndarray:
    return np.array(
        [1.0 if token in hazards else 0.0 for token in _HAZARD_TOKENS], dtype=np.float32
    )


def _one_hot(value: Optional[str], vocab: Sequence[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        vec[list(vocab).index(value)] = 1.0
    return vec


def _type_multiplier(
    attacking: PokemonType,
    defending_types: Tuple[PokemonType, ...],
    defending_ability: Optional[str] = None,
) -> float:
    """Type-chart multiplier, corrected for the defender's ability when it's
    known and grants a hard type immunity (see _TYPE_IMMUNITY_ABILITIES) or is
    Wonder Guard (blocks anything that isn't already super-effective - a
    different rule shape, since it depends on the raw multiplier itself
    rather than a fixed type, so it's handled as its own case rather than
    folded into the table).
    """
    if defending_ability == "wonderguard":
        d1 = defending_types[0]
        d2 = defending_types[1] if len(defending_types) > 1 else None
        raw = attacking.damage_multiplier(d1, d2, type_chart=_TYPE_CHART)
        return raw if raw > 1 else 0.0

    immune_type = _TYPE_IMMUNITY_ABILITIES.get(defending_ability or "")
    if immune_type is not None and attacking == immune_type:
        return 0.0

    d1 = defending_types[0]
    d2 = defending_types[1] if len(defending_types) > 1 else None
    return attacking.damage_multiplier(d1, d2, type_chart=_TYPE_CHART)


def _active_matchup_score(view: BattleView) -> float:
    """Mirrors evaluation.type_matchup_score's semantics (offense - defense,
    each the best multiplier available across a dual typing), but built from
    bare PokemonType tuples rather than full poke-env Pokemon objects, since
    a replay-derived PokemonView isn't a real Pokemon instance. Each side's
    ability (if known) is folded in via _type_multiplier.
    """
    my_types, opp_types = view.my_active.types, view.opp_active.types
    if not my_types or not opp_types:
        return 0.0
    offense = max(
        _type_multiplier(t, opp_types, view.opp_active.ability) for t in my_types
    )
    defense = max(
        _type_multiplier(t, my_types, view.my_active.ability) for t in opp_types
    )
    return offense - defense


def encode(view: BattleView) -> np.ndarray:
    vec = np.concatenate(
        [
            _encode_pokemon(view.my_active),
            *[_encode_pokemon(p) for p in view.my_bench],
            _encode_pokemon(view.opp_active),
            np.array([view.opp_remaining_fraction], dtype=np.float32),
            _hazard_vector(view.my_hazards),
            _hazard_vector(view.opp_hazards),
            _one_hot(view.weather, _WEATHER_NAMES),
            _one_hot(view.terrain, _TERRAIN_NAMES),
            np.array([_active_matchup_score(view)], dtype=np.float32),
        ]
    )
    assert vec.shape == (VECTOR_LEN,)
    return vec
