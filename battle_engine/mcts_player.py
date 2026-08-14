"""M5 (translator only) / M7 (full player, not yet built): converts a live
poke-env AbstractBattle into a be::BattleState for the C++ forward model /
search to operate on. Lives here, not in C++/bindings, per the plan's own
explicit decision ("Python-side is unambiguously easier") - fixing a
contradiction the original plan draft had between this file's stated
location (in the M4 section) and an M7 code sketch that implied a C++-side
`battle_state_from_poke_env`.

MctsPlayer itself (a poke-env Player wired to native.search()) is M7 scope,
built once M6's search() exists - this file currently holds only the
translator, needed now so tests/test_native_legality.py can exercise M5's
legal_actions() against real battle data.

Team-preview-order assumption (verified against poke-env's real source, not
guessed): AbstractBattle._team and ._opponent_team are populated by
ordinary dict insertion, and the only place either is ever reassigned
wholesale (get_pokemon()'s nickname-matching branch, abstract_battle.py)
replaces an existing entry's KEY at its EXISTING list index rather than
moving it - so dict insertion order is stable positionally for the whole
battle, for both dicts. For `battle.team` (always fully known from team
preview onward, since it's your own team) this gives a genuine, fixed
team-preview-order match to ActionId's switch targets (action.hpp). For
`battle.opponent_team`, insertion order is REVEAL order, not necessarily
the opponent's real team-preview order (poke-env has no way to know that
before a slot is revealed) - consistent with Tier 1's already-accepted
"opponent modeling is revealed-only" limitation (see action.hpp's own doc
comment on legal_actions()), not a new gap this translator introduces.
"""

from __future__ import annotations

from typing import Any, Optional

from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status as PokeEnvStatus

from battle_engine import _native

# Only the 18 real gen-9 types have a be::Type counterpart (see types.hpp's
# own comment on why Stellar is excluded - Terastallization-only, and
# poke-env's own damage_multiplier special-cases it to a no-op 1.0
# unconditionally). THREE_QUESTION_MARKS/STELLAR fall through
# _type_to_native's "unsupported" branch below, same as a real,
# named simplification elsewhere in this project - not a crash.
_TYPE_MAP = {t.name: getattr(_native.Type, t.name) for t in PokemonType if hasattr(_native.Type, t.name)}
_STATUS_MAP = {s.name: getattr(_native.Status, s.name) for s in PokeEnvStatus}

_TEAM_SIZE = 6
_MOVESET_SIZE = 4


def _type_to_native(poke_env_type: Optional[PokemonType]) -> Any:
    if poke_env_type is None:
        return _native.Type.NONE
    return _TYPE_MAP.get(poke_env_type.name, _native.Type.NONE)


def _status_to_native(poke_env_status: Optional[PokeEnvStatus]) -> Any:
    if poke_env_status is None:
        return _native.Status.NONE
    return _STATUS_MAP[poke_env_status.name]


def _stat_block(base_stats: dict) -> Any:
    block = _native.StatBlock()
    block.hp = base_stats["hp"]
    block.atk = base_stats["atk"]
    block.def_ = base_stats["def"]
    block.spa = base_stats["spa"]
    block.spd = base_stats["spd"]
    block.spe = base_stats["spe"]
    return block


def _pokemon_slot(mon: Pokemon) -> Any:
    slot = _native.PokemonSlot()
    slot.revealed = True
    slot.fainted = mon.fainted
    slot.level = mon.level
    slot.hp_fraction = mon.current_hp_fraction
    slot.status = _status_to_native(mon.status)
    slot.type1 = _type_to_native(mon.type_1)
    slot.type2 = _type_to_native(mon.type_2)
    slot.base_stats = _stat_block(mon.base_stats)
    slot.boost_spe = mon.boosts.get("spe", 0)
    known_spe = mon.stats.get("spe") if mon.stats else None
    slot.spe_stat = known_spe if known_spe is not None else -1

    move_ids = list(mon.moves.keys())[:_MOVESET_SIZE]
    slot.moves = move_ids + [""] * (_MOVESET_SIZE - len(move_ids))
    return slot


def _team_slots(team: dict) -> list:
    mons = list(team.values())
    if len(mons) > _TEAM_SIZE:
        # Should not happen for any real gen9 singles team - a real bug
        # (e.g. a doubles battle, or a malformed request) if it ever does.
        raise ValueError(f"team has {len(mons)} Pokemon, expected at most {_TEAM_SIZE}")
    slots = [_pokemon_slot(mon) for mon in mons]
    slots += [_native.PokemonSlot() for _ in range(_TEAM_SIZE - len(slots))]
    return slots


def _active_slot_index(team: dict, active: Optional[Pokemon]) -> int:
    if active is None:
        return -1
    mons = list(team.values())
    return mons.index(active)


def _side_conditions(conditions: dict) -> Any:
    result = _native.SideConditions()
    result.spikes_layers = conditions.get(SideCondition.SPIKES, 0)
    result.toxic_spikes_layers = conditions.get(SideCondition.TOXIC_SPIKES, 0)
    result.stealth_rock = SideCondition.STEALTH_ROCK in conditions
    result.sticky_web = SideCondition.STICKY_WEB in conditions
    return result


def battle_state_from_poke_env(battle: Any) -> Any:
    """Builds a be::BattleState snapshot of `battle`'s current turn. Raises
    ValueError if either team somehow has more than 6 Pokemon (a real bug
    upstream, not a case this translator papers over - mirrors
    encoding.py's own team-preview ValueError precedent for an analogous
    "this shouldn't be possible" state).
    """
    state = _native.BattleState()
    state.my_team = _team_slots(battle.team)
    state.opp_team = _team_slots(battle.opponent_team)
    state.my_active_slot = _active_slot_index(battle.team, battle.active_pokemon)
    state.opp_active_slot = _active_slot_index(battle.opponent_team, battle.opponent_active_pokemon)
    state.my_hazards = _side_conditions(battle.side_conditions)
    state.opp_hazards = _side_conditions(battle.opponent_side_conditions)
    return state
