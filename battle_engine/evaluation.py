"""Hand-crafted position evaluation: battle state -> a single score, positive
meaning good for the player whose perspective the battle object represents.

This is the "Stockfish 1.0" eval named in the project roadmap — every weight
below is hand-picked, not learned. That's the whole point of Phase 1: Phase 2
replaces this file's hand-tuned constants with a model trained on human
replays, while the search around it stays roughly the same shape.

Deliberately out of scope for now (each a named simplification, not an
oversight): item/ability effects, Trick Room, weather/terrain, and stat
boosts other than Speed (Attack/Defense boosts do matter for the eventual
matchup, but folding them in accurately means re-deriving the damage calc's
attack/defense ratio per boost stage — deferred until evidence from
benchmarking shows the coarse type-matchup signal isn't enough).
"""

from __future__ import annotations

from typing import Dict, Mapping

from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status

from battle_engine.damage import estimate_stat

HP_FRACTION_WEIGHT = 1.0
ALIVE_COUNT_WEIGHT = 0.5
TYPE_MATCHUP_WEIGHT = 0.5
STATUS_WEIGHT = 0.3
HAZARD_LAYER_WEIGHT = 0.15
HAZARD_PRESENCE_WEIGHT = 0.2
SPEED_CONTROL_WEIGHT = 0.2

# Hazards whose dict value is a stack count (more layers = worse) vs. hazards
# that are just present-or-not (their dict value is the turn they were set).
_STACKABLE_HAZARDS = {SideCondition.SPIKES, SideCondition.TOXIC_SPIKES}
_PRESENCE_HAZARDS = {SideCondition.STEALTH_ROCK, SideCondition.STICKY_WEB}


def _team_hp_fraction(team: Mapping[str, Pokemon]) -> float:
    return sum(mon.current_hp_fraction for mon in team.values())


def _alive_count(team: Mapping[str, Pokemon]) -> int:
    return sum(1 for mon in team.values() if not mon.fainted)


def _status_afflicted_count(team: Mapping[str, Pokemon]) -> int:
    return sum(
        1 for mon in team.values() if mon.status not in (None, Status.FNT)
    )


def type_matchup_score(mon: Pokemon, opponent: Pokemon) -> float:
    """How favorable mon's typing is against opponent's: offense - defense,
    each the best (super-effective-most) multiplier available across a dual
    typing. Positive means mon hits opponent harder than vice versa.
    """
    offense = max(opponent.damage_multiplier(t) for t in mon.types if t is not None)
    defense = max(mon.damage_multiplier(t) for t in opponent.types if t is not None)
    return offense - defense


def _hazard_score(side_conditions: Mapping[SideCondition, int]) -> float:
    score = 0.0
    for condition, stacks in side_conditions.items():
        if condition in _STACKABLE_HAZARDS:
            score += HAZARD_LAYER_WEIGHT * stacks
        elif condition in _PRESENCE_HAZARDS:
            score += HAZARD_PRESENCE_WEIGHT
    return score


def _boosted_speed(mon: Pokemon) -> float:
    stage = mon.boosts["spe"]
    multiplier = (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
    speed = estimate_stat(mon, "spe") * multiplier
    if mon.status == Status.PAR:
        speed *= 0.5
    return speed


def _speed_control_score(my_active: Pokemon, opp_active: Pokemon) -> float:
    my_speed = _boosted_speed(my_active)
    opp_speed = _boosted_speed(opp_active)
    if my_speed > opp_speed:
        return 1.0
    if opp_speed > my_speed:
        return -1.0
    return 0.0


def evaluate(battle) -> float:
    """Score `battle` from its own perspective (positive favors `battle`'s
    player). Accepts anything exposing the same attributes as poke-env's
    AbstractBattle (team, opponent_team, active_pokemon, opponent_active_pokemon,
    side_conditions, opponent_side_conditions) — duck-typed for testability.
    """
    my_team: Dict[str, Pokemon] = battle.team
    opp_team: Dict[str, Pokemon] = battle.opponent_team

    score = HP_FRACTION_WEIGHT * (
        _team_hp_fraction(my_team) - _team_hp_fraction(opp_team)
    )
    score += ALIVE_COUNT_WEIGHT * (_alive_count(my_team) - _alive_count(opp_team))
    score += STATUS_WEIGHT * (
        _status_afflicted_count(opp_team) - _status_afflicted_count(my_team)
    )
    score += _hazard_score(battle.opponent_side_conditions) - _hazard_score(
        battle.side_conditions
    )

    my_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    if my_active is not None and opp_active is not None:
        score += TYPE_MATCHUP_WEIGHT * type_matchup_score(my_active, opp_active)
        score += SPEED_CONTROL_WEIGHT * _speed_control_score(my_active, opp_active)

    return score
