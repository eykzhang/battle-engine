"""Translates an imitation-model action label (battle_engine.dataset's
13-way Metamon action scheme) into the action poke-env's Gymnasium
SinglesEnv actually expects - the reconciliation the roadmap deferred out
of Phase 2 for whenever the imitation model "feeds Phase 3's PPO policy
init" (see dataset.py's ACTION_SPACE_SIZE docstring).

poke-env's scheme (verified against poke_env.environment.singles_env.
SinglesEnv's own docstring and get_action_space_size, not assumed): 26 gen9
actions - 0-5 switch (target's raw, unsorted position in battle.team.
values()), 6-9 plain move, 10-13 move+mega, 14-17 move+z-move, 18-21
move+dynamax, 22-25 move+terastallize. Metamon's scheme only ever
recorded the two blocks that exist in gen9 OU (mega/z-move/dynamax don't -
Tera is the one gen9 gimmick), so move actions have a fixed positional
correspondence: plain move slot i -> 6 + i, tera move slot i -> 22 + i.

Switch actions have no such fixed correspondence and can't be resolved
without the live battle: dataset.py's switch labels are already remapped
(_remap_action) to point at a position in the species-sorted non-active
bench - the same ordering encoding.py's battle_view_from_poke_env sorts
battle.team into for the model's own input vector - while poke-env's switch
action is the target's raw, unsorted position in battle.team.values() (see
SinglesEnv.order_to_action's switch case, which does the same
species-identity lookup in the opposite direction). Both directions have to
go through species identity, not index arithmetic - there's no
state-independent switch mapping, which is why both translation functions
below take the live battle rather than being a pure lookup table.

Both directions are needed, not just metamon->poke-env: an independent
review (2026-07-31) found that rl_env.py's env needs to translate the
*opponent's* real order back into a metamon-space label too (poke-env's
SingleAgentWrapper calls env.order_to_action on the opponent's move to feed
it through the same env.step() plumbing) - without an inverse, that value
silently stayed in 26-way poke-env space while everything downstream
expected 13-way metamon space, corrupting the opponent's actual action
(e.g. a raw move-slot action getting reinterpreted as a switch label).

Sentinel values -1 (forfeit) and -2 (default order) are deliberately NOT
handled here - they're poke-env's own out-of-band control values, never
members of either Discrete action space (SinglesEnv.action_to_order /
order_to_action document them as special-cased before any of the real
26-way logic runs), so callers (rl_env.py) pass them through untranslated
rather than routing them through this module.
"""

from __future__ import annotations

from typing import Any

from battle_engine.dataset import _MOVE_ACTIONS, _SWITCH_ACTIONS, _TERA_MOVE_ACTIONS

_POKE_ENV_MOVE_OFFSET = 6
_POKE_ENV_TERA_MOVE_OFFSET = 22
_POKE_ENV_SWITCH_ACTIONS = range(0, 6)
_POKE_ENV_MOVE_ACTIONS = range(6, 10)
_POKE_ENV_TERA_MOVE_ACTIONS = range(22, 26)


def _require_active_pokemon(battle: Any) -> None:
    if battle.active_pokemon is None:
        # Matches encoding.py's battle_view_from_poke_env, which raises for
        # the same state (team preview / a queued forced-switch-after-faint
        # request) rather than silently defining "bench" some other way -
        # keeping this module and the observation vector in agreement about
        # when a well-defined bench even exists.
        raise ValueError(
            "no active_pokemon - switch actions aren't well-defined without "
            "one (team preview, or between a faint and the next switch)"
        )


def metamon_action_to_poke_env_action(action: int, battle: Any) -> int:
    """Converts one of dataset.py's 13-way action labels (0-3 move, 4-8
    switch, 9-12 move-while-terastallized) into poke-env's Gymnasium action
    index (0-25) for the given live battle. Raises ValueError for the -1
    "missing action" sentinel and for a switch label with no matching bench
    slot (bench smaller than the label implies) - both indicate a caller
    bug, not a legality question this function is responsible for (poke-env's
    own strict=True action_to_order still enforces real legality on top of
    this - a species-correct switch to a fainted or trapped-out mon is still
    an illegal order).
    """
    if action in _MOVE_ACTIONS:
        return _POKE_ENV_MOVE_OFFSET + action
    if action in _TERA_MOVE_ACTIONS:
        return _POKE_ENV_TERA_MOVE_OFFSET + (action - min(_TERA_MOVE_ACTIONS))
    if action in _SWITCH_ACTIONS:
        return _switch_action_to_poke_env(action, battle)
    raise ValueError(f"not a valid metamon action label: {action}")


def poke_env_action_to_metamon_action(action: int, battle: Any) -> int:
    """The inverse of metamon_action_to_poke_env_action: converts one of
    poke-env's Gymnasium action indices (0-25) into dataset.py's 13-way
    scheme for the given live battle. Raises ValueError for the sentinels
    (-1/-2, see module docstring - callers must handle those before calling
    this) and for the 12 gen9-gimmick actions with no metamon equivalent
    (mega 10-13, z-move 14-17, dynamax 18-21 - gen9 OU has none of these,
    Metamon's replay data never recorded them).
    """
    if action in _POKE_ENV_MOVE_ACTIONS:
        return action - _POKE_ENV_MOVE_OFFSET
    if action in _POKE_ENV_TERA_MOVE_ACTIONS:
        return min(_TERA_MOVE_ACTIONS) + (action - _POKE_ENV_TERA_MOVE_OFFSET)
    if action in _POKE_ENV_SWITCH_ACTIONS:
        return _poke_env_switch_to_metamon(action, battle)
    raise ValueError(f"no metamon action label for poke-env action: {action}")


def _switch_action_to_poke_env(action: int, battle: Any) -> int:
    _require_active_pokemon(battle)
    bench_slot = action - min(_SWITCH_ACTIONS)
    team = list(battle.team.values())
    bench_sorted = sorted(
        (mon for mon in team if mon is not battle.active_pokemon),
        key=lambda mon: mon.base_species,
    )
    if bench_slot >= len(bench_sorted):
        raise ValueError(
            f"switch action {action} (bench slot {bench_slot}) has no target - "
            f"only {len(bench_sorted)} non-active team members"
        )
    target = bench_sorted[bench_slot]
    return team.index(target)


def _poke_env_switch_to_metamon(action: int, battle: Any) -> int:
    _require_active_pokemon(battle)
    team = list(battle.team.values())
    if action >= len(team):
        raise ValueError(f"poke-env switch action {action} has no team member at that index")
    target = team[action]
    if target is battle.active_pokemon:
        raise ValueError(f"poke-env switch action {action} targets the active Pokemon")
    bench_sorted = sorted(
        (mon for mon in team if mon is not battle.active_pokemon),
        key=lambda mon: mon.base_species,
    )
    return min(_SWITCH_ACTIONS) + bench_sorted.index(target)
