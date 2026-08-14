"""M5: cross-checks legal_actions() (C++, action.hpp) against real poke-env
data translated through battle_engine.mcts_player.battle_state_from_poke_env
- the "translate N real poke-env states into BattleState, run legal_actions,
diff against poke-env's own switch/move availability, for revealed-mon
actions only" check named in plans/precious-crafting-bachman.md's M5
section. Must skip cleanly, not error, when _native hasn't been built yet -
same convention as tests/test_native_bindings.py.

Uses the same SimpleNamespace-battle-fixture pattern already established in
tests/test_encoding.py (real poke-env Pokemon objects via conftest.make_mon,
wrapped in a plain object exposing exactly the AbstractBattle attributes the
translator reads) rather than a live server connection - consistent with
every other non-integration test in this suite.

Real poke-env AbstractBattle.available_switches/available_moves ARE NOT
used as the comparison target here, deliberately: those properties depend
on live request/protocol state (PP, trapping, choice-lock, forced-switch)
that a SimpleNamespace fixture doesn't carry and this project's other tests
don't construct either. The comparison instead derives the expected set
directly from the same underlying Pokemon objects (non-active/non-fainted
mons for switches, known move ids for moves) - mathematically identical to
available_switches/available_moves for freshly-constructed mons with no
trapping/PP/choice-lock in play, which is exactly what these fixtures are.

Root-only ActionId -> poke-env BattleOrder translation (the OTHER half of
the plan's M5 legality check) is explicitly NOT tested here - that mapping
lives in MctsPlayer, which is M7 scope and doesn't exist yet (see
battle_engine/mcts_player.py's own module docstring).
"""

from types import SimpleNamespace

import pytest
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status

from conftest import make_mon

_native = pytest.importorskip("battle_engine._native")

from battle_engine.mcts_player import battle_state_from_poke_env  # noqa: E402


def _with_moves(mon, move_ids):
    for move_id in move_ids:
        mon._moves[move_id] = Move(move_id, gen=9)
    return mon


def _battle(my_team, my_active, opp_team, opp_active, my_hazards=None, opp_hazards=None):
    return SimpleNamespace(
        team={mon.species: mon for mon in my_team},
        opponent_team={mon.species: mon for mon in opp_team},
        active_pokemon=my_active,
        opponent_active_pokemon=opp_active,
        side_conditions=my_hazards or {},
        opponent_side_conditions=opp_hazards or {},
    )


def test_legal_actions_matches_real_switch_and_move_availability_for_my_side():
    active = _with_moves(make_mon("garchomp"), ["earthquake", "dragonclaw"])
    healthy_bench = make_mon("dragapult")
    fainted_bench = make_mon("toxapex", current_hp_fraction=0.0, status=Status.FNT)

    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active, healthy_bench, fainted_bench],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    state = battle_state_from_poke_env(battle)

    actions = _native.legal_actions(state, _native.Side.ME)

    # Switches: the healthy bench mon (slot 1) is legal; the active slot
    # (0, can't switch to self) and the fainted slot (2) are not.
    assert 1 in actions
    assert 0 not in actions
    assert 2 not in actions
    # Moves: both known moves are legal.
    assert _native.MOVE_ACTION_OFFSET + 0 in actions
    assert _native.MOVE_ACTION_OFFSET + 1 in actions
    assert _native.MOVE_ACTION_OFFSET + 2 not in actions
    assert _native.MOVE_ACTION_OFFSET + 3 not in actions


def test_legal_actions_restricts_opponent_to_revealed_data_only():
    my_active = _with_moves(make_mon("garchomp"), ["earthquake"])
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])  # only 1 of 4 moves seen
    opp_partial_bench = make_mon("toxapex")  # only 2 of 6 team members ever revealed

    battle = _battle(
        my_team=[my_active],
        my_active=my_active,
        opp_team=[opp_active, opp_partial_bench],
        opp_active=opp_active,
    )
    state = battle_state_from_poke_env(battle)

    actions = _native.legal_actions(state, _native.Side.OPP)

    # Switch: the revealed bench mon (slot 1) is legal - slots 2-5 were
    # never revealed at all and correctly can't appear (Tier 1's named
    # opponent-modeling limitation, not a bug).
    assert 1 in actions
    for slot in (2, 3, 4, 5):
        assert slot not in actions
    # Move: only the one seen move is legal.
    assert _native.MOVE_ACTION_OFFSET + 0 in actions
    for slot in (1, 2, 3):
        assert _native.MOVE_ACTION_OFFSET + slot not in actions


def test_battle_state_from_poke_env_reads_hazards_onto_the_correct_side():
    my_active = make_mon("garchomp")
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[my_active],
        my_active=my_active,
        opp_team=[opp_active],
        opp_active=opp_active,
        my_hazards={SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2},
        opp_hazards={SideCondition.STICKY_WEB: 1},
    )
    state = battle_state_from_poke_env(battle)

    assert state.my_hazards.stealth_rock is True
    assert state.my_hazards.spikes_layers == 2
    assert state.my_hazards.sticky_web is False
    assert state.opp_hazards.sticky_web is True
    assert state.opp_hazards.stealth_rock is False


def test_battle_state_from_poke_env_marks_no_active_pokemon_as_minus_one():
    my_fainted_active = make_mon("garchomp", current_hp_fraction=0.0, status=Status.FNT)
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[my_fainted_active],
        my_active=None,  # fainted mid-turn, no replacement chosen yet
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    state = battle_state_from_poke_env(battle)

    assert state.my_active_slot == -1
    assert _native.legal_actions(state, _native.Side.ME) == []
