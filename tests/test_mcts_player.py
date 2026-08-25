"""M7: MctsPlayer + the root ActionId -> poke-env BattleOrder translation
(_action_id_to_order). Must skip cleanly, not error, when _native hasn't
been built yet - same convention as tests/test_native_legality.py, which
this file's fixture style also mirrors (SimpleNamespace-wrapped real
poke-env Pokemon objects via conftest.make_mon, no live server connection).

Root-only translation is exactly what test_native_legality.py's own module
docstring calls out as NOT covered there ("M7 scope, doesn't exist yet") -
this file is that missing half.
"""

from types import SimpleNamespace

import pytest
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import DefaultBattleOrder

from conftest import make_mon

_native = pytest.importorskip("battle_engine._native")

from battle_engine.mcts_player import MctsPlayer, _action_id_to_order  # noqa: E402


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


# ---------------------------------------------------------------------------
# _action_id_to_order - pure translation, no live search needed.
# ---------------------------------------------------------------------------


def test_action_id_to_order_switch_maps_to_team_preview_slot():
    active = _with_moves(make_mon("garchomp"), ["earthquake"])
    bench1 = make_mon("dragapult")
    bench2 = make_mon("toxapex")
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active, bench1, bench2],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    # ActionId 0-5 is a direct index into battle.team's (team-preview-
    # stable) insertion order - slot 1 is bench1, slot 2 is bench2, NOT a
    # species-sorted mapping (that's action_space.py's different scheme).
    order1 = _action_id_to_order(1, battle)
    order2 = _action_id_to_order(2, battle)

    assert order1.order is bench1
    assert order2.order is bench2


def test_action_id_to_order_move_maps_to_active_moveset_slot():
    active = _with_moves(make_mon("garchomp"), ["earthquake", "dragonclaw", "swordsdance"])
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    # ActionId 6-9 = MOVE_ACTION_OFFSET + move slot - list(moves.keys())
    # ordering is the exact ordering _pokemon_slot() built the translated
    # state's PokemonSlot.moves from, so slot 0/1/2 must round-trip to the
    # same move ids search()'s legality check used.
    order0 = _action_id_to_order(_native.MOVE_ACTION_OFFSET + 0, battle)
    order2 = _action_id_to_order(_native.MOVE_ACTION_OFFSET + 2, battle)

    assert order0.order.id == "earthquake"
    assert order2.order.id == "swordsdance"


def test_action_id_to_order_switch_target_identity_is_not_species_sorted():
    # A regression guard against silently reusing action_space.py's
    # different (species-sorted, Metamon 13-way) switch mapping here - this
    # module's own scheme is insertion-order, not alphabetical. dragapult
    # sorts before garchomp alphabetically but is inserted second, so the
    # two mappings would disagree on slot 1 if this module ever
    # accidentally imported the wrong one.
    active = _with_moves(make_mon("garchomp"), ["earthquake"])
    bench = make_mon("dragapult")
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active, bench],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    assert _action_id_to_order(1, battle).order is bench


# ---------------------------------------------------------------------------
# MctsPlayer.choose_move - kNoAction fallback (stubbed search) and the
# team-preview/forced-switch edge case (real search).
# ---------------------------------------------------------------------------


def test_mcts_player_falls_back_to_default_order_on_no_action(monkeypatch):
    active = _with_moves(make_mon("garchomp"), ["earthquake"])
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    monkeypatch.setattr(
        _native,
        "search",
        lambda state, n_simulations, seed: SimpleNamespace(
            best_action=_native.NO_ACTION, root_visit_distribution=[]
        ),
    )

    player = MctsPlayer(battle_format="gen9ou", start_listening=False, n_simulations=10)
    order = player.choose_move(battle)

    assert isinstance(order, DefaultBattleOrder)
    assert order.message == "/choose default"


def test_mcts_player_search_called_with_configured_n_simulations(monkeypatch):
    active = _with_moves(make_mon("garchomp"), ["earthquake"])
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    calls = []

    def fake_search(state, n_simulations, seed):
        calls.append((n_simulations, seed))
        return SimpleNamespace(best_action=_native.NO_ACTION, root_visit_distribution=[])

    monkeypatch.setattr(_native, "search", fake_search)

    player = MctsPlayer(battle_format="gen9ou", start_listening=False, n_simulations=77, seed=0)
    player.choose_move(battle)
    player.choose_move(battle)

    assert [n for n, _ in calls] == [77, 77]
    # A fresh seed is drawn per choose_move() call (this module's own
    # documented design decision), not the same literal seed reused every
    # turn - two calls from one player instance must differ.
    assert calls[0][1] != calls[1][1]


def test_mcts_player_does_not_crash_with_no_active_pokemon():
    # The plan's own named edge case: active_pokemon is None (team preview,
    # or between a faint and the next switch request) - battle_state_from_
    # poke_env sets my_active_slot=-1, and legal_actions() restricts my
    # side to switch actions only. A real search (small n_simulations for
    # test speed) must complete and MctsPlayer must return a real switch
    # order, not crash and not propagate a move action with no active mon
    # to translate it against.
    bench1 = make_mon("garchomp")
    bench2 = make_mon("dragapult")
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[bench1, bench2],
        my_active=None,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    player = MctsPlayer(battle_format="gen9ou", start_listening=False, n_simulations=30, seed=1)
    order = player.choose_move(battle)

    assert isinstance(order.order, Pokemon)
    assert order.order in (bench1, bench2)
