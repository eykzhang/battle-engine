"""M7: MctsPlayer + the root ActionId -> poke-env BattleOrder translation
(_action_id_to_order). Must skip cleanly, not error, when _native hasn't
been built yet - same convention as tests/test_native_legality.py, which
this file's fixture style also mirrors (SimpleNamespace-wrapped real
poke-env Pokemon objects via conftest.make_mon, no live server connection).

Root-only translation is exactly what test_native_legality.py's own module
docstring calls out as NOT covered there ("M7 scope, doesn't exist yet") -
this file is that missing half.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from conftest import make_mon

_native = pytest.importorskip("battle_engine._native")

from battle_engine.mcts_player import MctsPlayer, MctsPuctPlayer, _action_id_to_order  # noqa: E402

_PPO_BIN = "data/cpp_weights/ppo.bin"


def _with_moves(mon, move_ids):
    for move_id in move_ids:
        mon._moves[move_id] = Move(move_id, gen=9)
    return mon


def _battle(my_team, my_active, opp_team, opp_active, my_hazards=None, opp_hazards=None, force_switch=False):
    return SimpleNamespace(
        team={mon.species: mon for mon in my_team},
        opponent_team={mon.species: mon for mon in opp_team},
        active_pokemon=my_active,
        opponent_active_pokemon=opp_active,
        side_conditions=my_hazards or {},
        opponent_side_conditions=opp_hazards or {},
        force_switch=force_switch,
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
    # active_pokemon is None - team preview, the only real state where
    # poke-env's own Battle.active_pokemon actually returns None (see the
    # fainted-active test below for the OTHER, previously-mishandled
    # no-well-defined-active-slot case). battle_state_from_poke_env sets
    # my_active_slot=-1, and legal_actions() restricts my side to switch
    # actions only. A real search (small n_simulations for test speed)
    # must complete and MctsPlayer must return a real switch order, not
    # crash and not propagate a move action with no active mon to
    # translate it against.
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


def test_active_slot_index_treats_a_fainted_active_pokemon_as_no_active_slot():
    # Real bug, found 2026-08-25 by instrumenting a live benchmark run:
    # poke-env's own Battle.active_pokemon property (battle.py) has NO
    # fainted check - it returns whichever team member has `.active ==
    # True`, so a just-fainted Pokemon is STILL battle.active_pokemon
    # (not None) until the replacement switch actually lands. The
    # `my_active=None` test above covers team preview, the only case
    # where active_pokemon really is None - this is the other, more
    # common case: a real fainted-but-still-"active" Pokemon handed
    # directly to the translator, exactly as poke-env would in real
    # play. _active_slot_index must treat this the same as None (-1),
    # not resolve it to that Pokemon's real team index - resolving it to
    # a real index let legal_actions() offer MOVE actions for a fainted
    # Pokemon's still-populated moveset, which the real server always
    # rejects, and poke-env's own retry loop only has a 1/1000 chance
    # per retry of giving up (Player.DEFAULT_CHOICE_CHANCE) - an
    # expected ~1000 wasted searches per faint before this fix.
    from battle_engine.mcts_player import battle_state_from_poke_env

    fainted_active = make_mon("garchomp", current_hp_fraction=0.0, status=Status.FNT)
    bench = make_mon("dragapult")
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[fainted_active, bench],
        my_active=fainted_active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    state = battle_state_from_poke_env(battle)

    assert state.my_active_slot == -1
    # switch-only, no move slots offered - and slot 0 (the fainted mon
    # itself) is correctly excluded too, same as any other already-active
    # slot would be; only the healthy bench slot (1) is a legal target.
    assert _native.legal_actions(state, _native.Side.ME) == [1]


def test_mcts_player_returns_a_switch_when_active_pokemon_has_fainted():
    # End-to-end version of the test above: a real search over a state
    # whose "active" Pokemon (per poke-env's own not-yet-switched
    # behavior) is fainted must return a real switch order, never a
    # move order the server would reject.
    fainted_active = _with_moves(make_mon("garchomp", current_hp_fraction=0.0, status=Status.FNT), ["earthquake"])
    bench = make_mon("dragapult")
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[fainted_active, bench],
        my_active=fainted_active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    player = MctsPlayer(battle_format="gen9ou", start_listening=False, n_simulations=30, seed=1)
    order = player.choose_move(battle)

    assert isinstance(order.order, Pokemon)
    assert order.order is bench


def test_battle_state_from_poke_env_sets_my_force_switch_and_keeps_real_active_slot():
    # Real bug, found 2026-08-25 in a follow-up hang investigation AFTER
    # 435f6a5 (the fainted-active fix above) already shipped: a real
    # 500-battle mcts_puct-vs-ppo benchmark run still stalled for over an
    # hour on a single battle. 435f6a5 only ever checked `active.fainted` -
    # it never covered poke-env's OTHER "no legal move this turn" signal,
    # `battle.force_switch`, which fires whenever the real request/response
    # protocol requires a switch-only response even though the active
    # Pokemon is alive and not fainted. The most common real trigger is a
    # pivot move (U-turn/Volt Switch/Baton Pass/...) that just resolved:
    # poke-env empties available_moves and sets force_switch=True for the
    # SAME still-active, still-healthy Pokemon. Before this fix,
    # my_active_slot resolved to that Pokemon's real team index (it's
    # neither None nor fainted), legal_actions() offered MOVE actions from
    # its still-populated moveset, and the server rejected every one of
    # them - hitting 435f6a5's own documented 1/1000-retry-chance mechanism
    # all over again. Confirmed live: a real repro stalled with search()
    # choosing the identical illegal move action on every retry (the
    # position doesn't change between retries), for 130+ consecutive
    # retries before being killed.
    #
    # The fix is a DEDICATED BattleState.my_force_switch field, not
    # resolving my_active_slot itself to -1 (the first attempt tried, and
    # this test would have caught it): my_active_slot also drives
    # legal_actions()'s switch-target exclusion ("can't switch into the
    # already-active slot") - wiping it to -1 for a Pokemon that's still
    # really active would make "switch into yourself" look legal too.
    from battle_engine.mcts_player import battle_state_from_poke_env

    active = _with_moves(make_mon("cinderace"), ["pyroball", "uturn", "willowisp", "courtchange"])
    bench = make_mon("dragapult")
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[active, bench],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
        force_switch=True,
    )

    state = battle_state_from_poke_env(battle)

    # my_active_slot stays the REAL index - active is alive, not fainted.
    assert state.my_active_slot == 0
    assert state.my_force_switch is True
    # switch-only, no move slots offered - the active mon (slot 0) is
    # correctly excluded as a switch target (it's still the real active
    # slot, "can't switch into yourself"); only the bench slot (1) is legal.
    assert _native.legal_actions(state, _native.Side.ME) == [1]


def test_mcts_player_returns_a_switch_when_force_switch_is_set_and_active_is_healthy():
    # End-to-end version of the test above: a real search over a state
    # whose active Pokemon is alive and healthy, but force_switch is True
    # (a pivot move just resolved), must return a real switch order,
    # never a move order the server would reject.
    active = _with_moves(make_mon("cinderace"), ["pyroball", "uturn", "willowisp", "courtchange"])
    bench = make_mon("dragapult")
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[active, bench],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
        force_switch=True,
    )

    player = MctsPlayer(battle_format="gen9ou", start_listening=False, n_simulations=30, seed=1)
    order = player.choose_move(battle)

    assert isinstance(order.order, Pokemon)
    assert order.order is bench


# ---------------------------------------------------------------------------
# _real_legal_action_ids / _real_legal_order_from_result - the root-level
# backstop added 2026-08-25 after a THIRD real-game mechanic (PP exhaustion)
# reproduced the identical multi-hour poke-env-retry-storm hang the
# fainted-active (435f6a5) and force_switch fixes above each closed one at a
# time. Confirmed live: a real mcts_puct-vs-ppo/gen9ou benchmark battle
# stalled at turn 63 with the search repeatedly picking a 0-PP move
# (malignantchain, 0/8 PP) every single retry, since the position doesn't
# change between retries. legal_actions() (action.hpp) has no PP/choice-
# lock/Disable/Encore/trapping model at all - rather than add a new
# BattleState field for every such mechanic (the my_force_switch pattern),
# this cross-checks the search's root pick against poke-env's own already-
# correct battle.available_moves/available_switches before ever submitting
# it.
# ---------------------------------------------------------------------------


def test_real_legal_action_ids_excludes_a_pp_exhausted_move_poke_env_already_excluded():
    from battle_engine.mcts_player import _real_legal_action_ids

    active = _with_moves(make_mon("pecharunt"), ["partingshot", "foulplay", "malignantchain", "recover"])
    bench = make_mon("dragapult")
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[active, bench],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    # malignantchain (move slot 2) is 0 PP - the real repro's exact shape:
    # poke-env already excludes it from available_moves, exactly as the
    # real server would.
    battle.available_moves = [active.moves["partingshot"], active.moves["foulplay"], active.moves["recover"]]
    battle.available_switches = [bench]

    legal = _real_legal_action_ids(battle)

    move_offset = _native.MOVE_ACTION_OFFSET
    assert move_offset + 2 not in legal  # malignantchain - PP exhausted, must be excluded
    assert {move_offset + 0, move_offset + 1, move_offset + 3} <= legal  # the other 3 moves
    assert 1 in legal  # bench switch target (team-preview slot 1)


def test_real_legal_order_from_result_salvages_the_next_best_real_legal_action():
    # The exact live-repro shape: best_action points at a PP-exhausted move
    # (malignantchain, move slot 2) that racked up the most search visits
    # before the search-time PP gap ever caught up with reality. The
    # backstop must not submit it - it must walk root_visit_distribution
    # (most-visited first) and salvage the next entry that IS real-legal,
    # not just give up and go straight to a context-free default.
    from battle_engine.mcts_player import _real_legal_order_from_result

    active = _with_moves(make_mon("pecharunt"), ["partingshot", "foulplay", "malignantchain", "recover"])
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    battle.available_moves = [active.moves["partingshot"], active.moves["foulplay"], active.moves["recover"]]
    battle.available_switches = []

    move_offset = _native.MOVE_ACTION_OFFSET
    malignant_chain = move_offset + 2  # illegal - 0 PP
    foul_play = move_offset + 1  # legal - second-most-visited
    result = SimpleNamespace(
        best_action=malignant_chain,
        root_visit_distribution=[(malignant_chain, 50), (foul_play, 30), (move_offset + 0, 20)],
    )

    order = _real_legal_order_from_result(result, battle)

    assert order.order is active.moves["foulplay"]


def test_real_legal_order_from_result_returns_none_when_nothing_searched_is_real_legal():
    from battle_engine.mcts_player import _real_legal_order_from_result

    active = _with_moves(make_mon("pecharunt"), ["malignantchain"])
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    battle.available_moves = []  # malignantchain is the only known move, and it's 0 PP
    battle.available_switches = []  # no bench to switch to either

    move_offset = _native.MOVE_ACTION_OFFSET
    result = SimpleNamespace(
        best_action=move_offset + 0, root_visit_distribution=[(move_offset + 0, 30)]
    )

    assert _real_legal_order_from_result(result, battle) is None


def test_mcts_player_never_submits_a_pp_exhausted_move(monkeypatch):
    # End-to-end version of the two unit tests above: search() itself has
    # no PP model and picks the PP-exhausted move as best_action (exactly
    # the live repro) - choose_move() must salvage a real-legal order, not
    # propagate the illegal one to poke-env.
    active = _with_moves(make_mon("pecharunt"), ["partingshot", "foulplay", "malignantchain", "recover"])
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    battle.available_moves = [active.moves["partingshot"], active.moves["foulplay"], active.moves["recover"]]
    battle.available_switches = []

    move_offset = _native.MOVE_ACTION_OFFSET
    malignant_chain = move_offset + 2

    monkeypatch.setattr(
        _native,
        "search",
        lambda state, n_simulations, seed: SimpleNamespace(
            best_action=malignant_chain,
            root_visit_distribution=[(malignant_chain, 50), (move_offset + 1, 30)],
        ),
    )

    player = MctsPlayer(battle_format="gen9ou", start_listening=False, n_simulations=10)
    order = player.choose_move(battle)

    assert order.order is not active.moves["malignantchain"]
    assert order.order is active.moves["foulplay"]


# ---------------------------------------------------------------------------
# MctsPuctPlayer (Phase 5 review fix, attempt 1) - mirrors MctsPlayer's own
# coverage above: NO_ACTION fallback, PolicyWeights loaded once at
# construction and reused (not reloaded) across choose_move() calls, and a
# small real search_puct() call against a fixture. Previously zero coverage
# anywhere in tests/ (the review's Issue 3), despite MctsPuctPlayer being
# this phase's actual Python-facing entry point.
# ---------------------------------------------------------------------------


def test_mcts_puct_player_falls_back_to_default_order_on_no_action(monkeypatch):
    active = _with_moves(make_mon("garchomp"), ["earthquake"])
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    monkeypatch.setattr(_native.PolicyWeights, "load", staticmethod(lambda path: object()))
    monkeypatch.setattr(
        _native,
        "search_puct",
        lambda state, weights, n_simulations, seed: SimpleNamespace(
            best_action=_native.NO_ACTION, root_visit_distribution=[]
        ),
    )

    player = MctsPuctPlayer(
        battle_format="gen9ou", start_listening=False,
        ppo_bin_path="unused-stubbed-out.bin", n_simulations=10,
    )
    order = player.choose_move(battle)

    assert isinstance(order, DefaultBattleOrder)
    assert order.message == "/choose default"


def test_mcts_puct_player_loads_weights_once_and_reuses_across_choose_move_calls(monkeypatch):
    active = _with_moves(make_mon("garchomp"), ["earthquake"])
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    sentinel_weights = object()
    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        return sentinel_weights

    search_calls = []

    def fake_search_puct(state, weights, n_simulations, seed):
        search_calls.append((weights, n_simulations, seed))
        return SimpleNamespace(best_action=_native.NO_ACTION, root_visit_distribution=[])

    monkeypatch.setattr(_native.PolicyWeights, "load", staticmethod(fake_load))
    monkeypatch.setattr(_native, "search_puct", fake_search_puct)

    player = MctsPuctPlayer(
        battle_format="gen9ou", start_listening=False,
        ppo_bin_path="a-real-looking-path.bin", n_simulations=77, seed=0,
    )
    player.choose_move(battle)
    player.choose_move(battle)

    # Constructed once (path passed through unchanged), never re-read per turn.
    assert load_calls == ["a-real-looking-path.bin"]
    # The SAME loaded weights object is reused on every choose_move() call -
    # identity, not just equality, is what proves it wasn't reloaded.
    assert search_calls[0][0] is sentinel_weights
    assert search_calls[1][0] is sentinel_weights
    assert [n for _, n, _ in search_calls] == [77, 77]
    # A fresh seed is drawn per choose_move() call, same convention as
    # MctsPlayer - two calls from one player instance must differ.
    assert search_calls[0][2] != search_calls[1][2]


@pytest.mark.skipif(
    not Path(_PPO_BIN).exists(),
    reason="requires data/cpp_weights/ppo.bin (scripts/export_weights.py), gitignored data/",
)
def test_mcts_puct_player_real_search_puct_call_returns_a_valid_order():
    # Small n_simulations for test speed - not a benchmark, just confirming
    # the real construction -> search_puct() -> BattleOrder path works
    # end-to-end against the real trained weights, no crash, no invalid
    # order propagated to poke-env.
    active = _with_moves(make_mon("garchomp"), ["earthquake", "dragonclaw"])
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )

    player = MctsPuctPlayer(
        battle_format="gen9ou", start_listening=False,
        ppo_bin_path=_PPO_BIN, n_simulations=20, seed=2,
    )
    order = player.choose_move(battle)

    assert isinstance(order, BattleOrder)
    assert not isinstance(order, DefaultBattleOrder)


@pytest.mark.skipif(
    not Path(_PPO_BIN).exists(),
    reason="requires data/cpp_weights/ppo.bin (scripts/export_weights.py), gitignored data/",
)
def test_mcts_puct_player_returns_a_switch_when_force_switch_is_set_and_active_is_healthy():
    # MctsPuctPlayer-specific regression for the force_switch bug (see
    # test_active_slot_index_treats_a_force_switch_active_pokemon_as_no_active_slot
    # and test_mcts_player_returns_a_switch_when_force_switch_is_set_and_active_is_healthy
    # above) - this is the class actually implicated in the real
    # 2026-08-25 multi-hour benchmark stall (mcts_puct vs ppo, gen9ou), so
    # it gets its own real-search end-to-end coverage, not just
    # MctsPlayer's. A pivot move just resolved (force_switch=True, active
    # alive and healthy) - search_puct() must return a real switch order,
    # never a move order the server would reject.
    active = _with_moves(make_mon("cinderace"), ["pyroball", "uturn", "willowisp", "courtchange"])
    bench = make_mon("dragapult")
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    battle = _battle(
        my_team=[active, bench],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
        force_switch=True,
    )

    player = MctsPuctPlayer(
        battle_format="gen9ou", start_listening=False,
        ppo_bin_path=_PPO_BIN, n_simulations=20, seed=2,
    )
    order = player.choose_move(battle)

    assert isinstance(order.order, Pokemon)
    assert order.order is bench
