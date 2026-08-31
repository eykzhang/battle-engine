"""Phase 6 / M5: the player - action translation, aggregation, and the
failure paths that decide whether a real ladder game survives.

The two things worth the most coverage here are the ones that cost this
project real time before. The root legality backstop, because Phase 4's
equivalent gap turned into an hour-long stall on a single battle
(notes/gotcha-legality-drift-needs-a-boundary-backstop-not-one-off-fixes.md).
And the `BaseException` catch, because a Rust panic crosses pyo3 as something
no `except Exception` sees, and on the ladder that is a forfeited game rather
than a stack trace
(notes/gotcha-poke-engine-encore-panics-without-last-used-move.md).

Fixtures are real `poke_env.battle.Battle` objects driven by real protocol
messages, matching tests/test_poke_engine_state.py: `available_moves` and
`available_switches` come from a real parsed `|request|`, which is exactly the
ground truth the backstop checks against.
"""

from __future__ import annotations

import logging
import random
from types import SimpleNamespace

import pytest

poke_engine = pytest.importorskip(
    "poke_engine", reason="poke-engine not built; run scripts/build_poke_engine.sh"
)

from poke_env.battle.battle import Battle  # noqa: E402

from battle_engine import set_search  # noqa: E402
from battle_engine.set_search import (  # noqa: E402
    Decision,
    SetSearchPlayer,
    aggregate,
    order_from_choice,
)
from battle_engine.usage_stats import find_cached, load_usage_stats  # noqa: E402


_REQUEST = {
    "active": [
        {
            "moves": [
                {"move": "Headlong Rush", "id": "headlongrush", "pp": 8, "maxpp": 8, "disabled": False},
                {"move": "Ice Spinner", "id": "icespinner", "pp": 24, "maxpp": 24, "disabled": False},
            ],
            # Sets Battle.can_tera, which is the only thing that makes a
            # "<move>-tera" choice submittable.
            "canTerastallize": "Ground",
        }
    ],
    "side": {
        "name": "p1user",
        "id": "p1",
        "pokemon": [
            {
                "ident": "p1: Tusk",
                "details": "Great Tusk, L100",
                "condition": "341/341",
                "active": True,
                "stats": {"atk": 359, "def": 249, "spa": 140, "spd": 140, "spe": 301},
                "moves": ["headlongrush", "icespinner"],
                "baseAbility": "protosynthesis",
                "ability": "protosynthesis",
                "item": "heavydutyboots",
            },
            {
                "ident": "p1: Gliscor",
                "details": "Gliscor, L100",
                "condition": "354/354",
                "active": False,
                "stats": {"atk": 216, "def": 246, "spa": 112, "spd": 196, "spe": 184},
                "moves": ["earthquake", "protect", "toxic", "uturn"],
                "baseAbility": "poisonheal",
                "ability": "poisonheal",
                "item": "toxicorb",
            },
        ],
    },
    "rqid": 2,
}


def _battle() -> Battle:
    battle = Battle("battle-gen9ou-1", "p1user", logging.getLogger("test"), gen=9)
    battle.player_role = "p1"
    battle.parse_request(_REQUEST)
    battle.parse_message(["", "switch", "p1a: Tusk", "Great Tusk, L100", "341/341"])
    battle.parse_message(["", "switch", "p2a: Gholdengo", "Gholdengo, L100", "100/100"])
    battle.parse_message(["", "turn", "1"])
    return battle


@pytest.fixture(scope="module")
def stats():
    path = find_cached(cutoff=1500)
    if path is None:
        pytest.skip("no cached usage stats; run scripts/fetch_usage_stats.py")
    return load_usage_stats(path)


def _player(stats, **kwargs) -> SetSearchPlayer:
    # start_listening=False builds a Player with no websocket, the same way
    # tests/test_mcts_player.py does.
    return SetSearchPlayer(
        battle_format="gen9ou", start_listening=False, usage_stats=stats, **kwargs
    )


def _result(options, total_visits=0):
    """A stand-in MctsResult. `side_two` is never read by the aggregation."""
    return SimpleNamespace(
        side_one=[
            SimpleNamespace(move_choice=name, visits=visits, total_score=score)
            for name, visits, score in options
        ],
        side_two=[],
        total_visits=total_visits or sum(v for _, v, _ in options),
    )


# ---------------------------------------------------------------------------
# Action translation. poke-engine renders a switch as "switch <species>" but
# parses one as the bare species id - the asymmetry this function exists for.
# ---------------------------------------------------------------------------


class TestOrderFromChoice:
    def test_a_move(self):
        order = order_from_choice("headlongrush", _battle())
        assert order is not None and "headlongrush" in order.message

    def test_a_switch_uses_the_rendered_prefix(self):
        order = order_from_choice("switch gliscor", _battle())
        assert order is not None and "switch" in order.message

    def test_a_tera_move_sets_the_tera_flag(self):
        order = order_from_choice("headlongrush-tera", _battle())
        assert order is not None and "terastallize" in order.message

    def test_tera_is_refused_once_the_battle_says_it_is_spent(self):
        battle = _battle()
        battle._can_tera = False
        assert order_from_choice("headlongrush-tera", battle) is None
        # ...and the plain move is still fine.
        assert order_from_choice("headlongrush", battle) is not None

    def test_a_move_the_server_did_not_offer_is_refused(self):
        # The backstop's real job. poke-engine will offer a move whose
        # unavailability the translated state could not represent - trapping is
        # the standing case - and submitting it makes poke-env re-prompt the
        # same turn forever.
        assert order_from_choice("earthquake", _battle()) is None

    def test_a_switch_the_server_did_not_offer_is_refused(self):
        battle = _battle()
        battle.parse_request({**_REQUEST, "side": {**_REQUEST["side"], "pokemon": [_REQUEST["side"]["pokemon"][0]]}})
        assert order_from_choice("switch gliscor", battle) is None

    @pytest.mark.parametrize("choice", ["none", "", "   ", "switch ", "notamove"])
    def test_unusable_choices_return_none_rather_than_raising(self, choice):
        assert order_from_choice(choice, _battle()) is None

    def test_case_is_normalized(self):
        assert order_from_choice("HEADLONGRUSH", _battle()) is not None
        assert order_from_choice("Switch Gliscor", _battle()) is not None


# ---------------------------------------------------------------------------
# Aggregation across sampled opponents.
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_sums_visits_across_states_and_ranks_by_them(self):
        ranked = aggregate(
            [
                _result([("knockoff", 10, 5.0), ("switch torkoal", 40, 18.0)]),
                _result([("knockoff", 60, 40.0), ("switch torkoal", 5, 2.0)]),
            ]
        )
        assert [name for name, _, _ in ranked] == ["knockoff", "switch torkoal"]
        assert ranked[0][1] == 70

    def test_value_is_visit_weighted_not_averaged_per_state(self):
        # A branch explored 10 times in one state and 1,000 in another must not
        # let the 10-visit estimate count as half the answer.
        ranked = aggregate([_result([("uturn", 10, 10.0)]), _result([("uturn", 990, 495.0)])])
        assert ranked[0][2] == pytest.approx(505.0 / 1000.0)

    def test_ties_break_deterministically(self):
        ranked = aggregate([_result([("bbb", 5, 1.0), ("aaa", 5, 1.0)])])
        assert [name for name, _, _ in ranked] == ["aaa", "bbb"]

    def test_an_action_only_some_states_offer_still_appears(self):
        ranked = aggregate([_result([("knockoff", 10, 5.0)]), _result([("uturn", 3, 1.0)])])
        assert dict((n, v) for n, v, _ in ranked) == {"knockoff": 10, "uturn": 3}

    def test_empty_results_are_empty_not_an_error(self):
        assert aggregate([]) == ()


# ---------------------------------------------------------------------------
# The player.
# ---------------------------------------------------------------------------


class TestChooseMove:
    def test_plays_the_action_with_the_most_summed_visits(self, stats, monkeypatch):
        monkeypatch.setattr(
            set_search.poke_engine,
            "monte_carlo_tree_search",
            lambda state, **kw: _result([("icespinner", 100, 60.0), ("headlongrush", 10, 4.0)]),
        )
        player = _player(stats, n_opponent_samples=2, search_time_ms=10)
        assert "icespinner" in player.choose_move(_battle()).message
        assert player.search_stats.root_pick_illegal == 0

    def test_walks_down_the_ranking_when_the_top_pick_is_not_really_legal(self, stats, monkeypatch):
        # Earthquake is on Gliscor, not on the active Great Tusk, so the server
        # never offered it. A search that ranks it first must not stall the
        # battle - it must play the best action that IS legal.
        monkeypatch.setattr(
            set_search.poke_engine,
            "monte_carlo_tree_search",
            lambda state, **kw: _result([("earthquake", 900, 500.0), ("icespinner", 100, 60.0)]),
        )
        player = _player(stats, n_opponent_samples=1, search_time_ms=10)
        assert "icespinner" in player.choose_move(_battle()).message
        assert player.search_stats.root_pick_illegal == 1
        assert player.search_stats.defaulted == 0

    def test_defaults_when_nothing_the_search_found_is_legal(self, stats, monkeypatch):
        monkeypatch.setattr(
            set_search.poke_engine,
            "monte_carlo_tree_search",
            lambda state, **kw: _result([("earthquake", 900, 500.0)]),
        )
        player = _player(stats, n_opponent_samples=1, search_time_ms=10)
        player.choose_move(_battle())
        assert player.search_stats.failures == {"no_legal_action_in_search": 1}

    def test_a_rust_panic_is_a_bad_move_not_a_forfeited_game(self, stats, monkeypatch):
        """A `PanicException` derives from BaseException and slips past every
        `except Exception`. On the ladder that ends the process mid-game."""

        class Panic(BaseException):
            pass

        def boom(state, **kw):
            raise Panic("Encore should not be active when last used move is not a move")

        monkeypatch.setattr(set_search.poke_engine, "monte_carlo_tree_search", boom)
        player = _player(stats, n_opponent_samples=1, search_time_ms=10)
        assert player.choose_move(_battle()) is not None
        assert player.search_stats.failures == {"sample:Panic": 1, "all_searches_failed": 1}

    def test_one_failed_sample_costs_a_sample_not_the_turn(self, stats, monkeypatch):
        """poke-engine's threaded search really does panic with `NonFinite` on
        about 0.2% of turns. The samples are independent, so losing one is a
        reason to drop that sample, not to throw away the turn and default."""

        class Panic(BaseException):
            pass

        calls = {"n": 0}

        def flaky(state, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise Panic("called `Result::unwrap()` on an `Err` value: NonFinite")
            return _result([("icespinner", 100, 60.0)], total_visits=100)

        monkeypatch.setattr(set_search.poke_engine, "monte_carlo_tree_search", flaky)
        seen = []
        player = _player(stats, n_opponent_samples=4, search_time_ms=40, on_decision=seen.append)

        assert "icespinner" in player.choose_move(_battle()).message
        assert player.search_stats.defaulted == 0
        assert player.search_stats.sample_failures == 1
        assert player.search_stats.samples_run == 3
        # The decision log reports what actually contributed, not what was asked for.
        assert seen[0].states_searched == 3

    def test_every_sample_failing_still_yields_a_move(self, stats, monkeypatch):
        monkeypatch.setattr(
            set_search.poke_engine,
            "monte_carlo_tree_search",
            lambda state, **kw: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        player = _player(stats, n_opponent_samples=3, search_time_ms=30)
        assert player.choose_move(_battle()) is not None
        assert player.search_stats.sample_failures == 3
        assert player.search_stats.failures["all_searches_failed"] == 1

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_interrupts_still_propagate(self, stats, monkeypatch, interrupt):
        # Catching BaseException must not make the process unkillable.
        def boom(state, **kw):
            raise interrupt()

        monkeypatch.setattr(set_search.poke_engine, "monte_carlo_tree_search", boom)
        with pytest.raises(interrupt):
            _player(stats, n_opponent_samples=1, search_time_ms=10).choose_move(_battle())

    def test_a_translation_failure_is_recorded_and_survived(self, stats, monkeypatch):
        monkeypatch.setattr(
            set_search,
            "state_from_poke_env",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("team preview")),
        )
        player = _player(stats, n_opponent_samples=1, search_time_ms=10)
        assert player.choose_move(_battle()) is not None
        assert player.search_stats.failures == {"translation:ValueError": 1}

    def test_a_turn_with_no_choice_never_reaches_the_search(self, stats, monkeypatch):
        def fail(state, **kw):
            raise AssertionError("searched a turn with nothing to choose")

        monkeypatch.setattr(set_search.poke_engine, "monte_carlo_tree_search", fail)
        battle = _battle()
        battle.parse_request({"wait": True, "side": _REQUEST["side"], "rqid": 9})
        player = _player(stats, n_opponent_samples=1, search_time_ms=10)
        assert player.choose_move(battle) is not None
        assert player.search_stats.defaulted == 1


class TestBudget:
    def test_the_budget_is_split_across_the_samples(self, stats):
        player = _player(stats, search_time_ms=1000, n_opponent_samples=8)
        assert player.per_state_ms == 125

    def test_a_budget_smaller_than_the_sample_count_still_searches(self, stats):
        # Integer division would otherwise hand poke-engine duration_ms=0.
        assert _player(stats, search_time_ms=4, n_opponent_samples=8).per_state_ms == 1

    def test_one_search_per_sampled_opponent_at_the_split_budget(self, stats, monkeypatch):
        calls = []

        def record(state, **kw):
            calls.append(kw)
            return _result([("icespinner", 5, 3.0)])

        monkeypatch.setattr(set_search.poke_engine, "monte_carlo_tree_search", record)
        player = _player(stats, search_time_ms=400, n_opponent_samples=4, threads=3)
        player.choose_move(_battle())
        assert len(calls) == 4
        assert {c["duration_ms"] for c in calls} == {100}
        assert {c["threads"] for c in calls} == {3}
        # A wall-clock budget, not a simulation count - passing both would let
        # iterations silently win.
        assert {c["iterations"] for c in calls} == {0}

    @pytest.mark.parametrize("kwargs", [{"n_opponent_samples": 0}, {"search_time_ms": 0}])
    def test_degenerate_configuration_is_rejected_at_construction(self, stats, kwargs):
        with pytest.raises(ValueError):
            _player(stats, **kwargs)


class TestSampling:
    def test_the_sampled_opponents_actually_differ(self, stats):
        player = _player(stats, n_opponent_samples=6, search_time_ms=10)
        states = player._sampled_states(_battle())
        # Identical draws would make K searches a K-fold waste of the budget.
        assert len({s.to_string() for s in states}) > 1

    def test_a_seed_makes_a_whole_battle_reproducible(self, stats):
        a = _player(stats, n_opponent_samples=3, search_time_ms=10, seed=7)
        b = _player(stats, n_opponent_samples=3, search_time_ms=10, seed=7)
        assert [s.to_string() for s in a._sampled_states(_battle())] == [
            s.to_string() for s in b._sampled_states(_battle())
        ]

    def test_successive_turns_are_not_correlated_to_one_seed(self, stats):
        player = _player(stats, n_opponent_samples=3, search_time_ms=10, seed=7)
        first = [s.to_string() for s in player._sampled_states(_battle())]
        second = [s.to_string() for s in player._sampled_states(_battle())]
        assert first != second


class TestDecisionLog:
    def test_every_turn_reports_one_decision(self, stats, monkeypatch):
        monkeypatch.setattr(
            set_search.poke_engine,
            "monte_carlo_tree_search",
            lambda state, **kw: _result([("icespinner", 100, 60.0)], total_visits=100),
        )
        seen = []
        player = _player(stats, n_opponent_samples=2, search_time_ms=10, on_decision=seen.append)
        player.choose_move(_battle())
        assert len(seen) == 1
        decision = seen[0]
        assert isinstance(decision, Decision)
        assert decision.chosen == "icespinner"
        assert decision.states_searched == 2
        assert decision.total_visits == 200
        assert decision.seconds > 0

    def test_a_fallback_says_why_in_the_log(self, stats, monkeypatch):
        monkeypatch.setattr(
            set_search.poke_engine,
            "monte_carlo_tree_search",
            lambda state, **kw: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        seen = []
        player = _player(stats, n_opponent_samples=1, search_time_ms=10, on_decision=seen.append)
        player.choose_move(_battle())
        assert seen[0].fallback_reason == "all_searches_failed"

    def test_stats_accumulate_across_turns(self, stats, monkeypatch):
        monkeypatch.setattr(
            set_search.poke_engine,
            "monte_carlo_tree_search",
            lambda state, **kw: _result([("icespinner", 10, 6.0)], total_visits=10),
        )
        player = _player(stats, n_opponent_samples=1, search_time_ms=10)
        for _ in range(3):
            player.choose_move(_battle())
        assert player.search_stats.turns == 3
        assert player.search_stats.visits == 30
        assert player.search_stats.ms_per_turn > 0
