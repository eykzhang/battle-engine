import asyncio
import json
import os
import signal
import socket
import sys
from pathlib import Path
from unittest import mock

import pytest

from battle_engine.benchmark import BenchmarkResult, run_benchmark, wilson_interval

# scripts/ isn't an installed package and no other test in THIS file imports
# from it - same local sys.path shim tests/test_export_weights.py already
# established for the identical reason (see that file's own comment).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_wilson_interval_symmetric_case():
    # Known textbook value (Agresti): 95% Wilson CI for 5/10.
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_interval_extreme_cases_stay_in_bounds():
    # 0 wins and n wins are the cases where the normal (Wald) approximation
    # breaks down (would allow bounds below 0 or above 1) — Wilson shouldn't.
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-3)
    assert hi == pytest.approx(0.2775, abs=1e-3)

    lo, hi = wilson_interval(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-3)


def test_wilson_interval_no_battles_returns_maximally_uncertain_bounds():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def _server_running(host: str = "localhost", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_run_benchmark_against_local_server():
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")
    result = asyncio.run(run_benchmark(p1, p2, n_battles=4))

    assert result.n_battles == 4
    assert result.p1_wins + result.p2_wins + result.ties == 4
    lo, hi = result.confidence_interval
    assert 0.0 <= lo <= result.p1_win_rate <= hi <= 1.0


# ---------------------------------------------------------------------------
# Progress visibility + survivable early exit, added 2026-08-25 after an
# 8+ hour run with zero incremental output had to be killed blind, losing
# everything - see notes/gotcha-benchmark-runs-need-empirical-timing-and-
# progress-visibility.md.
# ---------------------------------------------------------------------------


def test_benchmark_result_p1_win_rate_handles_zero_battles():
    # A run stopped (graceful_early_exit) before any battle completed
    # returns n_battles=0 - p1_win_rate must not raise ZeroDivisionError.
    result = BenchmarkResult(p1_name="A", p2_name="B", n_battles=0, p1_wins=0, p2_wins=0, ties=0)
    assert result.p1_win_rate == 0.0


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_run_benchmark_writes_checkpoint_file_after_every_battle(tmp_path):
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")
    checkpoint_path = tmp_path / "checkpoint.json"

    result = asyncio.run(run_benchmark(p1, p2, n_battles=2, checkpoint_path=checkpoint_path))

    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["n_completed"] == 2
    assert checkpoint["n_battles_target"] == 2
    assert checkpoint["p1_wins"] + checkpoint["p2_wins"] + checkpoint["ties"] == 2
    # The checkpoint's own tally must agree with the real final result, not
    # just be well-formed JSON - a checkpoint that silently drifted from
    # the truth would be worse than no checkpoint at all.
    assert checkpoint["p1_wins"] == result.p1_wins
    assert checkpoint["p2_wins"] == result.p2_wins
    assert checkpoint["ties"] == result.ties


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_run_benchmark_progress_interval_prints_a_running_tally(capsys):
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")

    asyncio.run(run_benchmark(p1, p2, n_battles=2, progress_interval=1))

    out = capsys.readouterr().out
    assert "[1/2]" in out
    assert "[2/2]" in out
    assert "elapsed" in out


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_run_benchmark_graceful_early_exit_returns_a_real_partial_result():
    # The actual capability the user asked for: `kill <pid>` (SIGTERM, not
    # -9) mid-run must stop cleanly after the in-flight battle and return
    # whatever's genuinely been completed, not crash or hang. Simulates
    # this by sending SIGTERM to our own process shortly after the run
    # starts, via a concurrent task (real asyncio concurrency, not a mock).
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")

    async def _run_and_interrupt():
        async def _send_sigterm_shortly():
            await asyncio.sleep(0.3)
            os.kill(os.getpid(), signal.SIGTERM)

        interrupt_task = asyncio.create_task(_send_sigterm_shortly())
        result = await run_benchmark(p1, p2, n_battles=500, graceful_early_exit=True)
        await interrupt_task
        return result

    result = asyncio.run(_run_and_interrupt())

    # Stopped well short of the 500-battle target, but with a real,
    # internally-consistent partial tally, not a crash or an empty result.
    assert 0 < result.n_battles < 500
    assert result.p1_wins + result.p2_wins + result.ties == result.n_battles


# ---------------------------------------------------------------------------
# _make_player (Phase 5 review fix, attempt 1): the "mcts_puct" branch
# (scripts/benchmark.py:149-158) had zero test coverage - the review's
# Issue 3. No live server connection needed - construction only, same
# scope as this file's other unit-level tests above.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("data/cpp_weights/ppo.bin").exists(),
    reason="requires data/cpp_weights/ppo.bin (scripts/export_weights.py), gitignored data/",
)
def test_make_player_mcts_puct_constructs_a_real_mcts_puct_player():
    pytest.importorskip("battle_engine._native")
    from scripts.benchmark import _make_player
    from battle_engine.mcts_player import MctsPuctPlayer

    player = _make_player(
        "mcts_puct",
        battle_format="gen9ou",
        model_path=Path("data/models/win_prob.pt"),
        ppo_model_path=Path("data/models/ppo.zip"),
        n_simulations=5,
        ppo_bin_path=Path("data/cpp_weights/ppo.bin"),
    )

    assert isinstance(player, MctsPuctPlayer)
    assert player._n_simulations == 5


# ---------------------------------------------------------------------------
# Phase 2 (benchmark-concurrency plan): run_benchmark's max_concurrent_battles
# and its scripts/benchmark.py CLI wiring. Per the dispatch prompt's GIL
# note, DW-2.2's "real bounded concurrency" is verified structurally - a
# lightweight instrumented player recording in-flight choose_move calls -
# never via a timing/speedup assertion against SetSearchPlayer/poke_engine
# (poke_engine.monte_carlo_tree_search holds the GIL for its whole duration
# and would show no such speedup regardless of scheduling correctness).
# ---------------------------------------------------------------------------


def _make_concurrency_tracking_random_player(**kwargs):
    """A RandomPlayer whose choose_move becomes async and records how many
    calls are in flight on this one instance at once - the structural
    signal DW-2.2 asks for. Deliberately never touches SetSearchPlayer/
    poke_engine.

    ConcurrencyTrackingRandomPlayer is defined here, not with a leading
    underscore: Player.__init__ derives the default account username from
    self.__class__.__name__ (see AccountConfiguration.generate(self.
    __class__.__name__) in poke_env/player/player.py) - a leading
    underscore produced a username the local Showdown server silently
    rewrote on login (stripping it), which then failed poke-env's own
    username-match assertion in ps_client.wait_for_login and hung the
    test. Caught by actually running this test, not assumed.

    Safe without a lock: choose_move is invoked from Player._handle_battle_
    request, which runs entirely on POKE_LOOP (poke-env's own single-
    threaded, cooperative background-thread event loop) - the same
    single-thread argument battle_engine.benchmark._run_concurrent_battles
    itself relies on for its _battle_finished_callback bookkeeping.
    """
    from poke_env.player import RandomPlayer

    class ConcurrencyTrackingRandomPlayer(RandomPlayer):
        async def choose_move(self, battle):
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            await asyncio.sleep(0.05)
            self.in_flight -= 1
            return self.choose_random_move(battle)

    player = ConcurrencyTrackingRandomPlayer(**kwargs)
    player.in_flight = 0
    player.max_in_flight = 0
    return player


class _RecordingCheckpointPath:
    """Duck-typed stand-in for checkpoint_path - run_benchmark only ever
    calls .write_text(data) on it (never anything Path-specific), so this
    captures every intermediate checkpoint write deterministically instead
    of racing a real file on disk mid-run.
    """

    def __init__(self):
        self.writes: list[dict] = []

    def write_text(self, data: str) -> None:
        self.writes.append(json.loads(data))


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_DW_2_1_default_max_concurrent_battles_is_one_and_sequential():
    # No max_concurrent_battles passed at all - the parameter's own default
    # (1) must still take the untouched sequential for-loop path.
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")

    result = asyncio.run(run_benchmark(p1, p2, n_battles=3))

    assert result.n_battles == 3
    assert result.p1_wins + result.p2_wins + result.ties == 3


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_DW_2_2_concurrent_battles_actually_overlap():
    from poke_env.player import RandomPlayer

    p1 = _make_concurrency_tracking_random_player(
        battle_format="gen9randombattle", max_concurrent_battles=3
    )
    p2 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=3)

    result = asyncio.run(run_benchmark(p1, p2, n_battles=6, max_concurrent_battles=3))

    assert result.n_battles == 6
    # Real overlap: more than one choose_move call was in flight on p1 at
    # once. Impossible under the strictly-sequential default path, where
    # exactly one battle (and so at most one choose_move call) exists at a
    # time - this is the structural signal that bounded concurrency is
    # actually happening, not just accepted as a parameter.
    assert p1.max_in_flight > 1
    assert p1.max_in_flight <= 3


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_DW_2_2_checkpoint_written_after_every_individual_battle():
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=3)
    p2 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=3)
    recorder = _RecordingCheckpointPath()

    result = asyncio.run(
        run_benchmark(p1, p2, n_battles=4, checkpoint_path=recorder, max_concurrent_battles=3)
    )

    # One write per completed battle, in completion order, not batched into
    # a single write at the end.
    assert len(recorder.writes) == 4
    assert [w["n_completed"] for w in recorder.writes] == [1, 2, 3, 4]
    # The final intermediate write must agree with the real result - a
    # checkpoint that silently drifted from the truth would be worse than
    # no checkpoint at all.
    assert recorder.writes[-1]["p1_wins"] == result.p1_wins
    assert recorder.writes[-1]["p2_wins"] == result.p2_wins
    assert recorder.writes[-1]["ties"] == result.ties


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_DW_2_3_graceful_early_exit_under_concurrency_returns_real_partial_result():
    # Same technique as test_run_benchmark_graceful_early_exit_returns_a_
    # real_partial_result (sequential path), at max_concurrent_battles=3:
    # cancels the background battle_against task promptly, lets in-flight
    # battles finish via _battle_count_queue.join(), and must return
    # (not hang on the cross-loop RuntimeError bug this plan's own notes
    # describe being caught and fixed) with an honest partial n_battles.
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=3)
    p2 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=3)

    async def _run_and_interrupt():
        async def _send_sigterm_shortly():
            await asyncio.sleep(0.3)
            os.kill(os.getpid(), signal.SIGTERM)

        interrupt_task = asyncio.create_task(_send_sigterm_shortly())
        result = await run_benchmark(
            p1, p2, n_battles=500, graceful_early_exit=True, max_concurrent_battles=3
        )
        await interrupt_task
        return result

    result = asyncio.run(_run_and_interrupt())

    assert 0 < result.n_battles < 500
    assert result.p1_wins + result.p2_wins + result.ties == result.n_battles


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_DW_2_4_uneven_division_completes_exact_battle_count():
    # n_battles=7 not evenly divisible by max_concurrent_battles=3 - the
    # single bulk battle_against(..., n_battles=7) call must still complete
    # exactly 7, with poke-env's own bounded queue pacing the unevenness,
    # not any manual chunking in this phase's code.
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=3)
    p2 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=3)

    result = asyncio.run(run_benchmark(p1, p2, n_battles=7, max_concurrent_battles=3))

    assert result.n_battles == 7
    assert result.p1_wins + result.p2_wins + result.ties == 7


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_DW_2_5_callback_restored_after_normal_completion():
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=2)
    p2 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=2)
    original = p1._battle_finished_callback

    asyncio.run(run_benchmark(p1, p2, n_battles=2, max_concurrent_battles=2))

    assert p1._battle_finished_callback == original


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_DW_2_5_callback_restored_when_battle_against_raises():
    # A real exception from battle_against (not a cancellation) must still
    # hit the try/finally's restoration - and must still propagate, not be
    # swallowed.
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=2)
    p2 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=2)
    original = p1._battle_finished_callback
    p1.battle_against = mock.AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run_benchmark(p1, p2, n_battles=2, max_concurrent_battles=2))

    assert p1._battle_finished_callback == original


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_DW_2_5_callback_restored_when_task_is_cancelled():
    # Same graceful-early-exit-under-concurrency trigger as DW-2.3, but
    # asserting specifically on the callback-restoration guarantee this
    # time (the try/finally around the cancel-and-drain sequence).
    from poke_env.player import RandomPlayer

    p1 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=2)
    p2 = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=2)
    original = p1._battle_finished_callback

    async def _run_and_interrupt():
        async def _send_sigterm_shortly():
            await asyncio.sleep(0.3)
            os.kill(os.getpid(), signal.SIGTERM)

        interrupt_task = asyncio.create_task(_send_sigterm_shortly())
        await run_benchmark(p1, p2, n_battles=500, graceful_early_exit=True, max_concurrent_battles=2)
        await interrupt_task

    asyncio.run(_run_and_interrupt())

    assert p1._battle_finished_callback == original


def test_DW_2_6_cli_has_max_concurrent_battles_flag_default_one(monkeypatch):
    from scripts.benchmark import parse_args

    monkeypatch.setattr(sys, "argv", ["benchmark.py"])
    args = parse_args()

    assert args.max_concurrent_battles == 1


def test_DW_2_6_cli_max_concurrent_battles_flag_parses_custom_value(monkeypatch):
    from scripts.benchmark import parse_args

    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--max-concurrent-battles", "4"])
    args = parse_args()

    assert args.max_concurrent_battles == 4


@pytest.mark.parametrize("name", ["random", "maxdamage", "heuristic", "search", "learned", "setsearch"])
def test_DW_2_6_max_concurrent_battles_reaches_every_make_player_branch(name):
    # Construction only, no live server - covers all 6 of _make_player's
    # code branches that don't need a gitignored model artifact this
    # checkout may not have (ppo/mcts/mcts_puct get their own skip-guarded
    # tests below, matching this file's existing convention for those).
    if name == "learned" and not Path("data/models/win_prob.pt").exists():
        pytest.skip("data/models/win_prob.pt not present in this checkout")
    if name == "setsearch" and not any(Path("data/usage_stats").glob("*gen9ou-1500.json")):
        pytest.skip("no cached gen9ou-1500 usage-stats file in this checkout")

    from scripts.benchmark import _make_player

    player = _make_player(
        name,
        battle_format="gen9randombattle",
        model_path=Path("data/models/win_prob.pt"),
        ppo_model_path=Path("data/models/ppo.zip"),
        n_simulations=5,
        ppo_bin_path=Path("data/cpp_weights/ppo.bin"),
        max_concurrent_battles=3,
    )

    assert player._max_concurrent_battles == 3


@pytest.mark.skipif(
    not Path("data/models/ppo.zip").exists(),
    reason="data/models/ppo.zip not present in this checkout",
)
def test_DW_2_6_make_player_ppo_forwards_max_concurrent_battles():
    # load_ppo_player forwards **player_kwargs through to FrozenPolicyPlayer
    # - this is the one branch that goes through a helper function rather
    # than constructing the Player subclass directly.
    from scripts.benchmark import _make_player

    player = _make_player(
        "ppo",
        battle_format="gen9ou",
        model_path=Path("data/models/win_prob.pt"),
        ppo_model_path=Path("data/models/ppo.zip"),
        n_simulations=5,
        ppo_bin_path=Path("data/cpp_weights/ppo.bin"),
        max_concurrent_battles=3,
    )

    assert player._max_concurrent_battles == 3


@pytest.mark.skipif(
    not Path("data/cpp_weights/ppo.bin").exists(),
    reason="requires data/cpp_weights/ppo.bin (scripts/export_weights.py), gitignored data/",
)
def test_DW_2_6_make_player_mcts_forwards_max_concurrent_battles():
    pytest.importorskip("battle_engine._native")
    from scripts.benchmark import _make_player

    player = _make_player(
        "mcts",
        battle_format="gen9ou",
        model_path=Path("data/models/win_prob.pt"),
        ppo_model_path=Path("data/models/ppo.zip"),
        n_simulations=5,
        ppo_bin_path=Path("data/cpp_weights/ppo.bin"),
        max_concurrent_battles=3,
    )

    assert player._max_concurrent_battles == 3


@pytest.mark.skipif(
    not Path("data/cpp_weights/ppo.bin").exists(),
    reason="requires data/cpp_weights/ppo.bin (scripts/export_weights.py), gitignored data/",
)
def test_DW_2_6_make_player_mcts_puct_forwards_max_concurrent_battles():
    pytest.importorskip("battle_engine._native")
    from scripts.benchmark import _make_player

    player = _make_player(
        "mcts_puct",
        battle_format="gen9ou",
        model_path=Path("data/models/win_prob.pt"),
        ppo_model_path=Path("data/models/ppo.zip"),
        n_simulations=5,
        ppo_bin_path=Path("data/cpp_weights/ppo.bin"),
        max_concurrent_battles=3,
    )

    assert player._max_concurrent_battles == 3
