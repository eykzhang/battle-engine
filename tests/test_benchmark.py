import asyncio
import json
import os
import signal
import socket
import sys
from pathlib import Path

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
