import asyncio
import socket
import sys
from pathlib import Path

import pytest

from battle_engine.benchmark import run_benchmark, wilson_interval

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
