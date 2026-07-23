import asyncio
import socket

import pytest

from battle_engine.benchmark import run_benchmark, wilson_interval


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
