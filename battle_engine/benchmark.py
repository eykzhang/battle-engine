"""Benchmark harness: pit two poke-env players against each other and report
p1's win rate with a confidence interval.

Every phase gate in this project is a head-to-head win rate measured by this
harness (see CLAUDE.md) — it exists before any bot does, on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from poke_env.player import Player


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95%-by-default Wilson score confidence interval for a binomial proportion.

    Preferred over the normal (Wald) approximation: Wald can produce bounds
    outside [0, 1] and its coverage degrades badly at small n or win rates near
    0/1 — exactly the regime early phase-gate benchmarks run in.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = wins / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class BenchmarkResult:
    p1_name: str
    p2_name: str
    n_battles: int
    p1_wins: int
    p2_wins: int
    ties: int

    @property
    def p1_win_rate(self) -> float:
        return self.p1_wins / self.n_battles

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return wilson_interval(self.p1_wins, self.n_battles)

    def __str__(self) -> str:
        lo, hi = self.confidence_interval
        return (
            f"{self.p1_name} vs {self.p2_name}: {self.p1_wins}/{self.n_battles} wins "
            f"({self.p1_win_rate:.1%}, 95% CI [{lo:.1%}, {hi:.1%}]), "
            f"{self.p2_wins} losses, {self.ties} ties"
        )


async def run_benchmark(p1: Player, p2: Player, n_battles: int = 500) -> BenchmarkResult:
    """Play p1 vs p2 for n_battles and report p1's win rate with a 95% CI.

    Resets both players' battle history first so repeated calls on the same
    player instances don't mix results across runs.
    """
    p1.reset_battles()
    p2.reset_battles()
    await p1.battle_against(p2, n_battles=n_battles)
    return BenchmarkResult(
        p1_name=p1.__class__.__name__,
        p2_name=p2.__class__.__name__,
        n_battles=n_battles,
        p1_wins=p1.n_won_battles,
        p2_wins=p1.n_lost_battles,
        ties=p1.n_tied_battles,
    )
