"""Benchmark harness: pit two poke-env players against each other and report
p1's win rate with a confidence interval.

Every phase gate in this project is a head-to-head win rate measured by this
harness (see CLAUDE.md) — it exists before any bot does, on purpose.
"""

from __future__ import annotations

import asyncio
import json
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path

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
        # n_battles can be 0 for a run stopped before any battle completed
        # (see run_benchmark's graceful-early-exit support) - guard against
        # ZeroDivisionError rather than assume n_battles is always positive,
        # which was safe to assume before that support existed.
        return self.p1_wins / self.n_battles if self.n_battles else 0.0

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


async def run_benchmark(
    p1: Player,
    p2: Player,
    n_battles: int = 500,
    progress_interval: int = 0,
    checkpoint_path: Path | None = None,
    graceful_early_exit: bool = False,
) -> BenchmarkResult:
    """Play p1 vs p2 for up to n_battles and report p1's win rate with a 95% CI.

    Resets both players' battle history first so repeated calls on the same
    player instances don't mix results across runs.

    Plays one battle at a time (not a single bulk poke-env
    battle_against(..., n_battles=N) call) specifically so real progress is
    observable and, with graceful_early_exit, interruptible - both were
    genuinely absent before 2026-08-25, when an 8+ hour run with zero
    incremental output and no way to see or keep partial progress had to be
    killed blind, discarding everything computed (see notes/gotcha-
    benchmark-runs-need-empirical-timing-and-progress-visibility.md).

    progress_interval: print a running tally every N completed battles.
    0 (the default) disables periodic printing - callers like
    ppo_eval.py's EvalVsOpponentCallback run frequent small internal evals
    during training and don't want this output.

    checkpoint_path: if given, overwrite this file with the current partial
    tally (JSON: p1/p2 names, wins/losses/ties, battles completed vs.
    target, elapsed seconds) after every single completed battle - survives
    even a hard kill of the process, unlike a return value that only exists
    once the whole run finishes.

    graceful_early_exit: if True, install a SIGINT/SIGTERM handler for the
    duration of this call that stops the loop after the in-flight battle
    finishes (never mid-battle) and returns whatever's been completed so
    far - BenchmarkResult.n_battles reflects the real completed count, not
    the original target, so `kill <pid>` (not -9) gets a real, honest
    partial answer instead of nothing. Off by default: installing a signal
    handler around every call would be a real behavior change for existing
    callers (e.g. training-loop code that has its own Ctrl+C handling) that
    never asked for it - scripts/benchmark.py's CLI opts in explicitly.
    """
    p1.reset_battles()
    p2.reset_battles()

    stop_requested = asyncio.Event()
    installed_signals: list[signal.Signals] = []
    if graceful_early_exit:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_requested.set)
                installed_signals.append(sig)
            except (NotImplementedError, RuntimeError):
                pass  # best-effort only - not every platform/context supports this

    start = time.monotonic()
    completed = 0
    try:
        for i in range(1, n_battles + 1):
            await p1.battle_against(p2, n_battles=1)
            completed = i
            elapsed = time.monotonic() - start

            if checkpoint_path is not None:
                checkpoint_path.write_text(
                    json.dumps(
                        {
                            "p1_name": p1.__class__.__name__,
                            "p2_name": p2.__class__.__name__,
                            "n_completed": completed,
                            "n_battles_target": n_battles,
                            "p1_wins": p1.n_won_battles,
                            "p2_wins": p1.n_lost_battles,
                            "ties": p1.n_tied_battles,
                            "elapsed_s": round(elapsed, 1),
                        }
                    )
                )

            if progress_interval and (i % progress_interval == 0 or i == n_battles):
                rate = elapsed / i
                eta = rate * (n_battles - i)
                print(
                    f"[{i}/{n_battles}] {p1.n_won_battles}W-{p1.n_lost_battles}L-"
                    f"{p1.n_tied_battles}T ({p1.n_won_battles / i:.1%}) | "
                    f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                    flush=True,
                )

            if stop_requested.is_set():
                print(
                    f"\nStop requested - stopping after {completed}/{n_battles} battles "
                    "(a real, partial result, not a crash).",
                    flush=True,
                )
                break
    finally:
        if graceful_early_exit:
            for sig in installed_signals:
                loop.remove_signal_handler(sig)

    return BenchmarkResult(
        p1_name=p1.__class__.__name__,
        p2_name=p2.__class__.__name__,
        n_battles=completed,
        p1_wins=p1.n_won_battles,
        p2_wins=p1.n_lost_battles,
        ties=p1.n_tied_battles,
    )
