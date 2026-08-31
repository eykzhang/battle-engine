#!/usr/bin/env python
"""Phase 6 / M5: watch the set-search player actually decide, turn by turn.

Not a benchmark - a diagnostic, and the same technique that found every
pathology this project has caught: the Phase-1 bot never switching, the
Phase-2 eval undervaluing a low-HP retreat, and Phase 3's protect-spam loop
were all found by reading real turns rather than by staring at a win rate
(notes/pattern-watch-real-replays-not-just-metrics.md). A win rate says a bot
is losing; it never says why.

    .venv/bin/python scripts/inspect_set_search.py --n-battles 3
    .venv/bin/python scripts/inspect_set_search.py --n-battles 1 --opponent search --top 6

Each turn prints what the search chose, how much of the total visit mass went
to it, its win-probability estimate, and the runners-up. The three columns
worth watching:

- **rank** is non-zero when the search's own top pick was not legal in the real
  game and a lower-ranked action was played instead. Trapping is the known
  cause (the translator cannot represent "cannot switch"); a steady stream of
  non-zero ranks means something else is also wrong.
- **value** is the search's win probability for the chosen action, averaged
  over the sampled opponents. It drifting to 0 while the bot still has Pokemon
  left is the shape a losing position looks like from inside.
- **visit share** near 1/n_actions means the search found nothing to prefer -
  either the position is genuinely balanced, or the budget was too small.

Needs the local Showdown server running, a cached usage-stats file
(scripts/fetch_usage_stats.py) and the gen9 poke-engine build.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import List

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

from battle_engine.benchmark import run_benchmark
from battle_engine.search import TwoPlySearchPlayer
from battle_engine.set_search import Decision, SetSearchPlayer
from battle_engine.teams import RandomTeamFromPool

OPPONENTS = {
    "random": RandomPlayer,
    "maxdamage": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
    "search": TwoPlySearchPlayer,
}


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-battles", type=int, default=3)
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="search")
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--search-time-ms", type=int, default=400)
    parser.add_argument("--opponent-samples", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cutoff", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=4, help="runners-up to print per turn")
    return parser.parse_args(argv)


def _print_decision(decision: Decision, top: int) -> None:
    if decision.fallback_reason:
        print(f"  turn {decision.turn:>3}  DEFAULTED ({decision.fallback_reason})  -> {decision.order}")
        return
    share = decision.visits / decision.total_visits if decision.total_visits else 0.0
    flag = f"  [rank {decision.rank_played}]" if decision.rank_played else ""
    print(
        f"  turn {decision.turn:>3}  {decision.chosen:<24} "
        f"share={share:5.1%}  value={decision.value:.3f}  "
        f"{decision.seconds * 1000:4.0f}ms{flag}"
    )
    for name, visits, value in decision.ranked[1 : top + 1]:
        alt_share = visits / decision.total_visits if decision.total_visits else 0.0
        print(f"          {name:<24} share={alt_share:5.1%}  value={value:.3f}")


async def _run(args: argparse.Namespace) -> int:
    logging.disable(logging.WARNING)
    decisions: List[Decision] = []

    player = SetSearchPlayer(
        battle_format=args.format,
        team=RandomTeamFromPool() if args.format != "gen9randombattle" else None,
        search_time_ms=args.search_time_ms,
        n_opponent_samples=args.opponent_samples,
        threads=args.threads,
        cutoff=args.cutoff,
        seed=args.seed,
        account_configuration=AccountConfiguration.generate("setsearchdbg", rand=True),
        on_decision=lambda d: (decisions.append(d), _print_decision(d, args.top)),
    )
    opponent: Player = OPPONENTS[args.opponent](
        battle_format=args.format,
        team=RandomTeamFromPool() if args.format != "gen9randombattle" else None,
        account_configuration=AccountConfiguration.generate("setsearchopp", rand=True),
    )

    print(
        f"{args.n_battles} battles vs {args.opponent} in {args.format}: "
        f"{args.search_time_ms}ms over {args.opponent_samples} sampled opponents, "
        f"{args.threads} threads",
        flush=True,
    )
    result = await run_benchmark(player, opponent, n_battles=args.n_battles, progress_interval=1)
    stats = player.search_stats

    print()
    print(result)
    print(
        f"  turns {stats.turns}   {stats.ms_per_turn:.0f} ms/turn   "
        f"{stats.visits:,} visits total ({stats.visits // max(stats.turns, 1):,}/turn)"
    )
    print(
        f"  root pick was real-illegal on {stats.root_pick_illegal} turns; "
        f"fell through to the default order on {stats.defaulted}"
    )
    if stats.failures:
        print(f"  failures: {stats.failures}")
    return 0


def main(argv: List[str]) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
