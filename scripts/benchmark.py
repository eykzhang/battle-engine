"""Phase-0 benchmark harness CLI: pit two baseline bots against each other.

Start the local Showdown server first (see README), then e.g.:

    .venv/bin/python scripts/benchmark.py --p1 maxdamage --p2 random --n-battles 500
"""

import argparse
import asyncio

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from battle_engine.benchmark import run_benchmark
from battle_engine.search import TwoPlySearchPlayer

PLAYERS = {
    "random": RandomPlayer,
    "maxdamage": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
    "search": TwoPlySearchPlayer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", choices=sorted(PLAYERS), default="maxdamage")
    parser.add_argument("--p2", choices=sorted(PLAYERS), default="random")
    parser.add_argument("--n-battles", type=int, default=500)
    parser.add_argument("--format", default="gen9randombattle")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    p1 = PLAYERS[args.p1](battle_format=args.format)
    p2 = PLAYERS[args.p2](battle_format=args.format)
    result = await run_benchmark(p1, p2, n_battles=args.n_battles)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
