"""Phase-0/1/2 benchmark harness CLI: pit two bots against each other.

Start the local Showdown server first (see README), then e.g.:

    .venv/bin/python scripts/benchmark.py --p1 maxdamage --p2 random --n-battles 500
    .venv/bin/python scripts/benchmark.py --p1 learned --p2 search --n-battles 500

"learned" loads the trained win-probability model (scripts/train_win_prob.py) and
wires it into TwoPlySearchPlayer's scoring via win_prob.make_eval_fn, in place of
evaluate() — same search shape as "search" (the Phase-1 bot), different eval.

Caveat worth knowing before trusting this number: the default --format is
gen9randombattle (Phase 0/1's format, auto-generated teams, no team-building
infra needed), but the model was trained on gen9ou human replays (constructed
OU teams). A "learned" benchmark on gen9randombattle is testing the model on
team compositions and movesets it never saw in training - a real distribution
mismatch, not just a formality. Interpret a loss here cautiously; it may
reflect the format mismatch more than the model's quality on its actual
target distribution.
"""

import argparse
import asyncio
from pathlib import Path

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from battle_engine.benchmark import run_benchmark
from battle_engine.search import TwoPlySearchPlayer
from battle_engine.win_prob import WinProbModel, make_eval_fn

PLAYERS = {
    "random": RandomPlayer,
    "maxdamage": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
    "search": TwoPlySearchPlayer,
}
CHOICES = sorted(PLAYERS) + ["learned"]


def _make_player(name: str, battle_format: str, model_path: Path) -> Player:
    if name == "learned":
        model = WinProbModel.load(model_path)
        return TwoPlySearchPlayer(
            battle_format=battle_format,
            eval_fn=make_eval_fn(model),
            # Not the default: see TwoPlySearchPlayer's docstring for why a
            # weight tuned for evaluate()'s scale would swamp a [0, 1]
            # probability output instead of complementing it.
            switch_urgency_weight=0.0,
        )
    return PLAYERS[name](battle_format=battle_format)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", choices=CHOICES, default="maxdamage")
    parser.add_argument("--p2", choices=CHOICES, default="random")
    parser.add_argument("--n-battles", type=int, default=500)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--model-path", type=Path, default=Path("data/models/win_prob.pt"))
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    p1 = _make_player(args.p1, args.format, args.model_path)
    p2 = _make_player(args.p2, args.format, args.model_path)
    result = await run_benchmark(p1, p2, n_battles=args.n_battles)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
