"""Phase-0/1/2 benchmark harness CLI: pit two bots against each other.

Start the local Showdown server first (see README), then e.g.:

    .venv/bin/python scripts/benchmark.py --p1 maxdamage --p2 random --n-battles 500
    .venv/bin/python scripts/benchmark.py --p1 learned --p2 search --n-battles 500

"learned" loads the trained win-probability model (scripts/train_win_prob.py) and
wires it into TwoPlySearchPlayer's scoring via win_prob.make_eval_fn, in place of
evaluate() — same search shape as "search" (the Phase-1 bot), different eval.

"ppo" loads a trained Phase-3 PPO checkpoint (scripts/train_ppo.py --save) via
battle_engine.ppo_eval.load_ppo_player — a genuinely different mechanism from
"search"/"learned": no lookahead at all, a direct state -> action policy learned
through self-play rather than ranking states with a 2-ply search. Needs
--ppo-model-path (default data/models/ppo.zip) and only makes sense on gen9ou
(what it was trained on) — same distribution-mismatch caveat "learned" has on
gen9randombattle, undocumented here since this project's only real PPO training
target has been gen9ou from the start.

The model was trained on gen9ou human replays (constructed OU teams), but the
default --format is gen9randombattle (Phase 0/1's format, auto-generated
teams, no team-building infra needed) - a "learned" benchmark on
gen9randombattle tests the model on team compositions and movesets it never
saw in training, a real distribution mismatch, not just a formality. Pass
`--format gen9ou` to benchmark on-distribution instead; teams are then drawn
from battle_engine.teams's small pool of real Smogon sample teams (gen9ou
requires a submitted team - poke-env has no built-in generator for it, unlike
gen9randombattle).
"""

import argparse
import asyncio
from pathlib import Path

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from battle_engine.benchmark import run_benchmark
from battle_engine.ppo_eval import load_ppo_player
from battle_engine.search import TwoPlySearchPlayer
from battle_engine.teams import RandomTeamFromPool
from battle_engine.win_prob import WinProbModel, make_eval_fn

# Best point estimate from a 7-point sweep (0.0-0.2, 80 battles each, gen9ou)
# run after diagnosing that the learned eval - with no switch-urgency
# compensation at all - undervalues switching away from a critically low-HP
# Pokemon relative to attacking with it (same structural blind spot the
# Phase-1 switch-urgency patch fixed for evaluate(), just never replaced for
# the learned eval's different scale). 0.0 scored 28.7%; 0.03/0.05/0.08 all
# clustered around 40-42.5%; 0.12 dipped to 31.2% (likely noise at N=80,
# CIs overlap) before 0.15/0.2 partially recovered. Worth re-sweeping with a
# larger N or a finer grid if the model is retrained again.
LEARNED_SWITCH_URGENCY_WEIGHT = 0.08

PLAYERS = {
    "random": RandomPlayer,
    "maxdamage": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
    "search": TwoPlySearchPlayer,
}
CHOICES = sorted(PLAYERS) + ["learned", "ppo"]


# Formats poke-env/Showdown generate a team for server-side - no submitted
# team needed. Everything else (gen9ou, the only other format this CLI
# actually targets today) needs one, hence RandomTeamFromPool below. An
# explicit allowlist rather than `!= "gen9randombattle"` - review flagged
# the negative check as a footgun: any *other* auto-team format (e.g.
# gen8randombattle) would have silently gotten an OU-team pool instead of
# no team, if this CLI's --format were ever pointed there.
_AUTO_TEAM_FORMATS = {"gen9randombattle"}


def _make_player(
    name: str, battle_format: str, model_path: Path, ppo_model_path: Path
) -> Player:
    team = None if battle_format in _AUTO_TEAM_FORMATS else RandomTeamFromPool()
    if name == "learned":
        model = WinProbModel.load(model_path)
        return TwoPlySearchPlayer(
            battle_format=battle_format,
            team=team,
            eval_fn=make_eval_fn(model),
            switch_urgency_weight=LEARNED_SWITCH_URGENCY_WEIGHT,
        )
    if name == "ppo":
        return load_ppo_player(ppo_model_path, battle_format=battle_format, team=team)
    return PLAYERS[name](battle_format=battle_format, team=team)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", choices=CHOICES, default="maxdamage")
    parser.add_argument("--p2", choices=CHOICES, default="random")
    parser.add_argument("--n-battles", type=int, default=500)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--model-path", type=Path, default=Path("data/models/win_prob.pt"))
    parser.add_argument("--ppo-model-path", type=Path, default=Path("data/models/ppo.zip"))
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    p1 = _make_player(args.p1, args.format, args.model_path, args.ppo_model_path)
    p2 = _make_player(args.p2, args.format, args.model_path, args.ppo_model_path)
    result = await run_benchmark(p1, p2, n_battles=args.n_battles)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
