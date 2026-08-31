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

"mcts" (Phase 4 M7) wires battle_engine.mcts_player.MctsPlayer — the C++
open-loop MCTS/DUCT search (cpp/src/mcts.cpp) with default_eval, plain
UCB1, no PPO prior/value (that's "mcts_puct" below, a different player
entirely). Needs --n-simulations (no built-in default — this project's
laptop-first hard rule requires a real measured ms/turn number before
picking one, not a guess baked into this CLI).

"mcts_puct" (Phase 4 M6b) wires battle_engine.mcts_player.MctsPuctPlayer —
the same C++ open-loop search tree as "mcts", but PUCT-guided: the trained
PPO actor supplies a per-node move prior and its critic supplies the leaf
value (cpp/src/mcts.cpp's search_puct(), see cpp/include/be/mcts.hpp's own
doc comment for the full design and the measured critic sign-backup
convention). Needs --n-simulations (same laptop-first rationale as "mcts")
and --ppo-bin-path (default data/cpp_weights/ppo.bin, scripts/
export_weights.py's output — a distinct artifact from --ppo-model-path's
data/models/ppo.zip, which "ppo" loads directly via PyTorch/stable-
baselines3 rather than this C++-facing binary format). Trained on gen9ou,
same distribution-mismatch caveat as "learned"/"ppo" — only meaningful with
--format gen9ou.

"setsearch" (Phase 6 M5) wires battle_engine.set_search.SetSearchPlayer — the
Foul Play architecture, and the first player here whose forward model is a
complete gen9 simulator rather than this project's own. poke-engine's MCTS run
over --opponent-samples independently sampled opponent teams (M4's Smogon
usage-statistics prior), each getting an equal slice of a --search-time-ms
wall-clock budget, with the action chosen by visits summed across all of them.
Unlike "mcts"/"mcts_puct" it takes a time budget rather than --n-simulations,
because the budget is what a real ladder turn actually constrains. Needs a
cached usage-stats file (scripts/fetch_usage_stats.py) and the gen9 poke-engine
build (scripts/build_poke_engine.sh). gen9ou is what the prior is for, so
--format gen9ou; on gen9randombattle the usage statistics describe a different
metagame entirely, a sharper distribution mismatch than "learned"/"ppo" have.

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
from poke_env.ps_client import AccountConfiguration

from battle_engine.benchmark import run_benchmark
from battle_engine.ppo_eval import load_ppo_player
from battle_engine.search import TwoPlySearchPlayer
from battle_engine.teams import RandomTeamFromPool
from battle_engine.win_prob import WinProbModel, make_eval_fn

# battle_engine.mcts_player imports battle_engine._native (the compiled C++
# extension) at module scope - importing MctsPlayer at THIS file's top
# level would make every benchmark matchup depend on ./scripts/build_cpp.sh
# having run, even a plain "random vs maxdamage" comparison that never
# touches "mcts". Deferred into _make_player's "mcts" branch instead, same
# "skip/fail only where actually needed" convention
# tests/test_native_legality.py's pytest.importorskip already established.

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
CHOICES = sorted(PLAYERS) + ["learned", "ppo", "mcts", "mcts_puct", "setsearch"]


# Formats poke-env/Showdown generate a team for server-side - no submitted
# team needed. Everything else (gen9ou, the only other format this CLI
# actually targets today) needs one, hence RandomTeamFromPool below. An
# explicit allowlist rather than `!= "gen9randombattle"` - review flagged
# the negative check as a footgun: any *other* auto-team format (e.g.
# gen8randombattle) would have silently gotten an OU-team pool instead of
# no team, if this CLI's --format were ever pointed there.
_AUTO_TEAM_FORMATS = {"gen9randombattle"}


def _make_player(
    name: str,
    battle_format: str,
    model_path: Path,
    ppo_model_path: Path,
    n_simulations: int,
    ppo_bin_path: Path,
    search_time_ms: int = 1000,
    opponent_samples: int = 8,
    threads: int = 4,
    usage_cutoff: int = 1500,
) -> Player:
    team = None if battle_format in _AUTO_TEAM_FORMATS else RandomTeamFromPool()
    # rand=True (a random 5-char suffix, not poke-env's own default per-process
    # incrementing counter) so repeated CLI invocations against the same
    # long-lived local server never collide with a stale username. Discovered
    # necessary during this phase's own DW-1.2 benchmark run: the local
    # Showdown server ties outstanding challenges to a userid, not a
    # connection, and never expires or clears them on disconnect (verified
    # against pokemon-showdown/server/ladders.ts's makeChallenge/
    # ladders-challenges.ts - a challenge is only cleared by acceptance,
    # cancellation, or the user renaming) - so a prior run's abrupt exit
    # (e.g. a killed process) left a real "already a challenge" popup
    # blocking every subsequent run that reused the same default username.
    account_configuration = AccountConfiguration.generate(name, rand=True)
    if name == "learned":
        model = WinProbModel.load(model_path)
        return TwoPlySearchPlayer(
            battle_format=battle_format,
            team=team,
            eval_fn=make_eval_fn(model),
            switch_urgency_weight=LEARNED_SWITCH_URGENCY_WEIGHT,
            account_configuration=account_configuration,
        )
    if name == "ppo":
        return load_ppo_player(
            ppo_model_path,
            battle_format=battle_format,
            team=team,
            account_configuration=account_configuration,
        )
    if name == "mcts":
        from battle_engine.mcts_player import MctsPlayer  # see the deferred-import comment above

        return MctsPlayer(
            battle_format=battle_format,
            team=team,
            n_simulations=n_simulations,
            account_configuration=account_configuration,
        )
    if name == "mcts_puct":
        from battle_engine.mcts_player import MctsPuctPlayer  # see the deferred-import comment above

        return MctsPuctPlayer(
            battle_format=battle_format,
            team=team,
            ppo_bin_path=str(ppo_bin_path),
            n_simulations=n_simulations,
            account_configuration=account_configuration,
        )
    if name == "setsearch":
        # Deferred for the same reason as "mcts" above, plus one of its own:
        # importing it parses a ~14 MB usage-stats file, which a benchmark that
        # never uses this player should not pay for.
        from battle_engine.set_search import SetSearchPlayer

        return SetSearchPlayer(
            battle_format=battle_format,
            team=team,
            search_time_ms=search_time_ms,
            n_opponent_samples=opponent_samples,
            threads=threads,
            cutoff=usage_cutoff,
            account_configuration=account_configuration,
        )
    return PLAYERS[name](battle_format=battle_format, team=team, account_configuration=account_configuration)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", choices=CHOICES, default="maxdamage")
    parser.add_argument("--p2", choices=CHOICES, default="random")
    parser.add_argument("--n-battles", type=int, default=500)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--model-path", type=Path, default=Path("data/models/win_prob.pt"))
    parser.add_argument("--ppo-model-path", type=Path, default=Path("data/models/ppo.zip"))
    # "mcts_puct" only - scripts/export_weights.py's C++-facing binary
    # format (Phase 2), a distinct artifact from --ppo-model-path above
    # (the raw PyTorch/stable-baselines3 checkpoint "ppo" loads directly).
    parser.add_argument("--ppo-bin-path", type=Path, default=Path("data/cpp_weights/ppo.bin"))
    # No principled default exists independent of measurement - this
    # project's laptop-first hard rule (CLAUDE.md) requires a real ms/turn
    # number before picking a simulation count, not a guess. 200 is this
    # phase's own measured choice: 501.9ms/turn (Debug/ASan,
    # cpp/tests/test_mcts.cpp's [!benchmark] case), checked against Phase
    # 3's DW-3.3 projection (6-9hr worst case for Phase 6's full sweep) and
    # found comfortably within budget (~2.1hr) - see
    # notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md for the
    # full comparison and arithmetic. Override for a different
    # laptop-feasibility tradeoff.
    parser.add_argument("--n-simulations", type=int, default=200)
    # "setsearch" only. A wall-clock budget, not a simulation count - see this
    # module's docstring. The defaults are measured on the M4 MacBook Air (the
    # laptop-first hard rule's target machine) rather than guessed: ~1,830
    # visits/ms at 4 threads and flat past 4, and splitting the budget across
    # sampled opponents is close to free (850k visits at 1 sample vs 906k at 8,
    # over the same 1,000 ms), so samples are chosen for opponent coverage
    # rather than against throughput.
    parser.add_argument("--search-time-ms", type=int, default=1000)
    parser.add_argument("--opponent-samples", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--usage-cutoff",
        type=int,
        default=1500,
        help=(
            "rating cutoff of the usage-stats file 'setsearch' predicts from. 1500 "
            "measured best on the M4 corpus; higher cuts describe a stronger "
            "population than the one being played."
        ),
    )
    # Real progress visibility + a survivable early exit - both genuinely
    # missing before 2026-08-25 (see notes/gotcha-benchmark-runs-need-
    # empirical-timing-and-progress-visibility.md): an 8+ hour run with
    # zero incremental output had to be killed blind, losing everything.
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print a running win/loss tally every N completed battles (0 disables).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Overwrite this file with the current partial tally (JSON) after every "
            "completed battle - survives a hard kill. Defaults to "
            "/tmp/benchmark_checkpoint_<p1>_vs_<p2>.json."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    extras = dict(
        search_time_ms=args.search_time_ms,
        opponent_samples=args.opponent_samples,
        threads=args.threads,
        usage_cutoff=args.usage_cutoff,
    )
    p1 = _make_player(
        args.p1, args.format, args.model_path, args.ppo_model_path, args.n_simulations, args.ppo_bin_path, **extras
    )
    p2 = _make_player(
        args.p2, args.format, args.model_path, args.ppo_model_path, args.n_simulations, args.ppo_bin_path, **extras
    )
    checkpoint_path = args.checkpoint_path or Path(f"/tmp/benchmark_checkpoint_{args.p1}_vs_{args.p2}.json")
    print(
        f"Checkpointing progress to {checkpoint_path} after every battle - "
        "Ctrl+C or `kill <pid>` (not -9) stops gracefully with a real partial result.",
        flush=True,
    )
    result = await run_benchmark(
        p1,
        p2,
        n_battles=args.n_battles,
        progress_interval=args.progress_interval,
        checkpoint_path=checkpoint_path,
        graceful_early_exit=True,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
