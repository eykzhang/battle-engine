"""Phase 6 / M6: plays SetSearchPlayer (the Foul Play architecture) on the
REAL Pokemon Showdown ladder (sim3.psim.us, via
poke_env.ps_client.ShowdownServerConfiguration) - not the local dev server
every other script in this repo defaults to. This is the phase gate itself:
GXE above 50%, against Phase 3's measured 26.3% baseline
(notes/phase-3-gate-met-overnight-run.md). The internal bot roster (search,
mcts, ...) stops being informative once the model changes this much - see
the phase-6 plan's M6 section.

Requires an already-registered Showdown account (this script cannot create
one - see scripts/ladder_ppo.py's module docstring, which verified this
against poke-env's own login code). Per this project's Hard Rules ("Ladder
runs of the bot itself follow bot etiquette (alt account, register as bot
where required)"): use a dedicated alt, not a personal account - the same
alt scripts/ladder_ppo.py used is fine to reuse. The password is never taken
as a plain CLI argument; set POKE_SHOWDOWN_USERNAME/POKE_SHOWDOWN_PASSWORD or
put them in the gitignored .env.local, exactly as ladder_ppo.py does.

GXE isn't computed here - poke-env's ladder() only reports this run's own
win/loss/tie count, same limitation ladder_ppo.py has. Check the account's
real GXE at https://pokemonshowdown.com/users/<username> after playing
(needs enough games for both the ladder rating and its displayed GXE to
settle - Showdown's own FAQ is the source for how many).

**Concurrency is 1 by default and should stay there for this player**,
unlike ladder_ppo.py's PPO policy. SetSearchPlayer.choose_move is a
synchronous, CPU-bound call that blocks poke-env's single asyncio event loop
for its entire search_time_ms budget (poke_engine.monte_carlo_tree_search is
not awaited - it can't be, it's a Rust extension call) - so a second
concurrent battle's server messages, including its own turn timer, cannot be
processed while a search is running. ladder_ppo.py's higher concurrency was
verified safe there because FrozenPolicyPlayer.choose_move has no blocking
work in it at all; that verification does not transfer to this player, and
raising --max-concurrent-battles here has not been tested against a real
opponent's turn timer.

Each ladder game is real and real-time, unlike the local benchmark harness's
batched near-instant battles, and --search-time-ms (default matches the M5
gate: 400 ms x 4 samples) is added directly to Showdown's own per-turn clock
from the bot's side - a real cost against a human opponent's patience, not
just wall-clock housekeeping. Start with a small --n-games.

Usage:
    echo 'POKE_SHOWDOWN_USERNAME=my-bot-alt' >> .env.local
    echo 'POKE_SHOWDOWN_PASSWORD=...' >> .env.local
    .venv/bin/python scripts/ladder_setsearch.py --n-games 10
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import time
from pathlib import Path

from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client import AccountConfiguration, ShowdownServerConfiguration

from battle_engine.set_search import DEFAULT_OPPONENT_SAMPLES, DEFAULT_SEARCH_TIME_MS, DEFAULT_THREADS, SetSearchPlayer
from battle_engine.teams import RandomTeamFromPool

DOTENV_PATH = Path(".env.local")


def _instrument_progress(player: SetSearchPlayer) -> None:
    """Same technique as ladder_ppo.py's _instrument_progress, for the same
    reason: a real ladder run with no live output is indistinguishable from
    a silent stall until it either finishes or a human notices nothing is
    happening (confirmed the hard way there - a 40-game run went dark for
    25+ minutes with piped stdout fully buffered).
    """
    seen_battles: set[str] = set()
    original_choose_move = player.choose_move

    def logged_choose_move(battle) -> BattleOrder:
        if battle.battle_tag not in seen_battles:
            seen_battles.add(battle.battle_tag)
            print(
                f"[{time.strftime('%H:%M:%S')}] battle started: {battle.battle_tag} "
                f"({len(seen_battles)} started so far)",
                flush=True,
            )
        return original_choose_move(battle)

    player.choose_move = logged_choose_move

    original_finished_callback = player._battle_finished_callback

    def logged_finished_callback(battle) -> None:
        original_finished_callback(battle)
        print(
            f"[{time.strftime('%H:%M:%S')}] battle finished: {battle.battle_tag} - "
            f"won={player.n_won_battles} lost={player.n_lost_battles} tied={player.n_tied_battles}",
            flush=True,
        )

    player._battle_finished_callback = logged_finished_callback


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader for a gitignored local secrets file - real
    env vars set before invocation always win. Copied from ladder_ppo.py
    rather than shared, same reasoning as that file: too small to be worth a
    new module.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


async def main() -> None:
    _load_dotenv(DOTENV_PATH)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--username",
        default=os.environ.get("POKE_SHOWDOWN_USERNAME"),
        required="POKE_SHOWDOWN_USERNAME" not in os.environ,
        help="an already-registered Showdown account (alt, not personal). Defaults to "
        "POKE_SHOWDOWN_USERNAME (env or .env.local) if set.",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=10,
        help="real, real-time ladder games against real humans - keep small, unlike the "
        "local benchmark harness's 500-battle runs (default: 10)",
    )
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--search-time-ms", type=int, default=DEFAULT_SEARCH_TIME_MS)
    parser.add_argument("--opponent-samples", type=int, default=DEFAULT_OPPONENT_SAMPLES)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--usage-cutoff",
        type=int,
        default=1500,
        help="rating cutoff of the usage-stats file this player predicts from - see "
        "battle_engine/set_search.py's module docstring for why 1500",
    )
    parser.add_argument(
        "--save-replays",
        action="store_true",
        help="save each battle's replay locally (poke-env's own save_replays option)",
    )
    parser.add_argument(
        "--max-concurrent-battles",
        type=int,
        default=1,
        help="how many ladder battles this account plays simultaneously (default: 1, "
        "sequential). See the module docstring for why this player has not been "
        "verified safe above 1, unlike ladder_ppo.py's policy player.",
    )
    args = parser.parse_args()

    if args.max_concurrent_battles > 1:
        print(
            f"WARNING: --max-concurrent-battles {args.max_concurrent_battles} raises "
            "poke-env's own battle concurrency, but SetSearchPlayer.choose_move blocks "
            "the single asyncio event loop for its whole search budget - unlike "
            "ladder_ppo.py's policy player, this has not been verified safe above 1. "
            "See this script's module docstring. Proceeding anyway.",
            flush=True,
        )

    password = os.environ.get("POKE_SHOWDOWN_PASSWORD") or getpass.getpass(
        f"Showdown password for {args.username}: "
    )

    player = SetSearchPlayer(
        battle_format=args.format,
        team=RandomTeamFromPool(),
        search_time_ms=args.search_time_ms,
        n_opponent_samples=args.opponent_samples,
        threads=args.threads,
        cutoff=args.usage_cutoff,
        account_configuration=AccountConfiguration(args.username, password),
        server_configuration=ShowdownServerConfiguration,
        save_replays=args.save_replays,
        max_concurrent_battles=args.max_concurrent_battles,
    )

    _instrument_progress(player)

    try:
        print(
            f"laddering as {args.username} on {args.format} for {args.n_games} games "
            f"({args.search_time_ms}ms over {args.opponent_samples} sampled opponents, "
            f"{args.threads} threads)...",
            flush=True,
        )
        await player.ladder(args.n_games)
        print(
            f"\n{player.n_won_battles}/{args.n_games} won "
            f"({player.n_won_battles / args.n_games:.1%}), "
            f"{player.n_lost_battles} lost, {player.n_tied_battles} tied",
            flush=True,
        )
        stats = player.search_stats
        print(
            f"  {stats.turns} turns, {stats.ms_per_turn:.0f} ms/turn, "
            f"root pick real-illegal on {stats.root_pick_illegal} turns, "
            f"defaulted on {stats.defaulted}",
            flush=True,
        )
        if stats.failures:
            print(f"  failures: {stats.failures}", flush=True)
        print(
            f"check the real GXE at https://pokemonshowdown.com/users/{args.username}",
            flush=True,
        )
    finally:
        # Same connected-player leak fixed everywhere else in this repo -
        # see ladder_ppo.py, inspect_ppo_replays.py, ppo_eval.py.
        await player.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
