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
where required)"): use a dedicated alt, not a personal account - and a
DIFFERENT alt from ladder_ppo.py's, not the same one. Reusing that account
was tried first and found to contaminate the reading: Glicko rating deviation
converges after enough games that a new bot's results barely move it, so this
player's GXE would really be reporting Phase 3's PPO endpoint, not this
player's own strength - see
notes/phase-6-m6-ladder-canary-and-account-contamination.md. The password is never taken
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
batched near-instant battles, and --search-time-ms (default: set_search.py's
own DEFAULT_SEARCH_TIME_MS/DEFAULT_OPPONENT_SAMPLES, 1000 ms x 8 samples - a
real ladder turn has no benchmark-budget pressure, only Showdown's own timer)
is added directly to Showdown's own per-turn clock from the bot's side - a
real cost against a human opponent's patience, not just wall-clock
housekeeping. Start with a small --n-games.

**Pause/resume**: games are played one at a time via a Python-level loop
around poke_env's own `Player.ladder(1)` (not a single `ladder(n)` call),
checking --pause-file between iterations - the only point in a ladder run
where stopping doesn't strand a real opponent mid-game. `Player.ladder`'s own
internals (poke_env/player/player.py's `_ladder`) already loop battle-by-
battle this way internally, so calling it once per game from here is
equivalent, not a behavior change. Touch the pause file (`touch <path>`) to
pause before the next game is searched for, `rm` it to resume; the process
polls for it every --pause-poll-seconds. A SIGSTOP/SIGCONT on the process
itself would NOT be safe here - it would freeze the websocket mid-battle,
which is exactly the "goes silent, opponent's connection times out, game gets
abandoned" failure this avoids.

Usage:
    echo 'POKE_SHOWDOWN_USERNAME=my-bot-alt' >> .env.local
    echo 'POKE_SHOWDOWN_PASSWORD=...' >> .env.local
    .venv/bin/python scripts/ladder_setsearch.py --n-games 10
    touch /tmp/pause-m6   # pauses before the next game once the current one ends
    rm /tmp/pause-m6      # resumes
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import time
from pathlib import Path
from typing import Optional

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


async def _play_with_pause_support(
    player: SetSearchPlayer, n_games: int, pause_file: Optional[Path], pause_poll_seconds: float
) -> None:
    """`player.ladder(n_games)`, but able to pause between games.

    `Player.ladder` has no hook to pause mid-run, so this calls its own
    `ladder(1)` once per game instead - equivalent to a single `ladder(n)`
    call (poke_env's `_ladder` already loops battle-by-battle internally;
    see this module's docstring), but with a real point to check a pause
    flag between iterations. Checked only between games, never mid-battle -
    pausing mid-battle would go silent on a live opponent's connection,
    the exact failure this exists to avoid.
    """
    for game_index in range(n_games):
        if pause_file is not None:
            announced = False
            while pause_file.exists():
                if not announced:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] paused ({pause_file} exists) - "
                        f"{game_index}/{n_games} games played so far",
                        flush=True,
                    )
                    announced = True
                await asyncio.sleep(pause_poll_seconds)
            if announced:
                print(f"[{time.strftime('%H:%M:%S')}] resumed", flush=True)
        await player.ladder(1)


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
    parser.add_argument(
        "--pause-file",
        type=Path,
        default=None,
        help="if this path exists, pause before searching for the next game (the current "
        "game, if any, always finishes first). Not created by this script - an external "
        "`touch`/`rm` controls it. Default: no pause file, never pauses.",
    )
    parser.add_argument(
        "--pause-poll-seconds",
        type=float,
        default=10.0,
        help="how often to check --pause-file while paused (default: 10s)",
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
        start_timer_on_battle_start=True,
    )

    _instrument_progress(player)

    try:
        pause_note = f", pause file: {args.pause_file}" if args.pause_file else ""
        print(
            f"laddering as {args.username} on {args.format} for {args.n_games} games "
            f"({args.search_time_ms}ms over {args.opponent_samples} sampled opponents, "
            f"{args.threads} threads{pause_note})...",
            flush=True,
        )
        await _play_with_pause_support(player, args.n_games, args.pause_file, args.pause_poll_seconds)
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
