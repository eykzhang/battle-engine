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

**Concurrency defaults to 1 but is now real.** SetSearchPlayer.choose_move is
`async def` (battle_engine/set_search.py) - translation and the post-search
legality check run on poke-env's event loop, and only the actual
poke_engine.monte_carlo_tree_search call is offloaded to a thread, so a
second concurrent battle's server messages (including its own turn timer)
are processed normally while a search is in flight. --max-concurrent-battles
raises the player's own bound on how many ladder games it has in flight at
once (poke-env's Player._ladder already paces `search_ladder_game` requests
against that bound internally - verified directly against
poke_env/player/player.py, the same mechanism battle_against uses). One
real, still-open caveat: poke_engine.monte_carlo_tree_search holds Python's
GIL for its whole call and does not release it (verified against the pinned
Rust source - see notes/gotcha-poke-engine-mcts-holds-the-gil.md), so
concurrent battles do not search in parallel - they interleave, each search
still fully serialized against the others. Concurrency here buys real games
in flight at once (server-side wait time, turn timers, opponent think-time
all overlap), not faster search. Not yet run against real opponents above
concurrency 1 - start small.

Each ladder game is real and real-time, unlike the local benchmark harness's
batched near-instant battles, and --search-time-ms (default: set_search.py's
own DEFAULT_SEARCH_TIME_MS/DEFAULT_OPPONENT_SAMPLES, 1000 ms x 8 samples - a
real ladder turn has no benchmark-budget pressure, only Showdown's own timer)
is added directly to Showdown's own per-turn clock from the bot's side - a
real cost against a human opponent's patience, not just wall-clock
housekeeping. Start with a small --n-games.

**Pause/resume**: one bulk `player.ladder(remaining_games)` call runs as a
background task, raced against a --pause-file watcher via
`asyncio.wait(..., return_when=FIRST_COMPLETED)` - the same design
battle_engine/benchmark.py's run_benchmark uses for graceful concurrent
early-exit. Real per-game completion is tracked via
`Player._battle_finished_callback` (poke-env's own native per-game hook),
not by counting `ladder()`'s own return, which only fires once ALL games in
that bulk call are done. When the pause file appears: the ladder task is
cancelled (safe - only stops issuing new `search_ladder_game` requests for
NEW games; already-in-flight games' own message handling runs as
independent tasks and is untouched), then in-flight games are drained via
`Player._battle_count_queue.join()` (bridged through poke-env's
`handle_threaded_coroutines`, since poke-env runs its own coroutines on a
separate event-loop thread - a bare `await` on that queue raises
`RuntimeError`). At `--max-concurrent-battles 1` this is one game, same as
before; above 1, pausing lets every currently in-flight game finish first,
which can be more than one - never a mid-battle stop. Once drained, the
process waits for the pause file to be removed, then relaunches `ladder()`
for whatever games remain - repeatable across multiple pause/resume cycles
in one run. A SIGSTOP/SIGCONT on the process itself would NOT be safe here -
it would freeze the websocket mid-battle, which is exactly the "goes
silent, opponent's connection times out, game gets abandoned" failure this
avoids.

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
import threading
import time
from pathlib import Path
from typing import Optional

from poke_env.concurrency import handle_threaded_coroutines
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


async def _wait_for_pause_file(pause_file: Path, poll_seconds: float) -> None:
    """Completes once `pause_file` exists - a wakeup signal to race against
    the in-flight `ladder()` task, not itself the paused-wait loop below."""
    while not pause_file.exists():
        await asyncio.sleep(poll_seconds)


async def _play_with_pause_support(
    player: SetSearchPlayer, n_games: int, pause_file: Optional[Path], pause_poll_seconds: float
) -> None:
    """Plays n_games on the ladder with real concurrency (bounded by the
    player's own max_concurrent_battles) and pause/resume support.

    poke_env's Player.ladder(n) already implements bounded concurrency
    correctly internally (`_ladder`'s `while self._battle_count_queue.full():
    wait` loop, verified directly against poke_env/player/player.py) - the
    same shape battle_engine/benchmark.py's run_benchmark uses via
    battle_against. So this calls `ladder(remaining)` in one bulk background
    task per pause/resume segment, rather than one `ladder(1)` call per game
    - a sequential ladder(1)-per-game loop would defeat max_concurrent_battles
    entirely, the same bug run_benchmark had before its own fix.

    Real per-game completion (for the pause-safe game count, not for
    `ladder()`'s own return - that only fires once every game in the whole
    bulk call is done) comes from wrapping `Player._battle_finished_callback`,
    poke-env's native per-game hook, fired synchronously and independently of
    `ladder()`'s own completion.
    """
    completed = 0
    # _on_finished runs on poke-env's own POKE_LOOP background thread (fired
    # from _handle_battle_message on the win/tie message - see
    # poke_env/concurrency.py), while the while-loop below reads `completed`
    # from this coroutine's own thread. A bare `completed += 1` happens to be
    # safe under CPython's GIL for a single int, but every other cross-thread
    # touch in this function goes through an explicit synchronization
    # primitive (handle_threaded_coroutines below) - match that rather than
    # lean on GIL incidental behavior, so a future refactor that batches the
    # increment can't silently drop or double-count completions.
    completed_lock = threading.Lock()
    original_callback = player._battle_finished_callback

    def _on_finished(battle) -> None:
        nonlocal completed
        original_callback(battle)
        with completed_lock:
            completed += 1

    player._battle_finished_callback = _on_finished
    try:
        while True:
            with completed_lock:
                if completed >= n_games:
                    break
                remaining = n_games - completed
            ladder_task = asyncio.create_task(player.ladder(remaining))

            if pause_file is not None:
                pause_signal = asyncio.create_task(_wait_for_pause_file(pause_file, pause_poll_seconds))
                try:
                    done, _pending = await asyncio.wait(
                        {ladder_task, pause_signal}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if ladder_task not in done:
                        ladder_task.cancel()
                        with completed_lock:
                            so_far = completed
                        print(
                            f"[{time.strftime('%H:%M:%S')}] pause requested - stopping after "
                            f"{so_far}/{n_games} games so far, letting in-flight games finish...",
                            flush=True,
                        )
                    else:
                        pause_signal.cancel()
                finally:
                    if not pause_signal.done():
                        pause_signal.cancel()

            try:
                await ladder_task
            except asyncio.CancelledError:
                pass

            # Cross-loop: ladder() runs (via handle_threaded_coroutines) on
            # POKE_LOOP, not this coroutine's own loop, so a bare
            # `await player._battle_count_queue.join()` here raises
            # RuntimeError (bound to a different event loop) - route through
            # the same bridge poke-env's own public methods use internally.
            # A no-op when ladder_task already completed normally; the real
            # drain only matters after the cancel-on-pause branch above.
            await handle_threaded_coroutines(player._battle_count_queue.join(), player.ps_client.loop)

            with completed_lock:
                so_far, target_reached = completed, completed >= n_games
            # Only wait to be un-paused if there's still something left to
            # resume - otherwise this segment's ladder(remaining) simply
            # finished on its own with the pause file still sitting there
            # (e.g. touched during the very last game), and there is nothing
            # left to pause before. Without this check the run hangs forever
            # here even though every requested game already completed.
            if pause_file is not None and pause_file.exists() and not target_reached:
                print(
                    f"[{time.strftime('%H:%M:%S')}] paused ({pause_file} exists) - "
                    f"{so_far}/{n_games} games played so far",
                    flush=True,
                )
                while pause_file.exists():
                    await asyncio.sleep(pause_poll_seconds)
                print(f"[{time.strftime('%H:%M:%S')}] resumed", flush=True)
    finally:
        player._battle_finished_callback = original_callback


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
        "sequential). Real as of this fix - poke-env's own Player.ladder() paces "
        "concurrent games against this bound internally. Not yet run against real "
        "opponents above 1 - see the module docstring's concurrency section.",
    )
    parser.add_argument(
        "--pause-file",
        type=Path,
        default=None,
        help="if this path exists, pause before searching for new games (any currently "
        "in-flight games - one at the default concurrency, possibly more above it - always "
        "finish first). Not created by this script - an external `touch`/`rm` controls it. "
        "Default: no pause file, never pauses.",
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
            f"NOTE: --max-concurrent-battles {args.max_concurrent_battles} - real concurrent "
            "games now (SetSearchPlayer.choose_move is async/thread-safe as of the Phase 1 "
            "concurrency fix), but this is the first time it's being run against real ladder "
            "opponents above 1 - watch the first few games. Search itself does not run in "
            "parallel across battles (poke_engine.monte_carlo_tree_search holds the GIL for "
            "its whole call) - this buys overlapping wait/think time, not faster search. See "
            "the module docstring.",
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
