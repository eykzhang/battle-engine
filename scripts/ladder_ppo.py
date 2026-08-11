"""Plays the trained PPO checkpoint on the REAL Pokemon Showdown ladder
(sim3.psim.us, via poke_env.ps_client.ShowdownServerConfiguration) - not the
local dev server every other script in this repo defaults to. This is the
roadmap's other Phase 3 gate component: "the final bot is benchmarked on the
real ladder ... for a GXE number to put on a resume."

Requires an already-registered Showdown account (this script cannot create
one - poke-env's client only logs into an existing account; verified against
the installed poke_env.ps_client.ps_client.log_in source, which sends a real
password-authenticated /trn only when a password is supplied, and has no
registration call at all). Per this project's Hard Rules ("Ladder runs of
the bot itself follow bot etiquette (alt account, register as bot where
required)"): use a dedicated alt, not a personal account. No specific
"register as bot"/alt-account rule was found codified in Showdown's own
rules page, FAQ, or its linked Bot FAQ as of 2026-08-11 - this is a
self-imposed precaution (keeps an experimental, unpolished automated
player's record off any personal identity), not a documented compliance
requirement.

The password is never taken as a plain CLI argument (would land in shell
history/process listings). Set POKE_SHOWDOWN_USERNAME/POKE_SHOWDOWN_PASSWORD
directly, or put them in a gitignored .env.local (KEY=VALUE per line) at the
repo root - loaded automatically if present, real env vars still take
precedence over it. Falls back to --username plus a secure getpass prompt if
neither supplies a password.

Each ladder game is a real, real-time match against a real human opponent -
nothing like the local benchmark harness's batched, near-instant battles.
Wall-clock cost scales with --n-games accordingly; start small.

--max-concurrent-battles (default 1) raises poke-env's own
Player(max_concurrent_battles=...) limit, letting this one account/connection
have that many ladder searches/battles in flight at once - a real, native
poke-env feature (see Player._ladder's _battle_count_queue/_battle_semaphore),
not something bolted on here. Checked safe for this script's specific setup
before exposing it: FrozenPolicyPlayer.choose_move has no `await` in it, and
poke-env's client loop is single-threaded/cooperative, so concurrent battles
can't corrupt each other's move computation mid-call; load_ppo_player also
seeds exactly one policy snapshot, so which battle is "active" doesn't affect
which weights get used regardless of interleaving.

KNOWN BUG at concurrency >= 5 (found the hard way, 2026-08-11 - see CLAUDE.md's
Phase 3 status for the full account): poke-env's own `_ladder` fires the next
ladder search BEFORE checking whether it's already at max_concurrent_battles
capacity, and the real Showdown server enforces a hard 5-concurrent-battles-
per-IP limit (server/monitor.ts's countConcurrentBattle) - so running at
exactly 5 lets poke-env's off-by-one push a 6th search past the server's
ceiling, which gets silently rejected (a |popup| poke-env only logs as a
warning and never retries). _ladder then waits forever on a signal that will
never arrive - a permanent, zero-CPU stall with the connection still alive,
confirmed via a real run that went completely dead for 25+ minutes before
being killed. Stay at 4 or below until poke-env's search-before-capacity-
check ordering is actually fixed upstream or patched here.

GXE isn't computed here - poke-env's ladder() only reports this run's own
win/loss/tie count. Check the account's real ladder rating (which needs
enough games for its rating deviation to settle) at
https://pokemonshowdown.com/users/<username> after playing.

Usage:
    echo 'POKE_SHOWDOWN_USERNAME=my-bot-alt' >> .env.local
    echo 'POKE_SHOWDOWN_PASSWORD=...' >> .env.local
    .venv/bin/python scripts/ladder_ppo.py --n-games 10
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

from battle_engine.ppo_eval import load_ppo_player
from battle_engine.teams import RandomTeamFromPool

DEFAULT_MODEL_PATH = Path("data/models/ppo.zip")
DOTENV_PATH = Path(".env.local")


def _instrument_progress(player) -> None:
    """Prints a flushed, real-time line on every battle start and finish -
    added after a real 40-game/5-concurrent run went silent for 25+ minutes
    (rating flat, near-zero CPU) with no way to tell live whether it was
    stalled or just playing slow real opponents, because Python fully
    buffers stdout when piped through `tee` and nothing flushes until a
    clean process exit (confirmed the hard way: even after SIGTERM, the
    piped log was still completely empty - buffering, not evidence either
    way). Player._battle_finished_callback is a real, public no-op hook
    meant to be overridden (see poke_env.player.player.Player) - not a
    private/internal detail. choose_move is also wrapped (same technique as
    inspect_ppo_replays.py's _instrument) so a start line prints on a
    battle's first move, since poke-env has no equivalent public
    battle-started callback.
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
    env vars set before invocation always win (os.environ.setdefault), no
    new dependency for something this small.
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
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--format", default="gen9ou")
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
        "sequential). See the module docstring for why this is safe to raise here.",
    )
    args = parser.parse_args()

    if args.max_concurrent_battles >= 5:
        print(
            f"WARNING: --max-concurrent-battles {args.max_concurrent_battles} is at or "
            "above the real Showdown server's own 5-concurrent-battles-per-IP limit. "
            "poke-env's _ladder has a known off-by-one (searches before checking "
            "capacity) that can push past this ceiling and cause a permanent, silent "
            "stall - see this script's module docstring / CLAUDE.md's Phase 3 status "
            "for the confirmed root cause. Proceeding anyway, but 4 or below is the "
            "known-safe range.",
            flush=True,
        )

    password = os.environ.get("POKE_SHOWDOWN_PASSWORD") or getpass.getpass(
        f"Showdown password for {args.username}: "
    )

    player = load_ppo_player(
        args.model_path,
        battle_format=args.format,
        team=RandomTeamFromPool(),
        account_configuration=AccountConfiguration(args.username, password),
        server_configuration=ShowdownServerConfiguration,
        save_replays=args.save_replays,
        max_concurrent_battles=args.max_concurrent_battles,
    )

    _instrument_progress(player)

    try:
        print(
            f"laddering as {args.username} on {args.format} for {args.n_games} games "
            f"(up to {args.max_concurrent_battles} concurrent)...",
            flush=True,
        )
        await player.ladder(args.n_games)
        print(
            f"\n{player.n_won_battles}/{args.n_games} won "
            f"({player.n_won_battles / args.n_games:.1%}), "
            f"{player.n_lost_battles} lost, {player.n_tied_battles} tied",
            flush=True,
        )
        print(
            f"check the real ladder rating at https://pokemonshowdown.com/users/{args.username}",
            flush=True,
        )
    finally:
        # Same leak this project has fixed elsewhere every time a connected
        # Player was left dangling after its work finished (see
        # inspect_ppo_replays.py, ppo_eval.py's EvalVsOpponentCallback).
        await player.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
