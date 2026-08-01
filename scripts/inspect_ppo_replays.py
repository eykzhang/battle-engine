"""Diagnostic (not a permanent benchmark tool): plays a handful of real
games between a trained PPO checkpoint and the Phase-1 search bot, logging
each turn's chosen order for both sides plus the final outcome/turn count -
the same "watch real replays" technique that found real bugs in Phase 1
(never-switch) and Phase 2 (damage-calc bugs), used here to look for
pathological behavior (stalling loops, degenerate move choices) before
tuning any training config in response to the Phase-3 sanity run's win-rate
plateau.

Usage:
    .venv/bin/python scripts/inspect_ppo_replays.py --n-battles 6
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from battle_engine.ppo_eval import load_ppo_player
from battle_engine.search import TwoPlySearchPlayer
from battle_engine.teams import RandomTeamFromPool

DEFAULT_MODEL_PATH = Path("data/models/ppo.zip")


def _instrument(player, label: str) -> None:
    """Wraps choose_move to print each turn's chosen order plus the
    deciding Pokemon's HP fraction and protect_counter (poke-env's own
    tracked count of consecutive protect-like moves - see
    Pokemon.protect_counter) at decision time - not just the move name, so
    a repeated move name can actually be distinguished as "still working"
    (HP holding, opponent's HP also not moving) vs. "failing/losing value"
    (HP dropping despite the repeated move) before concluding anything is
    actually a pathology.
    """
    original_choose_move = player.choose_move

    def logged_choose_move(battle):
        order = original_choose_move(battle)
        mon = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        my_hp = f"{mon.current_hp_fraction:.2f}" if mon else "?"
        opp_hp = f"{opp.current_hp_fraction:.2f}" if opp else "?"
        protect_ctr = mon.protect_counter if mon else "?"
        print(
            f"  turn {battle.turn:>4} [{label}] {order} "
            f"(my_hp={my_hp} opp_hp={opp_hp} protect_counter={protect_ctr})"
        )
        return order

    player.choose_move = logged_choose_move


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-battles", type=int, default=6)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    ppo_player = load_ppo_player(args.model_path, battle_format="gen9ou", team=RandomTeamFromPool())
    search_player = TwoPlySearchPlayer(battle_format="gen9ou", team=RandomTeamFromPool())
    _instrument(ppo_player, "ppo")
    _instrument(search_player, "search")

    try:
        for i in range(args.n_battles):
            print(f"\n=== battle {i + 1}/{args.n_battles} ===")
            ppo_player.reset_battles()
            search_player.reset_battles()
            await ppo_player.battle_against(search_player, n_battles=1)
            battle = list(ppo_player.battles.values())[-1]
            outcome = "ppo won" if ppo_player.n_won_battles else "ppo lost"
            print(f"--- {outcome}, {battle.turn} turns ---")
    finally:
        # Real bug caught by independent review (2026-08-01): without this,
        # a server disconnect or a choose_move exception mid-loop left both
        # websocket connections open for the rest of the process's life.
        await ppo_player.ps_client.stop_listening()
        await search_player.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
