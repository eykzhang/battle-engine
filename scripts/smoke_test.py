"""Phase-0 smoke test: two random bots battle on the local Showdown server.

Start the server first (see README), then:

    .venv/bin/python scripts/smoke_test.py
"""

import asyncio

from poke_env.player import RandomPlayer


async def main() -> None:
    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")
    await p1.battle_against(p2, n_battles=3)
    print(f"finished battles: {p1.n_finished_battles}, p1 wins: {p1.n_won_battles}")
    assert p1.n_finished_battles == 3, "not all battles finished"
    print("smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
