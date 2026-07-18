# battle-engine

An ML/search battle engine for competitive Pokémon — "Stockfish for Pokémon."

The engine plays (and analyzes) Pokémon Showdown battles using game-tree search and
machine learning, built in deliberate stages that mirror how chess engines actually
evolved: hand-crafted evaluation + search first, then a learned evaluation trained on
millions of human replays (the "NNUE moment"), then reinforcement learning via
self-play. It is the intelligence layer behind
[BattleBrain](../battle-brain), an iOS companion app that surfaces the engine's
per-turn win-probability analysis for replay review.

Built as a learning project: every stage has a version that trains on a laptop
(M4 MacBook Air), with objective strength gates (head-to-head win rates against the
previous stage's bot) instead of vibes.

## Roadmap

| Phase | What | Gate |
|---|---|---|
| 0 | Harness: local Showdown server, poke-env, baseline bots, benchmark script | max-damage vs random measured |
| 1 | Classical search + hand-crafted eval (no ML) | >70% vs max-damage over 500+ battles |
| 2 | Supervised learning: win-probability + imitation models on human replays | beats Phase-1 bot head-to-head |
| 3 | Reinforcement learning: PPO self-play from the imitation policy | beats Phase-2 bot head-to-head |
| 4+ | Stretch: C++ search core (MCTS + embedded NNUE-style inference), gen9OU | ladder GXE |

Format: `gen9randombattle` (the standard bot-development ladder), gen9OU as stretch.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Local Showdown server (cloned into the repo, gitignored)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install
```

## Run

```bash
# Terminal 1: start the local simulator (first run builds; takes a minute)
cd pokemon-showdown && node pokemon-showdown start --no-security

# Terminal 2: smoke test — two random bots play three battles
.venv/bin/python scripts/smoke_test.py

# Tests
.venv/bin/pytest
```

## Prior art this builds on

- [poke-env](https://github.com/hsahovic/poke-env) — Python/Gymnasium interface to Showdown
- [Foul Play](https://pmariglia.github.io/posts/foul-play/) — the strongest classical bot (Rust forward model + MCTS/DUCT)
- [Metamon](https://github.com/UT-Austin-RPL/metamon) — offline RL baselines + 3.5M+ parsed human replay trajectories
