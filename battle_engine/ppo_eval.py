"""Evaluating a PPO policy outside the training loop's SingleAgentWrapper
context - two distinct needs, both built the same session a real training
run was first being planned, since neither existed yet:

1. load_ppo_player: wraps a saved MaskablePPO checkpoint (scripts/
   train_ppo.py --save) as a real, benchmark-participating poke-env Player
   (via self_play.FrozenPolicyPlayer with a single seeded snapshot), so
   scripts/benchmark.py can actually measure a trained model head-to-head -
   the roadmap's Phase 3 gate ("RL-tuned policy beats the Phase-2
   supervised bot head-to-head") has no meaning without this; a policy that
   can only ever act as a SingleAgentWrapper opponent isn't benchmarkable.

2. EvalVsOpponentCallback: a real, interpretable progress signal *during*
   training. PPO's own logged metrics (loss, entropy, explained_variance,
   ep_rew_mean under whatever reward-shaping weights are configured) don't
   directly answer "is this actually getting better at winning" - this
   callback periodically freezes the live policy, plays it for a small
   number of real battles against a fixed reference opponent, and reports
   a win rate. Deliberately real games via the actual benchmark harness
   (battle_engine.benchmark.run_benchmark), not a proxy metric - the same
   harness every phase gate in this project is measured by.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from poke_env.player import Player
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from battle_engine.benchmark import BenchmarkResult, run_benchmark
from battle_engine.self_play import FrozenPolicyPlayer
from battle_engine.teams import RandomTeamFromPool


def load_ppo_player(
    model_path: Path,
    battle_format: str = "gen9ou",
    deterministic: bool = True,
    **player_kwargs,
) -> FrozenPolicyPlayer:
    """Loads a saved MaskablePPO checkpoint (env=None - only prediction is
    needed, not further training, so no live env has to be reconstructed;
    verified against stable_baselines3.common.base_class.BaseAlgorithm.load's
    own docs: "env ... can be None if you only need prediction from a
    trained model") into a FrozenPolicyPlayer with exactly one snapshot -
    "always play with this one trained policy", not a self-play pool.
    deterministic=True by default (unlike FrozenPolicyPlayer's own
    training-time default of False): evaluating a specific trained model's
    actual best-guess behavior is the more standard choice than a fresh
    stochastic sample per call, which is what training-time self-play wants
    instead for response variety.
    """
    saved = MaskablePPO.load(model_path, env=None)
    player = FrozenPolicyPlayer(
        battle_format=battle_format,
        max_snapshots=1,
        deterministic=deterministic,
        **player_kwargs,
    )
    player.add_snapshot(saved.policy.state_dict())
    return player


class EvalVsOpponentCallback(BaseCallback):
    """Every eval_interval timesteps (via model.learn(callback=...)),
    freezes the live training policy into a temporary deterministic
    FrozenPolicyPlayer and plays it for n_battles real games against a
    fixed, already-connected reference opponent (built once by the caller,
    reused across every eval - not reconstructed each time, unlike the
    temporary evaluator), reporting a win rate via the real benchmark
    harness. Appends each (timestep, win_rate) point to self.history so a
    caller can inspect the trend after training.

    The temporary evaluator's own connection is explicitly stopped after
    each eval batch (ps_client.stop_listening()) - an independent review
    found exactly this "build a real, connected Player and never close it"
    pattern leaking connections elsewhere in this project; a callback that
    fires every eval_interval steps for hours would leak one connection per
    fire if it repeated that mistake.
    """

    def __init__(
        self,
        opponent: Player,
        eval_interval: int,
        n_battles: int = 20,
        battle_format: str = "gen9ou",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.opponent = opponent
        self.eval_interval = eval_interval
        self.n_battles = n_battles
        self.battle_format = battle_format
        self.history: list[tuple[int, float]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_interval == 0:
            result = self._run_eval()
            self.history.append((self.num_timesteps, result.p1_win_rate))
            if self.verbose:
                print(f"[eval @ {self.num_timesteps} steps] {result}")
        return True

    def _run_eval(self) -> BenchmarkResult:
        evaluator = FrozenPolicyPlayer(
            battle_format=self.battle_format,
            team=RandomTeamFromPool(),
            max_snapshots=1,
            deterministic=True,
        )
        evaluator.add_snapshot(self.model.policy.state_dict())
        return asyncio.run(self._run_eval_battles(evaluator))

    async def _run_eval_battles(self, evaluator: FrozenPolicyPlayer) -> BenchmarkResult:
        try:
            return await run_benchmark(evaluator, self.opponent, n_battles=self.n_battles)
        finally:
            await evaluator.ps_client.stop_listening()
