"""Self-play: trains PPO against frozen snapshots of its own past policy
instead of only a fixed RandomPlayer - the roadmap's actual Phase 3 target
("self-play + play vs frozen past versions"), buildable now because
ppo_warm_start.py already made PPO's policy architecture-identical to
ImitationModel/WinProbModel's trunks (net_arch=[128, 64], ReLU,
observation-only features - see that module's docstring), so a frozen
opponent only needs one MaskableActorCriticPolicy built the same way, with
load_state_dict() swapped between snapshots - no separate opponent
architecture to maintain.

FrozenPolicyPlayer (a poke-env Player) samples a new frozen snapshot once
per battle, not per turn - checked via battle.battle_tag changing between
calls, since a policy that switched personality mid-battle wouldn't be a
coherent opponent. SelfPlaySnapshotCallback (an SB3 callback) periodically
pushes a copy of the live training policy's weights into the opponent's
snapshot pool during model.learn(). "Periodically", not "continuously
mirroring the trainee", is the whole point: an opponent that's always
exactly as strong as the trainee gives a weak learning signal (both sides
converging to ~50% win rate against each other says little about whether
either side is actually improving in absolute terms), whereas a small pool
of past snapshots creates a real difficulty ramp as training progresses and
avoids the trainee overfitting to beat one single frozen point (classic
self-play "strategy cycling", e.g. learning to beat pure-random-2-updates-ago
in a way that loses to what it beat 10 updates ago).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces
from poke_env.battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback

from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.encoding import VECTOR_LEN, battle_view_from_poke_env, encode
from battle_engine.ppo_warm_start import warm_start_policy_kwargs
from battle_engine.rl_env import MetamonActionSinglesEnv


def observation_space() -> spaces.Dict:
    """The same Dict shape MetamonActionSinglesEnv.observation_spaces
    builds per agent - duplicated here (not imported) because it's needed
    to construct a bare policy with no live env/connection at all, which
    MetamonActionSinglesEnv can't provide standalone (see
    build_inference_policy).
    """
    return spaces.Dict(
        {
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(VECTOR_LEN,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int64),
        }
    )


def action_space() -> spaces.Discrete:
    return spaces.Discrete(ACTION_SPACE_SIZE)


def build_inference_policy() -> MaskableActorCriticPolicy:
    """A bare MaskableActorCriticPolicy for inference only, never trained
    itself - built with warm_start_policy_kwargs() so its architecture
    matches the trainee's exactly and any of the trainee's saved
    state_dicts load into it directly via load_state_dict. The lr_schedule
    is a dummy constant since this instance is never optimized, only used
    for load_state_dict + predict.
    """
    return MaskableActorCriticPolicy(
        observation_space(),
        action_space(),
        lr_schedule=lambda _progress_remaining: 0.0,
        **warm_start_policy_kwargs(),
    )


class FrozenPolicyPlayer(Player):
    """A poke-env Player whose moves come from a frozen snapshot of a
    MaskableActorCriticPolicy, resampled from a small pool once per battle
    (not per turn - see module docstring).

    Needs at least one snapshot before it can play - seed one via
    add_snapshot(trainee.policy.state_dict()) before training starts (the
    trainee's own initial, possibly warm-started, weights are a perfectly
    reasonable first opponent - not a degenerate/empty one), then let
    SelfPlaySnapshotCallback add more as training progresses.

    Also doubles as the benchmark-facing player for a trained model (see
    battle_engine.ppo_eval.load_ppo_player): built with max_snapshots=1 and
    a single seeded snapshot, it's just "always play with this one policy" -
    no separate class needed for that role.
    """

    def __init__(
        self, *args, max_snapshots: int = 5, seed: int = 0, deterministic: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._policy = build_inference_policy()
        self._snapshots: Deque[Dict[str, torch.Tensor]] = deque(maxlen=max_snapshots)
        self._rng = random.Random(seed)
        self._active_battle_tag: Optional[str] = None
        # False during training (self-play wants a variety of responses
        # across the pool, same as PPO itself samples stochastically during
        # rollout collection); True is the more standard choice for
        # evaluating a single trained model's actual best-guess policy
        # (load_ppo_player defaults to True for exactly that reason).
        self._deterministic = deterministic

    def add_snapshot(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # .detach().clone().cpu() - the live training policy's tensors keep
        # getting mutated in place by the optimizer; a snapshot has to be a
        # real independent copy, not a view that would silently change
        # after being stored.
        self._snapshots.append({k: v.detach().clone().cpu() for k, v in state_dict.items()})

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.battle_tag != self._active_battle_tag:
            if not self._snapshots:
                raise RuntimeError(
                    "FrozenPolicyPlayer has no snapshots yet - seed one with "
                    "add_snapshot(...) before starting a battle"
                )
            self._policy.load_state_dict(self._rng.choice(self._snapshots))
            self._active_battle_tag = battle.battle_tag

        observation_vector = encode(battle_view_from_poke_env(battle))
        mask = np.array(MetamonActionSinglesEnv.get_action_mask(battle), dtype=bool)
        action, _ = self._policy.predict(
            {
                "observation": observation_vector[None, :],
                "action_mask": mask[None, :].astype(np.int64),
            },
            action_masks=mask[None, :],
            deterministic=self._deterministic,
        )
        # strict=False as defense in depth (consistent with the trainee's
        # own env) - the mask should already guarantee legality here.
        return MetamonActionSinglesEnv.action_to_order(
            int(action[0]), battle, fake=False, strict=False
        )


class MixedOpponentPlayer(Player):
    """Delegates each battle to one of several sub-players, sampled by
    weight and resampled once per battle (the same battle_tag-keyed
    granularity FrozenPolicyPlayer already uses for snapshot resampling) so
    a single battle is always played by one coherent opponent, never
    something that swaps identity mid-game.

    Built for the sanity-run plateau diagnosis (see CLAUDE.md's Phase 3
    status): pure self-play only ever tests the trainee against itself, so
    "beats recent snapshots" doesn't have to transfer to "beats
    TwoPlySearchPlayer" (Phase 3's actual gate opponent) - the trainee could
    converge on strategies that only work against its own policy family.
    Mixing a fraction of training battles against a real TwoPlySearchPlayer
    instance forces the trainee to also solve *that* matchup directly during
    training, not just at periodic eval checkpoints (--eval already does
    real games against it, but only ever *measures*, never *trains* against
    it).

    Sub-players are used exactly like FrozenPolicyPlayer already is in
    build_env - as pure choose_move/teampreview decision functions over a
    battle object neither actually owns a live connection for, never
    battle_against'd directly - so each should be built the same way
    (start_listening=False, no team=).
    """

    def __init__(self, *args, players: List[Tuple[float, Player]], seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if not players:
            raise ValueError("MixedOpponentPlayer needs at least one (weight, player) pair")
        self._players = [p for _, p in players]
        self._weights = [w for w, _ in players]
        self._rng = random.Random(seed)
        self._active_battle_tag: Optional[str] = None
        self._active_player: Optional[Player] = None

    def _select(self, battle_tag: str) -> Player:
        if battle_tag != self._active_battle_tag:
            self._active_player = self._rng.choices(self._players, weights=self._weights, k=1)[0]
            self._active_battle_tag = battle_tag
        return self._active_player

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self._select(battle.battle_tag).choose_move(battle)


class SelfPlaySnapshotCallback(BaseCallback):
    """Pushes a copy of the training policy's current weights into
    opponent's snapshot pool every snapshot_interval timesteps, via
    model.learn(callback=...).
    """

    def __init__(self, opponent: FrozenPolicyPlayer, snapshot_interval: int, verbose: int = 0):
        super().__init__(verbose)
        self.opponent = opponent
        self.snapshot_interval = snapshot_interval

    def _on_step(self) -> bool:
        if self.num_timesteps % self.snapshot_interval == 0:
            self.opponent.add_snapshot(self.model.policy.state_dict())
        return True
