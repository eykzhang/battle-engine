"""A SinglesEnv subclass exposing battle_engine.dataset's 13-way Metamon
action scheme as the Gymnasium action space, instead of poke-env's native
26-way one - the Phase 3 design decision recorded in action_space.py's
module docstring.

Why not the native 26-way space: PokeEnv.embed_battle is abstract
regardless of which action space is used, so any RL env here plugs in
battle_engine.encoding's observation vector either way - and that vector's
bench slots are species-sorted (a deliberate identity-stability choice, see
encoding.py's module docstring), not keyed by battle.team's raw insertion
order. poke-env's native switch actions (0-5) are raw team-list positions,
so a policy trained against the native space would need to predict a
positional index from features that don't encode position at all - not
just a harder init, a structurally mismatched target for the switch
sub-problem specifically. The 13-way scheme matches the observation
directly (bench slot N in the action label is bench slot N in the input
vector), and lets the already-trained imitation model (imitation.py) be
reused unchanged as a PPO policy initialization once that's wired up.

Both action_to_order and get_action_mask stay thin wrappers around
SinglesEnv's own real (26-way) implementations rather than reimplementing
switch/move/tera legality from scratch: action_to_order translates one
label via action_space.metamon_action_to_poke_env_action and delegates;
get_action_mask asks SinglesEnv for the real 26-length legality mask and
reads off the 13 positions this scheme's actions land on, so all the actual
legality logic (trapped, revealed-move-count, can_tera, ...) stays owned by
poke-env, not duplicated and left to drift out of sync with it.

order_to_action is also overridden (added after an independent review
caught its absence, 2026-07-31): poke-env's SingleAgentWrapper calls
env.order_to_action on the *opponent's* real move to feed it back through
this env's own step() plumbing (env.py's PokeEnv.step converts every
action via self.action_to_order, including the opponent's, for the
single-agent-vs-scripted-opponent training setup Phase 3 actually uses).
Without an override here, that call silently resolves to SinglesEnv's
native 26-way order_to_action, and the resulting index then gets
reinterpreted as a 13-way metamon label one step later - corrupting
whatever the opponent actually did (verified: a plain move order at slot 1
comes back as action 7, which action_space.py's translator reads as "switch
to bench slot 3").

Sentinels (-1 forfeit, -2 default order) are handled directly here rather
than via action_space.py (see that module's docstring) - passed straight
through to/from SinglesEnv's real implementation untranslated, matching how
poke-env itself special-cases them before its own 26-way logic runs.

embed_battle/calc_reward/observation_spaces (the remaining abstract methods
PokeEnv requires - action_to_order/order_to_action/get_action_mask/
get_action_space_size above cover the rest) reuse this project's existing
building blocks rather than inventing new ones: embed_battle is the same
battle_view_from_poke_env + encode pipeline win_prob.py and imitation.py
already train on, so a policy here sees exactly the state representation
this project has already validated; calc_reward is poke-env's own built-in
reward_computing_helper (fainted/HP/status/victory, weighted) rather than a
hand-rolled reward function - "working plumbing before hand-rolling
anything" (the roadmap's own words for this phase) applies to reward
shaping too, not just the SB3 wiring. The specific weights below are a
reasonable starting point, not a tuned choice - revisit once there's an
actual training run to look at.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from battle_engine.action_space import (
    metamon_action_to_poke_env_action,
    poke_env_action_to_metamon_action,
)
from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.encoding import VECTOR_LEN, battle_view_from_poke_env, encode

_SENTINEL_ACTIONS = (-1, -2)  # forfeit, default - see module docstring

# calc_reward weights, passed to PokeEnv.reward_computing_helper. Rebalanced
# (2026-08-09) after a real training run (see CLAUDE.md's Phase 3 status)
# confirmed the plateau this was flagged as a possible contributor to -
# derived from reward_computing_helper's actual source (poke_env/
# environment/env.py), not guessed: each side's 6 Pokemon contribute up to
# +-value to the state value (hp_value while healthy, fainted_value while
# fainted are mutually exclusive per mon - a mon is never both), so the
# worst-case shaping-only swing across a full battle is
# 6*value (my team) + 6*value (opponent team) = 12*value per side. At the
# old 0.15, that's 12*0.15 = 1.8 against the terminal +-1.0 victory_value -
# confirms the independent review's measured +-1.8 exactly, and means a
# decisive win/loss's shaping component alone could exceed the entire
# terminal win/loss signal. 0.05 brings the same worst-case swing to
# 12*0.05 = 0.6 - clearly subordinate to the +-1.0 terminal term (a decisive
# win now totals at most 1.6, and the terminal signal alone still ranges
# +-1.0 regardless of how the battle was played) - while keeping a real,
# nonzero per-turn dense signal rather than zeroing shaping out entirely.
_FAINTED_VALUE = 0.05
_HP_VALUE = 0.05
_VICTORY_VALUE = 1.0


class MetamonActionSinglesEnv(SinglesEnv):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.action_spaces = {
            agent: Discrete(ACTION_SPACE_SIZE) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(VECTOR_LEN,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return encode(battle_view_from_poke_env(battle))

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=_FAINTED_VALUE,
            hp_value=_HP_VALUE,
            victory_value=_VICTORY_VALUE,
        )

    @staticmethod
    def get_action_space_size(gen: int) -> int:
        return ACTION_SPACE_SIZE

    @staticmethod
    def action_to_order(
        action: np.int64, battle: AbstractBattle, fake: bool = False, strict: bool = True
    ) -> BattleOrder:
        action = int(action)
        if action in _SENTINEL_ACTIONS:
            poke_env_action = action
        else:
            try:
                poke_env_action = metamon_action_to_poke_env_action(action, battle)
            except ValueError as e:
                # Mirrors SinglesEnv.action_to_order's own strict/non-strict
                # contract for an illegal action - an independent review
                # found this translation layer's ValueErrors (no
                # active_pokemon, a switch label past the real bench size)
                # previously bypassed that contract and always raised
                # regardless of strict, unlike every other illegal-action
                # case SinglesEnv.action_to_order itself already handles.
                # Matters for training specifically: with strict=False (as
                # scripts/train_ppo.py uses), an untrained policy sampling
                # an out-of-range action should fall back to a random legal
                # move like any other illegal pick, not crash the run.
                if strict:
                    raise e
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                return Player.choose_random_singles_move(battle)
        # SinglesEnv.action_to_order calls .item() on the action (it expects
        # a numpy scalar, e.g. what Gymnasium hands back from a Discrete
        # space) - a plain Python int from our translation breaks that.
        return SinglesEnv.action_to_order(
            np.int64(poke_env_action), battle, fake=fake, strict=strict
        )

    @staticmethod
    def order_to_action(
        order: Any, battle: AbstractBattle, fake: bool = False, strict: bool = True
    ) -> np.int64:
        poke_env_action = int(SinglesEnv.order_to_action(order, battle, fake=fake, strict=strict))
        if poke_env_action in _SENTINEL_ACTIONS:
            return np.int64(poke_env_action)
        try:
            return np.int64(poke_env_action_to_metamon_action(poke_env_action, battle))
        except ValueError as e:
            # Same strict/non-strict contract as action_to_order above,
            # mirroring SinglesEnv.order_to_action's own fallback shape
            # (recurse on a freshly chosen random legal order rather than
            # trying to translate the untranslatable one).
            if strict:
                raise e
            if battle.logger is not None:
                battle.logger.warning(str(e) + " Defaulting to random move.")
            return MetamonActionSinglesEnv.order_to_action(
                Player.choose_random_singles_move(battle), battle, fake=fake, strict=strict
            )

    @staticmethod
    def get_action_mask(battle: AbstractBattle) -> List[int]:
        """13-length 0/1 legality mask, in this scheme's action order.

        battle._wait (a "no real decision this turn, but the PettingZoo
        step API still requires *an* action" placeholder state - see
        poke_env.player.player.Player._handle_battle_request, which never
        actually submits the order in that state) is handled explicitly
        rather than inferred from "every position came back illegal": an
        independent review found that inference was actually unsound - a
        genuinely all-illegal native mask can occur outside of _wait too
        (e.g. team position 0 just isn't the active Pokemon or a legal
        switch target right now), so collapsing both cases to the same
        fallback would silently mislabel a real all-illegal state as the
        harmless _wait placeholder.
        """
        if battle._wait:
            mask = [0] * ACTION_SPACE_SIZE
            mask[0] = 1
            return mask
        native_mask = SinglesEnv.get_action_mask(battle)
        mask = []
        for action in range(ACTION_SPACE_SIZE):
            try:
                poke_env_action = metamon_action_to_poke_env_action(action, battle)
            except ValueError:
                mask.append(0)
                continue
            mask.append(native_mask[poke_env_action])
        return mask


def sb3_contrib_action_mask_fn(wrapped_env: Any) -> np.ndarray:
    """The action_mask_fn sb3_contrib.common.wrappers.ActionMasker needs,
    for a MetamonActionSinglesEnv wrapped in poke-env's SingleAgentWrapper
    for single-agent PPO training (scripts/train_ppo.py). ActionMasker
    calls this with the SingleAgentWrapper instance it wraps (see
    ActionMasker.action_masks -> self._action_mask_fn(self.env), where
    self.env is whatever was passed to ActionMasker's constructor).
    wrapped_env.env is the underlying MetamonActionSinglesEnv
    (SingleAgentWrapper stores the raw multi-agent PokeEnv as .env - see
    single_agent_wrapper.py), and .battle1 is specifically the RL agent's
    own battle (agent1 - the one being asked to sample from a Discrete
    action space; battle2, the scripted opponent's, is never masked since
    it never samples from one).

    Exists mainly so this has the same unit-test coverage as everything
    else in this module - an independent review found that without real
    masking, letting illegal picks fall back to a randomly-substituted
    legal move is a real training-quality problem (56.2% of actions are
    illegal under a fresh near-uniform policy on real gen9ou states, and
    each gets a *different* move actually played, so PPO ends up crediting
    the wrong action for the outcome) - masking is what actually prevents
    the policy from sampling an illegal action in the first place.
    """
    return np.array(MetamonActionSinglesEnv.get_action_mask(wrapped_env.env.battle1), dtype=bool)
