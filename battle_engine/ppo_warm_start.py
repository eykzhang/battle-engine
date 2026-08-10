"""Initializes a fresh MaskablePPO policy from the two Phase-2 models the
roadmap always anticipated using here - `imitation.py` ("a policy warm-start
candidate") for the actor, `win_prob.py` ("a value-function warm-start
candidate") for the critic - instead of the random weights PPO would
otherwise start from.

Why this is possible at all: an independent review (2026-07-31) found that
stable-baselines3's default MultiInputPolicy concatenates the ACTION_SPACE_SIZE-dim
action_mask ahead of the VECTOR_LEN-dim observation (CombinedExtractor, Gymnasium's
spaces.Dict sorts keys alphabetically), so PPO's trunk would otherwise train
on a (VECTOR_LEN + ACTION_SPACE_SIZE)-dim, differently-ordered input that
doesn't match either Phase-2 model's trunk (VECTOR_LEN alone - check the real
current value, not a number hardcoded here, same "don't trust a stale
dimension in a comment" rule win_prob.py's own docstring states explicitly;
VECTOR_LEN has changed more than once already, 316->656->663->665 as of
2026-08-09's hazard-immunity addition). But masking (rl_env.py's
sb3_contrib_action_mask_fn, wired via sb3-contrib's MaskablePPO) now
enforces legality at the action-distribution level, not by having the
network learn it from a mask *input feature* - so the mask no longer needs
to be part of the observation the network actually sees.
`ObservationOnlyExtractor` below drops it, making the network's input
exactly VECTOR_LEN again, and - paired with net_arch=[128, 64] and
activation_fn=ReLU, matching both Phase-2 models exactly (verified against
the real saved checkpoints, not assumed) - makes PPO's trunk and Phase-2's
trunks literally the same shape, so weights can be copied directly rather
than approximated across a mismatch.

Layer-by-layer correspondence (verified against both real checkpoints'
nn.Sequential structure and sb3_contrib.common.maskable.policies.
MaskableActorCriticPolicy._build - not assumed from either library's
docs):
- ImitationModel.net[0] (Linear VECTOR_LEN->128) -> policy.mlp_extractor.policy_net[0]
- ImitationModel.net[3] (Linear 128->64)         -> policy.mlp_extractor.policy_net[2]
- ImitationModel.net[6] (Linear 64->13)          -> policy.action_net
- WinProbModel.net[0]   (Linear VECTOR_LEN->128) -> policy.mlp_extractor.value_net[0]
- WinProbModel.net[3]   (Linear 128->64)         -> policy.mlp_extractor.value_net[2]
- WinProbModel.net[6]   (Linear 64->1)           -> policy.value_net
(MlpExtractor's policy_net/value_net interleave Linear/activation with no
Dropout - SB3 doesn't use dropout in this network - so the Sequential
indices shift: index 2 not 3, skipping the activation but not a dropout
slot that isn't there.)

WinProbModel's raw output is a win-probability logit (pre-sigmoid,
BCEWithLogitsLoss - see that module's docstring), not a discounted-return
value estimate, which is what PPO's critic actually predicts - a
directionally-reasonable starting point (higher win-probability states
should indeed have higher value), not a calibrated one.

An independent review (2026-08-01) measured the actual consequence of not
rescaling it, rather than leaving it as a guess: stable-baselines3's PPO
applies one GLOBAL gradient-norm clip across every policy parameter
together (sb3_contrib.ppo_mask.MaskablePPO.train's
clip_grad_norm_(self.policy.parameters(), max_grad_norm)) - policy_net and
value_net are separate branches with no shared trunk, but they DO share
that one clip. Instrumented on a real training run (RandomPlayer opponent,
n_steps=256, 40 grad updates/rollout), against a matched-architecture
control (same warm_start_policy_kwargs(), random init instead of
warm-started, to isolate this effect from the architecture-vs-init confound
a separate review finding required fixing in scripts/train_ppo.py): the
oversized value-branch gradient (from the uncalibrated logit-vs-return
scale) does throttle the actor's effective gradient via that shared clip -
2.8x smaller on the very first update - but convergence to the
un-throttled control's magnitude happens by about update 4 (~1k timesteps
into a run planned for hundreds of thousands). Conclusion: real, measured,
and self-correcting early enough not to be worth guessing at a rescaling
factor for - left unscaled, as originally reasoned, now with numbers behind
that call instead of just an assumption.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from battle_engine.encoding import VECTOR_LEN
from battle_engine.imitation import ImitationModel
from battle_engine.win_prob import WinProbModel

WARM_START_NET_ARCH = [128, 64]
WARM_START_ACTIVATION_FN = nn.ReLU


class ObservationOnlyExtractor(BaseFeaturesExtractor):
    """Passes through only the "observation" key of the Dict observation
    space MetamonActionSinglesEnv exposes, dropping "action_mask" - see
    module docstring for why the network no longer needs to see the mask as
    an input feature now that masking is enforced at the distribution
    level, and why dropping it is what makes weight transplant from the
    Phase-2 models possible at all (their trunks were trained on VECTOR_LEN
    inputs, not VECTOR_LEN + ACTION_SPACE_SIZE).
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=VECTOR_LEN)

    def forward(self, observations: dict) -> torch.Tensor:
        return observations["observation"]


def warm_start_policy_kwargs() -> dict:
    """policy_kwargs for MaskablePPO that make its network shape-compatible
    with the Phase-2 models (see module docstring) - pass this to
    MaskablePPO(..., policy_kwargs=warm_start_policy_kwargs()), then call
    load_warm_start_weights on the constructed model's .policy before
    training.
    """
    return dict(
        features_extractor_class=ObservationOnlyExtractor,
        net_arch=WARM_START_NET_ARCH,
        activation_fn=WARM_START_ACTIVATION_FN,
    )


def _copy_linear(src: nn.Linear, dst: nn.Linear) -> None:
    if src.weight.shape != dst.weight.shape:
        raise ValueError(
            f"shape mismatch copying Linear weights: {src.weight.shape} -> {dst.weight.shape} "
            "- policy wasn't built with warm_start_policy_kwargs()?"
        )
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        dst.bias.copy_(src.bias)


def load_warm_start_weights(
    policy: MaskableActorCriticPolicy,
    imitation_model: ImitationModel,
    win_prob_model: WinProbModel,
) -> None:
    """Copies imitation_model's trunk+output into policy's actor
    (mlp_extractor.policy_net + action_net) and win_prob_model's trunk+
    output into policy's critic (mlp_extractor.value_net + value_net), in
    place. Raises ValueError (via _copy_linear) rather than silently
    mismatching if policy wasn't constructed with
    warm_start_policy_kwargs() - a shape mismatch here means the transplant
    target isn't what this function assumes, not something to paper over.
    """
    _copy_linear(imitation_model.net[0], policy.mlp_extractor.policy_net[0])
    _copy_linear(imitation_model.net[3], policy.mlp_extractor.policy_net[2])
    _copy_linear(imitation_model.net[6], policy.action_net)

    _copy_linear(win_prob_model.net[0], policy.mlp_extractor.value_net[0])
    _copy_linear(win_prob_model.net[3], policy.mlp_extractor.value_net[2])
    _copy_linear(win_prob_model.net[6], policy.value_net)
