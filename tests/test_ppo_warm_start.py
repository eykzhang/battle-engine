import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.encoding import VECTOR_LEN
from battle_engine.imitation import ImitationModel
from battle_engine.ppo_warm_start import load_warm_start_weights, warm_start_policy_kwargs
from battle_engine.win_prob import WinProbModel


class _StubEnv(gym.Env):
    """The minimal Gym env shape MaskablePPO's constructor needs (matching
    MetamonActionSinglesEnv's real observation/action spaces) - no real
    battle/server needed, since this module's tests only exercise policy
    network construction and weight transplant, never a rollout.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(VECTOR_LEN,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int64),
            }
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}

    def action_masks(self):
        return np.ones(ACTION_SPACE_SIZE, dtype=bool)


def _build_warm_started_model() -> MaskablePPO:
    model = MaskablePPO(
        "MultiInputPolicy", _StubEnv(), policy_kwargs=warm_start_policy_kwargs()
    )
    imitation_model = ImitationModel.load("data/models/imitation.pt")
    win_prob_model = WinProbModel.load("data/models/win_prob.pt")
    load_warm_start_weights(model.policy, imitation_model, win_prob_model)
    return model, imitation_model, win_prob_model


def test_warm_started_policy_trunk_matches_imitation_models_weights_exactly():
    model, imitation_model, _ = _build_warm_started_model()

    torch.testing.assert_close(
        model.policy.mlp_extractor.policy_net[0].weight, imitation_model.net[0].weight
    )
    torch.testing.assert_close(
        model.policy.mlp_extractor.policy_net[2].weight, imitation_model.net[3].weight
    )
    torch.testing.assert_close(model.policy.action_net.weight, imitation_model.net[6].weight)


def test_warm_started_policy_reproduces_imitation_models_actual_predictions():
    """A stronger check than comparing raw weights: feeds the same real
    input through both the transplanted policy's actor pathway and
    ImitationModel directly, and confirms they produce the SAME logits -
    this only holds if the features extractor (dropping action_mask),
    net_arch, and layer indexing all actually line up end to end, not just
    that individual Linear layers were copied to the intended-looking slot.
    """
    model, imitation_model, _ = _build_warm_started_model()
    model.policy.set_training_mode(False)  # dropout off, both models compared in eval mode
    imitation_model.eval()

    rng = np.random.default_rng(0)
    observation_vector = rng.standard_normal(VECTOR_LEN).astype(np.float32)
    obs = {
        "observation": torch.as_tensor(observation_vector).unsqueeze(0),
        "action_mask": torch.ones(1, ACTION_SPACE_SIZE, dtype=torch.int64),
    }

    with torch.no_grad():
        features = model.policy.extract_features(obs, model.policy.pi_features_extractor)
        latent_pi = model.policy.mlp_extractor.forward_actor(features)
        policy_logits = model.policy.action_net(latent_pi)

        imitation_logits = imitation_model(torch.as_tensor(observation_vector).unsqueeze(0))

    torch.testing.assert_close(policy_logits, imitation_logits)


def test_warm_started_policy_reproduces_win_prob_models_actual_predictions():
    model, _, win_prob_model = _build_warm_started_model()
    model.policy.set_training_mode(False)
    win_prob_model.eval()

    rng = np.random.default_rng(1)
    observation_vector = rng.standard_normal(VECTOR_LEN).astype(np.float32)
    obs = {
        "observation": torch.as_tensor(observation_vector).unsqueeze(0),
        "action_mask": torch.ones(1, ACTION_SPACE_SIZE, dtype=torch.int64),
    }

    with torch.no_grad():
        features = model.policy.extract_features(obs, model.policy.vf_features_extractor)
        latent_vf = model.policy.mlp_extractor.forward_critic(features)
        value = model.policy.value_net(latent_vf)

        win_prob_logit = win_prob_model(torch.as_tensor(observation_vector).unsqueeze(0))

    torch.testing.assert_close(value.squeeze(-1), win_prob_logit)


def test_load_warm_start_weights_raises_on_shape_mismatch_without_warm_start_kwargs():
    # A MaskablePPO built WITHOUT warm_start_policy_kwargs() keeps SB3's
    # default action_mask-concatenated, Tanh-activated architecture - the
    # transplant should refuse to silently copy into the wrong-shaped
    # layers rather than producing a policy that looks warm-started but
    # isn't.
    model = MaskablePPO("MultiInputPolicy", _StubEnv())
    imitation_model = ImitationModel.load("data/models/imitation.pt")
    win_prob_model = WinProbModel.load("data/models/win_prob.pt")

    import pytest

    with pytest.raises(ValueError):
        load_warm_start_weights(model.policy, imitation_model, win_prob_model)
