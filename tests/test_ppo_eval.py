import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from sb3_contrib import MaskablePPO

from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.ppo_eval import EvalVsOpponentCallback, load_ppo_player
from battle_engine.ppo_warm_start import warm_start_policy_kwargs
from battle_engine.self_play import action_space, observation_space


def _server_running(host: str = "localhost", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


class _StubEnv(gym.Env):
    """The minimal shape MaskablePPO.save/load needs - only used to build a
    real checkpoint file for load_ppo_player to load back, never trained.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = observation_space()
        self.action_space = action_space()

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}

    def action_masks(self):
        return [1] * ACTION_SPACE_SIZE


def test_load_ppo_player_seeds_a_single_snapshot_matching_the_saved_checkpoint(tmp_path):
    model = MaskablePPO("MultiInputPolicy", _StubEnv(), policy_kwargs=warm_start_policy_kwargs())
    checkpoint_path = tmp_path / "ppo.zip"
    model.save(checkpoint_path)

    player = load_ppo_player(checkpoint_path, battle_format="gen9ou")

    assert len(player._snapshots) == 1
    saved_state_dict = model.policy.state_dict()
    (loaded_snapshot,) = player._snapshots
    for key in saved_state_dict:
        torch.testing.assert_close(saved_state_dict[key], loaded_snapshot[key])


def test_load_ppo_player_defaults_to_deterministic():
    model = MaskablePPO("MultiInputPolicy", _StubEnv(), policy_kwargs=warm_start_policy_kwargs())

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "ppo.zip"
        model.save(checkpoint_path)
        player = load_ppo_player(checkpoint_path)

    assert player._deterministic is True


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_eval_callback_records_a_win_rate_and_closes_its_temporary_connection():
    """Real games against a cheap (RandomPlayer, not the slower search bot)
    opponent, to keep this test fast while still exercising the real
    benchmark harness end to end - the callback's whole point is that it's
    real games, not a mocked proxy.
    """
    from poke_env.player import RandomPlayer

    from battle_engine.teams import RandomTeamFromPool

    opponent = RandomPlayer(battle_format="gen9ou", team=RandomTeamFromPool())
    callback = EvalVsOpponentCallback(
        opponent, eval_interval=10, n_battles=4, battle_format="gen9ou"
    )
    callback.model = SimpleNamespace(
        policy=MaskablePPO(
            "MultiInputPolicy", _StubEnv(), policy_kwargs=warm_start_policy_kwargs()
        ).policy
    )

    callback.num_timesteps = 5
    callback._on_step()
    assert callback.history == []  # not yet at the interval

    callback.num_timesteps = 10
    callback._on_step()
    assert len(callback.history) == 1
    timestep, win_rate = callback.history[0]
    assert timestep == 10
    assert 0.0 <= win_rate <= 1.0
