"""Tests for scripts/export_weights.py (Phase 4 M2).

Compares the exported binary against the real data/models/ppo.zip
checkpoint's actual tensors, the same "check against the real artifact, not
a mock" standard tests/test_ppo_warm_start.py already applies to the
Phase-2 weight transplant.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.encoding import VECTOR_LEN
from battle_engine.ppo_warm_start import ObservationOnlyExtractor, WARM_START_ACTIVATION_FN

# scripts/ isn't an installed package (unlike battle_engine, which is
# editable-installed - see pyproject.toml's [tool.setuptools.packages.find]
# comment) and no other test imports from it yet, so it isn't on sys.path
# under plain `.venv/bin/pytest`. Add the repo root once, locally to this
# test module, rather than editing the shared tests/conftest.py for one
# script's sake.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.export_weights import (  # noqa: E402
    WEIGHTS_FORMAT_VERSION,
    WEIGHTS_MAGIC,
    export_weights,
    load_policy,
)

_REAL_CHECKPOINT = Path("data/models/ppo.zip")


class _StubEnv(gym.Env):
    """Minimal Gym env shape MaskablePPO's constructor needs - matches
    tests/test_ppo_warm_start.py's _StubEnv, no real battle/server needed
    since these tests only exercise policy construction and export, never
    a rollout.
    """

    def __init__(self, action_space_size: int = ACTION_SPACE_SIZE):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(VECTOR_LEN,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(action_space_size,), dtype=np.int64),
            }
        )
        self.action_space = spaces.Discrete(action_space_size)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}

    def action_masks(self):
        return np.ones(self.action_space.n, dtype=bool)


def _parse_ppo_bin(data: bytes) -> tuple[int, int, list[tuple[int, int, np.ndarray, np.ndarray]]]:
    """Parses the binary format export_weights.py's module docstring
    specifies, independently of any of export_weights.py's own internals -
    the point of a round-trip test is to check the actual byte contract,
    not to re-invoke the code under test to check itself.
    """
    magic, version, vector_len = struct.unpack_from("<4sII", data, 0)
    assert magic == WEIGHTS_MAGIC
    offset = struct.calcsize("<4sII")
    layers = []
    for _ in range(6):
        out_dim, in_dim = struct.unpack_from("<II", data, offset)
        offset += struct.calcsize("<II")
        weight = np.frombuffer(data, dtype="<f4", count=out_dim * in_dim, offset=offset).reshape(out_dim, in_dim)
        offset += out_dim * in_dim * 4
        bias = np.frombuffer(data, dtype="<f4", count=out_dim, offset=offset)
        offset += out_dim * 4
        layers.append((out_dim, in_dim, weight.copy(), bias.copy()))
    assert offset == len(data), "trailing bytes after the last layer - format contract violated"
    return version, vector_len, layers


@pytest.mark.skipif(not _REAL_CHECKPOINT.exists(), reason="data/models/ppo.zip not present in this checkout")
def test_export_weights_writes_binary_against_real_checkpoint(tmp_path):
    output_path = tmp_path / "ppo.bin"

    export_weights(_REAL_CHECKPOINT, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(not _REAL_CHECKPOINT.exists(), reason="data/models/ppo.zip not present in this checkout")
def test_export_weights_header_matches_format_contract(tmp_path):
    output_path = tmp_path / "ppo.bin"
    export_weights(_REAL_CHECKPOINT, output_path)

    version, vector_len, layers = _parse_ppo_bin(output_path.read_bytes())

    assert version == WEIGHTS_FORMAT_VERSION
    assert vector_len == VECTOR_LEN
    assert len(layers) == 6


@pytest.mark.skipif(not _REAL_CHECKPOINT.exists(), reason="data/models/ppo.zip not present in this checkout")
def test_export_weights_exactly_matches_real_checkpoint_tensors(tmp_path):
    """The DW-2.2 exact-equality check: the dumped weight/bias arrays for
    every layer must be byte-for-byte identical (np.array_equal, not
    np.allclose) to the real checkpoint's live tensors, in the plan's
    pinned actor-then-critic order.
    """
    output_path = tmp_path / "ppo.bin"
    export_weights(_REAL_CHECKPOINT, output_path)
    _, _, dumped_layers = _parse_ppo_bin(output_path.read_bytes())

    policy = load_policy(_REAL_CHECKPOINT)
    real_layers = (
        policy.mlp_extractor.policy_net[0],
        policy.mlp_extractor.policy_net[2],
        policy.action_net,
        policy.mlp_extractor.value_net[0],
        policy.mlp_extractor.value_net[2],
        policy.value_net,
    )

    assert len(dumped_layers) == len(real_layers)
    for (out_dim, in_dim, dumped_weight, dumped_bias), real_layer in zip(dumped_layers, real_layers):
        real_weight = real_layer.weight.detach().cpu().numpy().astype("<f4")
        real_bias = real_layer.bias.detach().cpu().numpy().astype("<f4")
        assert (out_dim, in_dim) == real_weight.shape
        assert np.array_equal(dumped_weight, real_weight)
        assert np.array_equal(dumped_bias, real_bias)


def test_export_weights_raises_on_shape_mismatch_before_writing_file(tmp_path):
    """DW-2.3: a checkpoint built with a different net_arch (so its Linear
    layers don't match _LAYER_SPECS's expected shapes) must raise
    ValueError, and must never leave a partial ppo.bin on disk.
    """
    mismatched_checkpoint = tmp_path / "mismatched_ppo.zip"
    output_path = tmp_path / "ppo.bin"

    model = MaskablePPO(
        "MultiInputPolicy",
        _StubEnv(),
        policy_kwargs=dict(
            features_extractor_class=ObservationOnlyExtractor,
            net_arch=[7, 5],  # deliberately NOT WARM_START_NET_ARCH
            activation_fn=WARM_START_ACTIVATION_FN,
        ),
    )
    model.save(mismatched_checkpoint)

    with pytest.raises(ValueError, match="expected weight shape"):
        export_weights(mismatched_checkpoint, output_path)

    assert not output_path.exists()


def test_export_weights_raises_clear_error_on_missing_checkpoint(tmp_path):
    missing_checkpoint = tmp_path / "does_not_exist.zip"
    output_path = tmp_path / "ppo.bin"

    with pytest.raises(FileNotFoundError, match="PPO checkpoint not found"):
        export_weights(missing_checkpoint, output_path)

    assert not output_path.exists()


def test_export_weights_creates_missing_output_directory(tmp_path):
    """data/cpp_weights/ doesn't exist on a fresh clone (it's gitignored) -
    export_weights must create its parent directory rather than requiring
    the caller to mkdir first.
    """
    if not _REAL_CHECKPOINT.exists():
        pytest.skip("data/models/ppo.zip not present in this checkout")
    output_path = tmp_path / "nested" / "dir" / "ppo.bin"

    export_weights(_REAL_CHECKPOINT, output_path)

    assert output_path.exists()
