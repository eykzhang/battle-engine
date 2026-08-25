"""Phase 4 M3: parity between the hand-written C++ forward pass
(cpp/include/be/mlp.hpp, exposed as _native.PolicyWeights/_native.MlpWeights)
and the real trained PyTorch policy it was ported from.

Loads data/models/ppo.zip directly and calls its mlp_extractor.policy_net /
action_net / mlp_extractor.value_net / value_net Sequentials on a raw
VECTOR_LEN-dim tensor - the same 3-layer chain each C++ MlpWeights branch
mirrors (see battle_engine/ppo_warm_start.py's module docstring for the
verified layer correspondence). This deliberately bypasses
ObservationOnlyExtractor/the Dict observation space/action masking
entirely: none of that is part of what mlp.hpp ports (masked softmax is
Phase 5's concern, per the plan's own OUT-of-scope note), so testing
against it here would test something this phase doesn't implement.

np.allclose, not exact equality, per the plan's own DW-3.2 instruction -
the C++ side accumulates in double precision (see mlp.hpp) specifically to
narrow, not eliminate, the summation-order drift against PyTorch's own
(different-order) float32 accumulation over ~665-2000 terms per output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from battle_engine.encoding import VECTOR_LEN

_native = pytest.importorskip("battle_engine._native")

_PPO_BIN = Path("data/cpp_weights/ppo.bin")
_PPO_CHECKPOINT = Path("data/models/ppo.zip")

pytestmark = pytest.mark.skipif(
    not (_PPO_BIN.exists() and _PPO_CHECKPOINT.exists()),
    reason="requires data/cpp_weights/ppo.bin (scripts/export_weights.py) and "
    "data/models/ppo.zip (scripts/train_ppo.py --save), neither committed (gitignored data/)",
)


def _random_input(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(VECTOR_LEN).astype(np.float32).tolist()


def _torch_forward(policy, input_vec: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Runs the same raw VECTOR_LEN input through the real policy's actor
    and critic branches directly (mlp_extractor.* + action_net/value_net),
    matching the exact chain mlp.hpp's MlpWeights::forward ports - not
    policy.predict()/forward(), which would additionally route through
    ObservationOnlyExtractor and the masked distribution.
    """
    x = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        actor_logits = policy.action_net(policy.mlp_extractor.policy_net(x))
        critic_value = policy.value_net(policy.mlp_extractor.value_net(x))
    return actor_logits.squeeze(0).numpy(), critic_value.squeeze(0).numpy()


@pytest.fixture(scope="module")
def torch_policy():
    model = MaskablePPO.load(_PPO_CHECKPOINT, env=None)
    model.policy.eval()
    return model.policy


@pytest.fixture(scope="module")
def cpp_weights():
    return _native.PolicyWeights.load(str(_PPO_BIN))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_actor_forward_matches_pytorch_policy(torch_policy, cpp_weights, seed):
    input_vec = _random_input(seed)

    cpp_logits = np.array(cpp_weights.actor.forward(input_vec), dtype=np.float32)
    torch_logits, _ = _torch_forward(torch_policy, input_vec)

    assert cpp_logits.shape == torch_logits.shape == (13,)
    assert np.allclose(cpp_logits, torch_logits, atol=1e-3, rtol=1e-3), (
        f"actor logits diverge beyond tolerance:\ncpp={cpp_logits}\ntorch={torch_logits}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_critic_forward_matches_pytorch_policy(torch_policy, cpp_weights, seed):
    input_vec = _random_input(seed)

    cpp_value = np.array(cpp_weights.critic.forward(input_vec), dtype=np.float32)
    _, torch_value = _torch_forward(torch_policy, input_vec)

    assert cpp_value.shape == torch_value.shape == (1,)
    assert np.allclose(cpp_value, torch_value, atol=1e-3, rtol=1e-3), (
        f"critic value diverges beyond tolerance: cpp={cpp_value}, torch={torch_value}"
    )


def test_policy_weights_load_reports_the_real_header_dimensions(cpp_weights):
    # A cheap regression guard on the loaded shapes themselves (not just
    # the forward-pass output) - if a future export_weights.py change ever
    # silently altered VECTOR_LEN or the actor/critic output widths, this
    # fails independently of whether the numeric parity checks above
    # happen to still pass.
    assert cpp_weights.actor.layer0.in_dim == VECTOR_LEN
    assert cpp_weights.actor.layer2.out_dim == 13
    assert cpp_weights.critic.layer0.in_dim == VECTOR_LEN
    assert cpp_weights.critic.layer2.out_dim == 1
