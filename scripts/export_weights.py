"""Phase 4 M2: dump a trained PPO checkpoint's actor+critic Linear layers to
a versioned binary format a later C++ phase (M3's hand-written forward pass)
can load without any PyTorch/ONNX/TorchScript runtime dependency - see
battle_engine/ppo_warm_start.py's module docstring for why the actor and
critic trunks are shape-compatible in the first place, and
docs/code-standards.md's "Technology Decisions" for why this is a raw
binary dump rather than ONNX/TorchScript.

    .venv/bin/python scripts/export_weights.py
    .venv/bin/python scripts/export_weights.py --checkpoint data/models/ppo.zip --output data/cpp_weights/ppo.bin

Binary format (see this module's WEIGHTS_MAGIC/WEIGHTS_FORMAT_VERSION and
_LAYER_SPECS below for the authoritative field-by-field contract - little-
endian throughout, no padding):

    magic: 4 bytes ("BEPP")
    version: uint32
    vector_len: uint32
    then 6 layers in fixed order (actor.net[0], actor.net[2],
    actor.action_net, critic.net[0], critic.net[2], critic.value_net), each:
        out_dim: uint32
        in_dim: uint32
        weight: out_dim*in_dim x float32, row-major (out, in) - this is
            PyTorch's own nn.Linear.weight layout, so no transpose is
            needed converting from the loaded checkpoint's tensors.
        bias: out_dim x float32
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.encoding import VECTOR_LEN
from battle_engine.ppo_warm_start import WARM_START_NET_ARCH

WEIGHTS_MAGIC = b"BEPP"
WEIGHTS_FORMAT_VERSION = 1  # bump on any future change to this binary layout

# PPO's critic always predicts a single scalar state-value estimate - this
# is inherent to how the actor-critic algorithm works, not a project
# constant pulled from elsewhere the way VECTOR_LEN/ACTION_SPACE_SIZE are.
_CRITIC_OUTPUT_DIM = 1

_HIDDEN0, _HIDDEN1 = WARM_START_NET_ARCH

# The 6 layers in the plan's pinned order, each as
# (name, expected_out_dim, expected_in_dim) - matches
# battle_engine/ppo_warm_start.py's documented layer correspondence
# exactly, not re-derived. A checkpoint whose real layer shape doesn't
# match its entry here means the checkpoint wasn't built with
# warm_start_policy_kwargs() (see ppo_warm_start.py's module docstring).
_LAYER_SPECS = (
    ("actor.net[0]", _HIDDEN0, VECTOR_LEN),
    ("actor.net[2]", _HIDDEN1, _HIDDEN0),
    ("actor.action_net", ACTION_SPACE_SIZE, _HIDDEN1),
    ("critic.net[0]", _HIDDEN0, VECTOR_LEN),
    ("critic.net[2]", _HIDDEN1, _HIDDEN0),
    ("critic.value_net", _CRITIC_OUTPUT_DIM, _HIDDEN1),
)


def load_policy(checkpoint_path: Path) -> MaskableActorCriticPolicy:
    """Loads a saved MaskablePPO checkpoint's policy (env=None - only the
    weights are needed, not further training - same pattern
    ppo_eval.load_ppo_player already uses). Raises a clear, caller-facing
    error rather than letting a missing path or a corrupt zip surface as a
    raw traceback into sb3_contrib internals - checkpoint_path is external
    input (a file path handed in from the CLI), so it gets validated at
    this boundary rather than assumed.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_path}")
    try:
        model = MaskablePPO.load(checkpoint_path, env=None)
    except Exception as e:
        raise RuntimeError(f"failed to load PPO checkpoint at {checkpoint_path}: {e}") from e
    return model.policy


def _extract_layers(policy: MaskableActorCriticPolicy) -> tuple[nn.Linear, ...]:
    """Pulls the 6 Linear layers out of policy in the exact order
    _LAYER_SPECS documents - the same correspondence
    ppo_warm_start.load_warm_start_weights copies into, read back out.
    """
    return (
        policy.mlp_extractor.policy_net[0],
        policy.mlp_extractor.policy_net[2],
        policy.action_net,
        policy.mlp_extractor.value_net[0],
        policy.mlp_extractor.value_net[2],
        policy.value_net,
    )


def _validate_layer_shapes(layers: tuple[nn.Linear, ...]) -> None:
    """Raises ValueError on the first layer whose real shape doesn't match
    _LAYER_SPECS's expectation. Called before any packing/writing begins so
    a shape-mismatched checkpoint can never produce a partial ppo.bin - a
    later C++ loader that only checks magic/version/vector_len would
    otherwise silently accept a truncated or wrongly-shaped file.
    """
    for (name, expected_out, expected_in), layer in zip(_LAYER_SPECS, layers):
        actual_out, actual_in = layer.weight.shape
        if (actual_out, actual_in) != (expected_out, expected_in):
            raise ValueError(
                f"{name}: expected weight shape ({expected_out}, {expected_in}), got "
                f"({actual_out}, {actual_in}) - checkpoint doesn't match the architecture "
                "ppo_warm_start.warm_start_policy_kwargs() builds"
            )


def _pack_layer(layer: nn.Linear) -> bytes:
    """Packs one Linear layer as out_dim/in_dim (uint32) followed by its
    weight (row-major (out, in), PyTorch's native nn.Linear.weight layout -
    no transpose) and bias, both float32 little-endian.
    """
    weight = layer.weight.detach().cpu().numpy().astype("<f4")
    bias = layer.bias.detach().cpu().numpy().astype("<f4")
    out_dim, in_dim = weight.shape
    return struct.pack("<II", out_dim, in_dim) + weight.tobytes() + bias.tobytes()


def export_weights(checkpoint_path: Path, output_path: Path) -> None:
    """Loads checkpoint_path, validates every layer's shape against
    _LAYER_SPECS, then writes output_path in one buffered write. Validation
    happens fully before any filesystem write to output_path, so a shape
    mismatch (ValueError) or a load failure never leaves a partial file
    behind - see _validate_layer_shapes's docstring.
    """
    policy = load_policy(checkpoint_path)
    layers = _extract_layers(policy)
    _validate_layer_shapes(layers)

    buffer = bytearray()
    buffer += struct.pack("<4sII", WEIGHTS_MAGIC, WEIGHTS_FORMAT_VERSION, VECTOR_LEN)
    for layer in layers:
        buffer += _pack_layer(layer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buffer))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/models/ppo.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/cpp_weights/ppo.bin"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_weights(args.checkpoint, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
