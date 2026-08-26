// M3: hand-written forward pass for the actor+critic MLP branches Phase
// 2's scripts/export_weights.py dumped from the trained PPO checkpoint
// (see PolicyWeights::load's doc comment below for the exact byte layout
// this header codes against - pinned by that script's own module
// docstring, not paraphrased here).
//
// Fixed 3-layer architecture (VECTOR_LEN -> 128 -> 64 -> out_dim, ReLU
// between the two hidden layers, NO activation after the final layer)
// rather than a generalized N-layer MLP framework: this project's PPO/
// imitation/win-prob architecture is fixed and known (net_arch=[128, 64],
// pinned in battle_engine/ppo_warm_start.py's WARM_START_NET_ARCH) - a
// templated general solution would be YAGNI here, and would fight
// PolicyWeights::load's own runtime dimension-chain validation (the
// artifact being read is itself runtime-generated and gitignored, so a
// compile-time template parameter isn't more correct than a load-time
// check for catching a future shape change).
//
// Internal accumulation is double-precision even though weights/inputs are
// float32 - this reduces float32 summation-order drift against PyTorch's
// own (different-order, likely SIMD-fused) accumulation enough that the
// Python-side parity test (tests/test_native_forward_pass.py) can use
// np.allclose (a tolerance band) rather than exact equality - which is
// also why that test doesn't assert exact equality: no two independent
// summation orders over ~665-2000 terms are expected to agree bit-for-bit.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace be {

// One Linear(in_dim -> out_dim) layer's weights, in the exact row-major
// (out, in) layout PyTorch's own nn.Linear.weight uses (see
// scripts/export_weights.py's _pack_layer docstring - no transpose needed
// converting from the checkpoint's saved tensors).
struct MlpLayer {
  int in_dim = 0;
  int out_dim = 0;
  std::vector<float> weight;  // out_dim * in_dim, row-major (out, in)
  std::vector<float> bias;    // out_dim
};

// Fixed 3-layer MLP: layer0 (VECTOR_LEN -> 128), layer1 (128 -> 64), layer2
// (64 -> out_dim - 13 for the actor branch, 1 for the critic branch, per
// ppo.bin's own pinned layer order). ReLU is applied after layer0 and
// layer1; NOT after layer2 - the actor's output is the raw pre-(masked)
// softmax logits (masked softmax is Phase 5's concern, needs a legal-
// action set a bare forward pass doesn't have), and the critic's output is
// its raw scalar value estimate.
struct MlpWeights {
  MlpLayer layer0;
  MlpLayer layer1;
  MlpLayer layer2;

  // input.size() must equal layer0.in_dim - a caller bug otherwise, NOT
  // checked here, same "callers own their inputs" convention as
  // legal_actions()/resolve_turn() elsewhere in this codebase (mcts.hpp,
  // forward_model.hpp). Returns a vector of layer2.out_dim floats.
  std::vector<float> forward(const std::vector<float>& input) const;
};

// Both branches loaded from one ppo.bin in a single call. Phase 5's PUCT
// expansion always needs actor (prior) and critic (leaf value) together at
// the same node - one load() call returning both halves the load-time
// validation surface a caller has to handle vs. two separate
// MlpWeights::load() calls, and matches how the two branches are always
// used together downstream.
struct PolicyWeights {
  MlpWeights actor;
  MlpWeights critic;

  // Reads path in scripts/export_weights.py's pinned binary format
  // (little-endian throughout, no padding):
  //   magic: 4 bytes ("BEPP")
  //   version: uint32 (must be 1 - the only format this loader supports)
  //   vector_len: uint32
  //   then 6 layers in fixed order (actor.net[0], actor.net[2],
  //   actor.action_net, critic.net[0], critic.net[2], critic.value_net),
  //   each: out_dim (uint32), in_dim (uint32), weight (out_dim*in_dim
  //   float32, row-major (out,in)), bias (out_dim float32)
  //
  // Checks SELF-consistency only: magic bytes, version == 1, every
  // declared layer dimension is positive, the layer chain is internally
  // consistent (layer0.out_dim == layer1.in_dim, layer1.out_dim ==
  // layer2.in_dim, per branch), and layer0.in_dim matches the header's
  // own declared vector_len for both branches. This is NOT a cross-check
  // against a C++-side VECTOR_LEN constant - none exists yet (a later
  // phase adds the 3-way encode_native()/encoding.VECTOR_LEN/ppo.bin
  // cross-check once encode_native() exists to check against).
  //
  // Throws std::runtime_error, with a caller-facing message naming what
  // failed (missing file, truncated read, bad magic, unsupported version,
  // non-positive dimension, or a dimension-chain mismatch), rather than
  // reading past the end of a short buffer or silently misinterpreting a
  // version-mismatched layout. This is the plan's only deserialization
  // boundary in hand-written C++, under an ASan-by-default build for
  // exactly this reason: a bug here should crash loudly (or, here, throw
  // cleanly) at load time, not silently misread weights into a forward
  // pass that then produces quiet garbage. An exception (not this
  // project's usual is_valid()-style predicate convention for hot-path
  // state invariants) is the right fit specifically because load() is a
  // one-time, call-once-at-construction I/O boundary, not a per-simulation
  // hot path - Phase 5 loads this exactly once per MctsPlayer construction
  // and reuses the handle for every subsequent search() call.
  static PolicyWeights load(const std::string& path);
};

}  // namespace be
