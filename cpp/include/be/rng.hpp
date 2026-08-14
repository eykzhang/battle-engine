// M5: a single, explicit-seed RNG wrapper - the actual mechanism M5's
// forward model (sampled damage/crit/accuracy/multi-hit) and M6's MCTS
// (rollout sampling) both need instead of damage.py's expected-value
// folding (see the Scope decision in plans/precious-crafting-bachman.md
// for why sampling, not averaging, is the point of this phase). A single
// seeded std::mt19937_64 per Rng instance makes every result reproducible
// - M5's fixed-seed determinism tests and M6's "same seed -> same root
// visit distribution" test both depend on this.
#pragma once

#include <cstdint>
#include <random>

namespace be {

class Rng {
 public:
  explicit Rng(uint64_t seed) : engine_(seed) {}

  // Uniform float in [0, 1).
  float uniform01() { return dist01_(engine_); }

  // true with probability p. p outside [0, 1] is a caller bug (not
  // clamped) - same "callers are responsible for their own inputs"
  // convention as forward_model.hpp's resolve_turn/legal_actions contract.
  bool bernoulli(float p) { return uniform01() < p; }

  // Uniform integer in [lo, hi], inclusive on both ends (e.g.
  // uniform_int(85, 100) for damage.py's 85-100 damage roll,
  // uniform_int(2, 5) for a variable multi-hit count).
  int uniform_int(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(engine_);
  }

 private:
  std::mt19937_64 engine_;
  std::uniform_real_distribution<float> dist01_{0.0f, 1.0f};
};

}  // namespace be
