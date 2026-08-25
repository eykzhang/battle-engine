# Discovery + Design: Phase 3 - M3 C++ NN forward pass

## Files Found
- `cpp/include/be/{mlp.hpp}`, `cpp/src/{mlp.cpp}`, `cpp/tests/test_mlp.cpp` — none exist yet, this phase creates them.
- `cpp/bindings/module.cpp` — exists (M1/M5/M7 bindings), extend with `MlpWeights`/`PolicyWeights`.
- `tests/test_native_forward_pass.py` — doesn't exist yet.
- `data/cpp_weights/ppo.bin` — exists (Phase 2 output), verified real: magic `BEPP`, version 1, vector_len 665, layer dims `(128,665)/(64,128)/(13,64)` actor, `(128,665)/(64,128)/(1,64)` critic — matches `scripts/export_weights.py`'s pinned layout exactly (checked by reading the raw bytes directly).
- `data/models/ppo.zip` — exists (2.3MB), the real checkpoint the parity test loads directly.
- `cpp/CMakeLists.txt` — `be_core` STATIC lib source list needs `src/mlp.cpp` added; `cpp/tests/CMakeLists.txt` needs `test_mlp.cpp` added to `be_tests`.

## Current State
M0-M7 done, Phase 1 (M7 wiring) and Phase 2 (weight export) both committed. `mcts.hpp`'s `EvalFn`/`default_eval` and `select_ucb1_action` are the only search-facing pieces that exist; nothing NN-related exists in C++ yet. `ppo_warm_start.py`'s layer correspondence and `export_weights.py`'s binary format are both already verified against the real checkpoint (Phase 2 exact-equality test), so this phase's job is a pure consumer of an already-pinned contract, not re-deriving anything.

## Gaps
None against the plan. All three DW items are buildable against what exists today.

## Code Standards
- C++: `snake_case` functions/vars, `PascalCase` types, `kCamelCase` constexpr. Project headers quoted+namespaced, stdlib angle-bracketed. `be` namespace only, no nesting. `cpp/include|src` stay pybind11-free; only `module.cpp` touches pybind11.
- Header comments: name the alternative considered and why it lost, state invariants at the field level, record any plan-correcting finding inline — `mcts.hpp` is the exemplar, read before writing `mlp.hpp`.
- Error handling: C++ uses `is_valid()`-style predicates for hot-path state invariants, not exceptions — but this project has no existing precedent for a *file-loading* boundary (pokedex/movedex tables are compiled-in, not loaded at runtime). `load()` here is a one-time construction-time call (Phase 5 loads once at `MctsPlayer.__init__`, never per-turn), not hot-path, so an exception (`std::runtime_error`, caller-facing message) is the right fit — matches the plan's own "load-time error, not a crash mid-forward-pass" framing.
- Naming: never silently drop a simplification — name it. Applies to the double-precision-accumulator choice and the "why exceptions here, not `is_valid()`" choice below.
- New C++ functionality lands as header+impl+test together, not staggered.

## Test Infrastructure
- Catch2 (`cpp/tests/`): one `TEST_CASE` per behavior, full-sentence name, tagged by module (`"[mlp]"`). Prefer a statistical/tolerance check for anything numeric; a fixed-seed exact case is a cheap smoke test alongside it, not the sole check — N/A here since the forward pass is deterministic given fixed weights, so a fixed known-input/known-output case is the right (and only needed) shape.
- pytest (`tests/`): `_native = pytest.importorskip("battle_engine._native")` as the first real line, native suite run via `./scripts/pytest_native.sh` only.
- No existing microbenchmark precedent in `cpp/tests/` (no `BENCHMARK`/`chrono` usage found anywhere in `cpp/`). Catch2 v3.15.3 (already `FetchContent`-pinned) ships `catch_benchmark_all.hpp`; using its `BENCHMARK()` macro inside a tagged `[!benchmark]` TEST_CASE is the idiomatic fit — Catch2's own convention is `!`-prefixed tags are hidden from a default `catch_discover_tests` run (won't add noise/flakiness to `ctest`'s normal green/red signal) while still being runnable explicitly for DW-3.3's number.

## DW Verification

| DW-ID | Done-When Item | Status | Test Cases |
|-------|---------------|--------|------------|
| DW-3.1 | Catch2 tests in `cpp/tests/test_mlp.cpp` cover a known-input/known-output case and a truncated/malformed weight file (load-time error, not a crash) | COVERED | `"MlpWeights::forward: matches a hand-computed 3-layer output"`, `"PolicyWeights::load: throws on a truncated weight file"`, `"PolicyWeights::load: throws on a bad magic header"`, `"PolicyWeights::load: throws on an unsupported version"`, `"PolicyWeights::load: throws on a dimension-chain mismatch"` |
| DW-3.2 | `tests/test_native_forward_pass.py` — same input vector through the real PyTorch policy (`data/models/ppo.zip`) and the C++ forward pass, `np.allclose` on both actor logits and critic value | COVERED | `test_actor_forward_matches_pytorch_policy`, `test_critic_forward_matches_pytorch_policy` (both against `MaskablePPO.load(...).policy`, called directly on `.mlp_extractor.policy_net`/`.action_net`/`.mlp_extractor.value_net`/`.value_net` — bypasses the mask/dict-obs plumbing entirely, since only the raw 3-layer chain each C++ branch mirrors is under test here) |
| DW-3.3 | microbenchmark forward pass (µs); project ms/turn at candidate `n_simulations`; state the fits/doesn't-fit threshold | COVERED (measurement, not a pass/fail assertion) | Catch2 `BENCHMARK()` inside a `[!benchmark]`-tagged `TEST_CASE` for the raw µs number; the ms/turn projection itself is arithmetic recorded below, not a test |

**All items COVERED:** YES

## Design Decisions

### Design: MlpWeights / PolicyWeights

**Approaches considered:**
1. **Fixed 3-layer struct (chosen)** — `MlpWeights` holds exactly 3 `MlpLayer` members (`layer0`, `layer1`, `layer2`), `forward()` hardcodes the Linear→ReLU→Linear→ReLU→Linear chain. `PolicyWeights::load()` reads the 6-layer file directly into two `MlpWeights`.
2. **Generic `std::vector<MlpLayer>` N-layer MLP** — `forward()` loops over an arbitrary-length layer vector, ReLU between all but the last.
3. **Templated compile-time-shaped MLP** (`Mlp<665, 128, 64, 13>`) — dimensions as template parameters, zero runtime dimension storage.

| Criterion | 1 (fixed struct) | 2 (generic vector) | 3 (templated) |
|---|---|---|---|
| Interface simplicity | 3 named fields, obvious | 1 field, but caller must know "3 layers, ReLU between all but last" is the real contract | Type signature IS the contract, but a template instantiated twice (actor/critic differ only in out_dim) for no real benefit |
| Information hiding | Good — `forward()`'s activation pattern is internal | Same, but generality invites a "what if N≠3" question this project's YAGNI stance already answered "never" | Compile-time dims add a rebuild-on-shape-change coupling `ppo.bin`'s own runtime-checked header explicitly exists to avoid |
| Caller ease of use (Phase 5) | Direct: `weights.actor.forward(input)` | Same call shape, no real win | Same call shape, but a shape mismatch becomes a compile error instead of `load()`'s already-designed runtime error — actively worse given the header's own version-checked-at-runtime philosophy |
| Matches plan's OUT scope | Yes — plan explicitly rejects "a generalized N-layer MLP framework" | No — this IS that rejected shape | No — even more generalized |

**Choice: 1 (fixed 3-layer struct).** Rationale: the plan's own Scope section already settled this ("a hardcoded 3-layer struct is the deep-enough, YAGNI-correct choice over a templated general solution") — recorded here for the depth-check, not re-litigated. A templated approach would also fight `load()`'s runtime dimension-chain validation, which is the actual defense against a future architecture change (a rebuild-time template parameter isn't more correct than a load-time check when the artifact being read is itself runtime-generated, gitignored, and can change shape without a recompile).

**Depth check:**
- Interface methods: `MlpWeights::forward(input) -> output` (1), `PolicyWeights::load(path) -> PolicyWeights` (1). Two methods total, both used in every real call site (Phase 5's PUCT expansion always calls `load()` once then `forward()` per node).
- Hidden details: row-major `(out,in)` weight layout, double-precision internal accumulation, ReLU placement (between hidden layers only), the exact byte layout of `ppo.bin` (magic/version/vector_len/6-layer chain) — none of this leaks to a caller, which only ever sees `std::vector<float> in -> std::vector<float> out` and a `path -> PolicyWeights` struct.
- Common case complexity: simple — `auto pw = PolicyWeights::load(path); auto logits = pw.actor.forward(encode_native(state));`.

### Load-time validation scope (per plan's Constraints)
Self-consistency only, no external `VECTOR_LEN` cross-check (none exists C++-side until Phase 4):
1. Magic bytes == `"BEPP"`.
2. `version == 1` (the one supported value; a future version bump is a load-time error, not a silent misread).
3. Every read is length-checked against the stream's actual remaining bytes (`read_exact` throws on short read) — this is what turns a truncated file into a clean error instead of reading uninitialized/garbage memory into `weight`/`bias` vectors.
4. Layer dimension chain: `layer0.out_dim == layer1.in_dim`, `layer1.out_dim == layer2.in_dim`, per branch.
5. `layer0.in_dim == header.vector_len` for both branches (the header's own declared width must match what its own first layer says it consumes — a self-consistency check on the file's own internal claims, not an external cross-check).
6. Declared `in_dim`/`out_dim` must be positive (rejects a corrupt header claiming a zero/negative-size layer before attempting an absurd `resize()`).

### Double-precision accumulation
Per the plan's explicit constraint. `forward()`'s inner dot-product loop accumulates in `double`, casting each `float` weight/input to `double` per multiply-add, then casts the final sum back to `float` (with ReLU applied in `double` before the cast, so a borderline-negative value isn't pushed across zero by the cast itself). This is strictly about reducing summation-order drift against PyTorch's own (different-order, likely SIMD-fused) accumulation — not a correctness requirement on its own, which is exactly why the parity test uses `np.allclose` (a tolerance band), not exact equality, per DW-3.2's own explicit instruction.

### Error handling: exception vs. `is_valid()`
Named explicitly per the codebase's "never leave a design choice silently implicit" convention. `is_valid()`-style predicates are this project's convention for *hot-path* state invariants (checked routinely, cheaply, without unwinding). `PolicyWeights::load()` is a one-time, call-once-per-`MctsPlayer`-construction I/O boundary (per the plan's Phase 5 note: "loads `_native.PolicyWeights.load(path)` **once** at construction... never re-reading the weight file per turn") — not hot-path, and the plan's own Edge cases line calls for "a load-time error, not a crash" specifically. `std::runtime_error` with a caller-facing message (which field, what was expected vs. found) is used throughout `mlp.cpp`'s loader.

## Prerequisites
- [x] Required files exist (or will be created)
- [x] Dependencies available (`data/cpp_weights/ppo.bin`, `data/models/ppo.zip` both real, verified above)
- [x] No missing prerequisites

## DW-3.3 measurement (recorded here per the plan's own instruction — feeds a Phase 5 design decision, not acted on now)

**Real per-battle numbers (Phase 1's own measured figures, `n_simulations=200`, `default_eval`):**
- default format: 1.0-1.2 s/battle
- gen9ou: 3.8-4.7 s/battle

**Turns/battle — measured fresh** (not in Phase 1's own record; needed to convert s/battle into a per-simulation figure, so measured directly rather than assumed, per this project's "evidence over assumption" hard rule): ran 6 real `mcts` vs `random` battles at default format and 6 real `mcts` vs `random` battles at gen9ou (local Showdown server, already running) via a throwaway script (not committed — outside this phase's file scope).
- default format: turns = [30, 52, 19, 49, 37, 32], avg 36.5
- gen9ou: turns = [36, 25, 128, 28, 110, 23], avg 58.3 (high variance at N=6 — two long/stall-ish battles; treated as an order-of-magnitude figure, not a precise one)

**Backed-out per-simulation cost of the existing `default_eval` search infrastructure** (resolve_turn + legal_actions + UCB1 selection + eval, everything search() does per simulation today):
- default format: (1.0-1.2 s / 36.5 turns) / 200 sims ≈ 137-164 µs/sim
- gen9ou: (3.8-4.7 s / 58.3 turns) / 200 sims ≈ 326-403 µs/sim

**Measured forward-pass cost** (Catch2 `BENCHMARK`, `cpp/tests/test_mlp.cpp`, real `ppo.bin` weights, same Debug+ASan/UBSan build every real 500-battle sweep in this project has actually been measured under — an apples-to-apples comparison with Phase 1's own numbers, not a Release-build best case):
- actor forward: 759.5 µs (mean, 100 samples)
- critic forward: 752.1 µs (mean, 100 samples)
- actor+critic combined (one PUCT node's worth, DW-3.3's literal ask): 1,509 µs ≈ **1.51 ms**

**Projection (DW-3.3's literal shape — one actor+critic pass per simulation):** one new node expansion per simulation is standard MCTS/DUCT behavior, so approximating "per-simulation cost" as "existing per-sim cost + one actor+critic forward pass" is a reasonable, named approximation (not exact — see caveat below).

| Format | existing per-sim (backed out) | + forward pass | projected per-sim | projected ms/turn @ 200 sims | projected s/battle (× measured turns/battle) |
|---|---|---|---|---|---|
| default | 137-164 µs | +1,509 µs | 1,646-1,673 µs | 329-335 ms | × 36.5 turns ≈ 12.0-12.2 s (vs. 1.0-1.2s baseline, ~10x) |
| gen9ou | 326-403 µs | +1,509 µs | 1,835-1,912 µs | 367-382 ms | × 58.3 turns ≈ 21.4-22.3 s (vs. 3.8-4.7s baseline, ~4.7-5.7x) |

**Caveat, named explicitly (per this project's "never silently drop a simplification" convention):** Phase 5's real per-node cost is likely higher than one actor+critic pair. The Approach notes call for `my_priors` (one actor call on the real state) AND `opp_priors` (a second actor call on the mirrored state) AND one critic call for the leaf value — roughly 2 actor + 1 critic ≈ 2×759.5 + 752.1 ≈ 2,271 µs per node, not 1,509 µs. Re-running the table at that figure:

| Format | projected per-sim | projected ms/turn @ 200 sims | projected s/battle |
|---|---|---|---|
| default | 2,408-2,435 µs | 482-487 ms | × 36.5 ≈ 17.6-17.8 s |
| gen9ou | 2,597-2,674 µs | 519-535 ms | × 58.3 ≈ 30.3-31.2 s |

**Threshold:** Phase 6 needs `mcts_puct` vs `ppo` and `mcts_puct` vs `mcts`, both at gen9ou, 500 battles each = 1,000 games total. Even at the more conservative (2 actor + 1 critic) ~31 s/battle figure: 1,000 × 31s ≈ 31,000s ≈ **8.6 hours** — comfortably inside "overnight" by this project's own precedent (Phase 3's PPO training run, accepted as fitting the laptop-first hard rule, took 10.2 hours). At the DW-3.3-literal single-actor+critic figure (~22s/battle gen9ou), the full sweep is ≈6.1 hours, even more headroom. **Conclusion: Approach A fits the laptop-first overnight budget at `n_simulations=200` under either estimate — Approach C's fallback is not triggered by this measurement.** (`n_simulations` itself is Phase 5's own call to make/re-measure once the real per-node code exists — this projection is an estimate to decide the Approach A vs. C fork now, not a promise the eventual number will match exactly.)

## Recommendation
BUILD. No gaps, no scope questions — proceed directly to stub/implement/validate.
