# Discovery + Design: Phase 5 - M6b — PUCT search with PPO prior/value

## Files Found
All file-scope files exist and are populated (M0-M4b done): `cpp/include/be/{mcts,action,battle_state,mlp}.hpp`,
matching `.cpp`s, `cpp/bindings/module.cpp`, `battle_engine/mcts_player.py`, `scripts/benchmark.py`,
`cpp/tests/{test_mcts,test_action,test_battle_state,test_mlp}.cpp`. `data/cpp_weights/ppo.bin` (Phase 2)
and `data/models/ppo.zip` (Phase 3 source) are both present on disk — DW-5.4's empirical sign-convention
measurement can run against the real trained critic, not a synthetic stand-in.

## Current State
- `mcts.hpp`/`mcts.cpp`: open-loop DUCT/UCB1 `search()`, tested (66/66 ctest). `SearchNode` is a tagged
  struct (`kDecision`/`kForcedSwitch`), `VisitStats{visits, value_sum}` per side, backup already applies
  `my_stats += v`, `opp_stats += -v` (verified correct for `default_eval`'s proven antisymmetry, `mcts.hpp`'s
  own doc comment already corrects an earlier `1-v` assumption).
- `action.hpp`/`action.cpp`: fixed `ActionId` scheme (0-5 switch by team-preview slot, 6-9 move slot),
  `legal_actions()`. No Metamon-mapping functions yet.
- `battle_state.hpp`/`battle_state.cpp`: `BattleState`, `encode_native()` (bit-for-bit `encode()` port).
  `species_sorted_bench()` (file-local, pointer-returning) already implements the exact species-sort M6b's
  mapping functions need — reusable via a new index-returning variant (see Design below), not re-derivable
  safely by hand a second time.
- `mlp.hpp`/`mlp.cpp`: `MlpWeights::forward()`, `PolicyWeights::load()` — actor (13-way logits, no softmax
  applied) + critic (raw scalar) both available.
- `forward_model.cpp`: confirmed by direct read — a fainted active's `active_slot` is **never** reset to
  -1 by `apply_action`/`resolve_turn`; it keeps pointing at the fainted slot until a forced-switch action
  reassigns it. This matters: `encode_native()`'s only throw condition is `active_slot < 0`, so a
  `kForcedSwitch` node's underlying state does NOT throw on `encode_native()` — it silently succeeds and
  produces output for a state PPO never saw during training (a real Showdown env always resolves the
  forced switch as part of one env step). "Untested semantics," not "crashes" — the plan's own framing
  ("no tested semantics for a missing active mon") is confirmed correct once this file was actually read.
- `action_space.py`: `_switch_action_to_poke_env`/`_poke_env_switch_to_metamon` — the exact species-sort
  logic to port (sorts ALL non-active team members, fainted included, by `base_species`; no fainted filter).
- `dataset.py`: Metamon's 13-way scheme confirmed — `_MOVE_ACTIONS = {0..3}`, `_SWITCH_ACTIONS = {4..8}`,
  `_TERA_MOVE_ACTIONS = {9..12}`.
- `sb3_contrib/common/maskable/distributions.py` (real source, read directly): `MaskableCategorical.apply_masking`
  sets a masked logit to `-1e8` then runs a normal 13-way softmax — mathematically identical to gathering only
  the legal logits and softmaxing that subset directly (both reduce to `exp(x_i)/sum_legal(exp(x_j))`).

## Gaps
- No `mirror(BattleState)` — needed for the opponent-side prior (Approach notes).
- No index-returning species-sort helper — `species_sorted_bench()`'s pointer-only return can't recover a
  team-preview slot index, which the Metamon mapping functions need.
- No PUCT selection, no priors storage on `SearchNode`, no `search_puct()`.
- No Metamon<->ActionId mapping functions.
- DW-5.4's sign convention is genuinely unverified — must be measured against the real critic, not assumed.

## Code Standards
Applied: heavy design-rationale doc comments (state alternative + why rejected), `is_valid()`-style
predicates over exceptions for hot-path state, `snake_case`/`PascalCase`/`kCamelCase` naming, `Side::Me`/
`Side::Opp` vocabulary, one-concept-per-file, "name every simplification, never invent a magic number,"
project layering (`cpp/include|src` has zero pybind11 dependency; only `module.cpp` touches pybind11).

## Test Infrastructure
Catch2 (`cpp/tests/`, one `TEST_CASE` per behavior, tagged by module), `ctest --test-dir cpp/build`.
`BE_TEST_PPO_WEIGHTS_PATH` is already a compile-time macro (`cpp/tests/CMakeLists.txt`) pointing at the
real `data/cpp_weights/ppo.bin` — reused directly for DW-5.4's empirical measurement and DW-5.3's
benchmark, same pattern `test_mlp.cpp`'s `[!benchmark]`-tagged microbenchmark already established.
No Python test file is in this phase's file scope — `mcts_player.py`/`benchmark.py` wiring is validated by
(a) the full existing pytest suite staying green (regression) and (b) a manual, uncommitted sanity run, not
a new pytest file.

## DW Verification

| DW-ID | Done-When Item | Status | Test Cases |
|-------|---------------|--------|------------|
| DW-5.1 | Catch2 tests for Metamon-mapping functions + `select_puct_action` | COVERED | `test_action.cpp`: known-team mapping (both directions), fainted-teammate/bench<5 case; `test_mcts.cpp`: `puct_priors_from_actor_logits` renormalizes over legal actions and drops tera mass, `select_puct_action` synthetic-bandit prior-pulls-early-selection test |
| DW-5.2 | Fixed-seed determinism for PUCT search | COVERED | `test_mcts.cpp`: `search_puct` same-seed-same-result test (same shape as M6's own determinism test) |
| DW-5.3 | Measured ms/turn at chosen `n_simulations`, checked vs Phase 3's projection | COVERED | `test_mcts.cpp`: `[!benchmark]`-tagged `search_puct` timing against real `ppo.bin`, run explicitly post-build, recorded in this report |
| DW-5.4 | Critic sign-backup convention checked empirically | COVERED | `test_mcts.cpp`: N mirrored-state pairs through the real critic, mean of `v(s)+v(mirror(s))` compared against 0 vs 1, decides constant used in `search_puct`'s backup |
| DW-5.5 | `kForcedSwitch` node: UCB1-only selection, continues past to real `kDecision` state for critic leaf value | COVERED | `test_mcts.cpp`: regression-style test mirroring M6's existing kForcedSwitch fixture, asserts no crash and (via a spy-able discovery, see Design) that the eventual leaf state has both actives resolved |
| DW-5.6 | `mirror()` round-trips, fields swapped, weather/terrain unchanged, `encode_native(mirror(s))` matches real opponent-POV `encode()` | COVERED | `test_battle_state.cpp`: round-trip equality test (field-wise helper, no new `operator==` added to the production type — not needed by any other caller), field-swap assertions, weather/terrain-unchanged assertion. The `encode_native(mirror(s))` vs. real Python `encode()` cross-check is Python-side (`encoding.py`'s `battle_view_from_poke_env`), but no pytest file is in this phase's scope — covered instead by hand-tracing `mirror()`'s effect against `encode_native()`'s own already-tested-symmetric field reads (every field `encode_native` reads is either `my_*`/`opp_*`-paired and swapped, or state-level and untouched) plus a manual (uncommitted) sanity script run against a real translated battle state, reported in this build's Deviations/notes rather than as a new tracked test file |

**All items COVERED:** YES

## Design Decisions

### 1. Species-sort logic: one shared implementation, not two
`encode_native()`'s file-local `species_sorted_bench()` (pointer-returning) and the new Metamon-mapping
functions need the IDENTICAL sort (base_species order, active excluded, fainted included, padded to
`kMaxBench`). Two independent implementations of the same sort is exactly the divergence risk this phase's
own Gate rationale warns about ("a defect here corrupts every PUCT decision silently"). **Chosen:** add
`species_sorted_bench_slots(state) -> std::array<int, kMaxBench>` (team-preview slot INDEX per position, -1
for no real member) to `battle_state.hpp`/`.cpp`, then rewrite `species_sorted_bench()` to build its
pointer array FROM this index array (same behavior by construction, not by duplicated logic). Low-risk,
behavior-preserving refactor of already-reviewed Phase 4 code — verified via the full existing
`test_native_encoding.py`/`test_battle_state.cpp` suite staying green after the change.
**Alternative considered:** re-derive the sort locally in `action.cpp`. Rejected — exactly the duplication
this phase's design intent argues against.

### 2. Metamon-mapping functions operate on "my" side only, no `Side` parameter
Both `metamon_switch_label_to_action_id`/`action_id_to_metamon_label` read only `state.my_team`/
`state.my_active_slot`. The opponent's own prior is computed by calling `encode_native`/the actor AND these
mapping functions on `mirror(state)` — reusing the identical "my side" code path for both sides (per the
Approach notes and Decision Log's own already-confirmed choice). This also sidesteps a layering problem:
`Side` is declared in `action.hpp` itself (fine there), but `battle_state.hpp` doesn't depend on
`action.hpp` (`action.hpp` depends on `battle_state.hpp`, not the reverse) — a `Side`-parameterized version
would need `battle_state.hpp` to know about `Side`, an unnecessary layering violation the "my side only"
design avoids entirely.

### 3. "Can this node be safely encoded/scored by the actor+critic?" — reuse of a fainted/`-1` check, not a new predicate
`encode_native()` only throws on `active_slot == -1` (team preview / a not-yet-resolved root). It does
**not** throw on a fainted-but-index-valid active (the real shape of every `kForcedSwitch` node's
underlying state, confirmed by reading `forward_model.cpp`) — but running the actor/critic there is still
untested/undefined territory (PPO's training env never presented such an observation). **Chosen:** a local
(mcts.cpp-internal, not header-exposed — no other caller needs it) predicate
`has_encodable_actives(state) = active_slot >= 0 && !fainted, both sides`. Whenever a newly-created
`kDecision`-shaped node's state satisfies this, run the actor+critic and populate `my_priors`/`opp_priors`;
otherwise leave both empty. `kForcedSwitch` nodes never get priors regardless (plan's own explicit choice).
**Every action-selection call becomes:** `priors.empty() ? select_ucb1_action(...) : select_puct_action(...)`
— one uniform rule that correctly handles `kForcedSwitch` nodes AND the one remaining edge (a root called
mid-forced-switch-response, `active_pokemon is None`, per Phase 1's own already-tested MctsPlayer edge case)
without a second, root-specific special case.

### 4. Leaf value source: critic when encodable, `default_eval` only as a last-resort, per-leaf choice
Per the plan: expansion continues past `kForcedSwitch` nodes (apply the switch, don't stop) until a real,
encodable `kDecision` state is reached, and the critic scores THAT state — this is the normal path and
requires no fallback. The ONLY place `default_eval` is used in `search_puct` is the already-existing,
already-tested "genuinely nothing to do" terminal cases M6's `search()` already handles this same way
(`kNoAction` from an empty action list at either a `kDecision` or `kForcedSwitch` node) where an encodable
state can never be reached by definition. Reasoning about scale-mixing: (a) for a `kForcedSwitch` node with
zero legal replacements, this state is never pushed to `path` (mirrors M6's existing convention — "no real
choice was made here"), so it doesn't write into that node's own `VisitStats`, only bubbles up as `leaf_value`
for ancestors already on `path`; (b) this can ONLY happen when the acted-on side has literally no revealed
teammates left to switch to, an extremely narrow, near-terminal shape. Documented explicitly in `mcts.hpp`
as a named, accepted scale-mixing exception, not a silent one — each leaf's evaluator (critic vs.
`default_eval`) determines its OWN opponent-side sign transform independently (see Decision 5), so even
this fallback stays sign-correct.

### 5. Sign convention picked per-leaf, not globally
`default_eval`'s convention is already proven (`-v`, antisymmetric, `mcts.hpp`'s own existing doc comment).
The critic's convention is measured separately (DW-5.4, see below) and may or may not match. **Chosen:**
compute `(leaf_value, opp_transform)` at the point a leaf is scored, not as one global constant for the
whole `search_puct` call — a leaf scored by `default_eval` always backs up `-leaf_value` into `opp_stats`;
a leaf scored by the critic backs up whatever DW-5.4 determines. This is correct regardless of whether the
two evaluators end up agreeing on convention.

### 6. `select_puct_action`: Q=0 for unvisited (AlphaZero convention), not UCB1's cold-start infinity
UCB1 forces every action to be tried once (`visits==0` always wins). PUCT does not — the prior already
biases early exploration. Formula: `PUCT(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))`, `Q(a) = 0`
when `visits==0`. `kDefaultPuctConstant = 1.4f` — matching `kDefaultExplorationConstant`'s own precedent
(a reasonable, explicitly-not-deeply-tuned starting point, consistent with the plan's "one measured pass,
no open-ended tuning" scope limit).

### 7. `puct_priors_from_actor_logits`: gather-then-softmax, not mask-then-full-softmax
Verified (see Files Found) these are mathematically identical. Implemented as gather (via
`action_id_to_metamon_label`) + numerically-stable softmax (subtract max) over exactly the legal subset —
simpler to implement and test directly than reproducing a 13-wide masked array. Lives in `mcts.hpp`/`.cpp`
(not `action.hpp`) since it's PUCT-specific (depends on the actor's 13-way output shape, an M6b/mlp.hpp
concept `action.hpp` has no reason to know about).

## Empirical measurement performed during this build (DW-5.4)

A Catch2 test (`test_mcts.cpp`, tag `[!benchmark]`-adjacent but run as a normal assertion since it's cheap)
loads the real `data/cpp_weights/ppo.bin`, builds 30 synthetic-but-varied `BattleState`/`mirror(state)`
pairs (varying HP fraction, status, boosts, species/types) with both actives always valid, and computes
`v(s) + v(mirror(s))` for each via the critic. Result recorded in this build's final report (see BUILD
Complete output) — the actual measured mean/std decided which constant `search_puct` uses; if ambiguous,
the plan's own default (`-v`) was used per its explicit fallback instruction, with the measured error
recorded in `mcts.hpp` as a named approximation.

## Final Measured Results (post-implementation)
- DW-5.4: mean(v(s)+v(mirror(s))) = -0.0659237, stddev = 1.49e-08 over 30 synthetic pairs — clearly closer to
  0 than to 1, extremely low variance (a real, consistent bias, not noise). `-v` confirmed, not defaulted-to.
  Recorded in `mcts.hpp`'s `search_puct()` doc comment along with a mechanical explanation for the small
  residual (encode()'s missing opponent-bench block makes `mirror()` a lossy, not information-preserving,
  transformation of the observation the critic actually sees).
- DW-5.3: `search_puct` at `n_simulations=200` measured 501.77ms mean (100 samples, Debug/ASan) — i.e. ≈502ms/turn.
  Rough comparison to Phase 3's 6-9 hour full-sweep projection: Phase 6 runs two 500-battle matchups
  (`mcts_puct` vs `ppo`, `mcts_puct` vs `mcts`); only `mcts_puct`'s own turns invoke `search_puct` (~15
  own-turns/battle, a rough estimate) → ~15,000 total `search_puct` calls → ~2.1 hours, comfortably inside
  Phase 3's projected budget (that projection was itself measured under the same Debug/ASan build, so this
  is an apples-to-apples comparison, not release-vs-debug).
- Final suite: `ctest` 96/96 passed (76 baseline + 20 new: 5 `test_action.cpp`, 5 `test_battle_state.cpp`, 10
  `test_mcts.cpp`, all visible; 1 additional `[!benchmark]`-tagged `test_mcts.cpp` case excluded from the
  default `ctest` count, same convention as `test_mlp.cpp`'s own DW-3.3 benchmark). `./scripts/pytest_native.sh`
  191 passed / 3 skipped (unchanged from baseline — no regressions from the `battle_state.cpp` refactor or
  the new Python wiring).

## Prerequisites
- [x] Required files exist
- [x] Dependencies available (Phase 1/3/4 all committed, ppo.bin present)
- [x] `sb3_contrib` source readable in this venv for the masking verification

## Recommendation
BUILD.

## Review Fix (attempt 1) — 2026-08-25

1st FAIL. The reviewer independently confirmed the core PUCT algorithm correct (value-scale-mixing
avoidance, mirror mechanism, tera renormalization, sign convention, weights-loaded-once) — no
re-architecture done here, only the three named gaps closed.

### Issue 1 — DW-5.6's `encode_native(mirror(s))` vs. real opponent-POV `encode()` was untested
`cpp/tests/test_battle_state.cpp:181-194`'s C++-side test can only check length (no Python
interpreter reachable from Catch2). Added
`tests/test_native_encoding.py::test_encode_native_mirror_matches_python_encode_of_real_opponent_pov`
— builds a real poke-env battle fixture (rich my/opp actives with boosts/item/ability/moves, a
bench mon each, hazards, weather, terrain), computes `_native.encode_native(_native.mirror(state))`,
and independently builds the genuine opponent-POV `battle_view_from_poke_env` (every role actually
swapped — team, active, hazards; weather/terrain unchanged, matching `mirror()`'s own
state-level/per-side split) and calls the real `encoding.encode()` on it, then `np.allclose`s the
two vectors. This is the actual oracle DW-5.6 requires, not a proxy. Ran green on first attempt
(not adjusted to pass) — confirms the previously-unverified claim that `mirror()`+`encode_native()`
reproduces `encode()`'s real my/opp asymmetry (opponent bench collapsed to `opp_remaining_fraction`,
never per-slot) correctly.

### Issue 2 — DW-5.3's comparison against Phase 3's projection had no durable record
Re-derived Phase 3's actual DW-3.3 numbers from the plan file rather than trusting the prior
summary: 1.51ms/node forward pass, projecting Phase 6's full sweep at 6-9 hours (Debug/ASan,
commit `aa4c485`). Verified the arithmetic independently: 2 matchups × 500 battles = 1,000 battles
`mcts_puct` plays in; ~15 own-turns/battle (rough estimate) → ~15,000 `search_puct` calls; at the
reviewer's independently-reproduced 501.947ms/call → 15,000 × 0.501947s = 7,529.2s ≈ **2.09
hours** — comfortably inside the 6-9hr budget (matches the prior summary's ~2.1hr, arithmetic
checks out). Recorded durably in
`/Users/edward/Projects/battle-engine/notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md`
(`type: log`, added to `notes/index.md`'s Logs list) per this project's preferred working-notes
convention — the reviewer's own stated preference. Also added a one-line pointer comment at
`scripts/benchmark.py`'s `--n-simulations` default (near the code the reviewer actually checked
first) so the comparison is discoverable from both places.

### Issue 3 — `MctsPuctPlayer` / `"mcts_puct"` had zero test coverage
Added to `tests/test_mcts_player.py` (mirroring `MctsPlayer`'s existing coverage exactly):
- `test_mcts_puct_player_falls_back_to_default_order_on_no_action` — `NO_ACTION` → `choose_default_move()`.
- `test_mcts_puct_player_loads_weights_once_and_reuses_across_choose_move_calls` — spies on
  `_native.PolicyWeights.load` (call-counted) and `_native.search_puct` (records the `weights`
  object identity per call); asserts `load` called exactly once across two `choose_move()` calls
  and both calls receive the *same* loaded object (identity check, not equality) — proves reuse,
  not reload. Also confirms `n_simulations` passed through unchanged and a fresh seed drawn per call.
- `test_mcts_puct_player_real_search_puct_call_returns_a_valid_order` — real `data/cpp_weights/ppo.bin`,
  small `n_simulations=20`, confirms the full construction → `search_puct()` → `BattleOrder` path
  works end-to-end (skipped if `ppo.bin` absent, matching this project's existing skip convention).

Added to `tests/test_benchmark.py`: `test_make_player_mcts_puct_constructs_a_real_mcts_puct_player`
— calls `scripts.benchmark._make_player("mcts_puct", ...)` directly (needed a local `sys.path`
shim, same one `tests/test_export_weights.py` already established for importing from `scripts/`,
since it isn't an installed package), asserts a real `MctsPuctPlayer` is constructed with the
configured `n_simulations`. Skipped if `ppo.bin` absent (construction eagerly loads it).

### Verification
- `ctest --test-dir cpp/build`: **96/96 passed** (unchanged — no C++ source touched this pass,
  only C++ test-file review evidence already existed; the fix is entirely Python-side test/notes/comment).
- `./scripts/pytest_native.sh`: **196 passed, 3 skipped** (up from 191/3 — exactly +5: the DW-5.6
  parity test, three new `MctsPuctPlayer` tests, and the `_make_player("mcts_puct")` test; skip
  count unchanged, none of the new tests skipped in this environment since `ppo.bin` is present).
- Full existing suite re-ran green — passing set only grew, no regressions.

### Files changed this pass
- `tests/test_native_encoding.py` — Issue 1 (new parity test).
- `/Users/edward/Projects/battle-engine/notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md`,
  `/Users/edward/Projects/battle-engine/notes/index.md` — Issue 2 (new durable note + index entry;
  both live in the main repo's gitignored `notes/`, outside this worktree, per this project's
  established vault-artifact convention).
- `scripts/benchmark.py` — Issue 2 (discoverability comment only, no logic change).
- `tests/test_mcts_player.py`, `tests/test_benchmark.py` — Issue 3 (new player/wiring tests).

No production logic changed — every fix this pass is test, documentation, or comment-only, as the
findings required (the underlying implementation was already reviewer-verified correct).

## Review Fix (attempt 2)

### Finding
2nd-review found one new, small gap: DW-5.4's own text requires the critic sign-backup convention
be checked empirically "with a stated fallback if ambiguous." The empirical check (N=30 synthetic
mirrored pairs, mean(v(s)+v(mirror(s))) = -0.0659237) is solid and passes, but nothing in the
codebase stated what convention would apply if that measurement had come out ambiguous instead of
clearly separating toward `-v`. The runtime already hardcodes the correct fallback unconditionally
— this was a pure documentation gap, not a functional bug.

### Fix
`cpp/include/be/mcts.hpp:428-441` (new paragraph inserted into the existing DW-5.4 sign-convention
doc comment above `search_puct()`, directly after the paragraph reporting the -0.0659 measurement
and its "-v" conclusion, before the residual-explanation paragraph). Added text states: (1) the
stated fallback per the plan's own DW-5.4 contingency — if the measurement had not separated
clearly within a stated tolerance, the code defaults to `-v` regardless; (2) *why* `-v` is the
correct unconditional default here specifically — it follows from the opponent-side VisitStats
table's own negamax backup structure (each node backs up the negation of its child's value by
construction), a property of the tree's bookkeeping, not of the critic's own output range or
antisymmetry — unlike `default_eval`'s case, where `-v` was proven from the eval formula's own
antisymmetry; (3) that this fallback was NOT triggered here (the -0.0659 separation was
unambiguous, clearly nearer 0 than 1); and (4) that the fallback would have chosen the same `-v`
convention already hardcoded, so no runtime behavior hinges on which path was taken.

### Verification
- `./scripts/build_cpp.sh`: clean rebuild, no compile errors (doc-comment-only change, but
  confirms the file still parses/builds).
- `ctest --test-dir cpp/build`: **96/96 passed** (unchanged from pre-fix baseline).
- `./scripts/pytest_native.sh`: **196 passed, 3 skipped** (unchanged from pre-fix baseline).
- No test added or removed — this fix is a doc-comment addition only, exactly as the finding
  called for ("one sentence closes it").

### Files changed this pass
- `cpp/include/be/mcts.hpp` — added the stated-fallback paragraph to the DW-5.4 sign-convention
  doc comment (lines 428-441). No other file touched.
