# Review: Phase 5 - M6b PUCT search (3rd/final attempt)

## Executed Results (Step 0)
- Build: `./scripts/build_cpp.sh` → clean build, no warnings surfaced.
- C++ suite: `ctest --test-dir cpp/build` → **96/96 passed** (4.77s), matches reported baseline.
- Python suite: `./scripts/pytest_native.sh` → **196 passed, 3 skipped** (3.76-4.96s), matches reported baseline. The 3 skips are pre-existing (`test_dataset.py`x2, `test_encoding.py`x1, all "no fetched replay sample at data/replays_raw" — unrelated to this phase).
- Independently re-ran the hidden `[!benchmark]` DW-5.3 microbenchmark directly: `./tests/be_tests "[puct][!benchmark]"` → **505.217 ms mean** (100 samples, 3.35ms stddev) against the real `data/cpp_weights/ppo.bin`, consistent with the 501.9-501.947ms recorded in `notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md` (within measurement noise for a wall-clock benchmark).
- `data/cpp_weights/ppo.bin` present on disk, so `test_mcts_player.py`'s real-weights integration test (`test_mcts_puct_player_real_search_puct_call_returns_a_valid_order`) actually executed against real trained weights, not skipped.

## Requirement Fulfillment

### DW-5.1
PREMISE:  Catch2 tests for the Metamon-mapping functions and `select_puct_action`.
EVIDENCE: `cpp/tests/test_action.cpp:154-221` (metamon_switch_label_to_action_id, action_id_to_metamon_label, fainted-teammate, bench<5); `cpp/tests/test_mcts.cpp:367-472` (select_puct_action x4, puct_priors_from_actor_logits x2)
TRACE:    `select_puct_action({0,1}, stats=[{0,0},{0,0}], priors=[0.1,0.9], parent_visits=1, c_puct=1.4)` → both Q=0, score = c_puct*prior*sqrt(1)/(1+0) → action 1's 0.9 prior wins → returns 1, matches `REQUIRE(picked == 1)` at line 374. `metamon_switch_label_to_action_id(state, 6)` on a 6-slot team maps to species-sorted position → `3` (Gengar), matches `REQUIRE(... == 3)` at line 160, confirmed passing in ctest run.
VERDICT:  PASS

### DW-5.2
PREMISE:  fixed-seed determinism holds for the PUCT search.
EVIDENCE: `cpp/tests/test_mcts.cpp:516-527`, implementation in `cpp/src/mcts.cpp:189-190` (`Rng rng(seed)`, single seeded RNG drives every stochastic choice).
TRACE:    `search_puct(state_a, weights, 40, seed=42)` and `search_puct(state_b, weights, 40, seed=42)` on two independently-constructed but field-identical states → identical `best_action` and identical sorted `root_visit_distribution` — test passed in ctest run (test #`search_puct: same seed...`, part of the 96/96).
VERDICT:  PASS

### DW-5.3
PREMISE:  measured ms/turn checked against Phase 3's projection, recorded durably.
EVIDENCE: `cpp/tests/test_mcts.cpp:676-684` (the measurement); `notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md` (the durable record and the arithmetic reconciling it against Phase 3's DW-3.3 6-9hr projection); `scripts/benchmark.py:174-184` (the resulting `--n-simulations=200` default, citing the note).
TRACE:    Independently re-ran the benchmark: 505.2ms/turn @ n_simulations=200 (Debug/ASan). The note computes 15,000 total `search_puct` calls for Phase 6's full 1,000-battle sweep at ~502ms/call → ~2.1 hours, compared against Phase 3's 6-9hr projection, concluding "comfortably inside budget." This is a real comparison with real arithmetic, not just a bare number.
VERDICT:  PASS

### DW-5.4
PREMISE:  the critic's sign-backup convention is checked empirically on N synthetic mirrored state pairs, with a stated fallback if ambiguous — per the plan: "If neither holds clearly within tolerance ..., default to `-v` (a property of the opponent table's structure, not the eval's shape) and record the measured antisymmetry error in `mcts.hpp` as a named [approximation]."
EVIDENCE: `cpp/include/be/mcts.hpp:414-458`; measurement test `cpp/tests/test_mcts.cpp:618-667` (30 synthetic mirror(state)/state pairs, varied HP/status/boosts/species/types); backup implementation `cpp/src/mcts.cpp:135-142` (`score_leaf_puct` always uses `{v, -v}` for both the critic and default_eval branches).
TRACE:    Measured `mean(v(s)+v(mirror(s))) = -0.0659` (mcts.hpp:419), judged clearly closer to 0 (the `-v` convention) than 1 (the `1-v` convention) — `score_leaf_puct` implements exactly `-v` (mcts.cpp:137-138), matching the measured result. The fallback clause is now explicitly stated (mcts.hpp:428-435): "had this measurement failed to separate clearly ... the code defaults to -v regardless. ... -v is a property of the opponent-side VisitStats table's own negamax structure ..., not of the critic's own output range or antisymmetry, so it holds even when the critic's antisymmetry is ambiguous." The measured residual (-0.0659) is recorded and named as an approximation with a mechanistic explanation (encode()'s missing opponent-bench block, mcts.hpp:441-458), satisfying the plan's "record the measured antisymmetry error ... as a named approximation" clause. This closes the specific gap the 2nd review found.
VERDICT:  PASS

### DW-5.5
PREMISE:  `kForcedSwitch` node handling.
EVIDENCE: `cpp/src/mcts.cpp:291-366` (search_puct's kForcedSwitch branch: `select_ucb1_action` only, `continue`s the loop rather than breaking after creating a child); `cpp/tests/test_mcts.cpp:529-559` (no-replacement → default_eval terminal), `561-581` (real replacement → continues to a real kDecision leaf).
TRACE:    A kForcedSwitch node's selection at mcts.cpp:300 unconditionally calls `select_ucb1_action(actions, stats, ...)` — no `priors.empty() ? ... : ...` branch exists here at all (unlike the kDecision branch at lines 203-212), because `my_priors`/`opp_priors` are never populated for a kForcedSwitch node (only `populate_decision_node`, called solely for kDecision nodes, sets them). After applying the chosen switch and creating a child, `created_forced_switch` at line 354 triggers `cur = ...; continue;` rather than `break`, so the walk descends through the forced-switch node into whatever comes next (chained forced-switch or a real kDecision state) — confirmed exercised by the "continues expansion... reaching a real kDecision state" test, which passed in ctest.
VERDICT:  PASS

### DW-5.6
PREMISE:  mirror round-trip/field-swap/weather-terrain AND the encode_native(mirror(s))-vs-real-opponent-POV-encode() value parity.
EVIDENCE: `cpp/src/battle_state.cpp:361-369` (mirror()); `cpp/tests/test_battle_state.cpp:152-193` (round-trip, field-swap, weather/terrain-unchanged, C++-side length check); `tests/test_native_encoding.py:217-270` (the value-parity test).
TRACE:    `mirror(mirror(state))` field-for-field equals `state` (test at line 152-156, passed). `mirror(state)` swaps `my_team↔opp_team`, `my_active_slot↔opp_active_slot`, `my_hazards↔opp_hazards`, leaves `weather`/`terrain` untouched (tests at 158-179, passed). The value-parity half lives in Python (the C++ test at 181-193 explicitly only checks length, by design — no Python interpreter in Catch2 scope): `test_encode_native_mirror_matches_python_encode_of_real_opponent_pov` builds a real poke-env battle, computes `encode_native(mirror(battle_state_from_poke_env(battle)))`, and compares via `np.allclose` against `encode()` of a genuinely-swapped opponent-POV `_battle(...)` object (my_team↔opponent_team, hazards swapped, weather/terrain unchanged) — this ran and passed in the pytest_native run (`test_native_encoding.py` showed 10/10 dots, no skips).
VERDICT:  PASS

**All requirements met:** YES

## Test-DW Coverage
- [x] DW-5.1 → `cpp/tests/test_action.cpp`, `cpp/tests/test_mcts.cpp` (Catch2, ran in ctest)
- [x] DW-5.2 → `cpp/tests/test_mcts.cpp:516-527` (Catch2, ran in ctest)
- [x] DW-5.3 → `cpp/tests/test_mcts.cpp:676-684` (measured, hidden `[!benchmark]` tag, independently re-run above) + `notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md` (durable record)
- [x] DW-5.4 → `cpp/tests/test_mcts.cpp:618-667` (Catch2, ran in ctest) + doc comment stating the fallback
- [x] DW-5.5 → `cpp/tests/test_mcts.cpp:529-581` (Catch2, ran in ctest)
- [x] DW-5.6 → `cpp/tests/test_battle_state.cpp` (Catch2) + `tests/test_native_encoding.py:217-270` (pytest, ran in pytest_native)
- [x] Coverage matches the stated "Targeted" level: C++ Catch2 correctness (test_mcts.cpp, test_action.cpp, test_battle_state.cpp), Python parity (test_native_encoding.py), player/integration wiring (test_mcts_player.py's 3 new MctsPuctPlayer tests, test_benchmark.py's 1 new `_make_player("mcts_puct", ...)` test) are all present and all ran.

No gaps found.

## Dead Code
None found. All three test-file diffs (`test_action.cpp` +108, `test_battle_state.cpp` +111, `test_mcts.cpp` +337) are pure additions — no existing test was modified, removed, or commented out. No unreachable code after early returns in the new C++ or Python source.

## Correctness Dimensions

| Dimension | Status | Evidence |
|-----------|--------|----------|
| Concurrency | PASS | `search_puct`'s pybind11 binding (`module.cpp:290`) uses `py::call_guard<py::gil_scoped_release>()`, same convention as `search()`. `PolicyWeights`/`MLP::forward` (pre-existing, unchanged this phase) is `const`, no mutable state — two concurrently-running battles sharing one loaded `PolicyWeights` (MctsPuctPlayer loads once at construction, reused per `choose_move`) can safely call `forward()` concurrently. No new shared mutable state introduced by this phase. |
| Error Handling | PASS | `PolicyWeights::load` failure (missing/malformed `ppo.bin`) propagates as a pybind11-translated `RuntimeError`, not swallowed. `metamon_switch_label_to_action_id`/`action_id_to_metamon_label` return an observable `-1` sentinel on a no-mapping case rather than throwing or silently returning garbage — documented, and both directions are tested for the failure case (bench<5). |
| Resources | N/A | No new file handles/sockets/locks in this phase's diff; `PolicyWeights::load` (pre-existing) is a one-shot read at construction, RAII-managed; `SearchNode` tree is `unique_ptr`-owned, freed automatically when `search_puct` returns. |
| Boundaries | PASS | Traced `n_simulations=0`: loop never executes, `root_node` (populated at construction) still yields a `best_action` from whatever `my_actions` it has (visits=0 all around, first action wins the `> -1` tie-break) — no crash. Traced the "root has no encodable actives" edge case (test at line 583-609, `my_active_slot=-1`): `populate_decision_node`'s `has_encodable_actives` gate correctly leaves priors empty at the root, so UCB1-only selection kicks in even at the top of the tree — passed. Traced fainted-slot handling in `species_sorted_bench_slots`: fainted slots ARE included (by design, matches encoding.py's bench semantics) — but the only runtime caller of `action_id_to_metamon_label` is `puct_priors_from_actor_logits`, always fed `node.my_actions`/`node.opp_actions` from `legal_actions()`, which excludes fainted, so no fainted ActionId ever actually reaches the Metamon-mapping functions at runtime. See Notes for a doc-comment inaccuracy this surfaced. |
| Security | N/A | No untrusted external input in this phase's diff — `ppo_bin_path` is a caller-supplied local file path (same trust model as every other model-loading path in this codebase, e.g. `WinProbModel.load`), not user/network-supplied. |

## Loaded-Skill Criteria

| Skill | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| aposd-designing-deep-modules | Interface depth / information hiding for the new PUCT search surface | PASS | `search_puct(root, weights, n_simulations, seed)` — 4 params, hides the entire tree-walk/backup machinery behind one call, matching `search()`'s existing depth. `MctsPuctPlayer` mirrors `MctsPlayer`'s shape exactly (construction-time load, one `choose_move` override) rather than adding a mode flag — avoids the "logical cohesion" shallow-interface trap explicitly called out in the code's own comments. |
| aposd-designing-deep-modules | Information leakage / silent failure | PASS (Note on one item) | No silent failures: `kNoAction`/`-1` sentinels are documented, observable, and tested return values, not swallowed errors. The `action.hpp` doc-comment claim about `species_sorted_bench_slots()`'s inclusion set is internally inconsistent with `battle_state.hpp`'s own doc-comment and with the project's own fainted-teammate test — see Notes; this is a documentation-accuracy issue, not an information-hiding or silent-failure defect (no functional consequence demonstrated). |
| cc-routine-and-class-design | Parameter count (7±2 threshold) | PASS | All new/changed routines this phase stay at 2-5 parameters (`search_puct`=4, `select_puct_action`=5, `puct_priors_from_actor_logits`=3, `populate_decision_node`=3, `mirror`=1, both Metamon-mapping functions=2). No violation, no warning-tier routine. |
| cc-routine-and-class-design | LSP / inheritance vs containment | PASS | `MctsPuctPlayer` IS-A `Player` (same substitutability as `MctsPlayer`/`FrozenPolicyPlayer`/`TwoPlySearchPlayer`) — one overridden method (`choose_move`), no empty overrides, no strengthened preconditions vs. the base class. `SearchNode`'s tagged-enum (not inheritance) design for `kDecision`/`kForcedSwitch` is containment, not an LSP question — correctly modeled as data-variant, not type hierarchy. |
| cc-routine-and-class-design | Cohesion classification | WARNING (non-blocking) | `search_puct()` (mcts.cpp:189-396, ~207 lines) is sequential/temporal cohesion — tree descent, node creation, kForcedSwitch continuation, leaf scoring, and backup all share the `state`/`path`/`cur` data across one required order. This is "ACCEPT w/caution" per the checklist, not a violation — and it exactly mirrors `search()`'s own already-existing shape (pre-dates this phase, presumably already accepted in the M6 review). Not flagged as a defect; noted for awareness only. |

## Notes (non-blocking)

1. **Doc-comment inaccuracy in `action.hpp` (low severity, high confidence).** `action.hpp:110-113`'s comment for `action_id_to_metamon_label` claims: "shouldn't happen for a legal switch ActionId (legal_actions()'s switch-legality contract - revealed, not fainted, not active - is exactly species_sorted_bench_slots()'s own inclusion set, by construction)". This is factually wrong: `species_sorted_bench_slots()` (`battle_state.cpp:341-359`) does **not** filter fainted slots — it only skips the active slot and unrevealed slots. `battle_state.hpp:283` even documents this correctly for the same function ("fainted members still counted as a real position"), and `test_action.cpp:186-197` explicitly tests that a fainted, revealed teammate still gets a real Metamon-label mapping. So `action.hpp`'s claim that "not fainted" is part of the inclusion set directly contradicts both the actual implementation and the project's own test. **No functional bug results** — the only runtime caller (`puct_priors_from_actor_logits` via `mcts.cpp:76`) always passes ActionIds sourced from `legal_actions()`, which does exclude fainted, so a fainted slot's ActionId never actually reaches this function during search. This is purely a misleading/self-contradictory comment, worth a one-line fix (drop "not fainted" from the inclusion-set description) but not a correctness defect.

2. **Minor redundant compute (low severity, medium confidence).** In `search_puct`'s kDecision branch, when `my_pick == kNoAction || opp_pick == kNoAction` fires at a node whose actives ARE encodable (e.g., the "opponent has 0 revealed moves yet" scenario), `score_leaf_puct(state, weights)` recomputes `encode_native(state)` and re-runs the critic forward pass, even though `populate_decision_node` already computed and cached this exact value in `decision_result.critic_value` when the node was created. Same state, same weights → same result, so not a correctness issue, just a wasted forward pass on an already-rare path. Not a DW item, not flagged as a defect.

3. **No dedicated `kTerminal` test for either `search()` or `search_puct()` (low severity, medium confidence).** Both functions have explicit `TurnResolution::kTerminal` handling (mcts.cpp:264-266, 510-512), but no test in `test_mcts.cpp` specifically drives a state to a real battle-ending turn to exercise that branch directly (it may be incidentally exercised inside the 40-500 simulation runs of other tests, but that's not verified/asserted). Not a DW item and not a listed edge case, so not a blocker — worth a targeted test if this code is touched again.

4. **`search_puct()`'s length (~207 lines) mirrors `search()`'s own pre-existing shape** rather than introducing new complexity — see Loaded-Skill Criteria above.

## Issues (if FAIL)
None.

**Verdict: PASS**
