# Review: Phase 5 - PUCT search (PPO actor/critic)

## Executed Results (Step 0)

- Build: `./scripts/build_cpp.sh` (Debug/ASan, default) → clean, up to date at review time.
- `ctest --test-dir cpp/build` → **96/96 passed** (matches reported baseline: 76 prior + 20 new, one `[!benchmark]`-tagged test excluded from the default run).
- `./scripts/pytest_native.sh` → **191 passed, 3 skipped** (matches reported baseline).
- `./be_tests "DW-5.4*" -s` (reproduced independently) → `mean(v(s)+v(mirror(s))) = -0.0659237, stddev = 1.49012e-08` over 30 pairs — matches the reported measurement exactly.
- `./be_tests "[puct][!benchmark]" -s` (reproduced independently) → `search_puct, n_simulations=200`: mean 501.947 ms (100 samples, Debug/ASan build) — matches the reported ≈502ms/turn.

## Requirement Fulfillment

### DW-5.1: Catch2 tests for Metamon-mapping + select_puct_action
PREMISE: Catch2 tests for the Metamon-mapping functions (known team → known mapping, fainted-teammate/bench<5, tera 9-12 dropped+renormalized) and `select_puct_action` (synthetic bandit, prior pulls selection toward high-prior arms early).
EVIDENCE: `cpp/tests/test_action.cpp:154-221` (known mapping, fainted-teammate, bench<5); `cpp/tests/test_mcts.cpp:367-472` (select_puct_action bandit tests; puct_priors_from_actor_logits tera-drop tests).
TRACE: `metamon_switch_label_to_action_id(state, 4)` on a 6-member team with active=Zapdos(slot0) → returns 1 (Bulbasaur, species-sorted) — matches hand-computed species order. `puct_priors_from_actor_logits` with `logits[9]=1000` (tera) and only move-label-0 legal → `priors[0]==Approx(1.0f)` (tera mass fully dropped, renormalized to 1 over the remaining legal set). `select_puct_action` with priors `{0.1,0.9}` on two unvisited actions, `parent_visits=1` → picks index 1 (higher prior).
VERDICT: PASS

### DW-5.2: fixed-seed determinism for PUCT search
PREMISE: fixed-seed determinism holds for the PUCT search, same standard as M6's `search()`.
EVIDENCE: `cpp/tests/test_mcts.cpp:516-527`, `cpp/src/mcts.cpp:189-396` (`search_puct`'s internal `Rng rng(seed)` drives every stochastic choice).
TRACE: Two independently-built identical `BattleState`s, `search_puct(state_a, weights, 40, seed=42)` vs `search_puct(state_b, weights, 40, seed=42)` → identical `best_action` and identical sorted root visit distributions. Ran via ctest, passed.
VERDICT: PASS

### DW-5.3: measured ms/turn at chosen `n_simulations`, checked against Phase 3's projection
PREMISE: measured ms/turn at the chosen `n_simulations` for this configuration, checked against Phase 3's projection.
EVIDENCE: `cpp/tests/test_mcts.cpp:676-684` (`[!benchmark]`-tagged microbenchmark); `scripts/benchmark.py:174-180` (`--n-simulations` default 200, comment says "this phase's own measured choice").
TRACE: Reproduced the microbenchmark myself: 501.947ms mean/turn at n_simulations=200 (Debug/ASan build, 100 samples). This half of the requirement has direct execution evidence.
The second half — "checked against Phase 3's projection" — has **no evidence in any reviewable artifact**. I searched `cpp/`, `battle_engine/`, `scripts/`, and the vault's `notes/` directory (`/Users/edward/Projects/battle-engine/notes/`, last touched 2026-08-18, nothing about this phase) for any mention of Phase 3's 570.6 steps/s or any other Phase-3-derived projection, and found nothing. The comparison is not present in a code comment, docstring, test, or committed note — only (per my own grep, which I am not using as grounding per this review's independence rule) in the build agent's own discovery scratch file, which is not durable project documentation and which I was instructed not to use as evidence.
VERDICT: PARTIAL — the ms/turn measurement is real, reproducible, and correctly executed; the required comparison against Phase 3's projection is unevidenced in any reviewable artifact.

### DW-5.4: critic sign-backup convention checked empirically
PREMISE: the critic's sign-backup convention is checked empirically (not assumed) on N synthetic mirrored state pairs, with a stated fallback if ambiguous.
EVIDENCE: `cpp/tests/test_mcts.cpp:618-667`; `cpp/include/be/mcts.hpp:414-445` (doc comment recording the measured result and the convention it selects); `cpp/src/mcts.cpp:120-142` (`score_leaf_puct`/`LeafScore`, both branches use `-v`).
TRACE: 30 synthetic `mirror(state)`/`state` pairs (varied HP/status/boosts/species/types) run through the real `data/cpp_weights/ppo.bin` critic. Reproduced independently: mean(v(s)+v(mirror(s))) = -0.0659237, stddev = 1.49012e-08 — clearly closer to 0 (the "-v" convention) than to 1 (the "1-v" convention). `score_leaf_puct` implements `{v, -v}` for both the critic and `default_eval` branches, matching the measured/proven conventions respectively.
VERDICT: PASS

### DW-5.5: kForcedSwitch node UCB1-only selection + continued expansion to a real kDecision leaf
PREMISE: a `kForcedSwitch` node during search correctly uses UCB1-only selection and continues expansion past it to a real `kDecision` state for the critic's leaf value (no crash, no invented encoding, no scale-mixed backup).
EVIDENCE: `cpp/src/mcts.cpp:291-366` (the `else` branch of `search_puct`'s tree walk); `cpp/tests/test_mcts.cpp:529-609`.
TRACE: Traced the code directly (not just the doc comment). At a `kForcedSwitch` node, selection always calls `select_ucb1_action(actions, stats, ...)` — `my_priors`/`opp_priors` are never consulted (kForcedSwitch nodes never populate them, `populate_decision_node` is only called for kDecision nodes). On a `kBothFainted` turn resolution, the code chains two `kForcedSwitch` nodes (`cur = cur->children[picks].get(); continue;`, `mcts.cpp:278-281` and `:352-357`) without computing any leaf value at creation time — the loop only computes a leaf (`decision_result.critic_value` if `decision_result.encodable`, else `default_eval`) once it reaches a node whose `new_node->kind` is `kDecision` (`mcts.cpp:359-364`). The one documented, narrow exception (a forced-switch node with zero legal replacements, `chosen == kNoAction`) falls back to `default_eval` via `score_leaf_puct` (`mcts.cpp:302-307`) and is not pushed to `path`, matching the doc comment's own named exception. Ran the two dedicated tests (`no revealed replacement falls back to default_eval`, `continues expansion past a forced-switch node with a real replacement`) — both passed (40 and 60 total visits recorded respectively, no crash).
VERDICT: PASS

### DW-5.6: mirror(BattleState) round-trip, field-swap, weather/terrain unchanged, encode_native(mirror(s)) parity
PREMISE: `mirror(BattleState)` round-trips (`mirror(mirror(s)) == s`), every `my_*`/`opp_*` field pair is actually swapped, weather/terrain unchanged, and `encode_native(mirror(s))` matches `encode()` of the real opponent-POV view on a fixture.
EVIDENCE: `cpp/src/battle_state.cpp:361-369` (`mirror()` implementation); `cpp/tests/test_battle_state.cpp:152-194`.
TRACE: `mirror()` is `std::swap(my_team, opp_team)` / `std::swap(my_active_slot, opp_active_slot)` / `std::swap(my_hazards, opp_hazards)`, weather/terrain untouched — trivially involutive by construction. Round-trip test (`mirror(mirror(state)) == state` field-for-field), field-swap test (every `my_team[i]`/`opp_team[i]`, active slots, hazards actually swapped), and weather/terrain-unchanged test all ran and passed.
The fourth sub-requirement — `encode_native(mirror(s))` matching `encode()` of the real opponent-POV view on a fixture — is **not tested**. The only test touching this (`test_battle_state.cpp:181-194`) checks only that `encode_native(mirror(state))` and `encode_native(state)` have the same *length* and don't throw; it does not compute the real Python `encode()` on an opponent-POV view of the same fixture and compare values. The test's own comment admits this: "Full cross-check against the real Python encode() on an opponent-POV view lives outside this phase's C++-only test scope... this Catch2-reachable check instead confirms mirror() produces a state encode_native() accepts... not [value parity]." I confirmed no such cross-check exists anywhere else either (grepped `tests/` for `mirror` — no hits outside this file and one unrelated comment).
This matters beyond documentation completeness: `mirror()` + `encode_native()` is exactly the mechanism `populate_decision_node` uses to compute `opp_priors` (mcts.cpp:180-182) — the DW-5.4 doc comment's own stated explanation for the measured -0.0659 residual is that `mirror(state)` is *not* an information-preserving transform of what the critic actually sees (encode()'s opponent-bench omission). Without ever comparing `encode_native(mirror(s))` against a real Python-side opponent-POV `encode()` call, this asymmetry claim — and the basic claim that "my"-side encoding logic, when handed a mirrored state, produces the same semantics as a genuine opponent-POV encode() — is unverified, not just under-documented.
VERDICT: FAIL — round-trip/field-swap/weather-terrain sub-clauses are demonstrated with execution evidence; the encode_native(mirror(s))-vs-real-opponent-POV-encode() sub-clause is not, and the test file's own comment concedes the gap.

**All requirements met:** NO — DW-5.3 and DW-5.6 both have unevidenced sub-clauses.

## Test-DW Coverage

- [x] DW-5.1, DW-5.2, DW-5.4, DW-5.5 have automated tests that ran in Step 0.
- [ ] DW-5.3's "checked against Phase 3's projection" clause has no automated test or recorded observed behavior anywhere in the reviewable codebase.
- [ ] DW-5.6's "matches encode() of the real opponent-POV view" clause has no automated test — the existing test explicitly checks a weaker property (length match) instead.
- **Test coverage vs. the stated level ("Targeted — C++ Catch2 correctness, Python parity tests, player/integration wiring")**: the C++ Catch2 correctness portion is genuinely strong. The other two named portions are largely absent for this phase's new surface:
  - **Python parity tests**: zero. `metamon_switch_label_to_action_id`, `action_id_to_metamon_label`, `puct_priors_from_actor_logits`, and `search_puct` are not exercised from Python anywhere in `tests/`. `module.cpp`'s own binding comment (`cpp/bindings/module.cpp:193-198`) states the Metamon-mapping functions are exposed to Python "mainly so a Python-side sanity check can cross-check them against `action_space.py`'s own poke-env-facing translation on a real battle" — no such test exists.
  - **Player/integration wiring**: `MctsPuctPlayer` (the new class in `battle_engine/mcts_player.py`) has **zero** test coverage. `tests/test_mcts_player.py` tests only the pre-existing `MctsPlayer` — nothing constructs `MctsPuctPlayer`, calls its `choose_move`, exercises its `NO_ACTION` fallback, or verifies its `PolicyWeights.load` integration. `scripts/benchmark.py`'s new `"mcts_puct"` branch in `_make_player` (lines 149-158) is likewise untested — `tests/test_benchmark.py` only covers `wilson_interval` and the generic `run_benchmark` path, never `_make_player` or the `"mcts_puct"` CLI choice.

This is a genuine, demonstrable gap against the dispatch prompt's own stated Test Coverage Level, on the single Python-facing entry point ("MctsPuctPlayer") that will actually run whatever benchmark measures this phase's strength bet.

## Dead Code

None found. Checked `cpp/src/mcts.cpp`, `cpp/include/be/mcts.hpp`, `cpp/src/action.cpp`, `cpp/src/battle_state.cpp`, `cpp/bindings/module.cpp`, `battle_engine/mcts_player.py`, `scripts/benchmark.py` for unused includes, unreachable code after early returns, debug statements, and commented-out blocks — none present. All `#include`s in `mcts.cpp` are used.

## Correctness Dimensions

| Dimension | Status | Evidence |
|-----------|--------|----------|
| Concurrency | PASS | `MlpWeights::forward` (`cpp/include/be/mlp.hpp:60`) is `const` and only reads immutable `layer0/1/2` member data into locally-allocated output vectors — no shared mutable state, safe if `search_puct` (GIL-released, `module.cpp:285-296`) were ever invoked concurrently from two poke-env battle threads sharing one `MctsPuctPlayer`'s `PolicyWeights`. Each `search_puct()` call builds its own stack-local `SearchNode` tree — no aliasing across concurrent calls. |
| Error Handling | PASS | `PolicyWeights::load` throws `std::runtime_error` on a bad file (translated to Python `RuntimeError`), uncaught in `MctsPuctPlayer.__init__` — fails loudly at construction rather than silently, matching the project's stated convention for this one-time I/O boundary. |
| Resources | N/A | No new resource acquisition in this phase's files beyond the one-time `PolicyWeights::load` at construction (already covered under Error Handling); no locks, threads, or long-lived handles introduced. |
| Boundaries | PASS | `puct_priors_from_actor_logits` guards the empty-`actions` case (`gathered.empty()` checked *before* `std::max_element`, which is UB on an empty range) — traced this explicitly since it's reachable when a newly-created kDecision node's own side has zero legal actions despite `has_encodable_actives` being true. `pack_action_pair`'s `uint8_t` cast of `kNoAction` (-1 → 0xFF) never collides with a real `ActionId` (0-9). |
| Security | N/A | No untrusted input in this phase's surface — `ppo_bin_path`/`n_simulations`/`seed` are all locally-controlled CLI/constructor arguments, not attacker-influenced. |

## Loaded-Skill Criteria

| Skill | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| aposd-designing-deep-modules | Interface depth / information hiding for `MctsPuctPlayer` and `search_puct` | PASS | `MctsPuctPlayer.__init__`/`choose_move` is a 2-method interface hiding the entire PUCT tree-walk, actor/critic forward passes, and mirror-based opponent-prior computation — a deep module by the skill's own standard (few methods, large hidden implementation). |
| aposd-designing-deep-modules | Avoid a flag-driven shallow branch (`MctsPlayer` vs `MctsPuctPlayer`, `select_ucb1_action` vs `select_puct_action`) | PASS | Confirmed as two genuinely separate types/functions, not a mode flag — matches the skill's "push specialization down into variants" guidance and the code's own stated cohesion rationale (logical-cohesion flag branches are rejected per the project's own `cc-routine-and-class-design` convention, cited directly in `mcts.hpp:296-300` and `mcts_player.py:381-387`). |
| cc-routine-and-class-design | LSP: `MctsPuctPlayer` IS-A `Player` | PASS | Only `choose_move` overridden, no empty overrides, no strengthened preconditions beyond `Player`'s own contract — same precedent as `MctsPlayer`/`FrozenPolicyPlayer`. |
| cc-routine-and-class-design | Parameter counts | PASS | `MctsPuctPlayer.__init__` (4 named params) and `search_puct` (4 params) are well under the 7±2 threshold. |
| cc-routine-and-class-design | Functional cohesion of `search_puct` | WARNING (non-blocking) | `search_puct` is a ~200-line tree-walk with two large branches (kDecision/kForcedSwitch) — a single "operation" at its declared abstraction level (run one simulation batch of PUCT), so it's defensible as functional cohesion at that level, but it's dense. This exact shape mirrors the already-gated M6 `search()` function, so it's not a new pattern introduced by this phase. Noted, not failed. |
| aposd-verifying-correctness | Requirements coverage, error handling, concurrency, boundaries, security | See Correctness Dimensions table above and DW items | — |

## Notes (non-blocking)

1. **DW-5.3's measurement is from a Debug/ASan build, not Release.** `./scripts/build_cpp.sh` defaults to Debug/ASan (per this project's own `cpp/CMakeLists.txt` convention, documented in `CLAUDE.md`), and both the reported and my reproduced ~502ms/turn figure come from that build. A real 500-battle benchmark run intended to finish "≤ overnight" would presumably use `--release`, which should be substantially faster. This isn't wrong, but if the `n_simulations=200` CLI default (`scripts/benchmark.py:180`) or any feasibility comparison against Phase 3 was made using the Debug/ASan number without noting the gap, it understates real throughput. Low severity, medium confidence — I did not find a place where this distinction caused an actual wrong conclusion, only that the artifact I can verify (the benchmark test) is Debug/ASan-only.
2. **Documentation inaccuracy, not a functional bug**: `action_id_to_metamon_label`'s doc comment (`action.hpp:108-113`) states `legal_actions()`'s switch-legality contract ("revealed, not fainted, not active") "is exactly `species_sorted_bench_slots()`'s own inclusion set." This is not literally true — `species_sorted_bench_slots()` (`battle_state.cpp:341-359`) includes fainted, revealed, non-active members (confirmed by its own doc comment and by the "fainted teammate still occupies a real species-sorted bench position" test), so `legal_actions()`'s set is a strict *subset*, not an exact match. The subset relation is what actually makes the "never returns -1 for a legal ActionId" guarantee hold, so there's no functional defect — just an imprecise comment. Low severity, high confidence.
3. The `has_encodable_actives` predicate (`mcts.cpp:113-118`) checks both `active_slot < 0` and `fainted` explicitly rather than relying on `encode_native`'s own (narrower) throw-on-`-1`-only guard — confirmed correct by reading `forward_model.cpp` isn't strictly re-verified by me in this review (out of file scope), but the claim ("resolve_turn/apply_action never resets active_slot to -1 on a faint") is consistent with everything else observed in `mcts.cpp`'s forced-switch handling (which always explicitly reassigns `active_slot` via a chosen switch, never resets it to -1 on faint).
4. Test coverage for the pre-existing `species_sorted_bench_slots()`/`species_sorted_bench()` refactor (design decision in the dispatch prompt) is confirmed sound — all 96 ctest cases and 191 pytest cases pass unchanged, including the pre-existing M4b encode_native parity tests in `tests/test_native_encoding.py` that exercise `species_sorted_bench()` indirectly.

## Issues (if FAIL)

1. DW-5.6's fourth sub-requirement (`encode_native(mirror(s))` matches `encode()` of the real opponent-POV view on a fixture) is untested — the actual test only checks vector length, not value parity, and the test's own comment admits this is out of scope for this phase's file set.
   - File: `cpp/tests/test_battle_state.cpp:181-194`
   - Demonstrated by: reading the test body directly — `REQUIRE(direct.size() == mirrored.size())` and `REQUIRE(mirrored.size() == kEncodeVectorLen)` are the only assertions; no comparison against a real Python `encode()` call on an opponent-POV fixture exists anywhere in the repo (grepped `tests/` for `mirror` — no relevant hits outside this file).
   - Fix: add a pytest-side test (in `tests/test_native_encoding.py` or a new file) that builds an opponent-POV `BattleView`/equivalent structure from the same fixture used on the C++ side, calls the real Python `encode()` on it, and compares the resulting vector element-for-element (or with `np.allclose`, matching this project's own established parity-test convention for `encode_native`) against `_native.encode_native(_native.mirror(state))`.
2. DW-5.3's "checked against Phase 3's projection" clause has no evidence in any reviewable, durable artifact (code comment, docstring, test, or committed vault note).
   - File: n/a (absence, not a specific line) — checked `cpp/`, `battle_engine/`, `scripts/`, and `/Users/edward/Projects/battle-engine/notes/`
   - Demonstrated by: grep across the above locations for "Phase 3", "570.6", "steps/s", "projection" — no hits tying the measured ms/turn number to any Phase-3-derived comparison.
   - Fix: record the comparison somewhere durable — a `notes/<slug>.md` per this project's own working-notes convention, or a code comment near `scripts/benchmark.py`'s `--n-simulations` default — stating what Phase 3 projection was used and what the resulting feasibility conclusion was.
3. `MctsPuctPlayer` (the new player class, this phase's actual runtime entry point) and `scripts/benchmark.py`'s new `"mcts_puct"` branch have zero test coverage, despite the dispatch prompt's own stated Test Coverage Level naming "player/integration wiring" as in scope.
   - File: `battle_engine/mcts_player.py:375-416`, `scripts/benchmark.py:149-158`
   - Demonstrated by: `grep -rln "MctsPuctPlayer" tests/` returns no hits; `tests/test_mcts_player.py` and `tests/test_benchmark.py` were read in full and neither references `MctsPuctPlayer` or `_make_player`/`"mcts_puct"`.
   - Fix: add tests mirroring `test_mcts_player.py`'s existing `MctsPlayer` coverage (NO_ACTION fallback, PolicyWeights loaded once and reused across `choose_move` calls, a small real `search_puct` call against a no-active-Pokemon fixture) for `MctsPuctPlayer`, and a `_make_player("mcts_puct", ...)` construction test in `test_benchmark.py`.

**Verdict: FAIL — blockers: DW-5.6's encode_native(mirror(s))-vs-opponent-POV-encode() parity claim is untested (Issue 1); DW-5.3's "checked against Phase 3's projection" clause is unevidenced (Issue 2). Issue 3 (MctsPuctPlayer/mcts_puct wiring has zero test coverage) is a real, stated-scope coverage gap that independently undermines confidence in this phase's Python-facing entry point, though it does not map to a specific numbered DW item's execution-evidence requirement the way Issues 1-2 do.**
