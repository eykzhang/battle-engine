# Review: Phase 5 - M6b PUCT search with PPO prior/value (attempt 2)

## Executed Results (Step 0)

- Clean rebuild: `rm -rf cpp/build && ./scripts/build_cpp.sh` -> succeeded, zero compiler warnings.
- `ctest --test-dir cpp/build` -> **96/96 passed** (matches the reported "unchanged, no C++ source touched in the fix" claim exactly). Re-ran after clean rebuild: still 96/96.
- `./scripts/pytest_native.sh` -> **196 passed, 3 skipped** (matches the reported "up from 191, +5 new tests, 0 skipped among them" claim). Re-ran after clean rebuild: still 196/3.
- `data/cpp_weights/ppo.bin` is present in this worktree, so every `pytest.mark.skipif(not Path(...).exists())`-gated test (the mirror-parity test, the `mcts_puct` benchmark-CLI test, the real `search_puct` player test) actually executed rather than skipping - verified by running `tests/test_native_encoding.py -v` and `tests/test_benchmark.py -v` individually and reading the per-test PASSED lines, not just the summary count.
- Independently re-ran `cpp/build/tests/be_tests "[mcts][puct][!benchmark]"` directly: measured 504.03ms mean (100 samples), consistent with the note's claimed 501.9ms / independently-reproduced 501.947ms, within run-to-run noise.

## Requirement Fulfillment

### DW-5.1
PREMISE:  Catch2 tests for the Metamon-mapping functions (known team -> known mapping, fainted-teammate/bench<5, tera 9-12 dropped+renormalized) and `select_puct_action` (synthetic bandit, prior pulls selection toward high-prior arms early).
EVIDENCE: `cpp/tests/test_action.cpp:154-221` (known-team mapping, fainted-teammate, bench<5); `cpp/tests/test_mcts.cpp:367-375` (prior wins among unvisited arms), `:427-471` (tera-label drop+renormalize).
TRACE:    `metamon_switch_label_to_action_id(state, 6)` on a 3-real-member team -> `-1` (test_action.cpp:214, `bench<5` case); `puct_priors_from_actor_logits` with `logits[9]=1000` and only move-label-0 legal -> `priors == [1.0]` (test_mcts.cpp:453-471, tera mass dropped not diluted); `select_puct_action` with priors `[0.1, 0.9]` on two unvisited arms, `parent_visits=1` -> picks index 1 (test_mcts.cpp:367-375).
VERDICT:  PASS

### DW-5.2
PREMISE:  fixed-seed determinism holds for the PUCT search.
EVIDENCE: `cpp/tests/test_mcts.cpp:516-527`.
TRACE:    Two independent `search_puct(state, weights, 40, seed=42)` calls on separately-constructed but field-identical states -> `result_a.best_action == result_b.best_action` and identical sorted visit distributions. Ran in ctest (test #passed, part of the 96/96).
VERDICT:  PASS

### DW-5.3
PREMISE:  measured ms/turn at the chosen `n_simulations`, checked against Phase 3's projection with the comparison recorded somewhere durable.
EVIDENCE: `cpp/tests/test_mcts.cpp:676-684` (the `[!benchmark]` case); `/Users/edward/Projects/battle-engine/notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md` (main checkout, untracked `notes/`, per this project's own convention - confirmed to exist there, dated 2026-08-25, `type: log`).
TRACE:    Ran `./be_tests "[mcts][puct][!benchmark]"` directly -> 504.03ms mean (100 samples), consistent with the note's 501.9ms/501.947ms. Re-verified the note's arithmetic independently: `15,000 x 0.501947 s = 7,529.205 s = 125.4868 min = 2.0914 hours` - matches the note's "~2.1 hours" and lands comfortably inside Phase 3's DW-3.3 projected 6-9 hour budget. The note correctly identifies it did not re-verify the "~15 mcts_puct turns/battle" estimate as a separate measurement (labeled a rough estimate, not overclaimed as measured).
VERDICT:  PASS

### DW-5.4
PREMISE:  the critic's sign-backup convention is checked empirically on N synthetic mirrored state pairs, with a stated fallback if ambiguous.
EVIDENCE: `cpp/tests/test_mcts.cpp:618-667` (N=30 synthetic pairs, real checkpoint); `cpp/include/be/mcts.hpp:414-445` (doc comment recording the measured result and the convention chosen).
TRACE:    Ran the ctest suite -> the DW-5.4 test case passes and its `INFO` output (visible with `-s`) reports `mean(v(s)+v(mirror(s)))` close to the doc comment's claimed -0.0659 (not independently re-verified to 4 decimal places here, but the test executes and the mechanism - `weights.critic.forward(encode_native(state))[0]` vs `...(mirror(state))...` summed across 30 varied HP/status/boost/species/type pairs - is real, not a stub). The "empirically checked on N pairs" clause is fully satisfied.
          The "with a stated fallback if ambiguous" clause is NOT satisfied anywhere in the deliverable. The plan (`.code-foundations/plans/2026-08-24-battle-engine-phase4-cpp-search-core.md:364-366`) explicitly specifies what this fallback must be: "If neither holds clearly within tolerance ..., default to `-v`" - and that same plan's Test Coverage checklist (line 451) names it as a required sub-item: `incl. the "neither holds" fallback`. Searched `cpp/include/be/mcts.hpp`, `cpp/src/mcts.cpp`, and `cpp/tests/test_mcts.cpp` for any statement of what convention would be used if the measured mean were NOT clearly closer to 0 or to 1 (grepped for "ambiguous", "neither", "fallback", "conservative", "default to" - none found outside the plan file itself). `mcts.hpp`'s doc comment (lines 414-445) only narrates the actual, unambiguous measured result and the convention it selected; it never states what would have been implemented had the measurement come out ambiguous. The code's runtime behavior happens to already match the plan's specified fallback (`score_leaf_puct` in `mcts.cpp:135-142` hardcodes `{v, -v}` unconditionally, which is also what "default to -v" would produce) - but nothing in the codebase *states* this as the ambiguity contingency; a future retrained checkpoint with an ambiguous measurement would silently keep using `-v` with no code or test documenting that this is a deliberate fallback rather than an unexamined leftover.
VERDICT:  FAIL - the empirical-check sub-clause passes; the "stated fallback if ambiguous" sub-clause has no evidence anywhere in the codebase.

### DW-5.5
PREMISE:  a `kForcedSwitch` node during search correctly uses UCB1-only selection and continues expansion past it to a real `kDecision` state for the critic's leaf value.
EVIDENCE: `cpp/src/mcts.cpp:291-300` (UCB1-only selection in the `else` branch, no priors table exists at a `kForcedSwitch` node); `:352-365` (continuation past a freshly-created `kForcedSwitch` child, `cur = cur->children[picks].get(); continue;`); `cpp/tests/test_mcts.cpp:561-581` (a real-replacement fixture, 60 sims, non-`kNoAction` best_action).
TRACE:    `search_puct` on a state where `opp_team[0]` is at 0.01 HP with only damaging/`protect` moves and `opp_team[1]` (Ivysaur) is a real revealed replacement, 60 sims seed=11 -> `result.best_action != kNoAction`, `total_visits == 60` (every simulation reached a real leaf, none silently dropped). Traced the code path: line 297-300 selects via `select_ucb1_action` (no `priors` argument passed at all, structurally cannot use PUCT at this node); line 337-350 determines whether a second forced-switch is owed or a real `kDecision` node should be populated via `populate_decision_node` (which runs the critic); line 352-357 continues descent rather than stopping.
VERDICT:  PASS

### DW-5.6
PREMISE:  `mirror(BattleState)` round-trips, every `my_*`/`opp_*` field pair is swapped, weather/terrain unchanged, and `encode_native(mirror(s))` matches `encode()` of the real opponent-POV view on a fixture.
EVIDENCE: `cpp/src/battle_state.cpp:361-369` (`mirror()` implementation); `cpp/tests/test_battle_state.cpp:152-191` (round-trip, field-swap, weather/terrain-unchanged, C++-side length check); `tests/test_native_encoding.py:217-270` (the value-level parity test).
TRACE:    Ran `tests/test_native_encoding.py::test_encode_native_mirror_matches_python_encode_of_real_opponent_pov` directly -> PASSED. Read the fixture construction line-by-line: `opponent_pov_battle` sets `my_team=[opp_active, opp_bench]`, `opp_team=[my_active, my_bench]`, `my_hazards=`(the original's `opp_hazards`), `opp_hazards=`(the original's `my_hazards`), weather/fields unchanged - every role genuinely swapped, not relabeled. `np.allclose(mirrored_vec, python_vec)` passes on a fixture with two genuinely distinct teams (Garchomp/Dragapult vs. Landorus-T/Toxapex, with boosts, item, ability, hazards, weather, terrain all populated and asymmetric between sides) - a no-op or partially-wrong `mirror()` would not coincidentally satisfy this given the teams differ.
VERDICT:  PASS

**All requirements met:** NO - DW-5.4's "stated fallback if ambiguous" sub-clause is undocumented anywhere.

## Test-DW Coverage
- [x] DW-5.1, 5.2, 5.3, 5.5, 5.6 all have corresponding executed tests (ctest and/or pytest_native).
- [x] DW-5.4's "checked empirically" clause has an executed test (`cpp/tests/test_mcts.cpp:618`).
- [ ] DW-5.4's "stated fallback if ambiguous" clause has no test and no documentation anywhere - not coverable by a test in the strict sense (the measurement wasn't ambiguous), but the plan explicitly requires it to be *stated*, which is a documentation deliverable this attempt still omits.
- [x] Coverage matches the stated "Targeted" level for every other item.

## Fix Verification (the three re-dispatched issues)

1. **DW-5.6 parity fix - CONFIRMED CLOSED.** `tests/test_native_encoding.py:217-270` is a genuine value-level `np.allclose` comparison against the real Python `encode()`, not a length check. Ran it directly: PASSED. The opponent-POV `BattleView` is constructed by actually swapping every field (team, active, hazards) on a `SimpleNamespace`-wrapped real `poke_env` battle object, not by relabeling the same one. Confirmed non-trivial (the two teams/hazards differ, so a broken `mirror()` would fail this test, not pass it vacuously).
2. **DW-5.3 comparison fix - CONFIRMED CLOSED.** `/Users/edward/Projects/battle-engine/notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md` exists in the main checkout (not the worktree), dated 2026-08-25, `type: log`, referencing Phase 3's DW-3.3 commit `aa4c485`. Re-derived its arithmetic independently: `15,000 x 0.501947 = 7,529.205s = 2.0914 hours` - matches the note's stated "~2.1 hours" conclusion. Correct.
3. **Test coverage fix - CONFIRMED CLOSED.** `tests/test_mcts_player.py:250-326` adds real `MctsPuctPlayer` coverage: `NO_ACTION` fallback to `DefaultBattleOrder` (monkeypatched `search_puct`), weights loaded once and reused across two `choose_move()` calls (identity check via a `sentinel_weights` object, not equality), and a real, unmocked `search_puct()` call against `data/cpp_weights/ppo.bin` returning a non-default `BattleOrder`. `tests/test_benchmark.py:73-89` adds `test_make_player_mcts_puct_constructs_a_real_mcts_puct_player`, confirmed to actually execute (not skip) since `data/cpp_weights/ppo.bin` exists in this worktree - verified via `pytest -v`.

## Edge Cases

- **A legal `ActionId` with no corresponding Metamon label (switches when bench < 5).** Traced: for a legal switch `ActionId`, `action_id_to_metamon_label` always finds a match in `species_sorted_bench_slots()`'s own output (by construction - `legal_actions()`'s switch-legality set is exactly that function's inclusion set), confirmed directly by `test_action.cpp:199-221`'s bench<5 fixture, where the two real legal actions (1, 2) both round-trip to real labels (4, 5) even though the team only has 3 revealed members. Handled correctly - this scenario cannot fire for a legal `ActionId` by construction, and the test demonstrates the invariant rather than merely asserting it.
- **The reverse: a Metamon label (tera, 9-12) with no `ActionId` counterpart - dropped and renormalized, not an error.** Confirmed via `test_mcts.cpp:453-471`: a tera label given `logits[9]=1000` (would dominate any naive full-13-way softmax) is simply never gathered by `puct_priors_from_actor_logits`, and the one real legal label gets probability 1.0 - correctly dropped and renormalized, not folded in and not an error.
- **`kForcedSwitch` nodes: UCB1-only selection, expansion continues, no crash, no invented encoding.** Confirmed via code trace (`mcts.cpp:291-365`) and two dedicated tests (`test_mcts.cpp:529-559` no-replacement -> falls back to `default_eval` terminal leaf; `:561-581` real-replacement -> continues to a real `kDecision` state). "No invented encoding" verified: `populate_decision_node`/`has_encodable_actives` is never called on the `kForcedSwitch` node's own (fainted-active) state - only after a real switch has been applied and the resulting state is checked for encodability.

## Dead Code
None found in the reviewed diff. `scripts/benchmark.py:197`'s `print(result)` is normal CLI output, not a debug leftover. No unreachable code after early returns; no unused imports found in the touched files. Compiler build produced zero warnings on a clean rebuild.

## Correctness Dimensions
| Dimension | Status | Evidence |
|-----------|--------|----------|
| Concurrency | N/A | Single-threaded C++ search; `search_puct`'s pybind11 binding releases the GIL (`module.cpp:291`) around the call as `mcts.hpp`'s own doc comment requires, but there is no shared mutable state across concurrent calls to reason about beyond that. |
| Error Handling | PASS | `select_ucb1_action`/`select_puct_action` both handle empty `actions` via `kNoAction` (traced, tested at `test_mcts.cpp:82`, `:411`); `encode_native` throws `std::invalid_argument` on an unset active (tested, `test_native_encoding.py:312-326`); `has_encodable_actives` additionally guards the fainted-but-index-valid case that `encode_native`'s own check misses (traced against `forward_model.cpp`'s confirmed behavior of never resetting `active_slot` to -1 on faint). |
| Resources | N/A | No new resource acquisition in this phase's diff; `PolicyWeights::load` is unchanged, called once at `MctsPuctPlayer` construction (verified via the identity-check test). |
| Boundaries | PASS | Traced `select_ucb1_action`'s `log(parent_visits)` for a `parent_visits==0` divide/log(0) risk: structurally unreachable, since `parent_visits` is the sum of all `stats[i].visits`, so `parent_visits==0` implies every `stats[i].visits==0`, which the function returns on before reaching the `log()` call. `pack_action_pair`'s `int8_t -> uint8_t` cast for `kNoAction=-1` traced and confirmed to produce a stable, collision-free bit pattern (tested at `test_mcts.cpp:98`). |
| Security | N/A | No untrusted/external input in this phase's diff - all inputs are internal `BattleState`/weights data. |

## Loaded-Skill Criteria

| Skill | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| aposd-designing-deep-modules | `SearchNode`'s tagged-enum design vs. a `std::variant` split-type: is the interface deep, or does complexity leak to callers? | PASS | The header's own doc comment (`mcts.hpp:59-72`) gives a genuine, verified technical reason (recursive-`variant`-with-`unique_ptr`-members incomplete-type problem) for the chosen design, not just asserted preference. `populate_decision_node` (`mcts.cpp:167-185`) is a real shared-logic extraction avoiding duplication between the two call sites that need it - a deep-module instinct applied correctly. |
| aposd-designing-deep-modules | Is `search_puct` a shallow pass-through or does it hide real complexity (tree walk, backup, priors) behind a 4-argument interface? | PASS | Interface is `(root, weights, n_simulations, seed) -> SearchResult` - 4 inputs, hides the entire open-loop tree/DUCT/PUCT machinery. Depth is high relative to interface size. |
| cc-routine-and-class-design | Parameter counts across the touched public functions. | PASS | `select_puct_action` (5 params), `search_puct` (4), `puct_priors_from_actor_logits` (3), `populate_decision_node` (3) - all well under the 7-param threshold. |
| cc-routine-and-class-design | `MctsPuctPlayer`/`MctsPlayer` inheritance (both `Player` subclasses) - LSP check. | PASS | Both override only `choose_move`, no empty overrides, no strengthened preconditions beyond `Player`'s own contract - "is-a Player" holds literally (both are usable anywhere a `Player` is expected). |
| cc-routine-and-class-design | `search_puct`'s main loop cohesion - is the ~200-line function doing "one operation" (functional cohesion) or several unrelated ones? | PASS w/caution | The function is long (`mcts.cpp:189-396`) but is a single MCTS simulation loop at its declared abstraction level ("run one PUCT search"); the two large `if (cur->kind == kDecision) {...} else {...}` branches are sequential/communicational (same tree-walk state, different node shapes), not logically unrelated operations selected by a control flag - closer to "ACCEPT w/caution" than a violation, and this shape (two node kinds needing genuinely different selection logic within one tree-walk loop) is inherent to the open-loop DUCT/PUCT design this phase committed to, not an incidental sprawl. Not flagged as a FAIL since no DW item names function length and the structure is traceable, not tangled - noted here as a legitimate design-quality observation rather than a demonstrated defect. |
| aposd-verifying-correctness | Boundary/error-handling dimensions (see Correctness Dimensions table above) | PASS | See table. |

## Notes (non-blocking)

- `search_puct`'s tree-walk function is long (~200 lines) with duplicated-looking blocks between the `kDecision`-branch's "reached a new node" logic and the `kForcedSwitch`-branch's own version of the same shape (create-node / determine-kind / continue-or-backup). Both blocks are real but structurally parallel; a future refactor could extract a shared "create and possibly continue past a new child" helper the way `populate_decision_node` already was extracted. Not a defect - no DW item requires it, and the parallel structure is easy to trace correctly (verified above), just an opportunity.
- DW-5.4's residual (~0.066, stddev 1.5e-8) is explained mechanically in `mcts.hpp`'s own doc comment (encoding asymmetry between POVs, opponent bench never encoded) - a good, falsifiable explanation rather than a hand-wave, and consistent with the measured tightness of the stddev.
- The `notes/phase-5-mcts-puct-ms-per-turn-vs-phase-3-projection.md` note correctly flags that its own number is from a Debug/ASan build, not `--release`, and states (without re-measuring) that a release build should only make the comparison more favorable. This is honest under-claiming, not a gap - the 2.1-hour number already clears the 6-9 hour budget by a wide margin even at the conservative number.
- `CLAUDE.md` in this worktree shows a large diff (1156 lines) unrelated to Phase 5's own scope - the plan's own Notes section already flags this as "a stray staged deletion... worth the user's own attention before committing," not something this review needs to act on.

## Issues (if FAIL)

1. DW-5.4's "stated fallback if ambiguous" sub-clause has no evidence anywhere in the codebase.
   - File: `cpp/include/be/mcts.hpp:414-445` (the doc comment that documents the measured result but not the ambiguity contingency); `cpp/tests/test_mcts.cpp:618-667` (the test that measures but doesn't state what would follow from an ambiguous result); confirmed absent via grep across `cpp/include/be/mcts.hpp`, `cpp/src/mcts.cpp`, `cpp/tests/test_mcts.cpp` for "ambiguous"/"neither"/"fallback"/"conservative"/"default to".
   - Demonstrated by: absence of any statement of the ambiguity-fallback convention, despite the plan itself (`.code-foundations/plans/2026-08-24-battle-engine-phase4-cpp-search-core.md:362-366`) explicitly specifying what it must say ("If neither holds clearly within tolerance..., default to `-v`") and explicitly listing it as a required Test Coverage sub-item (line 451: `incl. the "neither holds" fallback`).
   - Fix: add one sentence to `mcts.hpp`'s `search_puct()` doc comment (or the DW-5.4 test's own comment block) stating explicitly that the implementation defaults to `-v` regardless of measurement ambiguity, and that this is a deliberate contingency (matching the plan's own stated fallback), not merely an artifact of this checkpoint's clean measurement. Low effort, low risk - no code change needed, since the runtime behavior already implements the correct fallback; only the documentation is missing.

**Verdict: FAIL - DW-5.4's "stated fallback if ambiguous" sub-clause is undocumented anywhere in the codebase (Issue 1). All three previously-failed issues (DW-5.6 parity, DW-5.3 comparison, MctsPuctPlayer/benchmark test coverage) are genuinely fixed and independently re-verified. Every other Done-When item, every named edge case, and every loaded-skill criterion pass with execution evidence.**
