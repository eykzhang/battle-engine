# Review: Phase 1 - M7: MctsPlayer + benchmark wiring

## Executed Results (Step 0)
- `./scripts/pytest_native.sh` → 169 passed, 3 skipped, 0 failed (4.39s). Includes all 6 `tests/test_mcts_player.py` tests.
- `ctest --test-dir cpp/build` → 66/66 passed (0.12s), includes all 5 `test_mcts.cpp` `[mcts]`-tagged cases.
- No typecheck/lint command is defined for this project (no mypy/ruff config found); none run.

## Requirement Fulfillment

### DW-1.1
PREMISE:  `tests/test_mcts_player.py` passes — order translation correctness, `kNoAction` fallback, and the team-preview/forced-switch edge case.
EVIDENCE: `tests/test_mcts_player.py:48-193`; `battle_engine/mcts_player.py:151-181,234-245`
TRACE:    `_action_id_to_order(1, battle)` with `battle.team = {garchomp: active, dragapult: bench1, toxapex: bench2}` → `1 < NUM_SWITCH_ACTIONS(6)` → `list(battle.team.values())[1]` → `bench1` → `Player.create_order(bench1)` → asserted `order1.order is bench1` (test_action_id_to_order_switch_maps_to_team_preview_slot, passed). `_native.search` monkeypatched to return `NO_ACTION` → `choose_move` → `result.best_action == _native.NO_ACTION` → `self.choose_default_move()` → `DefaultBattleOrder`, `message == "/choose default"` (test_mcts_player_falls_back_to_default_order_on_no_action, passed). `active_pokemon=None` → `battle_state_from_poke_env` sets `my_active_slot=-1` → real `_native.search(n_simulations=30)` runs → `choose_move` returns a switch order, `isinstance(order.order, Pokemon)` and in `(bench1, bench2)` (test_mcts_player_does_not_crash_with_no_active_pokemon, passed).
VERDICT:  PASS

### DW-1.2
PREMISE:  real 500-battle benchmarks (`scripts/benchmark.py --p1 mcts`) run against random/maxdamage/heuristic/search at default format, and against `learned`/`ppo` with `--format gen9ou`, Wilson CIs, recorded in the plan's Execution Log; if ms/turn makes the full sweep exceed overnight, trim random/maxdamage/heuristic to 200 and keep search/learned/ppo at 500 — verify which was done and whether reasonable.
EVIDENCE: `/tmp/bench-mcts-vs-{random,maxdamage,heuristic,search,learned,ppo}.log`; plan Execution Log, `.code-foundations/plans/2026-08-24-battle-engine-phase4-cpp-search-core.md:506-524`
TRACE:    Cross-checked each of the six raw log files against the plan's Execution Log table line by line:
| Opponent | Log file | Log content | Plan table |
|---|---|---|---|
| random | bench-mcts-vs-random.log | 477/500, 95.4% [93.2,96.9] | 477/500, 95.4% [93.2%,96.9%] — match |
| maxdamage | bench-mcts-vs-maxdamage.log | 354/500, 70.8% [66.7,74.6] | 354/500, 70.8% [66.7%,74.6%] — match |
| heuristic | bench-mcts-vs-heuristic.log | 158/500, 31.6% [27.7,35.8] | 158/500, 31.6% [27.7%,35.8%] — match |
| search | bench-mcts-vs-search.log | 162/500, 32.4% [28.4,36.6] | 162/500, 32.4% [28.4%,36.6%] — match |
| learned | bench-mcts-vs-learned.log | 236/500, 47.2% [42.9,51.6] (printed as "vs TwoPlySearchPlayer" — correct, since `_make_player("learned", ...)` returns a `TwoPlySearchPlayer` instance per `scripts/benchmark.py:110-118`, and `run_benchmark` labels by `p1.__class__.__name__`, `battle_engine/benchmark.py:69-70`) | 236/500, 47.2% [42.9%,51.6%] — match |
| ppo | bench-mcts-vs-ppo.log | 151/500, 30.2% [26.3,34.4] (printed as "vs FrozenPolicyPlayer" — correct, `load_ppo_player` returns a `FrozenPolicyPlayer`, `battle_engine/ppo_eval.py:39-44`) | 151/500, 30.2% [26.3%,34.4%] — match |
All six numbers match exactly. File mtimes (`/tmp/bench-mcts-vs-*.log`, `ls -la`) run sequentially 01:05→02:26 (~81 min for the full six-matchup sweep), consistent with a real, non-fabricated run rather than a copy-pasted number. All six matchups were kept at full 500 battles (no trim); the plan states this was because the projected ~100-110 min fits comfortably within an overnight budget — reasonable, since 81 actual minutes is well under any overnight threshold, and the two required off-distribution opponents (`learned`, `ppo`) were correctly run at `--format gen9ou` per the DW's own distribution requirement.
VERDICT:  PASS

### DW-1.3
PREMISE:  `./scripts/pytest_native.sh` and `ctest --test-dir cpp/build` stay green.
EVIDENCE: executed directly, see Executed Results above.
TRACE:    `./scripts/pytest_native.sh` → 169 passed, 3 skipped, 0 failed. `ctest --test-dir cpp/build` → 66/66 passed, including all `[mcts]`-tagged `select_ucb1_action`, `pack_action_pair`, and `search()` tree-walk tests (lethal-move concentration, determinism, forced-switch-empty-bench terminal-leaf, root-empty-side terminal-leaf).
VERDICT:  PASS

**All requirements met:** YES

## Test-DW Coverage
- [x] DW-1.1 covered by `tests/test_mcts_player.py` (6 tests, all ran and passed in Step 0)
- [x] DW-1.2 covered by recorded observed behavior (six real benchmark runs — not automatable inside this test suite, appropriately treated as recorded observation per the plan's own design; the Execution Log's numbers were independently reconciled against raw log files rather than trusted)
- [x] DW-1.3 covered by direct execution of both commands in Step 0
- [x] Both edge cases (team-preview/forced-switch `active_pokemon is None`; `kNoAction` root fallback) have dedicated automated tests: `test_mcts_player_does_not_crash_with_no_active_pokemon`, `test_mcts_player_falls_back_to_default_order_on_no_action`

## Dead Code
None found in the touched files (`battle_engine/mcts_player.py`, `cpp/bindings/module.cpp`, `cpp/include/be/mcts.hpp`, `cpp/src/mcts.cpp`, `cpp/tests/test_mcts.cpp`, `scripts/benchmark.py`, `tests/test_mcts_player.py`, `pyproject.toml`). All imports are used; no unreachable code after early returns; no commented-out blocks or debug prints.

## Correctness Dimensions
| Dimension | Status | Evidence |
|-----------|--------|----------|
| Concurrency | PASS | `cpp/bindings/module.cpp:152-162` releases the GIL via `py::call_guard<py::gil_scoped_release>()` around the blocking `be::search` call, matching the documented need to not stall poke-env's asyncio loop for other concurrently-running battles. Argument conversion (Python `BattleState` → C++ `be::BattleState`) happens before the guard takes effect in pybind11's dispatch machinery, so no Python object is touched while the GIL is released. `MctsPlayer._rng` (a `random.Random`) is per-instance, accessed only from that instance's own single-threaded `choose_move` call — no shared-state race demonstrated. |
| Error Handling | PASS | `battle_state_from_poke_env` raises `ValueError` on a >6-Pokemon team (an explicitly documented "shouldn't be possible" upstream-bug case, unhandled by design, mirroring `encoding.py`'s precedent) — traced and consistent, no silent swallow. `kNoAction` from `search()` is explicitly checked and handled with a safe fallback (`mcts_player.py:237-244`) rather than silently propagating an invalid order. |
| Resources | PASS | `SearchNode` tree owned entirely by `unique_ptr` chains rooted in a stack-local `root_node` in `search()` — freed automatically via RAII when `search()` returns; no manual alloc/free, no leak path found. |
| Boundaries | PASS | Traced the empty-action-list case at both a plain `kDecision` node (`mcts.cpp:108-111`) and a `kForcedSwitch` node (`mcts.cpp:205-208`): both correctly treat `kNoAction` from `select_ucb1_action` as a terminal leaf (evaluate and stop, not pushed to `path`, not passed into `resolve_turn`/array-indexed) — confirmed by dedicated Catch2 regression tests (`test_mcts.cpp:232-347`) that reproduce the exact ASan-caught UB the comments describe, and both pass. Root-level empty-my_actions (`search()`'s final loop, `mcts.cpp:283-290`) correctly leaves `best_action = kNoAction`, which `MctsPlayer.choose_move` correctly detects and handles via `choose_default_move()`. |
| Security | N/A | No untrusted external input — `battle` comes from poke-env's own already-parsed protocol state, not raw network/user text. |

## Loaded-Skill Criteria

| Skill | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| cc-routine-and-class-design | LSP test: `MctsPlayer` IS-A `Player` | PASS | Only `choose_move` is overridden (`mcts_player.py:234`); no empty overrides, no strengthened preconditions, no new exceptions beyond what `Player.choose_move` callers already handle. Matches the established `FrozenPolicyPlayer` precedent in the same codebase. |
| cc-routine-and-class-design | Parameter count ≤7 | PASS | `search(root, leaf_eval, n_simulations, seed)` = 4; `select_ucb1_action(actions, stats, parent_visits, exploration_constant)` = 4; `MctsPlayer.__init__(*args, n_simulations, seed=0, **kwargs)` = 4 (variadic counted as 1 each). All well under threshold. |
| cc-routine-and-class-design | Functional cohesion | PASS w/caution (noted) | `MctsPlayer.choose_move` is functional (translate→search→translate, one operation at its declared abstraction level). `be::search()` itself (`mcts.cpp:66-292`, ~225 lines) is a single large routine handling selection, expansion, and backup for two structurally different node kinds inline, with real duplicated shape between the `kDecision` and `kForcedSwitch` branches (push-to-path / find-existing-child / create-and-insert-child, repeated). This is sequential/communicational cohesion (all steps operate on the same evolving `state`/`path`/`cur` in a required order), which the skill's classification treats as "ACCEPT w/caution," not a violation — but it is long enough, and its two branches similar enough, that a future maintainer would benefit from extracting the per-node-kind expand/backup logic into named helpers. Not a demonstrated defect; recorded as a Note, not a FAIL. |
| aposd-verifying-correctness | Requirements coverage | PASS | See DW-1.1 through DW-1.3 above — every stated requirement traces to code and an executed test/observation. |
| aposd-verifying-correctness | Boundary conditions (empty/max/invalid) | PASS | See Correctness Dimensions table — empty legal-action lists at both node kinds and at the root are traced and correctly handled, each backed by a dedicated regression test reproducing a real ASan-caught crash found during integration. |

## Notes (non-blocking)

1. **`be::search()` routine length/duplication** (confidence: high; severity: low). `cpp/src/mcts.cpp:66-292` — the `kDecision` and `kForcedSwitch` branches each independently implement "select via UCB1 → check kNoAction → push to path → look up or create child → evaluate leaf." The duplication is real but each branch has genuinely different mechanics (`resolve_turn` vs. direct switch application, `pack_action_pair` argument order, `TurnResolution` switch vs. a single `other_still_fainted` check), so a shared helper isn't a free win. Worth a look during a future refactor pass, not a defect now.

2. **`_pokemon_slot`/`_team_slots`/`_active_slot_index` rely on `Pokemon.__eq__`/list `.index()` behaving as identity comparison** (confidence: medium; severity: low). `mcts_player.py:118-122` — `mons.index(active)` assumes `active is` (or `==`s to) exactly one entry in `list(team.values())`. This holds under every test fixture and under the documented team-preview-order argument, but no test directly exercises what happens if `poke-env` ever handed back a *copy* rather than the same object reference for `active_pokemon` (would raise `ValueError` from `.index()`, uncaught, propagating out of `choose_move`). Not demonstrated as a real failure — the module's own docstring investigated and ruled this out via poke-env source, and it's outside the stated DW/edge-case scope — so this stays a Note, not a FAIL.

3. **`_team_slots` raising `ValueError` on `len(mons) > 6`** (confidence: high; severity: informational). `mcts_player.py:107-115` — unhandled by `choose_move`; a real 7+-Pokemon team would crash the bot's turn rather than degrade gracefully. Explicitly documented as an intentional "this shouldn't be possible, fail loud" choice mirroring existing precedent elsewhere in the codebase, and gen9 singles formats structurally cap teams at 6, so this isn't a demonstrated defect against any stated requirement.

4. **Benchmark log labels reflect implementation classes, not CLI names** (confidence: high; severity: none — verified correct). `bench-mcts-vs-learned.log` prints "vs TwoPlySearchPlayer" and `bench-mcts-vs-ppo.log` prints "vs FrozenPolicyPlayer" rather than "vs learned"/"vs ppo". Initially looked like a possible mislabeled/copy-pasted log, but traced to `battle_engine/benchmark.py:69-70`'s `p1.__class__.__name__ `labeling combined with `scripts/benchmark.py`'s `_make_player` wiring (`"learned"` constructs a `TwoPlySearchPlayer` with a learned eval; `"ppo"` constructs a `FrozenPolicyPlayer` via `load_ppo_player`) — correct and expected, not a bug.

## Verdict: PASS
