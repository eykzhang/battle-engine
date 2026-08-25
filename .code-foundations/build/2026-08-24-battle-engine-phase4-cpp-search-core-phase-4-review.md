# Review: Phase 4 (M4b) - encode_native() C++ port + MctsPlayer translator extension

## Executed Results (Step 0)

Build was stale (`.so` timestamp Aug 25 03:39:55, but `battle_state.hpp`/`module.cpp` timestamps Aug 25 03:42:xx — edited after the last build). Rebuilt before trusting any results:

- `./scripts/build_cpp.sh` → clean rebuild, no warnings, `_native.cpython-313-darwin.so` relinked.
- `ctest --test-dir cpp/build` → **76/76 passed** (100%), 3.32s. Includes all of `test_eval.cpp` (`hazard_score`, `type_matchup_score`, `speed_control_score` — default_eval's own tests) unaffected.
- `./scripts/pytest_native.sh -q -rs` → **191 passed, 3 skipped** (5.46s → 3.96s on rerun). The 3 skips are all `tests/test_dataset.py`/`tests/test_encoding.py` cases gated on a fetched replay sample (`data/replays_raw`), unrelated to this phase.
- `./scripts/pytest_native.sh tests/test_native_encoding.py tests/test_native_legality.py tests/test_mcts_player.py -v` → **19/19 passed**, including `test_encode_native_length_matches_ppo_bin_header_vector_len` (not skipped — `data/cpp_weights/ppo.bin` exists at 751,732 bytes on disk).

## Requirement Fulfillment

### DW-4.1
PREMISE: `tests/test_native_encoding.py` — `np.allclose(encode_native(state), encode(view))` on real battle states (live-battle fixtures, reusing the `conftest.make_mon`/`SimpleNamespace`-battle-fixture convention already established in `test_native_legality.py`/`test_mcts_player.py`).
EVIDENCE: `tests/test_native_encoding.py:62-77` (`_assert_parity`), exercised by 6 distinct fixtures (lines 85, 104, 123, 137, 154, 170) covering full team, hazards/weather/terrain/boosts, fainted bench, item/ability/protect_counter (including Heavy-Duty Boots/Levitate/Wonder Guard and an out-of-vocab item), a fainted opponent teammate, and bench species-sort order.
TRACE: `_battle(...)` → `battle_state_from_poke_env` (native) and `battle_view_from_poke_env` (python) built from the identical `SimpleNamespace` → `_native.encode_native(native_state)` vs `encode(python_view)` → `np.allclose` on the full 700-ish-dim vector. Ran via `./scripts/pytest_native.sh tests/test_native_encoding.py -v`: all 6 parity tests PASSED.
VERDICT: PASS

### DW-4.2
PREMISE: `tests/test_native_legality.py`/`test_mcts_player.py` still pass unchanged (extension doesn't regress `default_eval`'s existing fields).
EVIDENCE: `git diff --stat` confirms neither file is in this phase's changeset. `battle_state.hpp`/`battle_state.cpp` diffs (`git diff HEAD`) show every M4-era field, `is_valid`, and `slot_invariants_check` byte-identical — all additions are new fields/functions appended after the M4 code. `cpp/src/eval.cpp` (default_eval, `hazard_score`, `type_matchup_score`, `speed_control_score`) is untouched by this diff.
TRACE: Ran `test_native_legality.py` (4 tests) and `test_mcts_player.py` (6 tests) against the freshly rebuilt extension — all 10 PASSED. `ctest`'s `test_eval.cpp` suite (hazard_score/type_matchup_score/speed_control_score, part of the 76) also passed.
VERDICT: PASS

### DW-4.3
PREMISE: `encode_native()`'s output length equals both `encoding.VECTOR_LEN` (Python-side) and `data/cpp_weights/ppo.bin`'s header `vector_len` field — a 3-way cross-check.
EVIDENCE: `tests/test_native_encoding.py:208-216` (`test_encode_native_length_matches_vector_len`, asserts `len(result) == enc.VECTOR_LEN == _native.ENCODE_VECTOR_LEN`) and `:223-231` (`test_encode_native_length_matches_ppo_bin_header_vector_len`, asserts `len(...) == enc.VECTOR_LEN == header_vector_len` where `header_vector_len` comes from `_native.PolicyWeights.load(_PPO_BIN).actor.layer0.in_dim`).
TRACE: `data/cpp_weights/ppo.bin` exists (751,732 bytes) so the second test ran rather than skipped. Both tests PASSED, giving the actual 3-way cross-check (encoding.VECTOR_LEN, `_native.ENCODE_VECTOR_LEN`, and the real ppo.bin header), not just the first two legs.
VERDICT: PASS

**All requirements met:** YES

## Test-DW Coverage
- [x] DW-4.1 — 6 automated parity tests, ran in Step 0.
- [x] DW-4.2 — 10 automated tests (existing, unmodified files), ran in Step 0.
- [x] DW-4.3 — 2 automated tests (VECTOR_LEN cross-check + ppo.bin header cross-check), both ran (not skipped).
- [x] Edge case 1 (unrevealed opponent slots → sentinels) — covered implicitly by every parity test whose team fixtures are smaller than 6 (the untouched slots pad to default-constructed `PokemonSlot{}`, i.e. `species/item/ability == ""`, `revealed == false`) and explicitly by the fainted-opponent-teammate test's unencoded-opponent-bench path. All pass.
- [x] Edge case 2 (team-preview / no active mon → ValueError) — `test_encode_native_raises_when_my_side_has_no_active_pokemon` (line 241) PASSED, `pytest.raises(ValueError, match="team preview")` matches the actual thrown message's `"(not team preview)"` substring.

## Dead Code
None found. Grepped all five in-scope files for `TODO`/`FIXME`/`XXX`/stray prints — no hits. No unreachable code after early returns (`encode_pokemon_slot`'s `!revealed` early-return is a standard guard, not dead code below it).

## Correctness Dimensions

| Dimension | Status | Evidence |
|-----------|--------|----------|
| Concurrency | N/A | No new shared mutable state; `encode_native()` is a pure function of its `state` argument. |
| Error Handling | PASS | `encode_native()`'s boundary throws `std::invalid_argument` on either side missing an active mon (battle_state.cpp:345-350), mirroring `battle_view_from_poke_env`'s `ValueError`; verified pybind11 auto-translates to Python `ValueError` via the passing test. `_team_slots` (unmodified, pre-existing) still raises on >6-Pokemon teams. |
| Resources | N/A | No new file handles/connections/locks; the only I/O-adjacent code (`PolicyWeights::load`) is unmodified by this phase. |
| Boundaries | PASS | Traced the two hardest boundary cases by hand (see Loaded-Skill Criteria row below for the tie-break trace) — both match `encoding.py`'s reference exactly. Item vocab overflow (`"other known item"` bucket) and bench-padding-below-6 are both exercised by passing tests. |
| Security | N/A | No untrusted/external input in the reviewed diff — all inputs come from a live poke-env `AbstractBattle` object already trusted elsewhere in this codebase. |

## Loaded-Skill Criteria

| Skill | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| aposd-designing-deep-modules | Interface depth: `encode_native(state)` should hide substantial complexity behind a small interface | PASS | One function, one parameter, no caller-side wrapper logic needed; hides a ~700-line, bit-for-bit Python port (hazard tie-break, type-immunity abilities, item vocab, move-summary layout) entirely inside `battle_state.cpp`'s anonymous namespace. Genuinely deep. |
| aposd-designing-deep-modules | Information hiding: MoveSummary precomputation shouldn't leak C++-side derivation logic into the translator | PASS | `_move_summary_to_native` (mcts_player.py:146) calls `encoding.py`'s own `_move_summary_features` directly rather than re-deriving movedex flag logic — confirmed by import (`from battle_engine.encoding import _move_summary_features, ...`, line 62-67) and call site (line 156). Divergence between `encode()`'s live adapter and this translator is structurally impossible since both call the identical function. |
| cc-defensive-programming | External-boundary validation uses error handling, not assertions | PASS | `encode_native()`'s "no active Pokemon" check throws `std::invalid_argument` (a real error-handling boundary), while the two `assert(vec.size() == ...)` self-checks (battle_state.cpp:311, 379) are pure internal-invariant checks with no side effects inside the assert — correctly classified per the assertion-vs-error-handling table (internal bug vs. external/anticipated condition). |
| cc-defensive-programming | No empty catch blocks / no executable code in assertions | PASS | No `catch` blocks introduced by this diff. Both new `assert()` calls are non-side-effecting size comparisons. |
| aposd-verifying-correctness | Requirements coverage — every DW item has code + a passing test | PASS | See DW-4.1 through DW-4.3 above; each has an explicit test that ran and passed in Step 0. |

**Hand-trace demonstrating the hazard tie-break port (additional requirement (b))**: constructed `stealth_rock_turn = 5, tailwind_turn = 5` (a genuine tie between two turn-tracked conditions, not exercised by any existing fixture — the existing tests only exercise a single active turn-tracked hazard, or two turn-tracked hazards at *different* turns). Python's `_poke_env_hazards` (encoding.py:554-583) builds `turn_tracked = {stealthrock: 5, tailwind: 5}` in `_HAZARD_SIDE_CONDITIONS` insertion order (stealthrock, spikes, toxicspikes, stickyweb, reflect, lightscreen, auroraveil, tailwind, filtered to non-stackable → stealthrock, stickyweb, reflect, lightscreen, auroraveil, tailwind), then `max(turn_tracked, key=turn_tracked.get)` — CPython's `max()` only replaces the running maximum on strict `>`, so ties resolve to the *first*-encountered key, i.e. `stealthrock`. `most_recent_hazard_index` (battle_state.cpp:176-194) iterates the identical fixed order (`{0,stealth_rock_turn},{3,sticky_web_turn},{4,reflect_turn},{5,light_screen_turn},{6,aurora_veil_turn},{7,tailwind_turn}`) with `if (e.turn > best_turn)` — also strict `>`, so `stealthrock` (processed first, sets `best_turn=5`) is not overwritten when `tailwind` is later seen at the same value 5. Both algorithms independently resolve the tie to `stealthrock`. Verified `STACKABLE_CONDITIONS = {SPIKES, TOXIC_SPIKES}` directly against the installed poke-env source (`side_condition.py:95`), matching the header comment's claim exactly. The stackable-only fallback order (spikes before toxicspikes) was traced the same way and matches. This tie/stackable-interaction path is not covered by an automated test (see Notes) but the code is structurally identical to the Python reference at every branch point, so I'm confident in the trace.

## Notes (non-blocking)

- **Untested corner of the hazard reduction**: no automated test exercises (a) two turn-tracked hazards tied at the *same* turn number, or (b) both Spikes and Toxic Spikes active simultaneously with *no* turn-tracked hazard active (to confirm Spikes wins the stackable-only fallback). Both were hand-traced above and match the Python reference exactly, and neither is a DW item or a listed edge case, so this is not a FAIL — but it's the one place in this port where "the parity tests pass" doesn't by itself prove the tie-break logic, and a dedicated regression test would be cheap (`stealth_rock_turn == tailwind_turn` fixture) if a future change to this function needs a safety net.
- **Team-preview guard only tested for "my side" missing active**: `encode_native()`'s guard is `my_active_slot < 0 || opp_active_slot < 0`, a single boolean check with no divergent logic per operand. Only the `my_active_slot == -1` branch has a dedicated test (`test_encode_native_raises_when_my_side_has_no_active_pokemon`). Since both operands share identical code (same comparison, same throw), this is low-risk, but strictly speaking the `opp_active_slot == -1`-only case isn't independently exercised.
- **Per-simulation copy cost**: `mcts.cpp:75`'s `BattleState state = root;` deep-copies the full state (including the new `std::string species/item/ability` and `MoveSummary` fields) at the start of every MCTS simulation. Not a correctness issue and not something this milestone's DW items address, but worth knowing if a future latency budget gets tight — M4's struct was POD-ish (no heap-allocating members); M4b's isn't.
- `_active_slot_index` (mcts_player.py, pre-existing/unmodified) uses `list(team.values()).index(active)`, relying on `Pokemon.__eq__` defaulting to identity — fine since `active` is always the same object reference already present in `team.values()`, but this is inherited code outside this phase's scope, noted only for completeness.
- The `kTypeToAllTypesIndex` permutation table (battle_state.cpp:52-71) was independently re-verified against a live `list(PokemonType)` dump from the installed poke-env package (not just trusted from the comment) — all 18 entries correct.
- `kStatusToStatusesIndex`, `kItemVocab`, and the M4b `kEncode*` breakdown constants in `battle_state.hpp` were all cross-checked field-by-field against `encoding.py`'s `_STATUSES`, `_ITEM_VOCAB`, and `_POKEMON_VEC_LEN`/`VECTOR_LEN` — all match, and the runtime length-equality tests (DW-4.3) give this direct execution backing beyond the manual check.

## Issues (if FAIL)
None.

**Verdict: PASS**
