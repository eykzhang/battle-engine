# Discovery + Design: Phase 1 - M7 — MctsPlayer + benchmark wiring

## Files Found
- `cpp/include/be/mcts.hpp` / `cpp/src/mcts.cpp` — `search()`, `default_eval`, `SearchResult`, `kNoAction` all implemented and Catch2-tested (64 passing tests, confirmed by `test_mcts.cpp`'s real `search()` calls).
- `cpp/bindings/module.cpp` — binds `Type`/`Status`/`Side`/`StatBlock`/`SideConditions`/`PokemonSlot`/`BattleState`, `is_valid`, `legal_actions`, and the `NUM_SWITCH_ACTIONS`/`MOVE_ACTION_OFFSET`/`NUM_MOVE_ACTIONS` constants. No `mcts.hpp` include, no `search`/`SearchResult`/`kNoAction` binding yet.
- `battle_engine/mcts_player.py` — has `battle_state_from_poke_env` (M5 translator) only; no `Player` class yet, per its own module docstring.
- `battle_engine/ppo_eval.py` / `battle_engine/self_play.py` — `load_ppo_player`/`FrozenPolicyPlayer` are the `Player`-wrapping pattern to mirror: `__init__(*args, **kwargs)` → `super().__init__`, `choose_move(battle) -> BattleOrder` built from a translated action id, `self.create_order(...)`.
- `scripts/benchmark.py` — `PLAYERS` dict + `CHOICES`, `_make_player(name, format, model_path, ppo_model_path)`. `"ppo"`/`"learned"` are special-cased outside `PLAYERS`; `_AUTO_TEAM_FORMATS = {"gen9randombattle"}` gates whether a team is built.
- `tests/test_mcts_player.py` — does not exist yet (this phase creates it).
- `tests/test_native_legality.py` — established fixture pattern: `SimpleNamespace`-wrapped real poke-env `Pokemon` objects via `conftest.make_mon`, `_battle(...)` helper building `team`/`opponent_team`/`active_pokemon`/`opponent_active_pokemon`/`side_conditions`/`opponent_side_conditions`. Explicitly notes root ActionId→BattleOrder translation is *not* tested there — it's this phase's job.
- `poke_env/player/player.py` — `Player.create_order(Move|Pokemon) -> SingleBattleOrder` (staticmethod), `Player.choose_default_move() -> DefaultBattleOrder` (staticmethod, `"/choose default"`) — both callable without a live connection/instance state.
- `poke_env/player/battle_order.py` — `SingleBattleOrder.order` exposes the underlying `Move`/`Pokemon` for test assertions.

## Current State
M0–M6 (open-loop DUCT/MCTS core) are done and tested at the C++ layer only. `search()` takes `(root, leaf_eval, n_simulations, seed)` where `leaf_eval` is a `std::function<float(const BattleState&)>` — nothing Python-facing exists. `mcts_player.py`'s translator (`battle_state_from_poke_env`) already builds a `BattleState` whose `my_team`/`opp_team` order is `list(battle.team.values())` — the same order `action.hpp`'s ActionId 0-5 switch scheme assumes (dict-insertion-stable, per the module's own verified docstring), so the switch-id→order mapping is a direct index, not a species-sort translation like `action_space.py`'s (that module is Phase 2/3's different 13-way Metamon scheme, not this phase's fixed 0-9 scheme — do not reuse its species-sort logic here).

## Gaps
- No pybind11 binding for `search`/`SearchResult`/`kNoAction` — must add to `module.cpp`, `#include "be/mcts.hpp"`.
- Plan requires the exposed Python `search()` to fix `leaf_eval=default_eval` C++-side and never accept a Python callable (GIL-release safety) — the raw `be::search` signature must NOT be bound directly (pybind11's `functional.h` would silently accept a Python callable for the `EvalFn` parameter and crash when called GIL-released). A lambda wrapper is required.
- `MctsPlayer` class does not exist.
- `"mcts"` not in `scripts/benchmark.py`'s `PLAYERS`/`CHOICES`, no `--n-simulations` CLI plumbing.
- `tests/test_mcts_player.py` does not exist.
- No measured `n_simulations`/ms-per-turn number yet (DW-1.2's prerequisite).

## Code Standards
Applies directly: name every simplification (why `search()`'s Python binding pins `default_eval`, why the switch-id mapping is index-not-species-sort), `snake_case`/`kCamelCase` C++ naming already established, `py::gil_scoped_release` required per `mcts.hpp`'s own latency note, project headers before `<pybind11/...>` ordering already followed in `module.cpp`, one `TEST_CASE`-per-behavior discipline mirrored in pytest as one `test_` function per behavior with a descriptive name. `tests/test_mcts_player.py` must `pytest.importorskip("battle_engine._native")` first, matching `test_native_legality.py`.

## Test Infrastructure
pytest with `SimpleNamespace` battle fixtures wrapping real `poke_env.Pokemon` objects via `conftest.make_mon`; `Player` subclasses instantiated directly with `start_listening=False` for pure `choose_move` unit testing (confirmed via `tests/test_self_play.py`'s `FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False)` pattern — no live server needed for translation-only tests). C++ tests via `ctest --test-dir cpp/build`, Python native tests via `./scripts/pytest_native.sh` only (ASan runtime must be preloaded).

## DW Verification

| DW-ID | Done-When Item | Status | Test Cases |
|-------|---------------|--------|------------|
| DW-1.1 | `tests/test_mcts_player.py` passes — order translation correctness, `kNoAction` fallback, team-preview/forced-switch edge case | COVERED | `test_action_id_to_order_switch_maps_to_team_preview_slot`, `test_action_id_to_order_move_maps_to_active_moveset_slot`, `test_mcts_player_falls_back_to_default_order_on_no_action`, `test_mcts_player_does_not_crash_with_no_active_pokemon` (forced-switch/team-preview: `active_pokemon=None`, verifies `battle_state_from_poke_env` + `legal_actions` degrade to switch-only and a real `MctsPlayer.choose_move` call returns a valid switch order, not a crash) |
| DW-1.2 | real 500-battle benchmarks (`--p1 mcts`) vs random/maxdamage/heuristic/search at default format, vs learned/ppo at `--format gen9ou`, Wilson CIs, recorded in Execution Log; budget-aware trim rule if ms/turn makes the full 3,000-game run exceed overnight | COVERED (measurement step, not a unit test) | Run via `scripts/benchmark.py --p1 mcts --p2 <x> --n-battles N` against the local Showdown server; results recorded directly, not asserted in pytest |
| DW-1.3 | `./scripts/pytest_native.sh` and `ctest --test-dir cpp/build` stay green | COVERED | full-suite run of both after implementation |

**All items COVERED:** YES

## Design Decisions

**Python `search()` binding**: a `module.cpp` lambda `(BattleState, n_simulations, seed) -> SearchResult` that calls `be::search(state, be::default_eval, n_simulations, seed)` internally, with `py::call_guard<py::gil_scoped_release>()` on the `m.def` (idiomatic pybind11 GIL-release-for-call-duration, cleaner than manual RAII since it can't accidentally reacquire before the return-value conversion). No `EvalFn`/callable parameter is ever exposed to Python — satisfies the plan's explicit GIL-safety constraint. `SearchResult` bound as a `py::class_` with `def_readonly` for `best_action`/`root_visit_distribution`; `kNoAction` exposed as `m.attr("NO_ACTION")` (int), matching the existing `NUM_SWITCH_ACTIONS`-style constant exposure.

**Switch-id → order mapping**: `list(battle.team.values())[action_id]` for `action_id < NUM_SWITCH_ACTIONS` — direct index, since `battle_state_from_poke_env`'s own verified docstring establishes `battle.team`'s dict-insertion order as stable team-preview order, the same order `action.hpp`'s ActionId 0-5 scheme assumes. This is a different, simpler mapping than `action_space.py`'s species-sorted 13-way Metamon scheme — that module solves a different problem (Phase 2/3's imitation/PPO action space) and must not be reused here; conflating the two would silently mismap switches.

**Move-id → order mapping**: `active_pokemon.moves` dict, `list(...keys())[move_slot]` for `move_slot = action_id - MOVE_ACTION_OFFSET` — mirrors `_pokemon_slot`'s own `list(mon.moves.keys())[:4]` construction in the same file, so the ordering the C++ side legality-checked against is the exact ordering used to translate back.

**kNoAction fallback**: `result.best_action == _native.NO_ACTION` (or `< 0`, equivalent — `NO_ACTION` is `-1` and all real `ActionId`s are `>= 0`) routes to `self.choose_default_move()` (`DefaultBattleOrder`, `"/choose default"` — Showdown's own first-legal-order fallback), not `choose_random_move` — matches the plan's "safe default order" phrasing and doesn't require building an order from information `MctsPlayer` doesn't have in this state.

**MctsPlayer constructor**: `__init__(self, *args, n_simulations: int, seed: int = 0, **kwargs)`, an internal `random.Random(seed)` draws a fresh `getrandbits(64)` search-seed per `choose_move` call (not the same seed reused every turn) — mirrors `FrozenPolicyPlayer`'s own `seed`-param convention; per-turn seed variety avoids any risk of an unintentionally repeating simulation trace across turns while keeping the whole player's behavior reproducible given a fixed construction seed. LSP check: `MctsPlayer` IS-A `Player` (matches the plan's own confirmation and `FrozenPolicyPlayer`'s precedent) — no empty overrides, only `choose_move` is implemented, same shape as every existing `Player` subclass in this codebase.

**`n_simulations` benchmark CLI plumbing**: add `--n-simulations` (default TBD by measurement) to `scripts/benchmark.py`, threaded into `_make_player` only for `name == "mcts"` (mirrors how `--ppo-model-path` is only consulted for `name == "ppo"`).

## Prerequisites
- [x] `cpp/src/mcts.cpp` / `cpp/include/be/mcts.hpp` exist and are tested (M6 complete).
- [x] `battle_engine/mcts_player.py`'s translator exists (M5 complete).
- [x] Local Showdown server startable for DW-1.2's real benchmarks.
- [x] `./scripts/build_cpp.sh` rebuilds `_native*.so` after `module.cpp` changes, no reinstall step.

## Recommendation
BUILD. No gaps require a plan update — this phase's scope (bind `search`, build `MctsPlayer`, wire `"mcts"` into the benchmark CLI, measure real ms/turn) is fully buildable against what already exists.

## Additional Design Decision (resumed-session addendum, 2026-08-25)

**`scripts/benchmark.py`: randomized `AccountConfiguration` per player.** Discovered necessary while running DW-1.2's real sweep: `_make_player` had no explicit `account_configuration`, so poke-env fell back to its own default (`AccountConfiguration.generate(class_name, rand=False)` — a plain per-process-counter username like "MctsPlayer 1"). The local Showdown server ties an outstanding challenge to a userid, not a connection, and never clears it on disconnect (verified against `pokemon-showdown/server/ladders.ts`'s `makeChallenge` and `ladders-challenges.ts` — a challenge is only removed by acceptance, cancellation, or the user renaming). A benchmark process killed mid-battle (this resumed session's own cleanup of a stray orphaned process left by the prior turn) left a real "already a challenge" popup that deterministically blocked every subsequent run reusing the same default username — retrying, even after minutes of waiting, never resolved it, since there is no timeout in the server's challenge logic. Fix: `AccountConfiguration.generate(name, rand=True)` (a random 5-char suffix instead of the counter) passed explicitly to every player constructed by `_make_player`, so no two runs — successful or killed — can ever collide again. Applies uniformly to all player types, not just `mcts`, since any of them could be the reused/colliding side. Verified: `./scripts/pytest_native.sh` stayed green (169 passed, 3 pre-existing skips) after this change; `tests/test_benchmark.py` doesn't assert on player identity/usernames so it needed no update.

## Execution Log (resumed-session DW-1.2 sweep, 2026-08-25)

Sanity timing (n_simulations=200, before committing to the full sweep):
- `mcts` vs `random` (10 battles, default format): 11.85s total ≈ 1.0–1.2s/battle.
- `mcts` vs `search` (20 battles, default format): 16.17s total ≈ 0.81s/battle.
- `mcts` vs `learned` (10 battles, gen9ou): 47.2s total ≈ 4.72s/battle.
- `mcts` vs `ppo` (10 battles, gen9ou): 38.0s total ≈ 3.8s/battle.

Extrapolated full 6-matchup × 500-battle sweep ≈ 100–110 minutes total — comfortably inside the "few hours" budget, so **no trim was applied**: every matchup below ran the full 500 battles at `--n-simulations 200`.

Real DW-1.2 results (`.venv/bin/python scripts/benchmark.py --p1 mcts --p2 <opponent> --n-battles 500 --n-simulations 200`, default format unless noted):

| Opponent | Format | Record | Win rate | 95% Wilson CI |
|---|---|---|---|---|
| random | gen9randombattle | 477/500 | 95.4% | [93.2%, 96.9%] |
| maxdamage | gen9randombattle | 354/500 | 70.8% | [66.7%, 74.6%] |
| heuristic | gen9randombattle | 158/500 | 31.6% | [27.7%, 35.8%] |
| search | gen9randombattle | 162/500 | 32.4% | [28.4%, 36.6%] |
| learned | gen9ou | 236/500 | 47.2% | [42.9%, 51.6%] |
| ppo | gen9ou | 151/500 | 30.2% | [26.3%, 34.4%] |

Consistent with M0's finding (branching MCTS/DUCT does not beat the 1-ply projection): `mcts` beats the two non-search baselines (random, maxdamage) comfortably but loses to every search/learned/RL-based opponent (heuristic, search, learned, ppo) — heuristic and search land it in the same ~30-32% band M0 measured, and ppo (the strongest opponent in the roster) is its worst matchup at 30.2%. `n_simulations=200` was kept as the sweep's value — sanity timing showed it's cheap enough (well under 5s/battle even at gen9ou) that the CLI's chosen default did not need lowering to fit budget.
