<!-- base-commit: be5d78c401377de01e36f2797f0fa9016eddf467 -->
<!-- generated: 2026-08-24 -->

# Code Standards

## Forbidden Patterns

**Never assume a library/API's behavior — verify against real source or real data first.** This
project's hard rule ("Evidence over assumption," `CLAUDE.md`) exists because poke-env's API has
silently shifted across versions and cost real debugging time when trusted blind.
```python
# BAD — guesses poke-env's field semantics from the name alone
protect_streak = mon.protect_counter  # assumed: increments every turn Protect is chosen

# GOOD — from encoding.py's module docstring: verified against poke-env's actual source,
# with the exact wrong assumption named and the real behavior measured against ground truth
# (player_prev_move is "last move ever used," NOT "move used this transition" — a first
# version that assumed the latter was wrong on 59.4% of real streak-2 reconstructions)
```

**Never invent a magic number or silently drop a simplification — name it, and say why.** A
comment that states a scope decision (and often the bug/measurement that produced it) is this
codebase's default, not an exception.
```cpp
// BAD
if (mon.item == "boots") return true;  // heavy duty boots blocks hazards

// GOOD — from encoding.py's _is_hazard_immune: names the mechanic, the measurement that
// justified adding it (5.19% of a 20,610-state sample), and what's still deliberately out of
// scope, so a future reader can't mistake omission for oversight
```

**Never let a caller-owned invariant go unstated.** Functions across this codebase document
"caller is responsible for X" explicitly rather than defensively re-checking it (e.g.
`legal_actions()`'s "callers are responsible for restricting to what's actually choosable,"
`select_ucb1_action`'s "`actions` and `stats` must be index-aligned — a caller bug otherwise, not
checked here"). Don't add silent defensive checks that mask a caller bug instead of surfacing it.

## Code Examples

### C++ header: state the design decision, not just the shape

```cpp
// DO — from cpp/include/be/mcts.hpp: names the alternative considered, why it was
// rejected (a real C++ language limitation, not a style preference), and what invariant
// the reader must preserve
// A tagged single struct (below) was chosen over a std::variant<DecisionNode,
// ForcedSwitchNode>-style split-type design specifically because a recursive
// std::variant whose alternatives each hold unique_ptr<SearchNode> children hits real
// C++ trouble: forming variant<A, B> requires A and B to be COMPLETE types...
enum class NodeKind : uint8_t { kDecision, kForcedSwitch };

// DON'T — a plain struct with no rationale forces the next implementer to
// rediscover the same std::variant dead-end from scratch
struct SearchNode { int kind; /* ... */ };
```

### Python: adapter pattern with a verified-vs-assumed docstring

```python
# DO — from encoding.py: two small adapters converge on one shared BattleView, and the
# module docstring records what was checked against real data (not "should work")
def battle_view_from_poke_env(battle: AbstractBattle) -> BattleView: ...
def battle_view_from_replay_state(state: dict) -> BattleView: ...
def encode(view: BattleView) -> np.ndarray: ...

# DON'T — deriving the vector separately per data source duplicates every future bugfix
def encode_from_poke_env(battle): ...
def encode_from_replay(state): ...
```

## Error Handling

**Python: raise, with the caller-facing reason in the message — never a bare assert or a silent
default.** Reserved for real invariant violations (a state that "shouldn't be possible"), not
expected-failure control flow.
```python
# From encoding.py's _pad_bench — loud failure over silently truncating real data
assert len(bench) <= MAX_BENCH, (
    f"{len(bench)} bench slots exceeds MAX_BENCH={MAX_BENCH} - silently "
    "truncating would drop a real Pokemon rather than surface a bug"
)
```

**C++: `is_valid()`-style predicate functions for state invariants, not exceptions in the hot
path.** Search/forward-model code runs in a tight simulation loop — validity is a checkable
predicate (`be::is_valid(const BattleState&)`) callable at transition boundaries during testing,
not a per-call exception guard.

**Both layers: ASan/UBSan is the primary error-surfacing mechanism for C++.** `cpp/CMakeLists.txt`
builds Debug-with-sanitizers by default specifically because this is hand-written pointer/tree
code (`unique_ptr` + raw indices) — a use-after-free should crash loudly under ASan immediately,
not resurface as a confusing failure weeks later. Always run `./scripts/pytest_native.sh` (not
bare `pytest`) when `_native` is involved — it preloads the ASan runtime before the interpreter
starts, which `conftest.py` cannot do.

## Imports & Dependency Direction

Python: stdlib → `from __future__ import annotations` at top of file → third-party
(`poke_env`, `numpy`, `torch`, `sb3_contrib`) → internal `battle_engine.*`, absolute not relative.

C++: project headers quoted and namespaced (`#include "be/mcts.hpp"`), standard library
angle-bracketed, project headers after standard library. One namespace, `be`, for everything
under `cpp/include/be/` and `cpp/src/` — no nested namespaces.

Dependency direction: `cpp/include/be/*.hpp` + `cpp/src/*.cpp` are pure C++ with zero pybind11
dependency (only `cpp/bindings/module.cpp` includes `<pybind11/...>`) — keeps the engine testable
under plain Catch2 without a Python interpreter. `battle_engine/*.py` may depend on `_native`
(the compiled extension); nothing in `cpp/` may depend on `battle_engine/*.py`.

## Testing Patterns

**Catch2 (`cpp/tests/`): one `TEST_CASE` per behavior, named as a full sentence stating the
expected behavior, tagged by module.**
```cpp
// From cpp/tests/test_mcts.cpp — the name IS the spec; a comment above explains WHY this
// case exists when the behavior isn't self-evident from the name alone
TEST_CASE("select_ucb1_action: an untried action always wins, regardless of a visited "
          "action's average value", "[mcts]") {
  std::vector<ActionId> actions = {0, 1};
  std::vector<VisitStats> stats = {
      {/*visits=*/1, /*value_sum=*/1.0},   // action 0: perfect average, but only 1 visit
      {/*visits=*/0, /*value_sum=*/0.0},   // action 1: never tried
  };
  REQUIRE(select_ucb1_action(actions, stats, /*parent_visits=*/1, /*exploration_constant=*/1.4f) == 1);
}
```
Prefer (b) a statistical check over N≥10,000 samples with a tolerance band for anything
stochastic (damage rolls, MCTS sampling) over (a) a single fixed-seed exact-value assertion —
keep (a) only as a cheap determinism smoke test alongside (b), never as the sole check.

**pytest (`tests/`): any test touching the native extension must skip cleanly, not error, on a
fresh clone.**
```python
# From tests/test_native_legality.py — must be the first real line after the docstring/imports;
# a fresh clone shouldn't fail pytest just because scripts/build_cpp.sh hasn't run yet
_native = pytest.importorskip("battle_engine._native")
from battle_engine.mcts_player import battle_state_from_poke_env  # noqa: E402
```
Battle fixtures use `SimpleNamespace` wrapping real `poke_env` `Pokemon`/`Move` objects built via
`conftest.make_mon(...)`, exposing only the exact `AbstractBattle` attributes the code under test
reads — not a live server connection, and not a hand-rolled fake `Pokemon` class. Run the native
suite via `./scripts/pytest_native.sh`, never bare `pytest`, whenever `_native` is exercised.

## Naming Conventions

C++: `snake_case` functions/variables, `PascalCase` types, `kCamelCase` for `constexpr`
constants (`kNumSwitchActions`, `kDefaultExplorationConstant`, `kNoAction`). `Side::Me` /
`Side::Opp`, never `Player1`/`Player2` — matches `BattleState`'s own `my_*`/`opp_*` field naming
throughout the codebase; don't introduce a third vocabulary for the same distinction.

Python: `snake_case`, leading underscore for module-private helpers (`_poke_env_hazards`,
`_replay_pokemon_view`). Two-adapter-plus-shared-core files pair one exported function per data
source with a single shared function they both funnel into (see Code Examples above).

Domain terms: "hazards" (not "entry hazards" or "field conditions") for Stealth
Rock/Spikes/etc.; "leaf eval"/"leaf value" (not "heuristic") for a search-terminal scoring
function; "revealed" (not "known"/"visible") for what's been disclosed about the opponent's team.

## File Organization

```
cpp/
├── include/be/    # Pure C++ headers, one concept per file, heavy design-rationale comments
├── src/           # Matching .cpp per header
├── bindings/      # module.cpp only — thin pybind11 glue, no engine logic
└── tests/         # test_<module>.cpp per include/be/<module>.hpp, Catch2
battle_engine/     # Python package; _native*.so lands here directly (PEP 660 editable install,
                   # no reinstall step needed after ./scripts/build_cpp.sh)
scripts/           # Runnable entry points (benchmarks, training, C++ build)
tests/             # test_<module>.py per battle_engine/<module>.py, mirrors the package 1:1
data/              # Gitignored: replays, cached datasets, trained checkpoints — regenerate via
                   # scripts/, never commit
```
New C++ functionality: header in `include/be/`, implementation in `src/`, test in `tests/` — all
three land together, not staggered across commits. A new milestone's design rationale goes in
the header's top-of-file comment (see `mcts.hpp`), not a separate design doc.

## Technology Decisions

- Standalone CMake project (`cpp/CMakeLists.txt`), not wired into the Python build backend —
  `scripts/build_cpp.sh` runs `cmake` directly; there is no `[build-system]` table in
  `pyproject.toml` to hook into.
- **Debug build with `-fsanitize=address,undefined` is the default**, not an opt-in variant —
  `--release` is the explicit opt-out. Given hand-written tree/pointer code, catching a real bug
  immediately outweighs Debug's runtime cost.
- pybind11 v3.0.4 and Catch2 v3.x via CMake `FetchContent`, pinned to real tags — not
  system-installed, not vendored.
- C++20. STL containers over hand-rolled data structures (`std::unordered_map` keyed by a packed
  `uint32_t`, not a custom hash table) — this project's C++ is for search/tree logic, not for
  practicing infrastructure.
- Raw binary weight export (magic header + version + dims + float32 bytes), not ONNX/TorchScript,
  for any PyTorch→C++ weight transplant — keeps "hand-written forward pass" as a real learning
  artifact and avoids a second C++ runtime dependency.

## Exemplar Files

**`cpp/include/be/mcts.hpp`** — demonstrates the project's C++ documentation convention at its
fullest: every non-forced design choice explains the alternative considered and why it lost,
correctness invariants are stated at the field level (not just the function level), and a
plan-correcting finding (the eval-antisymmetry sign-convention bug) is recorded inline rather
than silently fixed. Read this before writing any new header in `cpp/include/be/`.

**`battle_engine/ppo_warm_start.py`** — demonstrates the "verify layer-by-layer against the real
checkpoint" standard for any weight transplant: exact `nn.Sequential` index correspondence
verified against real saved state, a measured (not assumed) answer to a real numerical-scale
question, `_copy_linear`'s shape-mismatch `ValueError` instead of a silent broadcast. The
required reference for `scripts/export_weights.py`/M3's C++ forward pass.

**`battle_engine/encoding.py`** — demonstrates the two-adapter-to-shared-core pattern, and the
project's convention of recording every simplification's rationale (and, where one exists, the
bug/measurement that produced it) directly in the module docstring rather than in an external
design doc.
