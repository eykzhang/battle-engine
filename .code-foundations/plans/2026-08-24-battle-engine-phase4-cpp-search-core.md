# Plan: battle-engine Phase 4 — finish the C++ search core (M7 + enhancement track)
**Created:** 2026-08-24
**Status:** complete
**Started:** 2026-08-25 00:00
**Resumed:** 2026-08-25 10:04 (user-directed, after the 07:45 hard cutoff)
**Completed:** 2026-08-26 07:15
**Current Phase:** 6 (final)
**Complexity:** complex
**Review cadence:** 2

---

## Context

battle-engine's Phase 4 C++ search core is half-built. M0-M6 (the open-loop DUCT/MCTS
algorithm itself, `cpp/src/mcts.cpp`) are done and tested — all 64 Catch2 tests pass — but
nothing wires it to a real player or benchmark (M7), and the enhancement track that gives the
search a real shot at beating Phase 3 PPO alone doesn't exist (M2 weight export, M3 hand-written
NN forward pass, M4b full state encoding, M6b PUCT search with the trained PPO actor as prior
and its critic as leaf value). The user wants this finished now, with Claude implementing
directly rather than scaffolding (this project's normal "user writes the C++" convention is
explicitly overridden for this push), so the phase closes with real measured numbers. Those
numbers are the input to a separate, later decision: whether further RL training is worth doing
before `battle-brain` defaults to a community engine (Foul Play) instead.

## Constraints

- Every existing convention in `plans/precious-crafting-bachman.md` and `docs/code-standards.md`
  stays in force: open-loop tree (nodes are action-index paths, never a stored canonical state),
  the fixed `ActionId` scheme (0-5 switch by team-preview slot, 6-9 move slot), Debug/ASan-by-
  default C++ builds, two-layer Catch2+pytest testing with clean skip when `_native` isn't built.
- M6b's PPO actor/critic port reuses `ppo_warm_start.py`'s already-verified layer correspondence
  (`policy.mlp_extractor.policy_net[0,2]` / `policy.action_net` / `policy.mlp_extractor.value_net[0,2]`
  / `policy.value_net`) — do not re-derive it.
- If M6b's PUCT search doesn't beat Phase 3 PPO alone on the first real 500-battle measurement,
  report it honestly and move to the inventory step. No open-ended tuning loop.
- No milestone or the phase itself is called done without a real measured number — every prior
  phase gate in this project was measured this way, and Phase 4 does not get a vibes exception.
- Setting up a Foul Play head-to-head is explicitly out of scope for this plan.

## Chosen Approach

**A. Per-node PUCT** — at every node the M6b search expands, run the actor+critic forward pass
on that node's real (open-loop-resampled) `BattleState` and map the 13-way Metamon action
distribution onto the node's actual legal `ActionId`s via a ported species-sort function,
producing a real prior at every depth, not just the root. **Rationale:** this is what the
project's own plan (`plans/precious-crafting-bachman.md`) and the user's confirmed success
criteria actually describe ("a PUCT variant (PPO actor+critic)") — a root-only prior or a
critic-only/no-prior configuration both quietly answer a smaller question. **Fallback:** if
measured ms/turn makes Approach A's simulation count impractical, descope to Approach C
(critic-only leaf value, plain UCB1, no prior) rather than a partial per-node implementation.

## Rejected Approaches

- **B. Root-only prior:** cheaper (no C++ port of the Metamon mapping needed), but every node
  past the first ply searches blind/uniform — undercuts the actual "PPO-guided search" claim
  being tested.
- **C. Critic-only, no actor prior:** simplest and cleanest architecturally, and a real,
  methodologically valid experiment (isolates "does a better value function help" from "does a
  prior help," same one-variable-at-a-time discipline M0 used) — kept as the named fallback for
  Approach A, not built by default.

---

## Implementation Phases

### Phase 1: M7 — MctsPlayer + benchmark wiring
**Model:** sonnet
**Skills:** cc-routine-and-class-design, aposd-verifying-correctness
**Gate:** Full — the root `ActionId ↔ BattleOrder` translation is the seam Phases 4, 5, and 6 all
build on; a silent wrong-slot mapping would surface only as a mediocre win rate, not a crash.
**Depends on:** none | **Unlocks:** Phase 3, Phase 4, Phase 5, Phase 6
**File scope:** `cpp/bindings/module.cpp`, `battle_engine/mcts_player.py`, `scripts/benchmark.py`, `tests/test_mcts_player.py`

**Goal:** Wire M6's already-tested `search()` to a real poke-env `Player` and the existing
benchmark harness, with a real measured simulation count and ms/turn.

**Scope:**
- IN: bind `search(state, n_simulations, seed) -> SearchResult` with `default_eval` selected
  C++-side (no Python callable crosses the `py::gil_scoped_release` boundary — passing a Python
  `leaf_eval` into a GIL-released call is a use-of-Python-without-the-GIL crash, not a valid
  design); `MctsPlayer(Player)` — legitimate inheritance (`MctsPlayer` IS-A `Player`, matches
  `FrozenPolicyPlayer`'s own precedent, LSP holds); root-only `ActionId → BattleOrder`
  translation; `"mcts"` benchmark entry.
- OUT: anything from the enhancement track (PUCT, PPO weights) — plain `default_eval` only.

**Constraints:** Reuse `ppo_eval.py`'s `load_ppo_player`/`FrozenPolicyPlayer` pattern for the
`Player` subclass shape, not a fresh design.
**Edge cases:** team-preview / forced-switch state (`active_pokemon is None`) must not crash —
`legal_actions()` already degrades correctly (`my_active_slot=-1` restricts to switches only,
verified against `action.hpp`'s own semantics), confirm with a test rather than assuming; a
`kNoAction` root result (empty legal action list) must fall back to a safe default order, not
propagate an invalid order to poke-env.

**Produces:** working `_native.search(state, n_simulations, seed) -> SearchResult` binding
(`default_eval` fixed C++-side, no Python callable parameter); `MctsPlayer`; `"mcts"` dispatch
branch in `scripts/benchmark.py`; a measured `n_simulations` value with its real ms/turn number,
recorded in this phase's commit/Execution Log entry and rolled into Phase 6's `notes/` writeup.
**File hints:** `battle_engine/ppo_eval.py` — the `Player`-wrapping pattern to mirror;
`battle_engine/mcts_player.py` — existing translator this phase builds on top of, not around.

**Done when:**
- [ ] DW-1.1: `tests/test_mcts_player.py` passes — order translation correctness, `kNoAction`
      fallback, and the team-preview/forced-switch edge case.
- [ ] DW-1.2: real 500-battle benchmarks (`scripts/benchmark.py --p1 mcts`) run against
      random/maxdamage/heuristic/search at default format, and against `learned`/`ppo` with
      `--format gen9ou` (their trained-on distribution — see `scripts/benchmark.py`'s own
      docstring on why an off-distribution comparison isn't valid), Wilson CIs, recorded in this
      phase's Execution Log entry. If the measured ms/turn makes all six 500-battle matchups
      (3,000 games) exceed an overnight run, keep `search`/`learned`/`ppo` at full 500 (the
      phase-gate-relevant comparisons) and trim random/maxdamage/heuristic to 200-battle sanity
      checks — state which was done.
- [ ] DW-1.3: `./scripts/pytest_native.sh` and `ctest --test-dir cpp/build` stay green.

**Difficulty:** MEDIUM
**Uncertainty:** Real ms/turn at a given `n_simulations` isn't known until measured — may need
2-3 candidate values tried before picking one that fits a reasonable benchmark window.

### Phase 2: M2 — Weight export tooling
**Model:** sonnet
**Skills:** cc-pseudocode-programming, cc-defensive-programming
**Gate:** Standard
**Depends on:** none | **Unlocks:** Phase 3
**File scope:** `scripts/export_weights.py`, `tests/test_export_weights.py`, `data/cpp_weights/`

**Goal:** Dump `data/models/ppo.zip`'s actor+critic Linear layers to a versioned binary format
the C++ side can load safely.

**Scope:**
- IN: load checkpoint via `sb3_contrib.MaskablePPO.load`; extract the 6 layers
  `ppo_warm_start.py` already names (`policy.mlp_extractor.policy_net[0,2]`, `policy.action_net`,
  `policy.mlp_extractor.value_net[0,2]`, `policy.value_net`); write one file with a magic header,
  a version field, `VECTOR_LEN` (imported from `encoding.py`, never hardcoded), then each layer's
  shape + float32 weight/bias bytes, actor branch then critic branch.
- OUT: the C++ loader itself (Phase 3); ONNX/TorchScript (rejected per `docs/code-standards.md`'s
  Technology Decisions).

**Edge cases:** checkpoint path missing/unreadable → clear error, not a stack trace; a layer
shape not matching the expected `(in, out)` from `ppo_warm_start.py`'s correspondence → raise
before writing anything (fail the whole export, don't emit a partial file — a partial file would
otherwise be silently accepted by a Phase 3 loader that only checks version/`VECTOR_LEN`).

**Produces:** `data/cpp_weights/ppo.bin`, pinned to this exact byte layout (Phase 3 codes against
this contract, not the script's prose): little-endian throughout — `magic: 4 bytes ("BEPP")`,
`version: uint32 = 1` (pinned; bump on any future format change), `vector_len: uint32` (from
`encoding.VECTOR_LEN`, never hardcoded), then 6
layers in fixed order (`actor.net[0]`, `actor.net[2]`, `actor.action_net`, `critic.net[0]`,
`critic.net[2]`, `critic.value_net`), each as `out_dim: uint32`, `in_dim: uint32`, `weight:
out_dim*in_dim × float32` (row-major `(out, in)`, matching PyTorch's own `Linear.weight` layout
directly — no transpose), `bias: out_dim × float32`; plus `tests/test_export_weights.py`.
**File hints:** `battle_engine/ppo_warm_start.py` — the exact layer correspondence to reuse, not
re-derive.

**Done when:**
- [ ] DW-2.1: `scripts/export_weights.py` runs against the real `data/models/ppo.zip` and
      produces `data/cpp_weights/ppo.bin`.
- [ ] DW-2.2: `tests/test_export_weights.py` passes — exact equality (not tolerance) between the
      dumped bytes and the real checkpoint's tensors.
- [ ] DW-2.3: a synthetic shape-mismatched checkpoint (dirty-path test) raises before any file is
      written — no partial `ppo.bin` is ever produced.

**Difficulty:** LOW
**Uncertainty:** None.

### Phase 3: M3 — C++ NN forward pass
**Model:** sonnet
**Skills:** aposd-designing-deep-modules, aposd-verifying-correctness
**Gate:** Standard
**Depends on:** Phase 1, Phase 2 | **Unlocks:** Phase 4, Phase 5
**File scope:** `cpp/include/be/mlp.hpp`, `cpp/src/mlp.cpp`, `cpp/tests/test_mlp.cpp`, `cpp/bindings/module.cpp`, `tests/test_native_forward_pass.py`

**Goal:** Hand-written MLP forward pass loading Phase 2's binary format, verified against the
real PyTorch policy's output.

**Scope:**
- IN: `MlpWeights::forward(input) -> output` for one branch (`VECTOR_LEN → 128 → 64 → out_dim`,
  ReLU between, double-precision internal accumulators); `PolicyWeights::load(path) ->
  {actor: MlpWeights (out_dim=13), critic: MlpWeights (out_dim=1)}` — one load call returning
  both branches, since Phase 5 always uses them together; **not** a shared trunk between them,
  verified `policy_net`/`value_net` are separate stacks in `ppo_warm_start.py`. `load()` checks
  self-consistency only (`header.version == 1`, `header.vector_len` matches every layer's
  declared `in_dim`/`out_dim` chain) — no C++-side `VECTOR_LEN` constant exists to check against
  until Phase 4; Phase 4 adds the 3-way cross-check (`encode_native()`'s length ==
  `encoding.VECTOR_LEN` == `ppo.bin`'s header field).
- OUT: masked softmax over the actor's output (Phase 5's concern, needs the legal-action set a
  bare forward pass doesn't have); a generalized N-layer MLP framework — the architecture is
  fixed and known, a hardcoded 3-layer struct is the deep-enough, YAGNI-correct choice over a
  templated general solution.

**Constraints:** Load-time dimension check against `ppo.bin`'s version/`VECTOR_LEN` fields — a
mismatch raises a clear error, never a silent misread (per the project's own already-stated
requirement, `VECTOR_LEN` has changed 4 times in this project's history). No `Security-sensitive`
marker: `data/cpp_weights/ppo.bin` is a locally generated, gitignored artifact under this
project's own control, never user-supplied — still dimension/version-validated at load per the
constraint above, so the untrusted-input concern that marker exists for doesn't apply here.
**Edge cases:** malformed/truncated weight file → load-time error, not a crash mid-forward-pass
(this is the plan's only deserialization boundary, in hand-written C++, under an ASan-by-default
build for exactly this reason).

**Produces:** `be::MlpWeights::forward()` + `be::PolicyWeights::load(path)` API + module.cpp
bindings for a parity test to call.
**File hints:** `battle_engine/ppo_warm_start.py` — required reference per
`docs/code-standards.md`'s Exemplar Files; `cpp/include/be/mcts.hpp` — required reading before
writing any new header in `cpp/include/be/`, same doc.

**Done when:**
- [ ] DW-3.1: Catch2 tests in `cpp/tests/test_mlp.cpp` cover a known-input/known-output case and
      a truncated/malformed weight file (load-time error, not a crash).
- [ ] DW-3.2: `tests/test_native_forward_pass.py` — same input vector through the real PyTorch
      policy and the C++ forward pass, `np.allclose` (not exact) on both actor logits and critic
      value.
- [ ] DW-3.3: microbenchmark one actor+critic forward pass (µs); combined with Phase 1's measured
      per-simulation cost, project ms/turn at candidate `n_simulations` for Phase 5's PUCT config.
      State the numeric threshold (fits a laptop-first overnight 500-battle run, or not) that
      selects Approach A vs. the Approach C fallback — available the moment this phase's forward
      pass exists, before Phase 5's own Metamon-mapping work is sunk.

**Difficulty:** MEDIUM
**Uncertainty:** None on the port itself — architecture and correspondence are already verified
in `ppo_warm_start.py`. DW-3.3's projected latency is the one open question this phase resolves.

### Phase 4: M4b — Full BattleState extension + encode() port
**Model:** fable
**Skills:** aposd-designing-deep-modules, cc-defensive-programming, aposd-verifying-correctness
**Gate:** Full
**Depends on:** Phase 1, Phase 3 | **Unlocks:** Phase 5
**File scope:** `cpp/include/be/battle_state.hpp`, `cpp/src/battle_state.cpp`, `cpp/bindings/module.cpp`, `battle_engine/mcts_player.py`, `tests/test_native_encoding.py`

**Goal:** Architect the extended state representation `encoding.py`'s `encode()` needs beyond the
hand-crafted eval's fields (species/item/ability/protect_counter/weather/terrain, the
single-most-recent-hazard view), and port `encode()` bit-for-bit onto it.

**Scope:**
- IN: add `species`/`item`/`ability`/`protect_counter` to `PokemonSlot`; state-level
  `weather`/`terrain`; extend `SideConditions` to all 8 tokens `encode()` tracks (vs.
  `default_eval`'s 4); `encode_native()` producing the exact 665-dim vector; extend
  `battle_state_from_poke_env`.
- OUT: any change to `default_eval`'s existing 4-field hazard scoring — it keeps using its own
  stack-count fields untouched.

**Approach notes:** `encode()`'s hazard dimension is single-most-recent-token, a different shape
than `default_eval`'s stack-count/presence fields — don't conflate them. Chosen design: extend
`SideConditions` to store per-token turn-set data for all 8 tokens (mirroring poke-env's own
real turn-tracked vs. stack-tracked split, verified via `STACKABLE_CONDITIONS`), and derive the
single-most-recent view on demand inside `encode_native()` — keeps `BattleState` the one stored
source of truth (matches this struct's own existing "one ordering stored, views computed on
demand, never a second stored copy" invariant) rather than having the Python translator
pre-reduce it before the fact ever reaches C++.
**Constraints:** `species` must exactly match the identity `action_space.py`'s species-sort uses
(`base_species`, not `species`/`name` — real bug precedent in `encoding.py`'s own history, form
changes like Terapagos renaming `name` but not `base_species`).
**Edge cases:** unrevealed opponent slots (no species/item/ability known) → `""`/unknown
sentinels, matching `PokemonSlot::moves`' existing convention; team-preview state (no active
mon) — same guard `encoding.py`'s `battle_view_from_poke_env` already applies.

**Produces:** extended `BattleState`; `encode_native()`; the `species` field Phase 5's Metamon
mapping depends on.
**File hints:** `battle_engine/encoding.py` — the exact vector layout and every named
simplification to preserve; `battle_engine/mcts_player.py` — existing translator to extend, not
rewrite.

**Done when:**
- [ ] DW-4.1: `tests/test_native_encoding.py` — `np.allclose(encode_native(state), encode(view))`
      on real battle states (live-battle fixtures per `test_native_legality.py`'s pattern).
- [ ] DW-4.2: `tests/test_native_legality.py`/`test_mcts_player.py` still pass unchanged
      (extension doesn't regress `default_eval`'s existing fields).
- [ ] DW-4.3: `encode_native()`'s output length equals both `encoding.VECTOR_LEN` (Python-side)
      and `ppo.bin`'s header `vector_len` field — the 3-way cross-check Phase 3 deferred here,
      since `VECTOR_LEN` has moved 4 times in this project's history and three places now have
      to agree.

**Difficulty:** HIGH
**Uncertainty:** Reconstructing `encode()`'s exact species-sorted bench view as a computed-on-
demand function (not a second stored ordering) is the highest-risk piece of this phase — verify
against real multi-faint battle states, not just a fresh-team fixture.

### Phase 5: M6b — PUCT search with PPO prior/value
**Model:** fable
**Skills:** aposd-designing-deep-modules, cc-routine-and-class-design, aposd-verifying-correctness
**Gate:** Full — new cross-phase seam (the Metamon mapping) every later measurement depends on;
a defect here corrupts every PUCT decision silently (see EXPLORE's pre-mortem), cheaper to catch
before the commit than after Phase 6's benchmarks are already run against it.
**Depends on:** Phase 1, Phase 3, Phase 4 | **Unlocks:** Phase 6
**File scope:** `cpp/include/be/mcts.hpp`, `cpp/src/mcts.cpp`, `cpp/include/be/action.hpp`, `cpp/src/action.cpp`, `cpp/include/be/battle_state.hpp`, `cpp/src/battle_state.cpp`, `cpp/bindings/module.cpp`, `battle_engine/mcts_player.py`, `scripts/benchmark.py`, `cpp/tests/test_mcts.cpp`, `cpp/tests/test_action.cpp`, `cpp/tests/test_battle_state.cpp`

**Goal:** Approach A (per-node PUCT, confirmed at EXPLORE): design and implement a novel
node-expansion pattern for this codebase — PUCT, distinct from the existing UCB1 mechanism —
giving a real PPO-informed prior and value at every expanded node, not just the root.

**Scope:**
- IN: `metamon_switch_label_to_action_id`/`action_id_to_metamon_label` (pure functions over
  `BattleState`, ported from `action_space.py`'s species-sort logic — move slots 0-3 map 1:1 to
  `ActionId` 6-9, no translation needed there); `select_puct_action` as its own function (not a
  branch inside `select_ucb1_action` — a flag-selected branch would be logical cohesion, REJECT
  per this project's own cohesion standard); node expansion runs the actor+critic forward pass
  and masked-softmax (verify `MaskableCategorical`'s real masking behavior, don't assume plain
  softmax-then-zero) to populate `my_priors`/`opp_priors` alongside existing `my_stats`/`opp_stats`,
  **renormalized over legal `ActionId`s only** (Metamon labels 9-12, tera moves, have no
  `ActionId` counterpart — `action.hpp` defers tera entirely — so that probability mass is
  dropped and the remaining distribution renormalized, named explicitly in `mcts.hpp` as a
  simplification, not silently); a PUCT-configured `MctsPlayer` variant
  (`battle_engine/mcts_player.py`) that loads `_native.PolicyWeights.load(path)` **once** at
  construction and passes the loaded handle into `search_puct(state, weights, n_simulations,
  seed)` on every call (never re-reading the weight file per turn — would blow DW-5.3's ms/turn
  budget); a `mcts_puct` branch in `scripts/benchmark.py`.
- OUT: Dirichlet root noise (an AlphaZero self-play-exploration technique — irrelevant here,
  this is inference-time search, not training); tuning `c_puct` beyond one measured pass (per
  the confirmed "report honestly, don't open-end tune" outcome policy).

**Approach notes:** `encode()`'s `VECTOR_LEN` layout has no opponent-bench block at all (it's
*my active, my bench, opponent active* only, per `encoding.py`) — the actor's 13-way switch
labels (4-8) are positions in **my** species-sorted bench, so `my_priors` at a node is a direct
`encode_native(state)` → actor forward pass → mapping. `opp_priors` has no equivalent path: there
is no vector shape for "the opponent's own bench, from their perspective." Resolved by mirroring
`BattleState` (swap `my_*`↔`opp_*` team/active/hazards) before calling `encode_native`/the actor
forward pass to compute the opponent's own prior — this reuses the exact same code path for both
sides and is consistent with this project's already-accepted Tier-1 opponent-modeling limitation
(the mirrored "my bench" is the opponent's actual revealed-only bench, not full information,
same asymmetry `legal_actions()` already accepts for opponent switches/moves). Name this
explicitly as a real, accepted simplification in `mcts.hpp`'s own doc comment, not silently.
`kForcedSwitch` nodes (the acted-on side's `active_slot == -1` at expansion time) use plain UCB1
for **selection only** (no prior — `encode()` has no tested semantics for a missing active mon,
and inventing an untested sentinel is worse than a documented narrower fallback) but do **not**
get their own leaf value: `default_eval` and the critic are different, unbounded scales
(hand-crafted HP/status/hazard score vs. a discounted-return estimate), and this search's backup
is path-wide (`mcts.cpp`'s existing backup loop updates every ancestor on the path, not just the
immediate parent) — mixing scales into shared ancestor `VisitStats` would corrupt every PUCT
comparison above a forced switch, not just the switch's own node. Instead, expansion continues
past a `kForcedSwitch` node (apply the switch, keep resolving) until a real `kDecision` state is
reached; the critic evaluates *that* state as the leaf value, keeping every backed-up value on
one consistent (critic) scale throughout this configuration.

**Constraints:** Verify the critic's value-backup sign convention empirically before assuming
it — `mcts.hpp`'s own doc comment already had to correct the plan's "1-v" assumption for
`default_eval`; the critic is a different, unbounded value estimate, not a [0,1] win probability,
so the correct convention isn't guaranteed to be either `1-v` or `-v` without checking.
**Edge cases:** a legal `ActionId` with no corresponding Metamon label (shouldn't happen for
moves; verify explicitly for switches when bench < 5 — fewer real Pokemon than metamon's fixed
9-label switch range); the reverse direction — a Metamon label (tera, 9-12) with no `ActionId`
counterpart — is the renormalization case above, not an error; `kForcedSwitch` nodes (see
Approach notes — UCB1-only selection, expansion continues past them rather than treating them
as a leaf, no crash and no invented encoding).

**Produces:** `_native.search_puct(state, weights, n_simulations, seed) -> SearchResult` (a
`PolicyWeights`-typed weights parameter, mirroring Phase 1's `search()` signature shape); a
`mcts_puct` benchmark entry; using Phase 3's `PolicyWeights`/`MlpWeights` and Phase 4's
`encode_native()`/`species` field.
**File hints:** `battle_engine/action_space.py` — the species-sort logic to port, don't
re-derive; `cpp/include/be/mcts.hpp` — existing open-loop tree/backup conventions this extends.

**Done when:**
- [ ] DW-5.1: Catch2 tests for the Metamon-mapping functions (known team → known mapping,
      including a fainted-teammate/bench<5 case, and a case confirming tera labels 9-12 are
      dropped with the remaining distribution renormalized) and `select_puct_action` (synthetic
      bandit, prior should pull selection toward high-prior arms early).
- [ ] DW-5.2: fixed-seed determinism holds for the PUCT search, same standard as M6's `search()`.
- [ ] DW-5.3: measured ms/turn at the chosen `n_simulations` for this configuration specifically,
      checked against Phase 3's DW-3.3 projection.
- [ ] DW-5.4: the critic's sign-backup convention is checked, not assumed — on N synthetic
      mirrored state pairs, confirm whether `v(state)` and `v(mirror(state))` sum to ~0 (`-v`
      convention) or to ~1 (`1-v` convention). If neither holds clearly within tolerance (a real
      possibility — the critic has no enforced mirror symmetry, unlike `default_eval`'s proven
      antisymmetry), default to `-v` (a property of the opponent table's structure, not the
      eval's shape) and record the measured antisymmetry error in `mcts.hpp` as a named
      approximation, not a silent assumption either way.
- [ ] DW-5.5: a `kForcedSwitch` node during search correctly uses UCB1-only selection and
      continues expansion past it to a real `kDecision` state for the critic's leaf value (no
      crash, no invented encoding, no scale-mixed backup) — covered by a Catch2 test.
- [ ] DW-5.6: `mirror(BattleState)` round-trips (`mirror(mirror(s)) == s`), every `my_*`/`opp_*`
      field pair is actually swapped (team, active slot, hazards), weather/terrain unchanged, and
      `encode_native(mirror(s))` matches `encode()` of the real opponent-POV view on a fixture —
      this is the phase's highest-leverage unverified seam, per the second review pass.

**Difficulty:** HIGH
**Uncertainty:** Phase 3's DW-3.3 projects whether per-node NN forward passes fit the
laptop-first budget before this phase's real work starts — if that projection says no, fall back
to Approach C (critic-only leaf value, plain UCB1, no prior) per the Rejected Approaches
fallback, rather than a half-built per-node implementation. The mirrored-opponent-prior
approximation (see Approach notes) is the other real unknown — its quality isn't verifiable
until real games are played (Phase 6).

### Phase 6: Benchmark + inventory writeup
**Model:** sonnet
**Skills:** none -- pure measurement/documentation phase, no new code design or clarity work
**Gate:** Minimal
**Depends on:** Phase 1, Phase 5 | **Unlocks:** none
**File scope:** `CLAUDE.md`, `notes/phase-4-m7-and-enhancement-track.md`, `notes/index.md`

**Goal:** Run the real head-to-head measurements both search configurations still need and
document where every asset in the project actually stands, following this project's own
working-history convention (`CLAUDE.md`: "Working history lives in `notes/`, not in this file").

**Scope:**
- IN: `"mcts_puct"` vs `"ppo"` at `--format gen9ou` (the critical comparison — the bar M6b has to
  clear, on the format both were trained on); `"mcts_puct"` vs `"mcts"` at `--format gen9ou`
  (same distribution-mismatch caveat applies — `mcts_puct`'s prior/value come from the gen9ou-
  trained checkpoint) — does the PPO prior/value help over the hand-crafted one; a new
  `notes/phase-4-m7-and-enhancement-track.md` (`type: log`, full tables and CIs for every asset —
  the **project roadmap's** Phase 1 search, Phase 2 supervised, Phase 3 PPO, plus this plan's
  Phase 1 `mcts` and Phase 5 `mcts_puct` — not to be confused with this plan's own phase
  numbering), linked from `notes/index.md`; `CLAUDE.md`'s Phase 4 status
  section reduced to the summary line + a link to that note, matching how Phases 0-3 are recorded.
- OUT: setting up or running any comparison against Foul Play — explicit follow-up decision for
  the user, not built here.

**Constraints:** Report results as measured — a losing result for `mcts_puct` vs `ppo` is
documented the same way a winning one would be, per the confirmed outcome policy.

**Produces:** `notes/phase-4-m7-and-enhancement-track.md` (the artifact the user's
training-continuation decision is made from) + the updated `CLAUDE.md` summary/link.

**Done when:**
- [x] DW-6.1: real 500-battle `mcts_puct` vs `ppo` result at `--format gen9ou`, with Wilson CI,
      in `notes/phase-4-m7-and-enhancement-track.md`. 179-317-4, 35.8% [31.7%, 40.1%] — loses.
- [x] DW-6.2: real 500-battle `mcts_puct` vs `mcts` result at `--format gen9ou`, with Wilson CI,
      in the same note. 302-193-5, 60.4% [56.0%, 64.6%] — wins.
- [x] DW-6.3: the note summarizes every asset's measured standing in one place, linked from
      `notes/index.md`; `CLAUDE.md`'s Phase 4 status carries the summary line + link, not the
      full tables.

**Difficulty:** LOW

---

## Test Coverage
**Level:** Targeted — C++ Catch2 correctness, Python parity tests (np.allclose/exact-equality
against real PyTorch/encode() output), and player/integration wiring (test_mcts_player.py-style).
Exhaustive per-edge-case boundary/dirty-path coverage explicitly trimmed — each phase's Edge
cases above are still handled in code, just not each pinned by a dedicated test unless already
named in a Done-when item.

## Test Plan
- [ ] `tests/test_mcts_player.py` — order translation, `kNoAction` fallback, team-preview edge case (Phase 1)
- [ ] `./scripts/pytest_native.sh` + `ctest --test-dir cpp/build` stay green (Phase 1, DW-1.3)
- [ ] real 500-battle `mcts` benchmark vs random/maxdamage/heuristic/search at default format,
      vs learned/ppo at `--format gen9ou`, budget-aware per DW-1.2 (Phase 1)
- [ ] `scripts/export_weights.py` runs against the real checkpoint and produces `ppo.bin` (Phase 2, DW-2.1)
- [ ] `tests/test_export_weights.py` — exact equality vs real checkpoint tensors; dirty-path
      shape-mismatch fails before any file write (Phase 2)
- [ ] `cpp/tests/test_mlp.cpp` — known-input/known-output forward pass + truncated-file dirty path (Phase 3)
- [ ] `tests/test_native_forward_pass.py` — np.allclose vs real PyTorch policy, actor + critic (Phase 3)
- [ ] forward-pass microbenchmark + ms/turn projection, A-vs-C threshold stated (Phase 3, DW-3.3)
- [ ] `tests/test_native_encoding.py` — np.allclose vs real `encode()` on live battle states (Phase 4)
- [ ] regression: `test_native_legality.py`/`test_mcts_player.py` unchanged after Phase 4's extension (Phase 4)
- [ ] `cpp/tests/test_action.cpp` — Metamon-mapping functions incl. fainted-teammate/bench<5 case (Phase 5)
- [ ] `cpp/tests/test_mcts.cpp` — `select_puct_action` synthetic bandit; fixed-seed determinism;
      `kForcedSwitch` UCB1-only selection + continue-past-to-critic behavior (Phase 5, DW-5.5)
- [ ] critic sign-convention check on synthetic mirrored state pairs, incl. the "neither holds" fallback (Phase 5, DW-5.4)
- [ ] `mirror(BattleState)` round-trip + field-swap + `encode_native` correctness (Phase 5, DW-5.6)
- [ ] measured ms/turn for `mcts_puct` at its chosen `n_simulations`, checked against Phase 3's projection (Phase 5)
- [x] real 500-battle `mcts_puct` vs `ppo` and vs `mcts`, both at `--format gen9ou`, documented in
      `notes/phase-4-m7-and-enhancement-track.md` with `CLAUDE.md` summary/link (Phase 6, DW-6.3)

## Assumptions
| Assumption | Confidence | Verify Before Phase | Fallback If Wrong |
|---|---|---|---|
| `MaskableCategorical`'s masking is "set illegal logits to -inf before softmax" | MED | Phase 5 | Read `sb3_contrib` source directly before implementing the mask |
| Per-node NN forward passes fit the laptop-first ms/turn budget at a workable `n_simulations` | MED | Phase 3 (DW-3.3) | Descope to Approach C (critic-only, no prior) per the plan's named fallback |
| `encode()`'s single-most-recent-hazard semantic is reconstructable from data `battle_state_from_poke_env` already has access to (turn numbers via `side_conditions`) | HIGH | Phase 4 | Re-derive from `encoding.py`'s `_poke_env_hazards` directly, same turn-ranking approach |
| Mirroring `BattleState` to compute the opponent's own PUCT prior via the same `encode_native`/actor path is a valid, accepted approximation (not silently wrong) | MED | Phase 5 | If mirrored-state priors measure as actively harmful (not just weaker), fall back to plain UCB1 on the opponent table only, keep the prior on my side |
| `scripts/train_ppo.py`'s PPO checkpoint was trained on `gen9ou` (per its default config), making that the valid comparison format for `learned`/`ppo`/`mcts_puct` matchups | HIGH | Phase 1 | Confirm the actual training format directly from `train_ppo.py`/checkpoint metadata before running benchmarks |

## Decision Log
| Decision | Alternatives Considered | Rationale | Phase |
|---|---|---|---|
| Approach A: per-node PUCT prior, not root-only | B: root-only prior; C: critic-only, no prior | Matches the actually-confirmed problem statement; B/C both quietly answer a smaller question | 5 |
| `select_puct_action` as a new function, not a flag branch on `select_ucb1_action` | Single function with a mode flag | Flag-selected branch is logical cohesion (REJECT per this project's own cohesion standard) | 5 |
| `SideConditions` extended to store per-token turn data for all 8 tokens; single-most-recent view derived on demand in `encode_native()` | Pre-reduce to one token in the Python translator | Keeps `BattleState` the one stored source of truth, matching its own existing invariant | 4 |
| Fixed 3-layer hardcoded `MlpWeights` struct, not a generalized N-layer MLP framework | Templated generic MLP class | Architecture is fixed and known — a generalized framework is YAGNI here | 3 |
| Header/module named `mlp.hpp`/`mlp.cpp`, not `nnue.hpp` | `nnue.hpp` (the original plan doc's placeholder name) | NNUE means an incrementally-updated sparse accumulator network; this is a plain dense 3-layer MLP forward pass — matches this project's domain-vocabulary-consistency naming standard | 3 |
| Opponent-side PUCT prior computed by mirroring `BattleState` through the same `encode_native`/actor path, not a separate mechanism | Root-only opponent prior; no opponent prior at all (plain UCB1 for `opp_stats`) | `encode()` has no opponent-bench block at all — mirroring reuses the one real code path and stays consistent with this project's existing Tier-1 revealed-only opponent-modeling limitation | 5 |
| `kForcedSwitch` nodes skip the PUCT prior/critic entirely (plain UCB1 + `default_eval`) | Invent a sentinel `encode()` semantic for a missing active mon | `encode()`'s real behavior for this case is untested/undefined — a narrower, documented fallback beats an unverified extension | 5 |
| Phase 6 writes `notes/phase-4-m7-and-enhancement-track.md` (`type: log`); `CLAUDE.md` keeps only a summary + link | All results directly in `CLAUDE.md` | Matches this project's own stated convention ("Working history lives in `notes/`, not in this file") and how Phases 0-3 are actually recorded | 6 |
| `kForcedSwitch` nodes use UCB1-only selection and expansion continues past them to a real `kDecision` state for the critic's leaf value | Fall back to `default_eval` at the forced-switch node itself | Backup is path-wide — mixing `default_eval`'s and the critic's different unbounded scales into shared ancestor `VisitStats` would corrupt every PUCT comparison above the forced switch, not just its own node | 5 |
| Metamon tera labels (9-12) dropped from the actor's prior, remaining distribution renormalized over legal `ActionId`s | Map tera labels onto their non-tera move counterpart | `action.hpp` defers tera actions entirely for this whole phase — no `ActionId` exists to map onto, and folding tera mass into a different action would misrepresent the actor's actual intent | 5 |
| `PolicyWeights::load(path)` returns both branches from one call | Two separate `MlpWeights::load()` calls, one per branch | Phase 5 always uses actor+critic together; one call matches that usage and halves the load-time validation surface | 3 |

---

## Notes

- `module.cpp` (and, for Phases 1/4/5, `mcts_player.py`) is a shared bottleneck file across
  Phases 1, 3, 4, and 5. The DAG is `{1, 2} → 3 (needs 1, 2) → 4 (needs 1, 3) → 5 (needs 1, 3, 4)
  → 6 (needs 1, 5)` specifically so every pair sharing one of those files has a direct or
  transitive dependency edge between them — only Phase 1 and Phase 2 ever share a wave, and they
  touch disjoint files. This is now enforced by the `Depends on` edges themselves, not a separate
  claim layered on top of them.
- `CLAUDE.md` currently has a stray staged deletion in git (`git status` shows `deleted:
  CLAUDE.md` staged, but the file still exists on disk unstaged) from before this plan — worth
  the user's own attention before committing Phase 6's edits to it, not resolved by this plan.

---

## Execution Log

### Phase 6: Benchmark + inventory writeup (Gate: Minimal)
- [x] BUILD: worked directly from the plan phase description (minimal gate, no discovery pass).
      Before either 500-battle run, verification surfaced two more real hang triggers beyond
      435f6a5's fainted-active fix — `force_switch` (pivot moves: U-turn/Volt Switch/Baton Pass)
      and PP-exhausted moves offered as legal — fixed in `ac78712` (dedicated
      `BattleState.my_force_switch` field, plus a general backstop: `MctsPlayer`/
      `MctsPuctPlayer.choose_move` now cross-checks the search's chosen action against poke-env's
      real `battle.available_moves`/`available_switches` before submitting, salvaging the
      next-best real-legal action from the search's visit distribution otherwise). Verified via a
      12-battle gen9ou timing slice that used to hang for over an hour, now completing cleanly and
      repeatably (401.1s, 301.1s). One non-blocking loose end: an intermittent, non-fatal UBSan
      report (addition-overflow in `<c++/v1/array>`) on one of the two verification runs, didn't
      reproduce on the second, checked and judged unrelated to this fix — flagged for follow-up,
      not a blocker.
- [x] Both real 500-battle runs completed with zero hangs on the same Debug/ASan build:
      DW-6.1 (`mcts_puct` vs `ppo`, gen9ou) 179-317-4, 35.8% [31.7%, 40.1%] — clear loss.
      DW-6.2 (`mcts_puct` vs `mcts`, gen9ou) 302-193-5, 60.4% [56.0%, 64.6%] — clear win.
      The PPO actor/critic prior genuinely helps the search over the hand-crafted eval, but the
      extra search on top of the same actor/critic doesn't outperform running the policy alone.
- [x] VERIFY: `./scripts/pytest_native.sh` 209 passed, 3 skipped (pre-existing, unrelated);
      `ctest --test-dir cpp/build` 100/100 passed.
- [x] Committed
Commit: (pending — see below)
Summary: Phase 6 done — the plan's final phase. Real 500-battle benchmarks for both remaining
matchups, `notes/phase-4-m7-and-enhancement-track.md` written (full asset inventory, every phase's
measured standing in one place), linked from `notes/index.md`, `CLAUDE.md`'s Phase 4 status
trimmed to a summary line + link matching Phases 0-3. Plan complete.

### Phase 5: M6b — PUCT search with PPO prior/value (Gate: Full)
- [x] BUILD: Discovery + design + implementation complete (fable-tier), plus two fix-forward
      passes (see below).
- [x] REVIEW: **fail → fail → pass (3rd attempt)**. 1st FAIL (3 issues: unevidenced DW-5.3
      comparison, untested DW-5.6 vector-parity claim, zero `MctsPuctPlayer`/`mcts_puct` CLI
      coverage) — all fixed with new tests + a durable `notes/` record, independently
      re-verified. 2nd FAIL (1 narrow doc gap: DW-5.4's plan-mandated fallback policy was never
      written down, though the runtime already implemented it correctly) — fixed with one
      doc-comment paragraph. 3rd attempt: PASS, all six DW items verified with execution
      evidence — see
      `.code-foundations/build/2026-08-24-battle-engine-phase4-cpp-search-core-phase-5-review-attempt3.md`.
      Throughout all three attempts the core PUCT algorithm itself (value-scale-mixing avoidance,
      mirror mechanism, tera renormalization, weights-loaded-once, the measured sign convention)
      was independently traced and reproduced as correct — every FAIL was on evidence/coverage
      gaps around already-correct logic, not on the logic itself.
- [x] Committed
Commit: 5979cb8
Summary: M6b done — the phase's actual strength bet. PUCT search using the PPO actor as a
per-node prior (via `encode_native`+forward pass, renormalized over legal `ActionId`s, tera
labels dropped) and its critic as leaf value (measured `-v` sign convention, mean -0.0659237,
stddev 1.49e-08). Opponent-side priors computed via a verified `mirror(BattleState)` (round-trips,
every field swapped, vector-parity-matched against a real opponent-POV `encode()`).
`kForcedSwitch` nodes use UCB1-only selection and continue expansion to a real decision state
rather than mixing `default_eval`/critic value scales. Measured 501.9ms/turn at
`n_simulations=200`; Phase 6's full sweep projects to ~2.1 hours (`notes/phase-5-mcts-puct-ms-
per-turn-vs-phase-3-projection.md`), well inside the laptop-first overnight budget. Next: Phase 6
— the real benchmarks and the `notes/` writeup the training-continuation decision is made from.

### Phase 4: M4b — Full BattleState extension + encode() port (Gate: Full)
- [x] BUILD: Discovery + design + implementation complete (fable-tier).
- [x] REVIEW: Verification passed — see
      `.code-foundations/build/2026-08-24-battle-engine-phase4-cpp-search-core-phase-4-review.md`.
      Hand-traced the hazard tie-break logic and species-identity threading independently; both
      correct. No FAILs.
- [x] Committed
Commit: 2fe0a54
Summary: M4b done — `BattleState` extended (species/item/ability/protect_counter/weather/
terrain/8-token hazards), `encode_native()` ports `encode()` bit-for-bit including the
single-most-recent-hazard reduction and species-sorted bench view (keyed on `base_species`).
Move-summary features computed once in Python via `encoding.py`'s own verified helper rather
than re-derived in C++ (movedex_table.hpp lacked the needed flags, out of scope to extend) —
divergence structurally impossible by construction. 3-way `VECTOR_LEN` cross-check confirmed.
Next: Phase 5 (M6b — the PUCT search itself, the phase's actual strength bet).

### Phase 3: M3 — C++ NN forward pass (Gate: Standard)
- [x] BUILD: Discovery + design + implementation complete.
- [x] REVIEW: DEFERRED — batch pending (tests green at commit: ctest 76/76, pytest_native 182
      passed / 3 pre-existing skips).
- Covered by batch review 2026-08-25 (phases 2-3) — PASS, no findings. Two non-blocking notes:
  DW-3.3's ms/turn projection was measured under Debug/ASan (likely conservative vs. release);
  re-measure under `--release` before Phase 5 tunes `n_simulations` against it.
- [x] Committed
Commit: aa4c485
Summary: M3 done — `MlpWeights`/`PolicyWeights` C++ forward pass reading M2's `ppo.bin`, verified
vs. real PyTorch (~9e-5 divergence). DW-3.3 measured 1.51ms/node forward-pass cost and projected
Phase 5's full sweep at 6-9 hours — **confirms Approach A (per-node PUCT) fits the laptop-first
budget**, Approach C's fallback is not needed. Un-reviewed set: {Phase 2, Phase 3} — cadence (2)
reached AND Phase 4 is Full gate, so a batch REVIEW fires now before Phase 4 opens.

### Phase 2: M2 — Weight export tooling (Gate: Standard)
- [x] BUILD: Discovery + design + implementation complete.
- [x] REVIEW: DEFERRED — batch pending (tests green at commit: 6/6 new tests, full suite 175
      passed / 3 pre-existing skips, no regressions).
- Covered by batch review 2026-08-25 (phases 2-3) — PASS, no findings.
- [x] Committed
Commit: 90434fb
Summary: M2 done — `scripts/export_weights.py` dumps the real PPO checkpoint's actor+critic
weights to `data/cpp_weights/ppo.bin` (magic/version/VECTOR_LEN header, 6 layers, row-major
float32), validated shape-mismatch-before-write. Un-reviewed set: {Phase 2}. Next: Phase 3 (M3
C++ NN forward pass), which will trip the batch-review trigger at cadence 2 once it also lands.

### Phase 1: M7 — MctsPlayer + benchmark wiring (Gate: Full)
- [x] BUILD: Discovery + design + implementation complete, verified by a resumed BUILD pass
      (see `.code-foundations/build/2026-08-24-battle-engine-phase4-cpp-search-core-phase-1-discovery.md`).
- [x] REVIEW: Verification passed — see
      `.code-foundations/build/2026-08-24-battle-engine-phase4-cpp-search-core-phase-1-review.md`.
      No FAILs; all three DW items verified with execution evidence.
- [x] Committed
Commit: 69f5059
Summary: M7 done — `search()`/`default_eval` bound (GIL-safe), `MctsPlayer` wired into the
benchmark harness, real 500-battle sweep measured (beats random/maxdamage, loses to
heuristic/search/ppo, ~even with learned) — this hand-crafted-eval configuration doesn't beat
search/learned/RL, consistent with M0. `n_simulations=200` held throughout. Next: Phase 2 (M2
weight export).

**Note:** `git add .` on the first commit attempt accidentally swept in an unrelated pre-existing
`CLAUDE.md` diff (a notes/-convention migration from before this session, unrelated to Phase 4).
Caught before reporting, fixed via `git reset --soft HEAD~1` + `git restore --staged CLAUDE.md`
+ re-commit — no content was lost, `CLAUDE.md` is back to its pre-existing uncommitted state in
the working tree. Stage explicitly by file scope for every phase from here on, not `git add .`.

DW-1.1: `./scripts/pytest_native.sh` — 169 passed, 3 skipped (pre-existing, unrelated), includes
6/6 `tests/test_mcts_player.py` tests. PASS.
DW-1.2: real 500-battle benchmark sweep, `n_simulations=200` (sanity-timed first, ~100-110 min
projected for the full sweep, within budget — no trim applied):

| Opponent | Format | Record | Win rate | 95% Wilson CI |
|---|---|---|---|---|
| random | gen9randombattle | 477/500 | 95.4% | [93.2%, 96.9%] |
| maxdamage | gen9randombattle | 354/500 | 70.8% | [66.7%, 74.6%] |
| heuristic | gen9randombattle | 158/500 | 31.6% | [27.7%, 35.8%] |
| search | gen9randombattle | 162/500 | 32.4% | [28.4%, 36.6%] |
| learned | gen9ou | 236/500 | 47.2% | [42.9%, 51.6%] |
| ppo | gen9ou | 151/500 | 30.2% | [26.3%, 34.4%] |

Consistent with M0's earlier Python-prototype finding: this hand-crafted-eval MCTS/DUCT
configuration beats non-search baselines comfortably but loses to every search/learned/RL
opponent, landing in the same ~30-32% band M0 measured. A real, honest result — not a blocker,
matches the plan's own "report honestly, move on" outcome policy.
DW-1.3: `ctest --test-dir cpp/build` — 66/66 passed. PASS.

One additional in-scope fix made during the sweep: `scripts/benchmark.py` now passes
`AccountConfiguration.generate(name, rand=True)` to every player it constructs — a killed
orphaned benchmark process (from this session's own process-management, see below) had left a
stale "already a challenge" state keyed by userid on the local Showdown server (challenges never
time out, verified against `pokemon-showdown/server/ladders.ts`), which repeated CLI invocations
with default usernames would otherwise collide on. Full suite re-verified green after this change.

**Process-management note (orchestrator, not a code issue):** the first BUILD dispatch for this
phase left a background benchmark process running past its own agent turn ending, and a
second, redundant dispatch briefly ran in parallel before being stopped — both were caught and
cleaned up (stray processes killed) before the final, clean resumed BUILD run above, which is
the authoritative one. No duplicate or conflicting file changes resulted; verified via `git
status` at each step.

Summary: Phase 1 (M7) is fully implemented, tested, and measured — hand-crafted-eval MCTS/DUCT
loses to search/learned/PPO (~30-32%), beats non-search baselines. Only the blocking REVIEW and
commit remain before Phase 2 can start.

