# Review: Phase 2 - M2: Weight export tooling

## Executed Results (Step 0)
- `./scripts/pytest_native.sh` (full suite) → 182 passed, 3 pre-existing skips
- `./scripts/pytest_native.sh tests/test_export_weights.py tests/test_native_forward_pass.py -v` → 13/13 passed
- `ctest --test-dir cpp/build` (clean rebuild via `cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug && cmake --build cpp/build -j`) → 76/76 passed, no compiler warnings emitted
- Direct CLI run: `.venv/bin/python scripts/export_weights.py --checkpoint data/models/ppo.zip --output /tmp/ppo_check.bin` → wrote 751732 bytes, byte-identical (`diff` on `xxd` output) to the checked-in `data/cpp_weights/ppo.bin`
- Independent standalone parse of `data/cpp_weights/ppo.bin` via a hand-written `struct.unpack` script (not `export_weights.py`'s own code) → header/6-layer shapes match the documented contract exactly (665→128→64→13 actor, 665→128→64→1 critic), consumed byte count equals file size exactly (no trailing/missing bytes)

## Requirement Fulfillment

### DW-2.1
PREMISE:  `scripts/export_weights.py` runs against the real `data/models/ppo.zip` and produces `data/cpp_weights/ppo.bin`.
EVIDENCE: `scripts/export_weights.py:130-147` (`export_weights`), `:150-160` (CLI defaults `data/models/ppo.zip`/`data/cpp_weights/ppo.bin`)
TRACE:    ran `.venv/bin/python scripts/export_weights.py --checkpoint data/models/ppo.zip --output /tmp/ppo_check.bin` directly → wrote 751732 bytes, byte-identical to the repo's existing `data/cpp_weights/ppo.bin`.
VERDICT:  PASS

### DW-2.2
PREMISE:  `tests/test_export_weights.py` passes — exact equality (not tolerance) between the dumped bytes and the real checkpoint's tensors.
EVIDENCE: `tests/test_export_weights.py:114-140` (`test_export_weights_exactly_matches_real_checkpoint_tensors`, uses `np.array_equal`, never `np.allclose`)
TRACE:    ran the test directly → PASSED. Independently re-verified: parsed the real `ppo.bin` myself via a standalone `struct.unpack` script and confirmed shapes/byte-length match the documented contract with no leftover bytes.
VERDICT:  PASS

### DW-2.3
PREMISE:  a synthetic shape-mismatched checkpoint (dirty-path test) raises before any file is written — no partial `ppo.bin` is ever produced.
EVIDENCE: `scripts/export_weights.py:130-139` (validation via `_validate_layer_shapes` happens fully before the `bytearray`/`write_bytes` block at :141-147); `tests/test_export_weights.py:143-165` (`test_export_weights_raises_on_shape_mismatch_before_writing_file`, builds a real `MaskablePPO` checkpoint with `net_arch=[7, 5]`, asserts `pytest.raises(ValueError, match="expected weight shape")` and `not output_path.exists()`)
TRACE:    ran the test directly → PASSED. Read the code path myself: `export_weights()` calls `load_policy` → `_extract_layers` → `_validate_layer_shapes` (raises `ValueError` on first shape mismatch) strictly before any `bytearray`/`output_path.write_bytes` occurs — no code path writes partial output.
VERDICT:  PASS

**All requirements met:** YES

## Test-DW Coverage
- [x] DW-2.1 — automated test (`test_export_weights_writes_binary_against_real_checkpoint`) + my own direct CLI execution
- [x] DW-2.2 — automated test (`test_export_weights_exactly_matches_real_checkpoint_tensors`, exact-equality via `np.array_equal`)
- [x] DW-2.3 — automated test (`test_export_weights_raises_on_shape_mismatch_before_writing_file`)
- Additional tests beyond the DW items: missing-checkpoint error message, missing-output-directory auto-creation, header/format contract shape — all green, targeted level matches the stated Test Coverage.

## Dead Code
None found. All imports in `scripts/export_weights.py` (`argparse`, `struct`, `Path`, `MaskablePPO`, `MaskableActorCriticPolicy`, `nn`, `ACTION_SPACE_SIZE`, `VECTOR_LEN`, `WARM_START_NET_ARCH`) are used. No unreachable code after early returns/raises.

## Correctness Dimensions
| Dimension | Status | Evidence |
|-----------|--------|----------|
| Concurrency | N/A | Single-threaded CLI script, no shared mutable state, no async |
| Error Handling | PASS | `load_policy` wraps `MaskablePPO.load`'s exception in a caller-facing `RuntimeError` (`scripts/export_weights.py:80-84`); missing-file path validated explicitly (`:78-79`, `FileNotFoundError`); shape mismatch is a clear `ValueError` naming the offending layer and both shapes (`:109-116`) |
| Resources | PASS | Single `write_bytes` call after `mkdir(parents=True, exist_ok=True)`; no file handles left open across error paths (uses `Path.write_bytes`, not a manually-managed handle) |
| Boundaries | PASS | `_validate_layer_shapes` checks every one of the 6 layers' `(out_dim, in_dim)` tuple against `_LAYER_SPECS` before any write; demonstrated via the shape-mismatch test with a real `net_arch=[7, 5]` checkpoint |
| Security | N/A | `checkpoint_path`/`output_path` are local CLI arguments from a trusted operator, not remote/untrusted input; `data/cpp_weights/ppo.bin` is explicitly documented in the plan as a locally-generated, gitignored artifact never exposed to untrusted parties |

## Loaded-Skill Criteria

| Skill | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| cc-defensive-programming | External input (checkpoint path) validated at entry | PASS | `load_policy` checks `.exists()` before calling `MaskablePPO.load`, raises `FileNotFoundError` with a clear message (`scripts/export_weights.py:78-79`) |
| cc-defensive-programming | No empty catch blocks | PASS | The one `except Exception as e:` (`:82-83`) re-raises as `RuntimeError(...) from e`, preserving the chain — not swallowed |
| aposd-designing-deep-modules | Interface depth / simplicity | PASS | `export_weights(checkpoint_path, output_path) -> None` is the only entry point a caller needs; internal helpers (`_extract_layers`, `_validate_layer_shapes`, `_pack_layer`) are private and hide the actual layer-correspondence/packing complexity |
| cc-pseudocode-programming | Design intent documented before/alongside code | PASS | Module docstring states the exact binary layout up front; each function's docstring explains rationale (e.g. why validation precedes writing) — reads as pseudocode-derived, not reverse-documented |

## Notes (non-blocking)
- The module docstring (`scripts/export_weights.py:1-27`) restates the binary format in full, duplicating the field list also in `PolicyWeights::load`'s doc comment (`cpp/include/be/mlp.hpp:73-91`) and the dispatch prompt itself. Triple-documented; not a defect, but a future format change has three places to keep in sync (`WEIGHTS_FORMAT_VERSION`/`version` check does guard against silent drift at runtime, so this is a comment-maintenance note only, not a correctness risk).
- `_pack_layer` computes `out_dim, in_dim = weight.shape` from the live tensor rather than trusting `_LAYER_SPECS`' pinned values — correct defensive choice (the packed header always reflects the tensor actually being written), just noting it as a deliberate, good design detail.

## Issues (if FAIL)
None.

**Verdict: PASS**

---

# Review: Phase 3 - M3: C++ NN forward pass

## Executed Results (Step 0)
Same run as Phase 2 above (single suite covers both):
- `./scripts/pytest_native.sh` → 182 passed, 3 pre-existing skips
- `./scripts/pytest_native.sh tests/test_native_forward_pass.py -v` → 7/7 passed (3 actor-parity seeds, 3 critic-parity seeds, 1 shape-regression test)
- `ctest --test-dir cpp/build` (clean rebuild) → 76/76 passed, including all `[mlp]`-tagged cases
- `./cpp/build/tests/be_tests "[mlp][!benchmark]"` → ran the hidden microbenchmark explicitly: actor forward 749µs, critic forward 738µs, actor+critic combined 1.501ms (mean, 100 samples)
- Independent standalone C++ program (not part of the test suite) compiled against `cpp/src/mlp.cpp` directly, loaded the real `data/cpp_weights/ppo.bin`, ran both branches → loaded shapes and forward-pass output matched expectations, no crash
- Adversarial probe I wrote myself (ASan+UBSan build): a header declaring `out_dim = 0x7FFFFFFF` on a file too short to back it → `PolicyWeights::load` threw a clean `std::runtime_error` ("truncated weight file - short read..."), not a crash

## Requirement Fulfillment

### DW-3.1
PREMISE:  Catch2 tests in `cpp/tests/test_mlp.cpp` cover a known-input/known-output case and a truncated/malformed weight file (load-time error, not a crash).
EVIDENCE: `cpp/tests/test_mlp.cpp:87-119` (two hand-computed-output cases, one exercising ReLU clamping), `:136-206` (six distinct malformed-file cases: truncated mid-layer, truncated header, bad magic, unsupported version, vector_len/layer0.in_dim disagreement, cross-layer dimension-chain mismatch, nonexistent path)
TRACE:    input `[1,1]` through a hand-specified 2→2→2→1 network → traced by hand in the test's own comment to `5.5`, code returns `Approx(5.5)`; ran all 8 `[mlp]`-tagged `PolicyWeights::load` cases → each `REQUIRE_THROWS_AS(..., std::runtime_error)` passed.
VERDICT:  PASS

### DW-3.2
PREMISE:  `tests/test_native_forward_pass.py` — same input vector through the real PyTorch policy and the C++ forward pass, `np.allclose` on both actor logits and critic value.
EVIDENCE: `tests/test_native_forward_pass.py:49-60` (`_torch_forward`, calls `policy.action_net(policy.mlp_extractor.policy_net(x))` / `policy.value_net(policy.mlp_extractor.value_net(x))` directly — the real trained policy, not a mock), `:75-98` (`np.allclose(..., atol=1e-3, rtol=1e-3)` on both branches, 3 random seeds each)
TRACE:    ran the 6 parametrized tests directly against the real `data/models/ppo.zip` and `data/cpp_weights/ppo.bin` → all 6 PASSED. Cross-checked the claimed correspondence (`policy.mlp_extractor.policy_net[0]`→`actor.net[0]` etc.) against `battle_engine/ppo_warm_start.py:32-36`'s own documented layer mapping — exact match, not re-derived independently by the test.
VERDICT:  PASS

### DW-3.3
PREMISE:  a microbenchmark of one actor+critic forward pass, combined with Phase 1's measured per-simulation cost, projecting ms/turn for a later PUCT-search phase and stating whether it fits a laptop-first overnight budget.
EVIDENCE: `cpp/tests/test_mlp.cpp:216-228` (Catch2 `BENCHMARK` cases, tagged `[!benchmark]` so they're excluded from a normal `ctest` run but runnable explicitly)
TRACE:    ran `./cpp/build/tests/be_tests "[mlp][!benchmark]"` myself → actor 749µs, critic 738µs, combined actor+critic 1.501ms mean over 100 samples on the real `ppo.bin`. This matches the plan's own Execution Log figure (1.51ms/node) to 3 significant figures, corroborating that the recorded number is real and reproducible rather than fabricated.
The combined "ms/turn projection + laptop-budget threshold statement" itself is documentation/analysis, not code — it lives in the plan's Execution Log (not something a test asserts). I independently sanity-checked the arithmetic rather than trusting that account: Phase 1's own measured `mcts` (plain `default_eval`, `n_simulations=200`) timing is ~0.8-4.7s/battle depending on format/opponent. Adding ~1.5ms/node for a PUCT variant that does one NN forward per node expansion (1 new node/simulation, matching this project's existing incremental-expansion MCTS design) gives roughly 200 sims x 1.5ms = 300ms/decision; over a full 6-matchup x 500-battle sweep at plausible turn counts this lands in the same single-digit-hours order of magnitude the plan claims (6-9 hours) — internally consistent, not verified to the ms.
VERDICT:  PASS (microbenchmark itself independently reproduced; the projection/threshold statement is a documentation deliverable satisfied via the plan's Execution Log, corroborated but not independently re-derived to full precision — see Notes)

**All requirements met:** YES

## Test-DW Coverage
- [x] DW-3.1 — automated tests (8 `PolicyWeights::load` cases + 2 `forward` cases in `cpp/tests/test_mlp.cpp`, all ran via `ctest`)
- [x] DW-3.2 — automated tests (`tests/test_native_forward_pass.py`, 6 parametrized parity cases, ran via `pytest_native.sh`)
- [x] DW-3.3 — no automated test can assert a "does this fit an overnight budget" business judgment; covered via my own recorded observed behavior (ran the microbenchmark directly, independently reproduced the claimed number) per Step 2's fallback path for non-testable items

## Dead Code
None found in `cpp/include/be/mlp.hpp`, `cpp/src/mlp.cpp`, or the `mlp`-related additions to `cpp/bindings/module.cpp`. Every helper (`read_exact`, `read_u32`, `read_layer`, `check_chain`, `check_input_width`) is used exactly once by `PolicyWeights::load`, no unreachable code after any `throw`.

## Correctness Dimensions
| Dimension | Status | Evidence |
|-----------|--------|----------|
| Concurrency | N/A | `PolicyWeights::load` is documented and used as a one-time, call-once-at-construction load; `forward()` is a pure function over its own `input` parameter and `const` member data, no shared mutable state |
| Error Handling | PASS | Every field read (`read_exact`) throws `std::runtime_error` naming what failed on short read; magic/version/dimension/chain checks each throw a distinct, caller-facing message (`cpp/src/mlp.cpp:92-129`); demonstrated via 6 distinct malformed-file test cases, all passing |
| Resources | PASS | `std::ifstream in(path, ...)` is a stack-local RAII object; every throw path unwinds through its destructor, closing the file handle — no leak on any error path |
| Boundaries | PASS (load path); Note (forward path) | `load()`'s dimension checks (`out_dim <= 0`, chain mismatches, vector_len disagreement) are demonstrated correct via the 6 malformed-file tests plus my own adversarial probe (a declared `0x7FFFFFFF` dimension on a too-short file threw cleanly, did not crash, matching the listed edge case). `forward()` does not validate `input.size() == layer0.in_dim` — demonstrated via a test I wrote that passes an empty vector and gets a SEGV under ASan/UBSan. This is explicitly documented in `mlp.hpp:56-59` as a deliberate "caller owns its inputs" convention matching existing codebase precedent (`legal_actions()`/`resolve_turn()`), and is outside the dispatch's listed Edge Cases (which scope specifically to "malformed/truncated weight file", i.e. `load()`, not `forward()`'s runtime input) — see Notes, not a FAIL |
| Security | N/A | `data/cpp_weights/ppo.bin` is explicitly documented (plan, EXPLORE section) as a locally-generated, gitignored, non-user-supplied artifact; still dimension/version-validated at load regardless, which the tests demonstrate |

## Loaded-Skill Criteria

| Skill | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| cc-defensive-programming | Assertion vs error handling: I/O boundary uses error handling (exceptions), not assertions | PASS | `PolicyWeights::load`'s own doc comment (`mlp.hpp:96-105`) explicitly reasons through this choice — a one-time I/O boundary gets exceptions, matching the project's stated convention; `forward()`'s in-process caller contract instead uses the assertion-free "caller owns inputs" convention already established elsewhere in the codebase, consistent with the skill's own table row ("Internal interface, same module → Assertion, No error handling") |
| cc-defensive-programming | No empty catch blocks | PASS | No catch blocks at all in `mlp.cpp` — errors propagate via `throw`, correctly per the module's exception-based strategy |
| cc-defensive-programming | External input validated at entry | PASS | `load()` validates magic/version/every dimension/every chain link before returning a usable `PolicyWeights`, demonstrated by all 6 malformed-file tests plus my own out-of-range-dimension probe |
| aposd-designing-deep-modules | Not a shared trunk between actor/critic (explicit design requirement) | PASS | `PolicyWeights` has two fully separate `MlpWeights actor;` / `MlpWeights critic;` members (`mlp.hpp:69-71`), each independently populated by its own 3 `read_layer` calls in `load()` (`mlp.cpp:114-119`) — verified structurally, not just by comment claim |
| aposd-designing-deep-modules | Interface depth | PASS | `MlpWeights::forward(input)` and `PolicyWeights::load(path)` are the only two public entry points; both hide the layer-chain/byte-layout complexity entirely |
| Naming-accuracy requirement (plan-pinned, checked per dispatch) | Module named `mlp.hpp`/`mlp.cpp`, not `nnue.hpp` | PASS | Confirmed via `ls cpp/include/be/ cpp/src/` — files are named `mlp.hpp`/`mlp.cpp`; plan's own decision table (line 468) records the rationale (NNUE means an incrementally-updated sparse accumulator net; this is a plain dense MLP) |

## Notes (non-blocking)
- **`forward()` has no bounds check on `input.size()`.** Confidence: high (demonstrated by a test I wrote and ran — an empty input vector produces an ASan SEGV, not a graceful error). Severity: low. Not a FAIL: it's explicitly documented as intentional in `mlp.hpp:56-59`, matches an established codebase convention (`legal_actions()`/`resolve_turn()` also don't validate caller-owned inputs), and is outside the dispatch's Edge Cases list (which is scoped to the weight *file*, not `forward()`'s runtime argument). Worth flagging because `MlpWeights::forward` is exposed directly to Python via `module.cpp:183-187` with no additional guard at the binding layer either — a future Python caller (e.g. Phase 5's PUCT expansion code) that passes a wrong-length encoded vector will crash the whole process rather than raising a catchable Python exception. Whether that's acceptable depends on how disciplined Phase 5's own encode()-to-forward() plumbing is; flagging now while it's cheap to note.
- **DW-3.3's combined "ms/turn projection + budget threshold" analysis has no durable, independently-checkable artifact** (no notes/ log entry, no code comment carrying the arithmetic) beyond the plan's own Execution Log summary paragraph. The raw microbenchmark number is independently reproducible (I did so, and it matched), but the downstream arithmetic combining it with Phase 1's per-battle timing data into an hours-long sweep estimate is not preserved anywhere I could re-verify to precision — only sanity-checked to order of magnitude. Not a FAIL (DW-3.3 doesn't mandate a specific recording location, and this project's own CLAUDE.md convention for "the session's own record, with exact numbers" is a `notes/*.md` file that wasn't written for this specific finding), but a real documentation-durability gap worth closing before Phase 5 needs to rely on this number.
- The microbenchmark was run under the Debug/ASan/UBSan build (project default; `--release` was not used for this measurement). ASan/UBSan instrumentation typically inflates wall-clock cost meaningfully (often 2x or more) relative to an optimized release build. This means the 1.5ms/node figure — and any ms/turn projection built on it — is likely a conservative (worse-than-real) upper bound, not an underestimate, so it doesn't undermine the "fits the budget" conclusion. But it does mean the true release-build number is unmeasured and unstated; worth a release-mode remeasurement before treating 1.5ms/node as the number Phase 5 tunes `n_simulations` against.

## Issues (if FAIL)
None.

**Verdict: PASS**

---

## Cross-Phase Coherence

- **Binary format contract match, verified independently.** Loaded the real `data/cpp_weights/ppo.bin` (already on disk, produced by Phase 2's script) through a hand-written standalone C++ program linked directly against `cpp/src/mlp.cpp` (not through the pytest fixture, not through `ctest`) — `PolicyWeights::load` parsed it without error and reported the exact expected shapes (665→128→64→{13,1} for both branches). Also independently parsed the same file with a hand-written Python `struct.unpack` script (not reusing `export_weights.py`'s own packing code) and got byte-for-byte consistent field values with zero trailing/missing bytes. Both loaders agree with each other and with the documented contract — the seam is solid, not just "the two test suites both happen to pass."
- **No regression.** `git show --stat` on both commits (90434fb for Phase 2, aa4c485 for Phase 3) confirms Phase 3's commit touched zero files from Phase 2's file list (`scripts/export_weights.py`, `tests/test_export_weights.py` untouched). Ran `tests/test_export_weights.py` after Phase 3's build was in place (clean `cpp/build` rebuild first) — still 6/6 passing.
- **Interface usage matches what Phase 2 actually exposes.** Phase 3's `mlp.hpp`/`mlp.cpp` read the binary format Phase 2's module docstring specifies (magic/version/vector_len/6-layers-in-order/out_dim,in_dim,weight,bias) — it does not call into `export_weights.py` at runtime at all (correctly; Phase 3 is a pure consumer of the artifact, not the script), so there's no live-interface coupling beyond the byte contract, which is verified above.
- **No contradictions between phase outputs.** Both phases agree on `VECTOR_LEN=665`, `ACTION_SPACE_SIZE=13`, and `WARM_START_NET_ARCH=[128,64]` — confirmed by importing all three constants directly from `battle_engine` and cross-checking against both the packed header and the C++ loader's reported shapes.

**No cross-phase defects found.**

---

## Summary

Phase 2: PASS — clean, no findings above Notes.
Phase 3: PASS — clean; one documented-and-accepted design tradeoff (`forward()`'s unchecked input size, matching existing codebase convention, outside the stated Edge Cases) and one documentation-durability gap (DW-3.3's projection isn't preserved anywhere precisely re-derivable) noted for awareness, neither rising to a blocker under the stated Verdict Rules.

**Overall Verdict: PASS**
