# Discovery + Design: Phase 2 - M2 — Weight export tooling

## Files Found
- `data/models/ppo.zip` exists (real trained Phase-3 checkpoint, warm-started + self-play trained).
- `battle_engine/ppo_warm_start.py` — the documented layer correspondence to reuse (module docstring
  lines 28-41 + `load_warm_start_weights`).
- `battle_engine/encoding.py` — `VECTOR_LEN` (currently 665, confirmed live against the real checkpoint below).
- `battle_engine/dataset.py` — `ACTION_SPACE_SIZE = 13`.
- `scripts/export_weights.py`, `tests/test_export_weights.py`, `data/cpp_weights/` — none exist yet; this
  phase creates all three.

## Current State
Loaded the real `data/models/ppo.zip` via `MaskablePPO.load(..., env=None)` (same pattern `ppo_eval.load_ppo_player`
already uses) to confirm actual shapes rather than assume them:

| Layer | shape (out, in) |
|---|---|
| `policy.mlp_extractor.policy_net[0]` | (128, 665) |
| `policy.mlp_extractor.policy_net[2]` | (64, 128) |
| `policy.action_net` | (13, 64) |
| `policy.mlp_extractor.value_net[0]` | (128, 665) |
| `policy.mlp_extractor.value_net[2]` | (64, 128) |
| `policy.value_net` | (1, 64) |

Confirms: `VECTOR_LEN=665` matches both trunk input dims; `WARM_START_NET_ARCH=[128,64]` matches every
hidden dim; `ACTION_SPACE_SIZE=13` matches `action_net`'s output dim; the critic's final layer is always
a scalar (1) — inherent to how PPO's value function works, not pulled from any project constant.
PyTorch's `nn.Linear.weight` is already `(out_features, in_features)` row-major — no transpose needed to
match the plan's pinned `(out, in)` byte layout.

## Gaps
None — the plan's file scope, format contract, and layer correspondence all match what's actually in the
checkpoint and in `ppo_warm_start.py`. No plan/reality mismatch found.

## Code Standards
- Raise with a caller-facing message in it, never a bare assert (`docs/code-standards.md` Error Handling).
  Applies to both edge cases: missing checkpoint path and shape mismatch.
- Never invent a magic number without naming it — `FORMAT_VERSION`, `MAGIC`, and the value-output-dim-is-1
  constant all get named with a one-line rationale comment.
- Import order: stdlib -> `from __future__ import annotations` -> third-party -> internal `battle_engine.*`.
- `scripts/*.py` convention (from `benchmark.py`/`train_win_prob.py`): module docstring with a runnable
  example, `argparse` via a `parse_args()` function, a `main()` guarded by `if __name__ == "__main__":`.
- Exemplar file per code-standards.md: `battle_engine/ppo_warm_start.py` is explicitly named as "the
  required reference for `scripts/export_weights.py`" — reuse its layer correspondence and its
  shape-mismatch `ValueError` pattern (`_copy_linear`) rather than re-deriving either.

## Test Infrastructure
- pytest, no native extension involved here (pure Python + file I/O) — plain `.venv/bin/pytest`, no
  `pytest_native.sh` wrapper needed.
- `tests/test_ppo_warm_start.py` is the closest sibling: loads the real `data/models/{imitation,win_prob}.pt`
  files directly (not mocks) and asserts `torch.testing.assert_close` against real weights — this project's
  established pattern for weight-transplant correctness is "compare against the real artifact," not a
  synthetic double. `tests/test_export_weights.py` follows the same shape for DW-2.1/DW-2.2: load the real
  `data/models/ppo.zip`, run the real export, and byte-compare against the real checkpoint's tensors.
- For DW-2.3 (dirty-path), a synthetic checkpoint is unavoidable — need a policy with a deliberately wrong
  layer shape. Cheapest approach: build a `MaskablePPO` with a stub env and `net_arch=[7, 5]` (deliberately
  NOT `WARM_START_NET_ARCH`), save it to a temp path, and confirm export raises `ValueError` before writing
  `ppo.bin`, using `tmp_path`.

## DW Verification

| DW-ID | Done-When Item | Status | Test Cases |
|-------|---------------|--------|------------|
| DW-2.1 | `scripts/export_weights.py` runs against the real `data/models/ppo.zip` and produces `data/cpp_weights/ppo.bin`. | COVERED | `test_export_weights_writes_binary_against_real_checkpoint` (invokes `export_weights()` against the real checkpoint path, asserts the output file exists and is non-empty) |
| DW-2.2 | `tests/test_export_weights.py` passes — exact equality (not tolerance) between the dumped bytes and the real checkpoint's tensors. | COVERED | `test_export_weights_writes_binary_against_real_checkpoint` (parses the written `ppo.bin` back into per-layer `(out_dim, in_dim, weight, bias)` tuples and asserts exact float32 byte/array equality against `MaskablePPO.load(...)`'s live tensors — `np.array_equal`, not `np.allclose`); `test_export_weights_header_matches_format_contract` (magic/version/vector_len exact match) |
| DW-2.3 | A synthetic shape-mismatched checkpoint (dirty-path test) raises before any file is written — no partial `ppo.bin` is ever produced. | COVERED | `test_export_weights_raises_on_shape_mismatch_before_writing_file` (synthetic `net_arch=[7,5]` checkpoint, asserts `ValueError` raised and the target output path does not exist afterward) |

**All items COVERED:** YES

## Design Decisions
- **Validate-then-write, single buffered write.** Build the full output in an in-memory `bytearray`/`BytesIO`
  and validate every layer's shape against the expected spec *before* touching the filesystem; write it to
  disk with one `Path.write_bytes()` call at the end. This makes "no partial file on failure" true by
  construction, not just true because validation happens to run first — matches `docs/code-standards.md`'s
  defensive-programming standard applied at the barricade (checkpoint I/O is the external boundary; internal
  packing after validation is assumed safe).
- **Expected shapes are derived from named constants, never hardcoded numbers.** `VECTOR_LEN` (from
  `encoding.py`), `WARM_START_NET_ARCH` (from `ppo_warm_start.py`), `ACTION_SPACE_SIZE` (from `dataset.py`),
  and one new local constant `CRITIC_OUTPUT_DIM = 1` (named with a comment: PPO's value head is always
  scalar — not derived from any shared project constant because none exists for it).
  Considered pulling `1` from `policy.value_net.out_features` of the *loaded* checkpoint instead of a
  constant, but that would validate the checkpoint against itself (always trivially true) rather than
  against an independent expectation — rejected.
- **Missing/unreadable checkpoint path**: check `Path.exists()` first and raise `FileNotFoundError` with the
  attempted path in the message; wrap `MaskablePPO.load()` itself in `try/except Exception as e: raise
  RuntimeError(...) from e` so any other load failure (corrupt zip, version mismatch) also surfaces as a
  clear message with the real cause chained, not a raw traceback into `sb3_contrib` internals.
  Confirmed real error text: an existing directory that isn't a valid checkpoint zip fails inside
  `MaskablePPO.load` with a low-level exception - wrapping it is what code-standards.md's "clear error, not
  a stack trace" languages actually protects against.
- **Layer order and naming**: reuse `ppo_warm_start.py`'s exact correspondence and naming vocabulary
  (`actor.net[0]`, `actor.net[2]`, `actor.action_net`, `critic.net[0]`, `critic.net[2]`,
  `critic.value_net`) as both the internal extraction order and the DW-2.3 error message's layer names —
  the plan's own "Produces" section pins this exact 6-name, fixed order.
- **Binary packing**: `struct.pack("<4s", MAGIC)` for the magic (already ASCII bytes, no encoding step
  needed), `struct.pack("<II", version, vector_len)` for the header, then per layer `struct.pack("<II",
  out_dim, in_dim)` followed by `weight.astype("<f4").tobytes()` (row-major, PyTorch's native layout, no
  transpose) and `bias.astype("<f4").tobytes()`. `<` (little-endian, no padding) throughout, matching the
  plan's "little-endian throughout" contract exactly — using `struct`'s explicit `<` avoids relying on the
  host's native byte order.
- **CLI shape**: matches `train_win_prob.py`'s pattern — module docstring with a runnable example,
  `parse_args()` with `--checkpoint` (default `data/models/ppo.zip`) and `--output` (default
  `data/cpp_weights/ppo.bin`), `main()` guarded by `if __name__ == "__main__":`. No asyncio needed (no
  Showdown server involved).

## Prerequisites
- [x] `data/models/ppo.zip` exists and loads (verified above).
- [x] `battle_engine.encoding.VECTOR_LEN`, `battle_engine.dataset.ACTION_SPACE_SIZE`,
      `battle_engine.ppo_warm_start.WARM_START_NET_ARCH` all importable and already used by the sibling
      `ppo_warm_start.py`/`test_ppo_warm_start.py`.
- [x] `sb3_contrib.MaskablePPO` available in the venv (already used by `ppo_eval.py`, `train_ppo.py`).

## Recommendation
BUILD. No gaps between plan and reality; the real checkpoint's shapes match every constant the plan and
`ppo_warm_start.py` already assume.
