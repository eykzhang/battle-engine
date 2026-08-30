#!/usr/bin/env bash
# Build the gen9 poke-engine Python bindings into .venv.
#
# WHY THIS SCRIPT EXISTS: `pip install poke-engine` does NOT work for this
# project. poke-engine selects its generation at COMPILE time via Cargo
# features, and the published PyPI wheel is built for gen4
# (poke-engine-py/Cargo.toml declares `default = ["poke-engine/gen4"]`, and the
# upstream Makefile publishes with gen4). A gen4 build accepts a gen9ou state
# and simulates it under gen4 mechanics with no error and no warning - wrong
# damage, wrong abilities, no Terastallization. It looks like it works.
#
# Terastallization is its own feature on top of gen9; `gen9` alone does not
# imply it.
#
# Full finding, the three checks that confirm a build really is gen9, and the
# three silent-failure footguns in the resulting Python API:
#   notes/gotcha-poke-engine-pypi-wheel-is-gen4-not-gen9.md
# The guard test that keeps a gen4 build from ever going unnoticed:
#   tests/test_poke_engine_is_gen9.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$REPO_ROOT/poke-engine"
VENV="$REPO_ROOT/.venv"
PINNED_TAG="v0.0.48"

if [ ! -d "$ENGINE_DIR" ]; then
  echo "error: $ENGINE_DIR not found. Clone it first (pinned, not latest):" >&2
  echo "  git clone --depth 1 --branch $PINNED_TAG \\" >&2
  echo "    https://github.com/pmariglia/poke-engine.git poke-engine" >&2
  exit 1
fi

# brew's rustup formula is keg-only and no longer ships rustup-init, so cargo
# lands here rather than on PATH or at the usual ~/.cargo/bin.
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found. Install with:" >&2
  echo "  brew install rustup && export PATH=\"/opt/homebrew/opt/rustup/bin:\$PATH\"" >&2
  echo "  rustup default stable" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/maturin" ]; then
  echo "error: maturin not found. Install with: $VENV/bin/pip install maturin" >&2
  exit 1
fi

# maturin refuses to run when both VIRTUAL_ENV and CONDA_PREFIX are set, which
# they both are in this project's default shell - hence the `env -u`.
cd "$ENGINE_DIR/poke-engine-py"
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV VIRTUAL_ENV="$VENV" \
  "$VENV/bin/maturin" develop --release \
  --no-default-features \
  --features "poke-engine/gen9,poke-engine/terastallization" \
  "$@"

echo
echo "Verifying the build is actually gen9 (this is the whole point - a gen4"
echo "build fails silently, so never skip this):"
"$VENV/bin/python" -m pytest "$REPO_ROOT/tests/test_poke_engine_is_gen9.py" -q
