#!/usr/bin/env bash
# Configure + build the Phase 4 C++ extension (cpp/). Debug by default (ASan/
# UBSan on - see cpp/CMakeLists.txt); pass --release for an optimized build
# with sanitizers off. Output lands directly in battle_engine/ as
# _native*.so - no reinstall step, see cpp/CMakeLists.txt's comment on why.
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_TYPE=Debug
if [[ "${1:-}" == "--release" ]]; then
  BUILD_TYPE=Release
fi

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build cpp/build -j
