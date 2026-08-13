#!/usr/bin/env bash
# Runs pytest with the ASan runtime preloaded via DYLD_INSERT_LIBRARIES -
# required on macOS whenever _native*.so was built with the Debug/ASan
# preset (cpp/CMakeLists.txt): importing a sanitized extension into a
# non-sanitized Python interpreter without preloading the runtime first
# aborts on import (`Fatal Python error: Aborted`, no useful traceback -
# confirmed real, hit during M1 bring-up). Plain `.venv/bin/pytest` still
# works for everything that doesn't import battle_engine._native; use this
# wrapper once the native extension is part of what's under test.
set -euo pipefail

cd "$(dirname "$0")/.."

ASAN_LIB=$(clang++ -print-file-name=libclang_rt.asan_osx_dynamic.dylib)
DYLD_INSERT_LIBRARIES="$ASAN_LIB" ASAN_OPTIONS=detect_leaks=0 .venv/bin/pytest "$@"
