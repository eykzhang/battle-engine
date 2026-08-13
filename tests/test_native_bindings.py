"""M1: proves the C++ toolchain (CMake + C++20 + pybind11 v3) and the
editable-install import path work end-to-end. Must skip cleanly, not error,
on a fresh clone where scripts/build_cpp.sh hasn't been run yet - see
plans/precious-crafting-bachman.md's build-system section for why.
"""

import pytest

_native = pytest.importorskip("battle_engine._native")


def test_ping():
    assert _native.ping() == 42
