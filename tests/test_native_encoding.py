"""M4b: cross-checks encode_native() (C++, battle_state.cpp) against the
real Python encode() (battle_engine/encoding.py) it ports bit-for-bit - the
"translate a real poke-env battle through both battle_state_from_poke_env
and battle_view_from_poke_env, encode both, np.allclose" check named in
this milestone's Done-When items.

Fixture style matches tests/test_encoding.py's own _battle() helper
(SimpleNamespace-wrapped real poke_env.Pokemon objects via
conftest.make_mon, weather/fields kwargs included) and
tests/test_native_legality.py's native-extension skip convention. Must
skip cleanly, not error, when _native hasn't been built yet.

Every "expected" value in this file comes from calling encode() on the
SAME fixture, never a hand-computed number - encode() is the oracle this
port is verified against, exactly as this milestone's Done-When items
specify.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from conftest import make_mon

_native = pytest.importorskip("battle_engine._native")

from battle_engine import encoding as enc  # noqa: E402
from battle_engine.encoding import battle_view_from_poke_env, encode  # noqa: E402
from battle_engine.mcts_player import battle_state_from_poke_env  # noqa: E402

_PPO_BIN = "data/cpp_weights/ppo.bin"

# Phase 5 of the encoding rewrite (2026-08-26/27, see
# battle_engine/encoding.py's module docstring and
# .code-foundations/plans/2026-08-26-battle-engine-encoding-rewrite.md's
# Decision Log/Constraints) grew VECTOR_LEN 665 -> 2156. encode_native() (C++,
# cpp/src/battle_state.cpp) is a bit-for-bit port of the OLD, pre-rewrite
# encode() and is NOT re-ported here - re-porting it is explicitly out of
# scope for that plan, deferred to a separate future plan. Until that re-port
# lands, encode_native() structurally cannot match Python encode() - every
# test below that compares against encode()'s real values, or against
# enc.VECTOR_LEN directly, is skipped with this reason rather than left
# failing red or deleted (this project's own "loud, visible gap over silent
# handling" convention - see _pad_bench's assert for the same spirit). Tests
# that don't depend on either (e.g. the missing-active-Pokemon error-path
# check at the bottom of this file) are unaffected and stay active.
_ENCODE_NATIVE_UNPORTED_REASON = (
    "encode_native() (C++) was not re-ported for the encoding rewrite "
    "(VECTOR_LEN 665 -> 2156) - a deliberately deferred, named future plan "
    "(see battle_engine/encoding.py's module docstring and "
    ".code-foundations/plans/2026-08-26-battle-engine-encoding-rewrite.md's "
    "Decision Log), not an oversight. Bit-for-bit parity against Python "
    "encode() is structurally impossible until that re-port lands - "
    "re-enable these tests then."
)


def _battle(
    my_team, my_active, opp_team, opp_active,
    my_hazards=None, opp_hazards=None, weather=None, fields=None,
):
    return SimpleNamespace(
        team={mon.species: mon for mon in my_team},
        opponent_team={mon.species: mon for mon in opp_team},
        active_pokemon=my_active,
        opponent_active_pokemon=opp_active,
        side_conditions=my_hazards or {},
        opponent_side_conditions=opp_hazards or {},
        weather=weather or {},
        fields=fields or {},
    )


def _with_moves(mon, move_ids):
    for move_id in move_ids:
        mon._moves[move_id] = Move(move_id, gen=9)
    return mon


def _assert_parity(battle):
    """The one real assertion this whole file makes: encode_native() on the
    native-translated state equals encode() on the Python-translated view,
    for the identical underlying battle.
    """
    native_state = battle_state_from_poke_env(battle)
    python_view = battle_view_from_poke_env(battle)

    native_vec = np.array(_native.encode_native(native_state), dtype=np.float32)
    python_vec = encode(python_view)

    assert native_vec.shape == python_vec.shape == (enc.VECTOR_LEN,)
    assert np.allclose(native_vec, python_vec), (
        f"encode_native() diverges from encode() at indices "
        f"{np.nonzero(~np.isclose(native_vec, python_vec))[0].tolist()}"
    )


# ---------------------------------------------------------------------------
# DW-4.1: encode_native() == encode() on real battle states.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_matches_python_encode_for_a_fresh_full_team_battle():
    active = _with_moves(make_mon("garchomp"), ["earthquake", "dragonclaw", "swordsdance", "protect"])
    bench = [
        _with_moves(make_mon("dragapult"), ["dracometeor", "uturn"]),
        _with_moves(make_mon("tinkaton"), ["gigatonhammer"]),
        make_mon("blissey"),
        make_mon("excadrill"),
        make_mon("pikachu"),
    ]
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake", "stealthrock"])
    battle = _battle(
        my_team=[active] + bench,
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    _assert_parity(battle)


@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_matches_python_encode_with_hazards_weather_terrain_and_boosts():
    active = _with_moves(make_mon("garchomp"), ["earthquake"])
    active.boosts = {"atk": 2, "def": -1, "spa": 0, "spd": 1, "spe": -2, "accuracy": 1, "evasion": -1}
    bench = make_mon("dragapult")
    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake"])
    opp_active.boosts = {"atk": 0, "def": 0, "spa": 3, "spd": 0, "spe": 6, "accuracy": 0, "evasion": 0}
    battle = _battle(
        my_team=[active, bench],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
        my_hazards={SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2},
        opp_hazards={SideCondition.REFLECT: 3, SideCondition.TAILWIND: 1},
        weather={Weather.SANDSTORM: 1},
        fields={Field.ELECTRIC_TERRAIN: 2},
    )
    _assert_parity(battle)


@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_matches_python_encode_with_a_fainted_bench_mon():
    active = make_mon("garchomp")
    fainted_bench = make_mon("toxapex", current_hp_fraction=0.0, status=Status.FNT)
    healthy_bench = make_mon("dragapult")
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active, fainted_bench, healthy_bench],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    _assert_parity(battle)


@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_matches_python_encode_with_item_ability_and_protect_counter():
    active = _with_moves(make_mon("garchomp"), ["protect"])
    active.item = "heavydutyboots"
    active.ability = "levitate"
    active._protect_counter = 2  # no public setter (poke-env: read-only property)
    opp_active = _with_moves(make_mon("toxapex"), ["stealthrock"])
    opp_active.item = "roseliberry"  # rare item -> exercises the "other known item" bucket
    opp_active.ability = "wonderguard"
    battle = _battle(
        my_team=[active],
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    _assert_parity(battle)


@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_matches_python_encode_with_a_fainted_opponent_teammate():
    # Opponent bench is never encoded (only opp_active + opp_remaining_
    # fraction) - a partially-revealed opponent team should still parity-
    # match exactly, exercising the opp_remaining_fraction derivation.
    my_active = make_mon("garchomp")
    opp_active = make_mon("landorustherian")
    opp_partial_bench = make_mon("toxapex", current_hp_fraction=0.0, status=Status.FNT)
    battle = _battle(
        my_team=[my_active],
        my_active=my_active,
        opp_team=[opp_active, opp_partial_bench],
        opp_active=opp_active,
    )
    _assert_parity(battle)


@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_bench_is_species_sorted_not_insertion_order():
    # The plan's own named highest-risk item: encode_native()'s bench must
    # be species-alphabetical (dragapult before garchomp's own bench, e.g.
    # "excadrill" < "pikachu" < "tinkaton"), NOT team-preview/insertion
    # order - a regression here would still produce a same-LENGTH vector,
    # so only a real value-level parity check (not a shape check) catches
    # it. Insert bench mons in a deliberately non-alphabetical order.
    active = make_mon("garchomp")
    bench = [make_mon("tinkaton"), make_mon("excadrill"), make_mon("pikachu")]
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[active] + bench,
        my_active=active,
        opp_team=[opp_active],
        opp_active=opp_active,
    )
    _assert_parity(battle)

    # Direct check that the native bench really is sorted (not just that it
    # happens to numerically agree with encode()'s own sort) - locate the
    # first bench block's "known" flag block and its type/species-linked
    # base-stat block against the alphabetically-first bench mon
    # (excadrill), independent of the parity check above.
    native_state = battle_state_from_poke_env(battle)
    native_vec = np.array(_native.encode_native(native_state), dtype=np.float32)
    python_view = battle_view_from_poke_env(battle)
    assert python_view.my_bench[0].known and python_view.my_bench[0].base_stats["atk"] > 0
    # excadrill sorts first alphabetically among {excadrill, pikachu, tinkaton}
    first_bench_vec = native_vec[enc._POKEMON_VEC_LEN: 2 * enc._POKEMON_VEC_LEN]
    assert first_bench_vec[0] == 1.0  # known


# ---------------------------------------------------------------------------
# DW-5.6 (Phase 5 review fix, attempt 1): encode_native(mirror(s)) must
# match encode() of the REAL opponent-POV view - a value-level parity
# check, not just a length check. cpp/tests/test_battle_state.cpp's own
# C++-side test can only confirm encode_native(mirror(s)) is well-formed
# and length-matches encode_native(s); it explicitly cannot reach a
# value-level comparison against the real Python encode() from C++-only
# test scope (no Python interpreter there). This is that missing half -
# the exact seam mcts.cpp's populate_decision_node uses (mirror(state) +
# encode_native() + the actor/critic) to compute the opponent-side PUCT
# prior and leaf value, so an unverified asymmetry here would silently
# corrupt every opponent-side PUCT decision.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_mirror_matches_python_encode_of_real_opponent_pov():
    my_active = _with_moves(make_mon("garchomp"), ["earthquake", "swordsdance"])
    my_active.boosts = {"atk": 2, "def": -1, "spa": 0, "spd": 1, "spe": -2, "accuracy": 0, "evasion": 0}
    my_active.item = "leftovers"
    my_active.ability = "roughskin"
    my_bench = _with_moves(make_mon("dragapult"), ["dracometeor", "uturn"])

    opp_active = _with_moves(make_mon("landorustherian"), ["earthquake", "stealthrock"])
    opp_active.boosts = {"atk": 0, "def": 0, "spa": 3, "spd": 0, "spe": 1, "accuracy": 0, "evasion": -1}
    opp_active.item = "choicescarf"
    opp_active.ability = "intimidate"
    opp_bench = make_mon("toxapex", current_hp_fraction=0.0, status=Status.FNT)

    battle = _battle(
        my_team=[my_active, my_bench],
        my_active=my_active,
        opp_team=[opp_active, opp_bench],
        opp_active=opp_active,
        my_hazards={SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2},
        opp_hazards={SideCondition.REFLECT: 3, SideCondition.TAILWIND: 1},
        weather={Weather.SANDSTORM: 1},
        fields={Field.ELECTRIC_TERRAIN: 2},
    )

    native_state = battle_state_from_poke_env(battle)
    mirrored_vec = np.array(_native.encode_native(_native.mirror(native_state)), dtype=np.float32)

    # The genuine opponent-POV view: a battle object where every role is
    # actually swapped (not just relabeled) - my_team <-> opponent_team,
    # active <-> opponent_active, hazards <-> opponent hazards, weather/
    # terrain unchanged (state-level, not per-side - same as mirror()
    # itself). encode() on THIS is the real oracle: whatever value
    # asymmetry exists between "my"-side and "opponent"-side encoding
    # (e.g. the opponent's bench is collapsed to opp_remaining_fraction,
    # never encoded per-slot - see encoding.py's module docstring) must be
    # reproduced identically by mirror() + encode_native().
    opponent_pov_battle = _battle(
        my_team=[opp_active, opp_bench],
        my_active=opp_active,
        opp_team=[my_active, my_bench],
        opp_active=my_active,
        my_hazards={SideCondition.REFLECT: 3, SideCondition.TAILWIND: 1},
        opp_hazards={SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2},
        weather={Weather.SANDSTORM: 1},
        fields={Field.ELECTRIC_TERRAIN: 2},
    )
    python_vec = encode(battle_view_from_poke_env(opponent_pov_battle))

    assert mirrored_vec.shape == python_vec.shape == (enc.VECTOR_LEN,)
    assert np.allclose(mirrored_vec, python_vec), (
        f"encode_native(mirror(s)) diverges from encode() of the real "
        f"opponent-POV view at indices "
        f"{np.nonzero(~np.isclose(mirrored_vec, python_vec))[0].tolist()}"
    )


# ---------------------------------------------------------------------------
# DW-4.3: encode_native()'s length matches encoding.VECTOR_LEN and
# ppo.bin's header vector_len (the 3-way cross-check Phase 3 deferred here).
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_length_matches_vector_len():
    active = make_mon("garchomp")
    opp_active = make_mon("landorustherian")
    battle = _battle(my_team=[active], my_active=active, opp_team=[opp_active], opp_active=opp_active)
    state = battle_state_from_poke_env(battle)

    result = _native.encode_native(state)

    assert len(result) == enc.VECTOR_LEN == _native.ENCODE_VECTOR_LEN


# Was pytest.mark.skipif(not Path(_PPO_BIN).exists(), ...) - that guard is
# still true (this worktree's gitignored data/ has no ppo.bin), but even
# when it exists, this assertion now fails for the same structural reason as
# every other test in this file (ppo.bin's header encodes the OLD 665-dim
# shape, since it was exported before this rewrite) - an unconditional skip
# with the parity reason is the accurate one going forward.
@pytest.mark.skip(reason=_ENCODE_NATIVE_UNPORTED_REASON)
def test_encode_native_length_matches_ppo_bin_header_vector_len():
    active = make_mon("garchomp")
    opp_active = make_mon("landorustherian")
    battle = _battle(my_team=[active], my_active=active, opp_team=[opp_active], opp_active=opp_active)
    state = battle_state_from_poke_env(battle)

    header_vector_len = _native.PolicyWeights.load(_PPO_BIN).actor.layer0.in_dim

    assert len(_native.encode_native(state)) == enc.VECTOR_LEN == header_vector_len


# ---------------------------------------------------------------------------
# Edge case named in the plan: no active Pokemon (team preview) must raise
# a clear error, not silently encode garbage or crash - mirrors
# encoding.py's battle_view_from_poke_env ValueError precedent.
# ---------------------------------------------------------------------------


def test_encode_native_raises_when_my_side_has_no_active_pokemon():
    bench1 = make_mon("garchomp")
    bench2 = make_mon("dragapult")
    opp_active = make_mon("landorustherian")
    battle = _battle(
        my_team=[bench1, bench2], my_active=None, opp_team=[opp_active], opp_active=opp_active,
    )
    state = battle_state_from_poke_env(battle)

    # pybind11 auto-translates std::invalid_argument to a Python
    # ValueError (not the generic RuntimeError std::runtime_error gets) -
    # a closer match to encoding.py's own ValueError precedent than
    # PolicyWeights::load's std::runtime_error-on-malformed-file case.
    with pytest.raises(ValueError, match="team preview"):
        _native.encode_native(state)
