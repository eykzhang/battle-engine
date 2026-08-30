"""Guard tests: the installed poke-engine must be a gen9 build with Tera.

This file exists because the failure mode it guards against is SILENT. Generation
is a compile-time Cargo feature in poke-engine, and the published PyPI wheel is a
**gen4** build. A gen4 build accepts a gen9ou state and simulates it under gen4
mechanics with no error, no warning, and no exception - wrong damage, wrong
abilities, no Terastallization. Every downstream number would be quietly wrong.

`pip install poke-engine` is therefore never correct for this project. Build with
`scripts/build_poke_engine.sh`, which runs this file as its final step.

Background, the build recipe, and the API footguns these tests also pin down:
notes/gotcha-poke-engine-pypi-wheel-is-gen4-not-gen9.md
"""

from __future__ import annotations

import pytest

poke_engine = pytest.importorskip(
    "poke_engine",
    reason="poke-engine not built; run scripts/build_poke_engine.sh",
)


def _mon(species: str, type1: str, type2: str = "typeless", moves=("tackle",), **kwargs):
    """A level-100 Pokemon with deliberately round, identical stats.

    Flat stats across both sides make a damage comparison between two defenders
    depend on typing alone, which is what the type-chart test below needs.
    """
    return poke_engine.Pokemon(
        id=species,
        level=100,
        types=(type1, type2),
        base_types=(type1, type2),
        hp=400,
        maxhp=400,
        attack=200,
        defense=200,
        special_attack=200,
        special_defense=200,
        speed=200,
        moves=[poke_engine.Move(id=m, pp=32, disabled=False) for m in moves],
        **kwargs,
    )


def _state(attacker, defender):
    """A 1v1 state; the other five slots on each side are fainted."""

    def side(active):
        return poke_engine.Side(
            pokemon=[active] + [poke_engine.Pokemon.create_fainted() for _ in range(5)],
            active_index="0",
        )

    return poke_engine.State(side_one=side(attacker), side_two=side(defender))


def test_steel_does_not_resist_dark():
    """The sharpest generation discriminator available.

    Steel resisted Dark through gen5 and stopped in gen6. If this build were the
    gen4 default, the Steel-typed defender would take roughly half the damage the
    Normal-typed one does. Equal damage means gen6+, which rules out the wheel
    that `pip install poke-engine` would have given us.
    """
    attacker = _mon("tyranitar", "dark", "rock", moves=("knockoff",))
    into_steel = poke_engine.calculate_damage(
        _state(attacker, _mon("skarmory", "steel")), "knockoff", "tackle", True
    )[0]
    into_normal = poke_engine.calculate_damage(
        _state(attacker, _mon("snorlax", "normal")), "knockoff", "tackle", True
    )[0]

    assert into_steel == into_normal, (
        f"Steel took {into_steel} from a Dark move vs {into_normal} for Normal. "
        "Steel resisting Dark means this is a gen5-or-earlier build - almost "
        "certainly the gen4 PyPI wheel. Rebuild with scripts/build_poke_engine.sh."
    )


@pytest.mark.parametrize("move", ["ivycudgel", "collisioncourse"])
def test_gen9_only_moves_deal_damage(move):
    """Ivy Cudgel and Collision Course were both introduced in gen9.

    Note the footgun this also pins: an unknown move id does NOT raise from
    `calculate_damage`, it silently returns [0, 0]. That is exactly why this
    asserts on real damage rather than on the absence of an exception.
    """
    state = _state(_mon("ogerpon", "grass", moves=(move,)), _mon("snorlax", "normal"))
    rolls = poke_engine.calculate_damage(state, move, "tackle", True)[0]

    assert rolls and max(rolls) > 0, (
        f"{move} dealt no damage, so this build's move data predates gen9."
    )


def test_unknown_move_is_silently_zero_not_an_error():
    """Pins the footgun above so the translation layer can rely on it.

    If a future poke-engine version starts raising here instead, this test fails
    and the validation code that currently compensates can be simplified.
    """
    state = _state(_mon("ogerpon", "grass", moves=("tackle",)), _mon("snorlax", "normal"))
    assert poke_engine.calculate_damage(state, "notarealmove", "tackle", True)[0] == [0, 0]


@pytest.mark.parametrize("species", ["greattusk", "gholdengo", "ironvaliant"])
def test_gen9_only_species_resolve(species):
    """Species ids must resolve to themselves, not to the NONE fallback.

    An unrecognised species is silently accepted and serializes as NONE, so this
    checks the serialized id rather than merely that construction succeeded.
    """
    serialized_id = _state(_mon(species, "normal"), _mon("snorlax", "normal")).to_string().split(",")[0]
    assert serialized_id == species.upper(), (
        f"{species!r} serialized as {serialized_id!r}. NONE means this build's "
        "species data predates gen9."
    )


def test_unknown_species_falls_back_to_none_silently():
    """The species-side twin of the unknown-move footgun, pinned for the same reason.

    Also covers the formatting trap: poke-engine ids are lowercase and
    space-free, so poke-env's display names ("Great Tusk") must be normalised
    before they are handed over, or they land here as NONE.
    """
    for bad in ("zzzznotamon", "Great Tusk", "great tusk"):
        serialized_id = _state(_mon(bad, "normal"), _mon("snorlax", "normal")).to_string().split(",")[0]
        assert serialized_id == "NONE"


def test_terastallization_is_compiled_in():
    """Tera is a separate Cargo feature; gen9 alone does not imply it."""
    tera_mon = _mon("ogerpon", "grass", terastallized=True, tera_type="water")
    assert tera_mon.terastallized is True
    assert tera_mon.tera_type.lower() == "water"

    # And it must survive a serialization round trip, which is how states are
    # handed to the engine's search entry points.
    state = _state(tera_mon, _mon("snorlax", "normal"))
    assert poke_engine.State.from_string(state.to_string()).to_string() == state.to_string()
