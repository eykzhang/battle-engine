"""Phase 6 / M3: the poke-env Battle -> poke-engine State translator.

Every silent-failure mode this module guards against is pinned here, because
none of them raises on its own: an unknown species becomes a statless `NONE`,
an unknown move deals no damage, an unknown item becomes `UNKNOWNITEM`, and a
bad status/weather/terrain/nature panics through pyo3 as something an
`except Exception` will not catch. A test is the only thing that notices.

Fixtures are REAL `poke_env.battle.Battle` objects driven by real Showdown
protocol messages through `parse_message`/`parse_request`, not SimpleNamespace
stand-ins. That is deliberate and stronger than the convention the rest of
tests/ uses (see docs/code-standards.md's testing section): the opponent's
percent-scale `max_hp`, the item sentinel, and the post-Tera typing flip are
all poke-env behaviors this translator has to survive, and a hand-built fake
would reproduce whatever this module already assumes about them. No server
connection is involved.
"""

from __future__ import annotations

import logging

import pytest

poke_engine = pytest.importorskip(
    "poke_engine",
    reason="poke-engine not built; run scripts/build_poke_engine.sh",
)

from poke_env.battle.battle import Battle  # noqa: E402
from poke_env.battle.field import Field  # noqa: E402
from poke_env.battle.status import Status  # noqa: E402
from poke_env.battle.weather import Weather  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402

from battle_engine.poke_engine_state import (  # noqa: E402
    PLACEHOLDER_MAX_HP,
    SlotFill,
    TranslationResult,
    UnknownToPokeEngine,
    _STATUS_TO_ENGINE,
    _TERRAIN_TO_ENGINE,
    _WEATHER_TO_ENGINE,
    is_known_ability,
    is_known_item,
    is_known_move,
    is_known_species,
    require_ability,
    require_item,
    require_move,
    require_species,
    state_from_poke_env,
)


# ---------------------------------------------------------------------------
# Fixtures: real Battle objects, driven by real protocol messages.
# ---------------------------------------------------------------------------

_MY_TEAM_REQUEST = {
    "active": [
        {
            "moves": [
                {"move": "Headlong Rush", "id": "headlongrush", "pp": 8, "maxpp": 8, "disabled": False},
                {"move": "Ice Spinner", "id": "icespinner", "pp": 24, "maxpp": 24, "disabled": True},
            ]
        }
    ],
    "side": {
        "name": "p1user",
        "id": "p1",
        "pokemon": [
            {
                "ident": "p1: Tusk",
                "details": "Great Tusk, L100",
                "condition": "341/341",
                "active": True,
                "stats": {"atk": 359, "def": 249, "spa": 140, "spd": 140, "spe": 301},
                "moves": ["headlongrush", "icespinner"],
                "baseAbility": "protosynthesis",
                "ability": "protosynthesis",
                "item": "heavydutyboots",
            },
            {
                "ident": "p1: Gliscor",
                "details": "Gliscor, L100",
                "condition": "354/354",
                "active": False,
                "stats": {"atk": 216, "def": 246, "spa": 112, "spd": 196, "spe": 184},
                "moves": ["earthquake", "protect", "toxic", "uturn"],
                "baseAbility": "poisonheal",
                "ability": "poisonheal",
                "item": "toxicorb",
            },
        ],
    },
    "rqid": 2,
}


_FORCE_SWITCH_REQUEST = {
    "forceSwitch": [True],
    "side": {
        "name": "p1user",
        "id": "p1",
        "pokemon": [
            dict(_MY_TEAM_REQUEST["side"]["pokemon"][0], condition="0 fnt"),
            _MY_TEAM_REQUEST["side"]["pokemon"][1],
        ],
    },
    "rqid": 3,
}


def _battle(with_request: bool = True) -> Battle:
    """A gen9ou battle one turn in: Great Tusk out against Gholdengo."""
    battle = Battle("battle-gen9ou-1", "p1user", logging.getLogger("test"), gen=9)
    battle.player_role = "p1"
    if with_request:
        battle.parse_request(_MY_TEAM_REQUEST)
    battle.parse_message(["", "switch", "p1a: Tusk", "Great Tusk, L100", "341/341"])
    battle.parse_message(["", "switch", "p2a: Gholdengo", "Gholdengo, L100", "100/100"])
    battle.parse_message(["", "turn", "1"])
    return battle


def _side_one(state) -> list:
    return state.to_string().split("/")[0].split("=")


def _side_two(state) -> list:
    return state.to_string().split("/")[1].split("=")


def _slot(state, side: str, index: int) -> list:
    fields = _side_one(state) if side == "p1" else _side_two(state)
    return fields[index].split(",")


# ---------------------------------------------------------------------------
# Vocabulary probes and validation. Footguns 1-3.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "species",
    ["greattusk", "gholdengo", "ironvaliant", "urshifurapidstrike", "landorustherian", "ogerponwellspring"],
)
def test_real_gen9_species_are_known(species):
    assert is_known_species(species)


def test_poke_env_display_names_are_rejected_not_silently_accepted():
    """The multi-word trap: poke-engine takes "Great Tusk" without complaint
    and turns it into a `NONE` Pokemon with no stats and no typing.
    """
    for display_name in ("Great Tusk", "great tusk", "Iron Valiant"):
        assert not is_known_species(display_name)
        with pytest.raises(UnknownToPokeEngine) as excinfo:
            require_species(display_name)
        assert display_name in str(excinfo.value)
        assert "to_id_str" in str(excinfo.value)
        # ...and normalizing is all it takes.
        assert is_known_species(to_id_str(display_name))


def test_unknown_move_is_rejected_by_name():
    assert is_known_move("closecombat")
    assert not is_known_move("notarealmove")
    with pytest.raises(UnknownToPokeEngine) as excinfo:
        require_move("notarealmove")
    assert excinfo.value.kind == "move"
    assert excinfo.value.value == "notarealmove"


def test_unknown_ability_is_rejected_by_name():
    assert is_known_ability("protosynthesis")
    assert not is_known_ability("notanability")
    with pytest.raises(UnknownToPokeEngine):
        require_ability("notanability")


def test_item_display_name_does_not_survive_normalization():
    assert is_known_item("heavydutyboots")
    assert not is_known_item("Heavy-Duty Boots")
    with pytest.raises(UnknownToPokeEngine):
        require_item("Heavy-Duty Boots")


def test_unmodelled_items_are_a_real_category_not_a_typo():
    """poke-engine's Items enum only covers items it implements mechanics
    for. These are legal, common gen9 OU items with no poke-engine entry,
    measured against pokemon-showdown/data/items.ts (347 of 583 absent).
    They are why the translator downgrades rather than raising by default.
    """
    for item in ("heatrock", "redcard", "stickybarb", "safetygoggles"):
        assert not is_known_item(item)


# ---------------------------------------------------------------------------
# Enum vocabularies. Footguns 4 and 5 - these panic rather than fall back.
# ---------------------------------------------------------------------------


def test_status_vocabulary_covers_every_poke_env_status_except_fainted():
    """poke-env's names (BRN/FRZ/PAR/PSN/SLP/TOX) are not poke-engine's
    (BURN/FREEZE/PARALYZE/POISON/SLEEP/TOXIC), and a wrong one panics.
    """
    assert set(_STATUS_TO_ENGINE) == set(Status) - {Status.FNT}
    for status, engine_id in _STATUS_TO_ENGINE.items():
        mon = poke_engine.Pokemon(id="pikachu", level=100, hp=100, maxhp=100, status=engine_id)
        side = poke_engine.Side(pokemon=[mon] + [poke_engine.Pokemon.create_fainted() for _ in range(5)])
        # A bad spelling raises pyo3_runtime.PanicException here, which is
        # not a ValueError - so this asserts on the round-tripped value
        # rather than merely on the absence of an exception.
        assert poke_engine.State(side_one=side, side_two=side).to_string().split(",")[18] == status.name.replace(
            "BRN", "BURN"
        ).replace("FRZ", "FREEZE").replace("PAR", "PARALYZE").replace("PSN", "POISON").replace(
            "SLP", "SLEEP"
        ).replace("TOX", "TOXIC")


@pytest.mark.parametrize("poke_env_weather", list(_WEATHER_TO_ENGINE))
def test_weather_vocabulary_is_accepted_by_the_engine(poke_env_weather):
    """poke-env spells these SUNNYDAY/RAINDANCE/SANDSTORM/SNOWSCAPE and
    poke-engine spells them SUN/RAIN/SAND/SNOW. Passing poke-env's spelling
    through would panic, not fall back.
    """
    engine_id = _WEATHER_TO_ENGINE[poke_env_weather]
    side = poke_engine.Side(pokemon=[poke_engine.Pokemon.create_fainted() for _ in range(6)])
    state = poke_engine.State(side_one=side, side_two=side, weather=engine_id, weather_turns_remaining=5)
    assert state.to_string().split("/")[2] == f"{engine_id.upper()};5"


@pytest.mark.parametrize("terrain_field", list(_TERRAIN_TO_ENGINE))
def test_terrain_vocabulary_is_accepted_by_the_engine(terrain_field):
    engine_id = _TERRAIN_TO_ENGINE[terrain_field]
    side = poke_engine.Side(pokemon=[poke_engine.Pokemon.create_fainted() for _ in range(6)])
    state = poke_engine.State(side_one=side, side_two=side, terrain=engine_id, terrain_turns_remaining=5)
    assert state.to_string().split("/")[3] == f"{engine_id.upper()};5"


def test_delta_stream_has_no_engine_counterpart_and_is_dropped():
    """poke-engine's Weather enum stops at HEAVYRAIN. Delta Stream would
    panic if passed through, so it is dropped with an assumption recorded.
    """
    assert Weather.DELTASTREAM not in _WEATHER_TO_ENGINE
    battle = _battle()
    battle.parse_message(["", "-weather", "DeltaStream"])
    result = state_from_poke_env(battle)

    assert result.state.to_string().split("/")[2] == "NONE;0"
    dropped = [a for a in result.assumed() if a.attribute == "weather"]
    assert dropped and "DELTASTREAM" in dropped[0].source


# ---------------------------------------------------------------------------
# End-to-end translation of a real battle.
# ---------------------------------------------------------------------------


def test_translates_a_real_battle_into_a_usable_state():
    battle = _battle()
    result = state_from_poke_env(battle)

    assert isinstance(result, TranslationResult)
    ours = _slot(result.state, "p1", 0)
    theirs = _slot(result.state, "p2", 0)
    assert ours[0] == "GREATTUSK"
    assert theirs[0] == "GHOLDENGO"
    assert ours[10] == "HEAVYDUTYBOOTS"  # from the request, a real observation
    assert ours[8] == "PROTOSYNTHESIS"
    # Our side is side_one, matching what poke-engine's own mcts() reports on.
    assert _side_one(result.state)[6] == "0"


def test_the_engine_accepts_the_translated_state():
    """The strongest single check available: `generate_instructions` raises
    ValueError on a move the side does not own and panics on a malformed
    state, so a clean transition means the whole state was well-formed.
    """
    battle = _battle()
    battle.parse_message(["", "move", "p2a: Gholdengo", "Make It Rain", "p1a: Tusk"])
    instructions = poke_engine.generate_instructions(
        state_from_poke_env(battle).state, "headlongrush", "makeitrain"
    )
    assert instructions


def test_state_survives_a_to_string_from_string_round_trip():
    """Exact for a state with no volatile statuses. See
    test_a_non_empty_volatile_set_gains_a_spurious_none_on_the_first_round_trip
    for the upstream bug that makes the volatile case different.
    """
    battle = _battle()
    battle.parse_message(["", "-damage", "p2a: Gholdengo", "62/100"])
    battle.parse_message(["", "-status", "p2a: Gholdengo", "brn"])
    battle.parse_message(["", "-boost", "p1a: Tusk", "atk", "2"])
    battle.parse_message(["", "-sidestart", "p1: p1user", "move: Spikes"])
    battle.parse_message(["", "-weather", "Sandstorm"])
    state = state_from_poke_env(battle).state

    assert not state.side_one.volatile_statuses and not state.side_two.volatile_statuses
    assert poke_engine.State.from_string(state.to_string()).to_string() == state.to_string()


def test_team_preview_is_rejected_rather_than_guessed():
    battle = Battle("battle-gen9ou-2", "p1user", logging.getLogger("test"), gen=9)
    battle.player_role = "p1"
    battle.parse_request(_MY_TEAM_REQUEST)
    with pytest.raises(ValueError, match="no active Pokemon"):
        state_from_poke_env(battle)


# ---------------------------------------------------------------------------
# The percent-scale opponent HP trap.
# ---------------------------------------------------------------------------


def test_opponent_hp_is_translated_to_real_points_not_percent():
    """poke-env reports the opponent's max_hp on a 0-100 percent scale
    (notes/gotcha-opponent-max-hp-is-on-a-percent-scale.md - this project's
    most expensive bug). Handing 62/100 to poke-engine would make every
    attack into that Pokemon look like a near-certain KO.
    """
    battle = _battle()
    battle.parse_message(["", "-damage", "p2a: Gholdengo", "62/100"])
    assert battle.opponent_active_pokemon.max_hp == 100  # the trap, still live upstream

    theirs = _slot(state_from_poke_env(battle).state, "p2", 0)
    hp, maxhp = int(theirs[6]), int(theirs[7])
    assert maxhp == 315  # Gholdengo, 0 EVs / 31 IVs / neutral, the estimate_stat convention
    assert hp == round(0.62 * maxhp)


def test_our_own_hp_uses_the_real_pool_from_the_request():
    battle = _battle()
    battle.parse_message(["", "-damage", "p1a: Tusk", "170/341"])
    ours = _slot(state_from_poke_env(battle).state, "p1", 0)
    assert (int(ours[6]), int(ours[7])) == (170, 341)


def test_our_own_stats_are_observed_and_the_opponents_are_estimated():
    result = state_from_poke_env(_battle())
    ours = next(a for a in result.for_slot("p1:0") if a.attribute == "stats")
    theirs = next(a for a in result.for_slot("p2:0") if a.attribute == "stats")

    assert ours.observed and ours.source == "poke-env"
    assert ours.value["spe"] == 301  # the exact spread from the request
    assert not theirs.observed


# ---------------------------------------------------------------------------
# Boosts, status, hazards, weather, terrain.
# ---------------------------------------------------------------------------


def test_boosts_live_on_the_side_not_the_pokemon():
    battle = _battle()
    battle.parse_message(["", "-boost", "p1a: Tusk", "atk", "2"])
    battle.parse_message(["", "-unboost", "p1a: Tusk", "spe", "1"])
    state = state_from_poke_env(battle).state

    # Side::serialize order after the six Pokemon, active index, side
    # conditions, wish and volatiles: attack, defense, spa, spd, speed.
    fields = _side_one(state)
    assert "2" in fields and "-1" in fields
    round_tripped = poke_engine.State.from_string(state.to_string()).side_one
    assert round_tripped.attack_boost == 2
    assert round_tripped.speed_boost == -1


def test_status_uses_poke_engines_spelling():
    battle = _battle()
    battle.parse_message(["", "-status", "p2a: Gholdengo", "brn"])
    assert _slot(state_from_poke_env(battle).state, "p2", 0)[18] == "BURN"


def test_toxic_counter_lands_on_the_side_not_the_pokemon():
    """poke-engine keeps the badly-poisoned counter as SideConditions
    .toxic_count, so reading it off the Pokemon would lose it entirely.
    """
    battle = _battle()
    battle.parse_message(["", "-status", "p2a: Gholdengo", "tox"])
    for _ in range(3):
        battle.parse_message(["", "turn", "2"])
    assert battle.opponent_active_pokemon.status_counter == 3

    side_two = poke_engine.State.from_string(state_from_poke_env(battle).state.to_string()).side_two
    assert side_two.side_conditions.toxic_count == 3


def test_hazards_and_screens_reach_the_engine():
    battle = _battle()
    battle.parse_message(["", "-sidestart", "p2: p2user", "move: Stealth Rock"])
    battle.parse_message(["", "-sidestart", "p2: p2user", "move: Spikes"])
    battle.parse_message(["", "-sidestart", "p2: p2user", "move: Spikes"])
    battle.parse_message(["", "-sidestart", "p1: p1user", "Reflect"])
    result = state_from_poke_env(battle)
    state = poke_engine.State.from_string(result.state.to_string())

    assert state.side_two.side_conditions.stealth_rock == 1
    assert state.side_two.side_conditions.spikes == 2
    assert state.side_one.side_conditions.reflect > 0
    # Screen duration is derived from the start turn, not observed - Light
    # Clay is invisible until the item is revealed, so it must be recorded.
    reflect = next(a for a in result.assumed() if a.attribute == "reflect")
    assert reflect.source == "assumed-base-duration"


def test_weather_and_terrain_translate_with_turn_counts():
    battle = _battle()
    battle.parse_message(["", "-weather", "RainDance"])
    battle.parse_message(["", "-fieldstart", "move: Electric Terrain"])
    state = state_from_poke_env(battle).state.to_string().split("/")

    assert state[2].startswith("RAIN;")
    assert state[3].startswith("ELECTRICTERRAIN;")


def test_trick_room_is_not_mistaken_for_a_terrain():
    """battle.fields carries non-terrain effects too, so a naive "first
    entry wins" read would report Trick Room as the terrain.
    """
    battle = _battle()
    battle.parse_message(["", "-fieldstart", "move: Trick Room"])
    parts = state_from_poke_env(battle).state.to_string().split("/")

    assert Field.TRICK_ROOM in battle.fields
    assert parts[3] == "NONE;0"
    assert parts[4].startswith("true;")


# ---------------------------------------------------------------------------
# Terastallization.
# ---------------------------------------------------------------------------


def test_tera_keeps_the_original_typing_and_sets_the_tera_flag():
    """poke-env's mon.types already RETURNS the Tera type once a Pokemon has
    terastallized; poke-engine expects the original typing plus the flag and
    substitutes the Tera type itself in damage_calc.rs. Passing mon.types
    through would double-apply Tera and lose the original STAB types.
    """
    battle = _battle()
    battle.parse_message(["", "-terastallize", "p2a: Gholdengo", "Flying"])
    assert [t.name for t in battle.opponent_active_pokemon.types] == ["FLYING"]  # upstream flip

    theirs = _slot(state_from_poke_env(battle).state, "p2", 0)
    assert (theirs[2], theirs[3]) == ("STEEL", "GHOST")
    assert (theirs[4], theirs[5]) == ("STEEL", "GHOST")
    assert theirs[27] == "true"
    assert theirs[28] == "FLYING"


def test_untera_pokemon_carry_no_tera_type():
    theirs = _slot(state_from_poke_env(_battle()).state, "p2", 0)
    assert theirs[27] == "false"
    assert theirs[28] == "TYPELESS"


# ---------------------------------------------------------------------------
# Fainted and unrevealed slots.
# ---------------------------------------------------------------------------


def test_a_fainted_pokemon_is_zero_hp_not_a_fainted_status():
    """poke-engine has no FAINTED status variant; hp == 0 is the whole
    representation, and passing poke-env's Status.FNT through would panic.
    """
    battle = _battle()
    battle.parse_message(["", "faint", "p2a: Gholdengo"])
    theirs = _slot(state_from_poke_env(battle).state, "p2", 0)

    assert battle.opponent_active_pokemon.status is Status.FNT
    assert int(theirs[6]) == 0
    assert theirs[18] == "NONE"


def test_unrevealed_opponent_slots_are_placeholders_not_fainted():
    """A fainted placeholder would tell the search the opponent is down to
    one Pokemon and let it evaluate an even position as nearly won. The
    baseline filler uses a neutral, stated placeholder instead, and says so.
    """
    result = state_from_poke_env(_battle())
    fields = _side_two(result.state)

    for index in range(1, 6):
        slot = fields[index].split(",")
        assert slot[0] == "NONE"
        assert int(slot[6]) == PLACEHOLDER_MAX_HP
        assert (slot[2], slot[3]) == ("NORMAL", "TYPELESS")
        assert any(a.attribute == "species" and not a.observed for a in result.for_slot(f"p2:{index}"))


def test_unrevealed_slots_do_not_claim_to_be_observed():
    result = state_from_poke_env(_battle())
    for index in range(1, 6):
        assert all(not a.observed for a in result.for_slot(f"p2:{index}"))


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------


def test_assumed_and_observed_partition_the_ledger():
    result = state_from_poke_env(_battle())
    # Compared as lists, not sets: an Attribution's value can be a dict of
    # stats, so the records are deliberately not hashable.
    assert len(result.assumed()) + len(result.observed()) == len(result.attributions)
    assert all(not a.observed for a in result.assumed())
    assert all(a.observed for a in result.observed())
    assert all(a.source for a in result.attributions)


def test_a_revealed_move_is_observed_and_an_unrevealed_one_is_not():
    battle = _battle()
    battle.parse_message(["", "move", "p2a: Gholdengo", "Make It Rain", "p1a: Tusk"])
    result = state_from_poke_env(battle)

    moves = next(a for a in result.for_slot("p2:0") if a.attribute == "moves")
    assert moves.observed and moves.value == ("makeitrain",)


def test_an_unrevealed_item_is_recorded_as_assumed():
    result = state_from_poke_env(_battle())
    theirs = next(a for a in result.for_slot("p2:0") if a.attribute == "item")
    ours = next(a for a in result.for_slot("p1:0") if a.attribute == "item")

    assert not theirs.observed and theirs.value == "unknownitem"
    assert ours.observed and ours.value == "heavydutyboots"


def test_a_knocked_off_item_is_an_observation_of_nothing():
    """poke-env's unrevealed item is the truthy string "unknown_item" and a
    consumed one is a real None, so a "falsy means unknown" read gets both
    backwards.
    """
    battle = _battle()
    battle.parse_message(["", "-enditem", "p1a: Tusk", "Heavy-Duty Boots"])
    result = state_from_poke_env(battle)

    ours = next(a for a in result.for_slot("p1:0") if a.attribute == "item")
    assert ours.observed and ours.value is None
    assert _slot(result.state, "p1", 0)[10] == "NONE"


# ---------------------------------------------------------------------------
# The unknown-filler seam - the interface M4 plugs into.
# ---------------------------------------------------------------------------


class _StubUsageStatsFiller:
    """Stands in for M4: names a species, item, ability and moveset for
    every slot, whether or not the battle revealed one.
    """

    name = "stub-usage-stats"

    def __init__(self):
        self.seen = []

    def fill_side(self, observation):
        self.seen.append(observation)
        return [
            SlotFill(
                species="kingambit",
                item="leftovers",
                ability="supremeoverlord",
                moves=("suckerpunch", "kowtowcleave"),
            )
            for _ in range(observation.team_size)
        ]


def test_a_filler_populates_unrevealed_slots():
    filler = _StubUsageStatsFiller()
    result = state_from_poke_env(_battle(), filler=filler)

    filled = _slot(result.state, "p2", 3)
    assert filled[0] == "KINGAMBIT"
    assert (filled[2], filled[3]) == ("DARK", "STEEL")  # dex typing, not a placeholder
    assert filled[10] == "LEFTOVERS"
    assert filled[8] == "SUPREMEOVERLORD"
    assert filled[22].startswith("SUCKERPUNCH")
    assert all(not a.observed for a in result.for_slot("p2:3"))
    assert all(a.source == filler.name for a in result.for_slot("p2:3"))


def test_a_fill_can_never_overwrite_an_observation():
    """The invariant that makes the ledger worth reading. The stub above
    would happily replace Gholdengo with Kingambit if the merge let it.
    """
    result = state_from_poke_env(_battle(), filler=_StubUsageStatsFiller())

    assert _slot(result.state, "p2", 0)[0] == "GHOLDENGO"
    assert _slot(result.state, "p1", 0)[10] == "HEAVYDUTYBOOTS"
    species = next(a for a in result.for_slot("p2:0") if a.attribute == "species")
    assert species.observed and species.source == "poke-env"


def test_a_filler_sees_only_what_the_battle_revealed():
    filler = _StubUsageStatsFiller()
    state_from_poke_env(_battle(), filler=filler)
    opponent_view = next(o for o in filler.seen if not o.is_ours)

    assert opponent_view.team_size == 6
    assert opponent_view.slots[0].species == "gholdengo"
    assert opponent_view.slots[0].item is None  # not revealed, and says so
    assert all(slot.species is None for slot in opponent_view.slots[1:])


def test_a_filler_that_returns_the_wrong_number_of_slots_is_a_loud_error():
    class _Short:
        name = "short"

        def fill_side(self, observation):
            return [SlotFill()]

    with pytest.raises(ValueError, match="exactly team_size fills"):
        state_from_poke_env(_battle(), filler=_Short())


def test_a_filler_cannot_smuggle_an_unknown_species_past_validation():
    class _Bogus:
        name = "bogus"

        def fill_side(self, observation):
            return [SlotFill(species="zzzznotamon") for _ in range(observation.team_size)]

    with pytest.raises(UnknownToPokeEngine, match="species"):
        state_from_poke_env(_battle(), filler=_Bogus())


def test_a_filler_may_hand_back_a_display_name_and_it_is_normalized():
    """Normalizing the filler's output rather than demanding pre-normalized
    ids keeps the seam usable for M4, whose usage-statistics source spells
    species the way Smogon does.
    """

    class _DisplayNames:
        name = "display-names"

        def fill_side(self, observation):
            return [SlotFill(species="Great Tusk") for _ in range(observation.team_size)]

    result = state_from_poke_env(_battle(), filler=_DisplayNames())
    assert _slot(result.state, "p2", 3)[0] == "GREATTUSK"


# ---------------------------------------------------------------------------
# The spread seam (M4). `SlotFill.stats` is final stats, not a nature and EVs -
# see the field's own docstring for why the seam is shaped that way.
# ---------------------------------------------------------------------------


class _SpreadFiller:
    """Supplies one spread's worth of final stats to every slot."""

    name = "spread"

    def __init__(self, stats):
        self._stats = stats

    def fill_side(self, observation):
        return [SlotFill(stats=self._stats) for _ in range(observation.team_size)]


_BULKY = {"hp": 450, "atk": 200, "def": 350, "spa": 150, "spd": 300, "spe": 120}


def test_a_spread_replaces_poke_envs_neutral_estimate_for_the_opponent():
    """The error M3 measured. poke-env's `estimate_stat` returns a 0-EV,
    neutral-nature guess for anything it has not been told, and that guess is
    not an observation - it is the systematic damage error set prediction
    exists to fix.
    """
    plain = _slot(state_from_poke_env(_battle()).state, "p2", 0)
    filled = _slot(state_from_poke_env(_battle(), filler=_SpreadFiller(_BULKY)).state, "p2", 0)

    # Serialized slot layout: 6 hp, 7 maxhp, 13-17 atk/def/spa/spd/spe.
    assert plain[7] != "450"  # the neutral estimate, whatever it is
    assert filled[7] == "450"  # max HP
    assert filled[13] == "200"  # attack
    assert filled[14] == "350"  # defense


def test_a_spread_never_touches_our_own_observed_stats():
    """Our team's real stats come from the request JSON. A fill that could
    overwrite them would be the seam's one unbreakable rule broken.
    """
    ours = _slot(state_from_poke_env(_battle(), filler=_SpreadFiller(_BULKY)).state, "p1", 0)
    assert ours[13] == "359"  # from _MY_TEAM_REQUEST, not the fill
    assert ours[7] != "450"


def test_a_spread_rescales_current_hp_and_keeps_the_fraction():
    """poke-env reports an opponent's HP as a percentage, so the *fraction* is
    the only part of it that was ever real. Changing max HP has to carry the
    current value with it or the model starts the turn at the wrong health.
    """
    battle = _battle()
    battle.parse_message(["", "-damage", "p2a: Gholdengo", "50/100"])

    plain = _slot(state_from_poke_env(battle).state, "p2", 0)
    filled = _slot(state_from_poke_env(battle, filler=_SpreadFiller(_BULKY)).state, "p2", 0)

    assert int(plain[6]) / int(plain[7]) == pytest.approx(0.5, abs=0.01)
    assert int(filled[6]) / int(filled[7]) == pytest.approx(0.5, abs=0.01)
    assert filled[6] == "225"


def test_a_spread_on_an_unrevealed_slot_beats_the_dex_default():
    result = state_from_poke_env(
        _battle(),
        filler=_LayerFill(SlotFill(species="kingambit", stats=_BULKY)),
    )
    filled = _slot(result.state, "p2", 3)
    assert filled[0] == "KINGAMBIT"
    assert filled[7] == "450" and filled[6] == "450"  # unrevealed slots start full


def test_nature_and_evs_stay_at_the_engine_default_and_are_genuinely_inert():
    """Why the seam is final stats and not a nature plus six EVs: poke-engine
    recomputes from base stats only on a form change, so these two fields are
    carried along and ignored. The serialized state shows both - a SERIOUS
    nature and 85 EVs everywhere - next to the explicit stats that actually
    apply."""
    filled = _slot(state_from_poke_env(_battle(), filler=_SpreadFiller(_BULKY)).state, "p2", 0)
    assert filled[11] == "SERIOUS"
    assert filled[12] == "85;85;85;85;85;85"
    assert filled[13] == "200"


def test_a_spread_is_recorded_as_an_assumption_with_its_source():
    result = state_from_poke_env(_battle(), filler=_SpreadFiller(_BULKY))
    stats = next(a for a in result.for_slot("p2:0") if a.attribute == "stats")
    assert not stats.observed and stats.source == "spread:spread"
    ours = next(a for a in result.for_slot("p1:0") if a.attribute == "stats")
    assert ours.observed and ours.source == "poke-env"


class _LayerFill:
    name = "one-fill"

    def __init__(self, fill):
        self._fill = fill

    def fill_side(self, observation):
        return [self._fill for _ in range(observation.team_size)]


# ---------------------------------------------------------------------------
# Observation building, exposed for M4's evaluation.
# ---------------------------------------------------------------------------


def test_observe_side_matches_what_the_filler_is_handed():
    """`observe_side` and the translator's internal path must not drift: M4's
    evaluation scores the filler through the first and the search runs it
    through the second."""
    from battle_engine.poke_engine_state import observe_side

    filler = _StubUsageStatsFiller()
    battle = _battle()
    state_from_poke_env(battle, filler=filler)
    from_translator = next(o for o in filler.seen if not o.is_ours)
    assert observe_side(battle, ours=False) == from_translator


# ---------------------------------------------------------------------------
# Item strictness.
# ---------------------------------------------------------------------------


def test_unmodelled_items_downgrade_by_default_and_say_so():
    battle = _battle()
    battle.parse_message(["", "-item", "p2a: Gholdengo", "Heat Rock"])
    result = state_from_poke_env(battle)

    assert _slot(result.state, "p2", 0)[10] == "UNKNOWNITEM"
    item = next(a for a in result.for_slot("p2:0") if a.attribute == "item")
    assert not item.observed and "unmodelled(heatrock)" in item.source


def test_strict_items_turns_the_downgrade_back_into_an_error():
    battle = _battle()
    battle.parse_message(["", "-item", "p2a: Gholdengo", "Heat Rock"])
    with pytest.raises(UnknownToPokeEngine, match="heatrock"):
        state_from_poke_env(battle, strict_items=True)


def test_strict_items_accepts_a_modelled_item():
    state_from_poke_env(_battle(), strict_items=True)


# ---------------------------------------------------------------------------
# Move slots.
# ---------------------------------------------------------------------------


def test_disabled_moves_come_from_the_servers_own_answer():
    """battle.available_moves already accounts for Choice lock, Taunt,
    Encore, Disable, Torment and 0 PP, so none of that is reimplemented.
    """
    battle = _battle()
    moves = _slot(state_from_poke_env(battle).state, "p1", 0)[22:26]

    assert moves[0].split(";") == ["HEADLONGRUSH", "false", "8"]
    assert moves[1].split(";") == ["ICESPINNER", "true", "24"]


def test_a_forced_switch_turn_does_not_disable_every_move():
    """available_moves is empty on a forced switch, which means "you cannot
    pick a move right now", not "every move is disabled".
    """
    battle = _battle()
    battle.parse_message(["", "faint", "p1a: Tusk"])
    battle.parse_request(_FORCE_SWITCH_REQUEST)
    assert battle.force_switch and battle.available_moves == []

    result = state_from_poke_env(battle)
    moves = _slot(result.state, "p1", 0)[22:26]
    assert all(field.split(";")[1] == "false" for field in moves if not field.startswith("NONE"))
    # The forced switch itself is a real signal poke-engine carries per side.
    assert poke_engine.State.from_string(result.state.to_string()).side_one.force_switch


def test_only_revealed_opponent_moves_are_carried_over():
    battle = _battle()
    battle.parse_message(["", "move", "p2a: Gholdengo", "Make It Rain", "p1a: Tusk"])
    moves = _slot(state_from_poke_env(battle).state, "p2", 0)[22:26]

    assert moves[0].startswith("MAKEITRAIN")
    assert all(field.startswith("NONE") for field in moves[1:])


# ---------------------------------------------------------------------------
# Volatile statuses.
# ---------------------------------------------------------------------------


def test_substitute_becomes_a_volatile_plus_side_health():
    """poke-engine splits a Substitute across two places: the volatile on
    the side's set and `substitute_health` as a separate i16.
    """
    battle = _battle()
    battle.parse_message(["", "-start", "p1a: Tusk", "Substitute"])
    result = state_from_poke_env(battle)
    side_one = poke_engine.State.from_string(result.state.to_string()).side_one

    assert "SUBSTITUTE" in side_one.volatile_statuses
    assert side_one.substitute_health == 341 // 4
    health = next(a for a in result.assumed() if a.attribute == "substitute_health")
    assert health.source == "assumed-full"


def test_an_effect_poke_engine_does_not_model_is_dropped_and_recorded():
    """An unrecognized volatile is inserted into poke-engine's set AS
    `NONE` rather than rejected, so dropping it is the only safe option -
    and the drop is recorded rather than silent. `Effect.PROTOSYNTHESIS`
    is a real poke-env effect with no poke-engine counterpart (poke-engine
    spells the boosted-stat variants PROTOSYNTHESISATK and so on).
    """
    battle = _battle()
    battle.parse_message(["", "-start", "p1a: Tusk", "Protosynthesis"])
    result = state_from_poke_env(battle)

    dropped = next(a for a in result.assumed() if a.attribute == "volatile_statuses_dropped")
    assert "protosynthesis" in dropped.value
    assert "protosynthesis" not in {v.lower() for v in result.state.side_one.volatile_statuses}


def test_a_non_empty_volatile_set_gains_a_spurious_none_on_the_first_round_trip():
    """An upstream poke-engine bug, pinned rather than worked around.

    `Side::serialize` writes each volatile followed by a trailing ":", and
    `Side::deserialize` splits on ":" and feeds the resulting empty final
    element to `PokemonVolatileStatus::from_str`, which defaults to NONE.
    So a state with any volatile does NOT survive to_string/from_string
    unchanged - it gains a NONE and only then becomes stable. Anything in
    M3's fidelity harness that compares serialized states has to normalize
    for this, and if a future poke-engine version fixes it, this test fails
    and the normalization can go.
    """
    battle = _battle()
    battle.parse_message(["", "-start", "p1a: Tusk", "Substitute"])
    first = state_from_poke_env(battle).state.to_string()
    second = poke_engine.State.from_string(first).to_string()
    third = poke_engine.State.from_string(second).to_string()

    assert first.split("=")[8] == "SUBSTITUTE:"
    assert second.split("=")[8] == "NONE:SUBSTITUTE:"
    assert second == third


def test_the_engine_is_not_told_the_opponent_is_down_to_one_pokemon():
    """The reason unrevealed slots are placeholders rather than fainted:
    poke-engine reads hp == 0 as "this Pokemon is gone", and a side with
    five gone Pokemon evaluates as nearly won.
    """
    state = state_from_poke_env(_battle()).state
    alive = sum(1 for index in range(6) if int(_slot(state, "p2", index)[6]) > 0)
    assert alive == 6
