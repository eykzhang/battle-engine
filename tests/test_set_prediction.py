"""Phase 6 / M4: the usage-statistics filler and the layering that composes it.

Two things are worth pinning here. The `UnknownFiller` contract, because the
translator validates it at runtime and a violation is a `ValueError` mid-battle.
And `LayeredFiller`'s species coherence, because getting it wrong produces a
state that is perfectly well-formed and silently describes a Gholdengo with
Great Tusk's spread and moveset.
"""

from __future__ import annotations

import random

import pytest

from battle_engine.poke_engine_state import (
    TEAM_SIZE,
    RevealedOnlyFiller,
    SideObservation,
    SlotFill,
    SlotObservation,
)
from battle_engine.set_prediction import (
    LayeredFiller,
    UsageStatsFiller,
    spread_to_stats,
)
from battle_engine.usage_stats import Spread, find_cached, load_usage_stats


@pytest.fixture(scope="module")
def stats():
    path = find_cached(cutoff=1500)
    if path is None:
        pytest.skip("no cached usage stats; run scripts/fetch_usage_stats.py")
    return load_usage_stats(path)


@pytest.fixture
def filler(stats):
    return UsageStatsFiller(stats=stats)


def observation(*slots: SlotObservation, is_ours: bool = False) -> SideObservation:
    filled = list(slots) + [
        SlotObservation(index=i) for i in range(len(slots), TEAM_SIZE)
    ]
    return SideObservation(
        is_ours=is_ours, format_id="gen9ou", team_size=TEAM_SIZE, slots=tuple(filled)
    )


class TestSpreadToStats:
    def test_matches_a_hand_checked_showdown_spread(self):
        # Jolly 252 Atk / 252 Spe Great Tusk, the metagame's modal spread.
        stats = spread_to_stats("greattusk", Spread("jolly", (0, 252, 4, 0, 0, 252)))
        assert stats["atk"] == 361
        assert stats["spe"] == 300
        assert stats["hp"] == 371  # 0 HP EVs, base 115

    def test_ev_investment_actually_moves_the_number(self):
        bulky = spread_to_stats("greattusk", Spread("impish", (252, 0, 252, 0, 4, 0)))
        neutral = spread_to_stats("greattusk", Spread("serious", (0, 0, 0, 0, 0, 0)))
        assert bulky["hp"] > neutral["hp"]
        assert bulky["def"] > neutral["def"]


class TestFillerContract:
    def test_returns_exactly_team_size_fills(self, filler):
        # The translator raises if this is wrong, mid-battle.
        assert len(list(filler.fill_side(observation()))) == TEAM_SIZE

    def test_names_a_species_only_for_unrevealed_slots(self, filler):
        fills = list(filler.fill_side(observation(SlotObservation(index=0, species="greattusk"))))
        assert fills[0].species is None
        assert all(f.species for f in fills[1:])

    def test_respects_species_clause_across_the_whole_side(self, filler):
        fills = list(filler.fill_side(observation(SlotObservation(index=0, species="gholdengo"))))
        invented = [f.species for f in fills if f.species]
        assert "gholdengo" not in invented
        assert len(set(invented)) == len(invented)

    def test_fills_nothing_it_was_already_told(self, filler):
        slot = SlotObservation(
            index=0,
            species="greattusk",
            ability="protosynthesis",
            item="heavydutyboots",
            tera_type="steel",
        )
        fill = list(filler.fill_side(observation(slot)))[0]
        # A fill can only supply what was never observed; the translator would
        # discard these anyway, and returning them would make the provenance
        # ledger claim an assumption where there was an observation.
        assert fill.ability is None and fill.item is None and fill.tera_type is None

    def test_an_item_revealed_as_nothing_is_an_observation_not_a_gap(self, filler):
        # poke-env's three item states stay distinct: None = never revealed,
        # "" = revealed to be holding nothing (Knock Off).
        knocked_off = SlotObservation(index=0, species="greattusk", item="")
        never_shown = SlotObservation(index=0, species="greattusk", item=None)
        assert list(filler.fill_side(observation(knocked_off)))[0].item is None
        assert list(filler.fill_side(observation(never_shown)))[0].item is not None

    def test_fills_the_moveset_out_to_four(self, filler):
        slot = SlotObservation(index=0, species="greattusk", moves=("headlongrush", "icespinner"))
        fill = list(filler.fill_side(observation(slot)))[0]
        assert len(fill.moves) == 2
        assert not set(fill.moves) & {"headlongrush", "icespinner"}

    def test_supplies_a_spread_as_final_stats(self, filler):
        fill = list(filler.fill_side(observation(SlotObservation(index=0, species="greattusk"))))[0]
        assert set(fill.stats) == {"hp", "atk", "def", "spa", "spd", "spe"}
        # The whole point: not the translator's 0-EV neutral convention.
        assert fill.stats["atk"] > spread_to_stats("greattusk", Spread("serious", (0,) * 6))["atk"]

    def test_a_species_outside_the_metagame_file_falls_through(self, filler):
        # Below the usage cut, or a format mismatch. Nothing to predict from, so
        # the translator's own conventions stand rather than a wrong guess.
        fill = list(filler.fill_side(observation(SlotObservation(index=0, species="pichu"))))[0]
        assert fill == SlotFill()

    def test_switches_can_turn_each_prediction_off_independently(self, stats):
        no_species = UsageStatsFiller(stats=stats, fill_species=False)
        assert all(f.species is None for f in no_species.fill_side(observation()))
        no_spreads = UsageStatsFiller(stats=stats, fill_spreads=False)
        fills = list(no_spreads.fill_side(observation(SlotObservation(index=0, species="greattusk"))))
        assert fills[0].stats is None


class TestModes:
    def test_modal_is_deterministic(self, stats):
        filler = UsageStatsFiller(stats=stats)
        first = [f.species for f in filler.fill_side(observation())]
        second = [f.species for f in filler.fill_side(observation())]
        assert first == second

    def test_sampling_is_reproducible_from_a_seed(self, stats):
        a = UsageStatsFiller(stats=stats, rng=random.Random(11))
        b = UsageStatsFiller(stats=stats, rng=random.Random(11))
        assert [f.species for f in a.fill_side(observation())] == [
            f.species for f in b.fill_side(observation())
        ]

    def test_sampling_actually_varies(self, stats):
        filler = UsageStatsFiller(stats=stats, rng=random.Random(5))
        teams = {tuple(f.species for f in filler.fill_side(observation())) for _ in range(10)}
        # M5 samples K opponent states per turn; identical draws would make the
        # root parallelism a K-fold waste of the search budget.
        assert len(teams) > 1

    def test_teammate_conditioning_changes_which_species_get_invented(self, stats):
        seen = observation(SlotObservation(index=0, species="greattusk"))
        conditioned = UsageStatsFiller(stats=stats)
        flat = UsageStatsFiller(stats=stats, condition_on_teammates=False)
        assert [f.species for f in conditioned.fill_side(seen)] != [
            f.species for f in flat.fill_side(seen)
        ]

    def test_the_name_records_every_choice_that_changes_the_answer(self, stats):
        # The name lands in the provenance ledger and in every report header, so
        # two runs of a report can never be confused for one another.
        name = UsageStatsFiller(stats=stats, rng=random.Random(0), condition_on_teammates=False).name
        assert "gen9ou-1500" in name and "sampled" in name and "no-teammates" in name


class _FixedFiller:
    def __init__(self, name, fill):
        self.name = name
        self._fill = fill

    def fill_side(self, observation):
        return [self._fill] + [SlotFill() for _ in range(observation.team_size - 1)]


class TestLayeredFiller:
    def test_the_earlier_layer_wins_field_by_field(self):
        top = _FixedFiller("top", SlotFill(item="choicescarf"))
        bottom = _FixedFiller("bottom", SlotFill(item="leftovers", ability="levitate"))
        merged = list(LayeredFiller(top, bottom).fill_side(observation()))[0]
        assert merged.item == "choicescarf"
        assert merged.ability == "levitate"

    def test_moves_concatenate_in_layer_order_and_stay_unique(self):
        top = _FixedFiller("top", SlotFill(moves=("earthquake", "icespinner")))
        bottom = _FixedFiller("bottom", SlotFill(moves=("icespinner", "rapidspin", "knockoff", "bulkup")))
        merged = list(LayeredFiller(top, bottom).fill_side(observation()))[0]
        # The oracle's move has to survive the four-move cap, so it goes first.
        assert merged.moves[:2] == ("earthquake", "icespinner")
        assert len(merged.moves) == 4 == len(set(merged.moves))

    def test_a_layer_naming_a_different_species_contributes_nothing_else(self):
        # The trap. Merging field by field would give Gholdengo Great Tusk's
        # spread and moveset - a well-formed state describing a Pokemon that
        # does not exist.
        top = _FixedFiller("top", SlotFill(species="gholdengo"))
        bottom = _FixedFiller(
            "bottom",
            SlotFill(species="greattusk", moves=("headlongrush",), stats={"atk": 361}, item="heavydutyboots"),
        )
        merged = list(LayeredFiller(top, bottom).fill_side(observation()))[0]
        assert merged.species == "gholdengo"
        assert merged.moves == () and merged.stats is None and merged.item is None

    def test_a_layer_that_names_the_same_species_still_contributes(self):
        top = _FixedFiller("top", SlotFill(species="greattusk"))
        bottom = _FixedFiller("bottom", SlotFill(species="greattusk", moves=("headlongrush",)))
        merged = list(LayeredFiller(top, bottom).fill_side(observation()))[0]
        assert merged.moves == ("headlongrush",)

    def test_a_species_less_layer_always_contributes(self):
        top = _FixedFiller("top", SlotFill(species="gholdengo"))
        bottom = _FixedFiller("bottom", SlotFill(ability="goodasgold"))
        merged = list(LayeredFiller(top, bottom).fill_side(observation()))[0]
        assert merged.ability == "goodasgold"

    def test_the_name_lists_the_layers(self, filler):
        assert LayeredFiller(RevealedOnlyFiller(), filler).name.startswith("revealed-only+usage-stats")

    def test_rejects_a_layer_that_breaks_the_contract(self):
        class Broken:
            name = "broken"

            def fill_side(self, observation):
                return [SlotFill()]

        with pytest.raises(ValueError, match="broken"):
            LayeredFiller(Broken()).fill_side(observation())

    def test_needs_at_least_one_layer(self):
        with pytest.raises(ValueError):
            LayeredFiller()
