"""Phase 6 / M4: the set-prediction evaluation's own judgment calls.

The measurement decides what counts as a fair question, and every one of those
decisions can flatter or bury a condition. They are pinned here so a later
change to the scoring shows up as a failing test rather than as a moved number
in a report nobody can reproduce.
"""

from __future__ import annotations

import pytest

from battle_engine.poke_engine_state import (
    TEAM_SIZE,
    RevealedOnlyFiller,
    SideObservation,
    SlotFill,
    SlotObservation,
)
from battle_engine.set_eval import RevealedSet, score_snapshot


class StubFiller:
    """Returns a canned fill for slot 0 and a canned species for the rest."""

    name = "stub"

    def __init__(self, first=SlotFill(), species=()):
        self._first = first
        self._species = list(species)

    def fill_side(self, observation):
        fills = [self._first]
        remaining = list(self._species)
        for _ in range(observation.team_size - 1):
            fills.append(SlotFill(species=remaining.pop(0)) if remaining else SlotFill())
        return fills


def observation(*slots: SlotObservation) -> SideObservation:
    filled = list(slots) + [SlotObservation(index=i) for i in range(len(slots), TEAM_SIZE)]
    return SideObservation(
        is_ours=False, format_id="gen9ou", team_size=TEAM_SIZE, slots=tuple(filled)
    )


TUSK = RevealedSet(
    species="greattusk",
    ability="protosynthesis",
    item="heavydutyboots",
    tera_type="steel",
    moves=frozenset({"headlongrush", "icespinner", "rapidspin", "knockoff"}),
)


class TestOnlyHiddenFactsAreScored:
    def test_an_already_revealed_attribute_is_never_asked(self):
        slot = SlotObservation(
            index=0,
            species="greattusk",
            ability="protosynthesis",
            item="heavydutyboots",
            tera_type="steel",
            moves=tuple(TUSK.moves),
        )
        counts = score_snapshot(observation(slot), {"greattusk": TUSK}, StubFiller())
        # Counting these would inflate every condition equally and make the
        # comparison between conditions smaller than it is.
        assert (counts.ability_asked, counts.item_asked, counts.tera_asked, counts.move_asked) == (
            0,
            0,
            0,
            0,
        )

    def test_an_attribute_the_battle_never_reveals_is_never_asked(self):
        unknown = RevealedSet(species="greattusk", moves=frozenset())
        counts = score_snapshot(
            observation(SlotObservation(index=0, species="greattusk")),
            {"greattusk": unknown},
            StubFiller(SlotFill(ability="protosynthesis", item="leftovers")),
        )
        # There is no truth to score against, so a guess is neither right nor
        # wrong. 40.5% of abilities and 26.8% of items are in this state.
        assert counts.ability_asked == 0 and counts.item_asked == 0

    def test_an_item_knocked_off_is_not_recorded_as_a_reveal(self):
        # `""` means "revealed to be holding nothing", which is an observation
        # about the slot, not a set fact the prior could have predicted.
        slot = SlotObservation(index=0, species="greattusk", item="")
        counts = score_snapshot(observation(slot), {"greattusk": TUSK}, StubFiller())
        assert counts.item_asked == 0


class TestScoring:
    def test_a_correct_prediction_scores(self):
        counts = score_snapshot(
            observation(SlotObservation(index=0, species="greattusk")),
            {"greattusk": TUSK},
            StubFiller(SlotFill(ability="protosynthesis", item="heavydutyboots", tera_type="steel")),
        )
        assert (counts.ability_hit, counts.item_hit, counts.tera_hit) == (1, 1, 1)

    def test_a_wrong_prediction_is_charged_not_ignored(self):
        counts = score_snapshot(
            observation(SlotObservation(index=0, species="greattusk")),
            {"greattusk": TUSK},
            StubFiller(SlotFill(ability="levitate", item="leftovers", tera_type="water")),
        )
        assert (counts.ability_asked, counts.ability_hit) == (1, 0)
        assert (counts.item_asked, counts.item_hit) == (1, 0)
        assert (counts.tera_asked, counts.tera_hit) == (1, 0)

    def test_declining_to_predict_is_charged_the_same_as_being_wrong(self):
        counts = score_snapshot(
            observation(SlotObservation(index=0, species="greattusk")),
            {"greattusk": TUSK},
            RevealedOnlyFiller(),
        )
        assert (counts.ability_asked, counts.ability_hit) == (1, 0)

    def test_move_recall_counts_only_the_moves_still_hidden(self):
        slot = SlotObservation(index=0, species="greattusk", moves=("headlongrush", "icespinner"))
        counts = score_snapshot(
            observation(slot),
            {"greattusk": TUSK},
            StubFiller(SlotFill(moves=("rapidspin", "bulkup"))),
        )
        assert counts.move_asked == 2  # rapidspin, knockoff
        assert counts.move_hit == 1

    def test_species_recall_is_over_what_is_still_hidden(self):
        truth = {"greattusk": TUSK, "gholdengo": RevealedSet(species="gholdengo")}
        counts = score_snapshot(
            observation(SlotObservation(index=0, species="greattusk")),
            truth,
            StubFiller(species=["gholdengo", "kingambit"]),
        )
        assert counts.species_asked == 1 and counts.species_hit == 1
        # Precision is over every guess, including ones about slots the battle
        # never showed - a lower bound, since a Pokemon held in the back all
        # game is indistinguishable here from one that was never on the team.
        assert counts.species_guesses == 2 and counts.species_guesses_correct == 1

    def test_the_baseline_scores_zero_on_everything_by_construction(self):
        truth = {"greattusk": TUSK, "gholdengo": RevealedSet(species="gholdengo")}
        counts = score_snapshot(
            observation(SlotObservation(index=0, species="greattusk")), truth, RevealedOnlyFiller()
        )
        assert counts.species_hit == 0 and counts.species_guesses == 0
        assert counts.move_hit == 0 and counts.item_hit == 0
        # ...but the denominators are not zero, so the comparison is against a
        # real set of questions rather than an empty one.
        assert counts.species_asked == 1 and counts.move_asked == 4

    def test_a_species_the_truth_has_never_heard_of_is_skipped(self):
        counts = score_snapshot(
            observation(SlotObservation(index=0, species="greattusk")), {}, StubFiller()
        )
        assert counts.ability_asked == 0 and counts.move_asked == 0


class TestDistinctCases:
    def test_distinct_keys_are_collected_for_the_sample_size_line(self):
        asked: dict = {}
        from collections import defaultdict

        asked = defaultdict(set)
        slot = SlotObservation(index=0, species="greattusk", moves=("headlongrush",))
        score_snapshot(observation(slot), {"greattusk": TUSK}, StubFiller(), asked)
        # Turn-weighted totals repeat the same slot every turn; these keys are
        # what stops a report reading 111 asks as 111 independent trials.
        assert asked["ability"] == {"greattusk"}
        assert asked["moves recall"] == {
            ("greattusk", "icespinner"),
            ("greattusk", "rapidspin"),
            ("greattusk", "knockoff"),
        }
