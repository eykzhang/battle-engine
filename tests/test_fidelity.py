"""Phase 6 / M3: the forward-model fidelity harness.

What is worth pinning here is not "the harness runs" - the corpus run proves
that - but the judgment calls it makes, because every one of them changes a
published number and none of them would fail loudly if it silently changed:

- which turns are excluded, and that each exclusion has a *named* reason
  rather than being absorbed into the denominator;
- that `OracleFiller` supplies the turn's action and nothing else, so a
  fidelity number is not quietly an omniscience number;
- that the comparison ignores what a replay cannot see, and that the two
  categories which measure the harness rather than the model
  (`item_revealed`, `volatile`) stay out of the verdict.

Fixtures are real `poke_env.battle.Battle` objects driven by real Showdown
protocol messages, for the same reason
tests/test_poke_engine_state.py gives: a hand-built fake would echo this
module's own assumptions back at it.
"""

from __future__ import annotations

import json
import logging

import pytest

poke_engine = pytest.importorskip(
    "poke_engine",
    reason="poke-engine not built; run scripts/build_poke_engine.sh",
)

from poke_env.battle.battle import Battle  # noqa: E402

from battle_engine import fidelity  # noqa: E402
from battle_engine.poke_engine_state import (  # noqa: E402
    RevealedOnlyFiller,
    SideObservation,
    SlotObservation,
    state_from_poke_env,
)
from battle_engine.replay_log import Action, ActionKind, parse_replay_json  # noqa: E402

LOGGER = logging.getLogger("test-fidelity")


# ---------------------------------------------------------------------------
# A complete two-turn gen9ou log, as Showdown actually emits one.
#
# Turn 1: both sides attack. Turn 2: p1 switches, p2 attacks. The log carries
# both sides' actions, which is the whole reason M2 fetched replays from
# Showdown's API instead of reusing the Metamon corpus.
# ---------------------------------------------------------------------------

LOG = """|player|p1|alice|101|1500
|player|p2|bob|clerk|1520
|gen|9
|tier|[Gen 9] OU
|rule|HP Percentage Mod: HP is shown in percentages
|clearpoke
|poke|p1|Great Tusk|
|poke|p1|Dragapult|
|poke|p2|Gholdengo|
|poke|p2|Kingambit|
|teampreview
|teamsize|p1|2
|teamsize|p2|2
|start
|switch|p1a: Tusk|Great Tusk, L100|100/100
|switch|p2a: Gholdengo|Gholdengo, L100|100/100
|turn|1
|move|p1a: Tusk|Headlong Rush|p2a: Gholdengo
|-damage|p2a: Gholdengo|55/100
|move|p2a: Gholdengo|Make It Rain|p1a: Tusk
|-damage|p1a: Tusk|61/100
|
|upkeep
|turn|2
|switch|p1a: Pult|Dragapult, L100|100/100
|move|p2a: Gholdengo|Shadow Ball|p1a: Pult
|-damage|p1a: Pult|72/100
|
|upkeep
|turn|3
|win|bob
"""


def _replay_payload(log: str = LOG) -> dict:
    return {"id": "gen9ou-test-1", "format": "gen9ou", "log": log, "rating": 1500}


def _battle_at(marker: str, turn: int, log: str = LOG) -> Battle:
    """The live battle at one instant, driven by the same code the harness uses."""
    for seen_marker, seen_turn, battle in fidelity.ReplayDriver(log, "battle-gen9ou-test"):
        if (seen_marker, seen_turn) == (marker, turn):
            return battle
    raise AssertionError(f"log never reached {marker} of turn {turn}")


def _transitions(log: str = LOG):
    return {t.turn: t for t in parse_replay_json(_replay_payload(log)).transitions}


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------


def test_driver_reaches_both_turn_boundaries():
    markers = [(marker, turn) for marker, turn, _ in fidelity.ReplayDriver(LOG, "battle-gen9ou-test")]
    assert ("turn", 1) in markers
    assert ("upkeep", 1) in markers
    assert markers.index(("turn", 1)) < markers.index(("upkeep", 1)) < markers.index(("turn", 2))


def test_driver_counts_unhandled_messages_instead_of_dropping_them():
    """The two poke-env refuses on real gen9ou logs are `|t:|` and `|win|`.

    Counted rather than swallowed: a third one appearing means a protocol
    message with battle state in it is being silently discarded.
    """
    driver = fidelity.ReplayDriver(LOG, "battle-gen9ou-test")
    list(driver)
    assert set(driver.unhandled) <= {"t:", "win"}
    assert driver.unhandled["win"] == 1


def test_driver_reads_p1_username_from_the_log():
    """poke-env infers `player_role` from a username match and picks the
    *opposite* role when it does not match.

    A placeholder username lands on the right role only by accident, after
    both opening `|player|` lines cancel out - and a mid-battle re-announcement
    (a disconnect/reconnect emits another `|player|p1|...`) then flips it, so
    p1's Pokemon start being filed into `opponent_team`. Two of the 300 corpus
    replays died that way with "p2's team already has 6 pokemons"; the failure
    is silent until a seventh Pokemon arrives, which is late in a battle and
    only in some battles.
    """
    driver = fidelity.ReplayDriver(LOG, "battle-gen9ou-test")
    assert driver.player_username == "alice"
    _, _, battle = next(iter(driver))
    assert battle.player_role == "p1"


def test_driver_survives_a_mid_battle_player_reannouncement():
    reconnect = LOG.replace("|turn|2\n", "|player|p1|alice|101|\n|turn|2\n")
    for marker, turn, battle in fidelity.ReplayDriver(reconnect, "battle-gen9ou-test"):
        if (marker, turn) == ("upkeep", 2):
            assert battle.player_role == "p1"
            assert set(battle.team) == {"p1: Tusk", "p1: Pult"}
            assert set(battle.opponent_team) == {"p2: Gholdengo"}
            return
    raise AssertionError("log never reached upkeep of turn 2")


def test_driver_state_before_and_after_differ_by_the_turn():
    before = state_from_poke_env(_battle_at("turn", 1)).state
    after = state_from_poke_env(_battle_at("upkeep", 1)).state
    assert before.side_two.pokemon[0].hp == before.side_two.pokemon[0].maxhp
    assert after.side_two.pokemon[0].hp < after.side_two.pokemon[0].maxhp


# ---------------------------------------------------------------------------
# Action translation - what gets scored and what gets excluded, by name
# ---------------------------------------------------------------------------


def test_move_action_translates_to_its_move_id():
    transition = _transitions()[1]
    action = fidelity.engine_action(transition.p1_action, transition)
    assert action.kind == "move"
    assert action.text == "headlongrush"
    assert action.needs_move == "headlongrush"


def test_switch_action_is_addressed_by_species_not_by_slot():
    """poke-engine's `MoveChoice::from_string` resolves a switch by species id.

    Not by index, and specifically not by `base_species` - that collapses
    formes poke-engine stats separately, producing a real id with the wrong
    stats and no error (the M3 finding).
    """
    transition = _transitions()[2]
    action = fidelity.engine_action(transition.p1_action, transition)
    assert action.kind == "switch"
    assert action.text == "dragapult"
    assert action.needs_species == "dragapult"


@pytest.mark.parametrize(
    "kind,reason",
    [
        (ActionKind.UNOBSERVED, "unobserved_action"),
        (ActionKind.BLOCKED, "blocked_action"),
        (ActionKind.DRAGGED, "dragged"),
        (ActionKind.PIVOT, "forced_pivot"),
        (ActionKind.REPLACEMENT, "replacement_in_action_slot"),
    ],
)
def test_unreplayable_action_kinds_are_excluded_with_a_named_reason(kind, reason):
    """Every exclusion is named, so none of them can hide in the denominator.

    These are turns where the log records no decision that could be replayed -
    not turns the model got wrong - and conflating the two is how a fidelity
    number stops meaning anything.
    """
    transition = _transitions()[1]
    with pytest.raises(fidelity.UnscorableTurn) as excinfo:
        fidelity.engine_action(Action(player="p1", kind=kind), transition)
    assert excinfo.value.reason == reason


def test_tera_move_carries_the_tera_type_it_will_need():
    transition = _transitions()[1]
    action = fidelity.engine_action(
        Action(player="p1", kind=ActionKind.MOVE, move="icebeam", terastallized=True, tera_type="Ghost"),
        transition,
    )
    assert action.text == "icebeam-tera"
    assert action.needs_tera_type == "Ghost"


# ---------------------------------------------------------------------------
# The oracle - it must supply the turn's action and nothing more
# ---------------------------------------------------------------------------


def test_move_is_unaddressable_before_it_is_revealed():
    """The measurement that makes M4's case, pinned as behaviour.

    On turn 1 nothing has used a move, so no Pokemon in a revealed-only state
    has any moves at all, and poke-engine - which resolves an action by name
    against the active Pokemon's current moveset - cannot be asked what
    Headlong Rush does. That is a set-prediction gap, not a forward-model
    failure, and the harness has to be able to tell them apart.
    """
    battle = _battle_at("turn", 1)
    baseline = state_from_poke_env(battle, filler=RevealedOnlyFiller()).state
    action = fidelity.engine_action(_transitions()[1].p1_action, _transitions()[1])

    assert not fidelity._addressable(baseline, "side_one", action)
    with pytest.raises(ValueError):
        poke_engine.generate_instructions(baseline, action.text, "none")


def test_oracle_makes_exactly_that_move_addressable():
    battle = _battle_at("turn", 1)
    transition = _transitions()[1]
    a1 = fidelity.engine_action(transition.p1_action, transition)
    a2 = fidelity.engine_action(transition.p2_action, transition)
    oracle = fidelity.OracleFiller(
        fidelity._side_oracle(battle, a1, ours=True),
        fidelity._side_oracle(battle, a2, ours=False),
    )
    state = state_from_poke_env(battle, filler=oracle).state

    assert [m.id.lower() for m in state.side_one.pokemon[int(state.side_one.active_index)].moves] == ["headlongrush"]
    assert [m.id.lower() for m in state.side_two.pokemon[int(state.side_two.active_index)].moves] == ["makeitrain"]
    assert poke_engine.generate_instructions(state, a1.text, a2.text)


def test_oracle_does_not_lift_the_rest_of_the_fog():
    """A fidelity number must not quietly become an omniscience number.

    The action oracle supplies one move; it must leave unrevealed slots as
    placeholders and unrevealed items as `unknownitem`, exactly as
    `RevealedOnlyFiller` would.
    """
    battle = _battle_at("turn", 1)
    transition = _transitions()[1]
    oracle = fidelity.OracleFiller(
        fidelity._side_oracle(battle, fidelity.engine_action(transition.p1_action, transition), ours=True),
        fidelity._side_oracle(battle, fidelity.engine_action(transition.p2_action, transition), ours=False),
    )
    state = state_from_poke_env(battle, filler=oracle).state

    unrevealed = [p for p in state.side_two.pokemon if p.id.lower() == fidelity.PLACEHOLDER_ID]
    assert len(unrevealed) == 5, "the opponent's unrevealed slots must stay placeholders"
    assert state.side_two.pokemon[0].item.lower() == "unknownitem"


def test_oracle_cannot_overwrite_an_observation():
    """The property the whole harness leans on.

    `state_from_poke_env` merges observation over fill structurally, so an
    oracle that names the wrong move cannot erase a revealed one - it can
    only add. If that ever stopped holding, the oracle could paper over a
    real divergence and the fidelity number would be silently inflated.
    """
    battle = _battle_at("turn", 2)  # Gholdengo has now revealed Make It Rain
    oracle = fidelity.OracleFiller(
        fidelity._SideOracle(),
        fidelity._SideOracle(active_species="gholdengo", move="shadowball"),
    )
    state = state_from_poke_env(battle, filler=oracle).state
    moves = {m.id.lower() for m in state.side_two.pokemon[int(state.side_two.active_index)].moves}
    assert "makeitrain" in moves, "a revealed move must survive the oracle"
    assert "shadowball" in moves


def test_hindsight_supplies_only_monotone_attributes():
    """`Hindsight` back-fills from the battle's end, so it may only carry
    attributes that were already true on turn 1.

    Items are excluded on purpose: they are consumed, eaten and knocked off,
    so a battle-final item is not a valid turn-N item, and back-filling one
    would fabricate an observation.
    """
    knowledge = fidelity.hindsight_knowledge(parse_replay_json(_replay_payload()))
    assert set(knowledge) == {"p1", "p2"}
    assert "makeitrain" in knowledge["p2"]["gholdengo"].moves
    assert "shadowball" in knowledge["p2"]["gholdengo"].moves
    assert not hasattr(fidelity.Hindsight(), "item")


# ---------------------------------------------------------------------------
# Comparison - only what a replay can see
# ---------------------------------------------------------------------------


def test_identical_states_diverge_nowhere():
    state = state_from_poke_env(_battle_at("turn", 1)).state
    assert fidelity.compare_states(state, state, state) == ()


def test_hp_inside_the_percent_quantization_is_not_a_divergence():
    """A replay reports HP to the nearest whole percent on both sides, so an
    exactly-right prediction can still miss by a rounding step."""
    before = state_from_poke_env(_battle_at("turn", 1)).state
    observed = state_from_poke_env(_battle_at("upkeep", 1)).state
    nudged = observed.apply_instructions(poke_engine.generate_instructions(observed, "none", "none")[0])
    hp_divergences = [d for d in fidelity.compare_states(observed, nudged, before) if d.category == "hp"]
    assert all(d.magnitude > fidelity.HP_TOLERANCE for d in hp_divergences)


def test_an_unknownitem_prediction_is_reported_but_not_scored():
    """The model saying "I do not know this item" is a declined prediction,
    not a wrong one - the item became visible during the turn being scored.
    Its real cost already lands in `hp`, so counting it again as a modelling
    error would double-charge a set-prediction gap.
    """
    divergences = (
        fidelity.Divergence("item_revealed", "side_one", "gholdengo", "leftovers", "unknownitem"),
        fidelity.Divergence("hp", "side_one", "gholdengo", 0.9, 0.8, 0.1),
    )
    assert [d.category for d in fidelity.scoring_divergences(divergences)] == ["hp"]
    assert "item_revealed" in fidelity.INFORMATIONAL
    assert "volatile" in fidelity.INFORMATIONAL


def test_an_item_prediction_the_replay_cannot_check_is_reported_but_not_scored():
    """The mirror of the case above, and the one M4 introduced. A replay never
    reveals 73.2% of items, so a prior that names one is making a claim the log
    has no way to confirm or deny. Scoring it would charge set prediction for
    the fog of war; without this arm every turn under a usage-stats prior
    counted as a miss, which was a measurement artifact and not a finding. If
    the guess is wrong, its cost still lands in `hp`.
    """
    divergences = (
        fidelity.Divergence("item_predicted", "side_two", "kingambit", "unknownitem", "leftovers"),
        fidelity.Divergence("hp", "side_two", "kingambit", 0.9, 0.8, 0.1),
    )
    assert [d.category for d in fidelity.scoring_divergences(divergences)] == ["hp"]
    assert "item_predicted" in fidelity.INFORMATIONAL


def test_two_real_items_that_disagree_are_still_a_real_divergence():
    """The informational arms are about *unknowns*, not about items generally.
    When the battle showed one item and the model has another, that is the
    prior being wrong and it is scored."""
    divergence = fidelity.Divergence("item", "side_two", "kingambit", "leftovers", "airballoon")
    assert fidelity.scoring_divergences((divergence,)) == (divergence,)


def test_weather_expiry_is_a_separate_finding_from_a_missed_weather():
    """How long a weather has left is not observable (poke-env records only
    when it started, and poke-engine implements none of the four rocks that
    extend it), so the model clearing one early is a different finding from
    the model failing to notice one starting."""
    assert fidelity._field_divergence("weather", "sand", "sand", "none").category == "weather_expired_early"
    assert fidelity._field_divergence("weather", "sand", "none", "sand").category == "weather_expired_late"
    assert fidelity._field_divergence("weather", "none", "sand", "none").category == "weather"
    assert fidelity._field_divergence("weather", "sand", "sand", "sand") is None


def test_placeholder_slots_are_not_compared():
    """Several unrevealed slots share the id `none`, and there is nothing to
    compare in them anyway - a placeholder is the harness's own invention."""
    state = state_from_poke_env(_battle_at("turn", 1)).state
    assert fidelity.PLACEHOLDER_ID not in fidelity._slots_by_species(state.side_two)
    assert set(fidelity._slots_by_species(state.side_two)) == {"gholdengo"}


# ---------------------------------------------------------------------------
# Encore: a poke-engine invariant that panics rather than raising
# ---------------------------------------------------------------------------


def test_encore_without_a_last_used_move_is_dropped_rather_than_guessed():
    """poke-engine panics - through pyo3, past every `except Exception` - if a
    side carries `encore` while `last_used_move` is not a move.

    The translator now supplies `last_used_move` from poke-env, and drops
    `encore` when it cannot. Pinned here because the failure mode is a hard
    process crash, which on the real ladder is a forfeited game, and because
    the alternative fix (point `last_used_move` at an arbitrary index) would
    force the wrong move every turn with no error at all.
    """
    battle = _battle_at("turn", 2)
    result = state_from_poke_env(battle)
    assert "encore" not in {v.lower() for v in result.state.side_one.volatile_statuses}
    # Nothing has moved on the incoming Dragapult, so there is no last move
    # to name; the field must still be the shape poke-engine's own default is.
    assert result.state.side_one.last_used_move.startswith("move:")
    poke_engine.generate_instructions(result.state, "none", "none")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_scoring_a_replay_produces_one_score_per_replayable_turn(tmp_path):
    path = tmp_path / "gen9ou-test-1.json"
    path.write_text(json.dumps(_replay_payload()))

    report = fidelity.score_replay(path)

    assert report.replays == 1
    # Three transitions: turns 1 and 2, plus the turn the battle ends on,
    # where nobody is recorded as having acted.
    assert report.turns_seen == 3
    assert {s.turn for s in report.scores} == {1, 2}
    assert report.skipped["unobserved_action"] == 1
    assert not report.panics
    # Nothing had been revealed on turn 1, so neither action was addressable
    # without the oracle - the M4 baseline, on a battle small enough to check
    # by hand.
    assert not report.scores[0].representable
    assert sorted(report.scores[0].missing) == ["p1:move", "p2:move"]


def test_hindsight_condition_scores_the_same_turns():
    """The two conditions must differ only in what the oracle knows.

    If hindsight also changed which turns are scorable, the delta between the
    conditions would not be attributable to knowledge.
    """
    payload = _replay_payload()
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        path = handle.name

    action = fidelity.score_replay(path)
    hindsight = fidelity.score_replay(path, hindsight=True)
    assert [s.turn for s in action.scores] == [s.turn for s in hindsight.scores]
    assert action.skipped == hindsight.skipped
    assert action.condition == "action-oracle"
    assert hindsight.condition == "hindsight-oracle"


def test_report_renders_without_scores():
    """A corpus that produces nothing scorable must still explain itself
    rather than dividing by zero."""
    report = fidelity.FidelityReport(backend="poke-engine")
    report.turns_seen = 3
    report.skipped["unobserved_action"] = 3
    rendered = report.render()
    assert "unobserved_action" in rendered
