"""Phase 6 / M7 (BattleBrain): the replay-analysis CLI's core module.

Two things are worth real coverage here, beyond schema shape: that a turn
whose every sampled search panics is still emitted (`winProbability: null`),
not dropped - dropping it would be indistinguishable from "this turn was
never reached" - and that `BaseException` (not `Exception`) is what is
actually caught, the same real gap `set_search.py`'s own tests guard
(`notes/gotcha-poke-engine-encore-panics-without-last-used-move.md`). A
monkeypatched panic proves this directly: if the implementation caught only
`Exception`, this test would fail with an unhandled exception rather than a
clean assertion.

Runs against one small real corpus replay at a reduced search budget (not
ladder parity) - real `poke_engine`/`state_from_poke_env`/usage-stats
translation, kept fast by a small `--search-time-ms`/`--opponent-samples`
rather than by mocking the pipeline itself.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

poke_engine = pytest.importorskip(
    "poke_engine", reason="poke-engine not built; run scripts/build_poke_engine.sh"
)

from battle_engine.replay_analysis import (  # noqa: E402
    REASON_ACTION_NOT_IN_SEARCH_RESULTS,
    REASON_BASELINE_TRANSLATION_FAILED,
    REASON_NO_SEARCH_RESULTS,
    REASON_NO_TRANSITION,
    REASON_NOT_ADDRESSABLE,
    ReplayAnalysisError,
    analyze_replay,
    coverage,
)
from battle_engine.replay_log import ReplayParseError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
# `scripts/` is deliberately excluded from the installed `battle_engine`
# package (`pyproject.toml`'s `packages.find` includes only `battle_engine*`,
# so `scripts.analyze_replay` doesn't resolve via the editable install the
# way `battle_engine.*` does above) and carries no `__init__.py`. DW-2.15
# needs to import it directly as the plain script it is; this mirrors how
# `.venv/bin/python scripts/analyze_replay.py` already runs it, cwd-relative
# from the repo root, rather than as an installed package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORPUS = REPO_ROOT / "data" / "replays_showdown"
# Shortest of the plan's pre-scanned candidates (24 turns) - fast enough for a
# reduced-budget pytest run without needing to mock the search itself.
SHORT_REPLAY = CORPUS / "gen9ou-2672927429.json"

# A deliberately small budget: fast, but still exercises the real
# translate-and-search pipeline (not the ladder-parity default, which is a
# fixture-generation-time concern, not a unit-test one).
FAST_KWARGS = dict(search_time_ms=100, n_opponent_samples=2, threads=2)


def _payload() -> dict:
    return json.loads(SHORT_REPLAY.read_text())


@pytest.mark.skipif(not SHORT_REPLAY.exists(), reason="replay corpus not fetched")
class TestSchemaShape:
    def test_DW_2_1_and_DW_2_3_produces_schema_v1_document_with_bounded_win_probability(self) -> None:
        doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)

        assert doc["schemaVersion"] == 1
        assert doc["replayId"] == "gen9ou-2672927429"
        assert doc["format"] == "gen9ou"
        assert isinstance(doc["players"], list) and len(doc["players"]) == 2
        assert doc["perspective"] == "p1"
        assert doc["engine"] == {
            "searchBudgetMsPerTurn": 100,
            "opponentSamples": 2,
            "threads": 2,
            "usageStatsCutoff": 1500,
            "pokeEngineTag": "v0.0.48",
        }
        assert doc["totalTurns"] == len(doc["turns"]) > 0
        assert 0 <= doc["gradableTurns"] <= doc["totalTurns"]

        for turn in doc["turns"]:
            assert set(turn) == {
                "turn",
                "winProbability",
                "gradable",
                "ungradableReason",
                "samplesUsed",
                "playedAction",
                "playedActionValue",
                "costOfPlayed",
                "topActions",
            }
            wp = turn["winProbability"]
            assert wp is None or (isinstance(wp, float) and 0.0 <= wp <= 1.0)
            assert isinstance(turn["gradable"], bool)
            # DW-2.9: samplesUsed is the count that actually contributed,
            # bounded by the configured opponentSamples - never negative,
            # never more than configured.
            assert isinstance(turn["samplesUsed"], int)
            assert 0 <= turn["samplesUsed"] <= doc["engine"]["opponentSamples"]
            # One direction only: zero surviving samples guarantees no
            # winProbability. The converse does not hold - a zero-visit
            # search (DW-2.14) can have samplesUsed > 0 and still produce a
            # null winProbability, since every sample completed without
            # panicking but the search explored nothing. Asserting the full
            # biconditional here would fail on that correct behavior the
            # moment this unmocked run happened to hit it.
            if turn["samplesUsed"] == 0:
                assert wp is None
            # DW-2.13: a reason exactly when ungradable, never when gradable.
            assert (turn["ungradableReason"] is None) == turn["gradable"]
            if not turn["gradable"]:
                assert turn["playedActionValue"] is None
                assert turn["costOfPlayed"] is None
                assert isinstance(turn["ungradableReason"], str) and turn["ungradableReason"]
            else:
                # DW-2.12: a gradable turn's cost is never negative - the
                # played action's value can be at most the best ranked value.
                assert turn["costOfPlayed"] >= 0.0
            for action in turn["topActions"]:
                assert set(action) == {"action", "visitShare", "value"}
                assert 0.0 <= action["visitShare"] <= 1.0

    def test_DW_2_2_reports_eval_bar_and_grading_coverage(self) -> None:
        doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)

        eval_bar, grading = coverage(doc)
        assert 0.0 <= eval_bar <= 1.0
        assert 0.0 <= grading <= 1.0
        # Independently derivable from the document itself, not just trusted
        # from the helper: both numbers must be recoverable from the shipped
        # JSON alone (what an app fixture actually ships).
        non_null = sum(1 for t in doc["turns"] if t["winProbability"] is not None)
        assert eval_bar == pytest.approx(non_null / doc["totalTurns"])
        assert grading == pytest.approx(doc["gradableTurns"] / doc["totalTurns"])

    def test_p2_perspective_produces_a_valid_document_from_the_other_side(self) -> None:
        doc = analyze_replay(_payload(), perspective="p2", **FAST_KWARGS)
        assert doc["perspective"] == "p2"
        assert doc["totalTurns"] > 0
        # p1 and p2 see the same battle, so the turn count should agree.
        doc_p1 = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)
        assert doc["totalTurns"] == doc_p1["totalTurns"]

    def test_DW_2_12_switch_turn_is_gradable_and_matches_canonical_name(self) -> None:
        # Turn 5 of this replay is a switch into moltres - fidelity.py names
        # it "moltres" (EngineAction.text), set_search.aggregate names the
        # same option "switch moltres". Before the fix-forward, the lookup
        # compared the two forms directly and no switch was ever gradable.
        doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)
        switch_turn = next(t for t in doc["turns"] if t["turn"] == 5)
        assert switch_turn["playedAction"] == "moltres"
        assert switch_turn["gradable"] is True
        assert switch_turn["ungradableReason"] is None
        assert switch_turn["playedActionValue"] is not None
        assert switch_turn["costOfPlayed"] is not None
        assert switch_turn["costOfPlayed"] >= 0.0
        assert any(a["action"] == "switch moltres" for a in switch_turn["topActions"])


@pytest.mark.skipif(not SHORT_REPLAY.exists(), reason="replay corpus not fetched")
class TestDW_2_13_UngradableReason:
    """DW-2.13: every ungradable turn names why, not just that."""

    def test_reason_is_none_exactly_when_gradable(self) -> None:
        doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)
        for turn in doc["turns"]:
            assert (turn["ungradableReason"] is None) == turn["gradable"]

    def test_not_addressable_is_the_reason_for_a_known_own_team_unrevealed_turn(self) -> None:
        # Turn 2: the played switch's canonical name is present in that
        # turn's search results (the search draws sometimes place the
        # species on our team), but the turn's own deterministic baseline
        # draw does not - see the DW-2.12 Decision Log entry for why that
        # disagreement is expected rather than a bug.
        doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)
        turn2 = next(t for t in doc["turns"] if t["turn"] == 2)
        assert turn2["gradable"] is False
        assert turn2["ungradableReason"] == REASON_NOT_ADDRESSABLE

    def test_unscorable_turn_reason_is_surfaced_verbatim_not_collapsed(self) -> None:
        # Turn 13's action is UNOBSERVED (no playedAction at all) - engine_action
        # raises UnscorableTurn("unobserved_action"), and that string - not a
        # generic "unscorable" label - is what ungradableReason carries, reusing
        # UnscorableTurn's own already-closed vocabulary instead of inventing a
        # second one.
        doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)
        turn13 = next(t for t in doc["turns"] if t["turn"] == 13)
        assert turn13["playedAction"] is None
        assert turn13["ungradableReason"] == "unobserved_action"

    def test_reason_taxonomy_is_a_small_closed_vocabulary(self) -> None:
        doc_p1 = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)
        doc_p2 = analyze_replay(_payload(), perspective="p2", **FAST_KWARGS)
        known = {
            REASON_NO_TRANSITION,
            REASON_BASELINE_TRANSLATION_FAILED,
            REASON_NOT_ADDRESSABLE,
            REASON_NO_SEARCH_RESULTS,
            REASON_ACTION_NOT_IN_SEARCH_RESULTS,
            # fidelity.UnscorableTurn's own closed vocabulary, surfaced
            # verbatim rather than collapsed into one generic label - every
            # reason `engine_action` (and the `_switch_target_species` helper
            # it calls for a switch) can raise, per fidelity.py:301-354. The
            # five turn-boundary/state-translation reasons `_prepare` raises
            # (fidelity.py:1028-1080) are a different code path - used by
            # fidelity measurement, never reachable from `_grade_turn`'s
            # `engine_action` call - and are deliberately excluded.
            "unobserved_action",
            "blocked_action",
            "dragged",
            "forced_pivot",
            "replacement_in_action_slot",
            "move_unknown",
            "move_not_in_engine",
            "switch_without_target",
            "switch_target_not_in_snapshot",
            "species_not_in_engine",
            "switch_target_is_placeholder",
        }
        reasons = {
            t["ungradableReason"] for doc in (doc_p1, doc_p2) for t in doc["turns"] if t["ungradableReason"]
        }
        assert reasons, "no ungradable turn produced a reason to check the taxonomy against"
        assert reasons <= known, f"unrecognized ungradableReason(s): {reasons - known}"

    def test_DW_2_9_and_search_result_reasons_agree_under_total_panic(self, monkeypatch) -> None:
        def _panic(*args, **kwargs):
            raise _AlwaysPanics("simulated poke-engine NonFinite panic")

        monkeypatch.setattr("poke_engine.monte_carlo_tree_search", _panic)
        doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)

        # Every turn with a translated, addressable action has no search
        # results to grade against - that specific reason, not some other one.
        for turn in doc["turns"]:
            if turn["ungradableReason"] == REASON_ACTION_NOT_IN_SEARCH_RESULTS:
                pytest.fail("action_not_in_search_results should be unreachable when ranked is always empty")
        assert any(t["ungradableReason"] == REASON_NO_SEARCH_RESULTS for t in doc["turns"])


@pytest.mark.skipif(not SHORT_REPLAY.exists(), reason="replay corpus not fetched")
class TestValidation:
    def test_invalid_perspective_raises_replay_analysis_error(self) -> None:
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(_payload(), perspective="p3", **FAST_KWARGS)

    def test_missing_log_raises_replay_parse_error(self) -> None:
        payload = _payload()
        del payload["log"]
        with pytest.raises(ReplayParseError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_non_int_rating_is_reported_as_null_not_trusted_verbatim(self) -> None:
        payload = _payload()
        payload["rating"] = "not-a-number"  # malformed external input
        doc = analyze_replay(payload, perspective="p1", **FAST_KWARGS)
        assert doc["rating"] is None

    def test_DW_2_10_bool_rating_is_reported_as_null_not_as_a_json_boolean(self) -> None:
        # bool subclasses int in Python - isinstance(True, int) is True - so
        # this needs its own guard beyond the plain isinstance(rating, int)
        # check the previous test exercises.
        payload = _payload()
        payload["rating"] = True
        doc = analyze_replay(payload, perspective="p1", **FAST_KWARGS)
        assert doc["rating"] is None

    def test_DW_2_10_non_iterable_players_is_rejected(self) -> None:
        payload = _payload()
        payload["players"] = 5
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_wrong_arity_players_is_rejected(self) -> None:
        payload = _payload()
        payload["players"] = ["a", "b", "c"]
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_non_string_player_elements_are_rejected(self) -> None:
        payload = _payload()
        payload["players"] = [1, 2]
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_falsy_players_falls_back_to_the_log_not_rejected(self) -> None:
        # Distinguishes "malformed" from "absent": an empty players array
        # still resolves, from the log's own |player| lines, exactly like a
        # missing field does - it isn't a rejection case.
        payload = _payload()
        payload["players"] = []
        doc = analyze_replay(payload, perspective="p1", **FAST_KWARGS)
        assert len(doc["players"]) == 2

    def test_DW_2_10_non_string_id_is_rejected(self) -> None:
        payload = _payload()
        payload["id"] = {"nested": "obj"}
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_non_string_formatid_is_rejected(self) -> None:
        payload = _payload()
        payload["formatid"] = ["gen9ou"]
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_malformed_turn_line_is_rejected_not_a_raw_traceback(self) -> None:
        payload = _payload()
        assert "|turn|2" in payload["log"], "fixture no longer has a turn 2 marker to corrupt"
        payload["log"] = payload["log"].replace("|turn|2", "|turn|not-a-number", 1)
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_11_non_positive_threads_raises(self) -> None:
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(_payload(), perspective="p1", search_time_ms=100, n_opponent_samples=2, threads=-1)

    # --- Round-3 review: log *content*, not just parseability, is external
    # input. Each of these previously escaped as an untyped exception
    # (KeyError/IndexError) or reached the document verbatim. ---

    def test_DW_2_10_unknown_species_in_switch_line_is_rejected_not_a_raw_traceback(self) -> None:
        payload = _payload()
        target = "|switch|p1a: Meowscarada|Meowscarada, F|100/100"
        assert target in payload["log"]
        payload["log"] = payload["log"].replace(
            target, "|switch|p1a: Meowscarada|NotARealPokemon, L50, M|100/100", 1
        )
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_truncated_switch_line_is_rejected_not_a_raw_traceback(self) -> None:
        payload = _payload()
        target = "|switch|p1a: Meowscarada|Meowscarada, F|100/100"
        assert target in payload["log"]
        payload["log"] = payload["log"].replace(target, "|switch|p1a: Meowscarada", 1)
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_unknown_tera_type_is_rejected_not_a_raw_traceback(self) -> None:
        payload = _payload()
        payload["log"] = payload["log"] + "\n|-terastallize|p1a: Meowscarada|NotAType"
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_crlf_payload_analyzes_correctly_instead_of_erroring(self) -> None:
        # CRLF is ordinary third-party network output, not malformed data -
        # the fix is to normalize it, not merely reject it more cleanly.
        payload = _payload()
        clean_doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)
        payload["log"] = payload["log"].replace("\n", "\r\n")
        crlf_doc = analyze_replay(payload, perspective="p1", **FAST_KWARGS)
        assert crlf_doc["totalTurns"] == clean_doc["totalTurns"]
        assert crlf_doc["players"] == clean_doc["players"]

    def test_DW_2_10_out_of_range_turn_number_is_rejected(self) -> None:
        payload = _payload()
        payload["log"] = payload["log"].replace("|turn|2", "|turn|999999999999999999999", 1)
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_negative_turn_number_is_rejected(self) -> None:
        payload = _payload()
        payload["log"] = payload["log"].replace("|turn|2", "|turn|-5", 1)
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_duplicate_turn_number_is_rejected(self) -> None:
        payload = _payload()
        payload["log"] = payload["log"].replace("|turn|3", "|turn|2", 1)
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_missing_player_line_for_the_non_analyzed_side_is_rejected(self) -> None:
        # DW-2.10 round-3 (attempt-3 review, sample 1): payload["players"] is
        # left INTACT here, deliberately - a real replay-API payload always
        # carries it, and `replay_log.py`'s own fallback
        # (`players or (tracker.p1.username, tracker.p2.username)`) means a
        # truthy payload array wins, so `parsed.players` never reflects the
        # log for a real payload. The earlier version of this test forced
        # `payload["players"] = []` to reach the log-derived fallback path,
        # which only the fallback (never the common case) ever exercised -
        # exactly why the original guard missed the defect this round fixes.
        payload = _payload()
        payload["log"] = "\n".join(
            line for line in payload["log"].split("\n") if not line.startswith("|player|p2|")
        )
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_corrupted_player_line_for_the_non_analyzed_side_is_rejected(self) -> None:
        # Not merely absent - the |player|p2| line is present but its role
        # field is corrupted so it no longer matches "p2" at all. Same
        # defect class, demonstrated with a different mutation.
        payload = _payload()
        target = "|player|p2|kangarooteam1|1|1659"
        assert target in payload["log"]
        payload["log"] = payload["log"].replace(target, "|player|p2kangarooteam1|1|1659", 1)
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_missing_player_line_for_the_analyzed_side_is_still_rejected(self) -> None:
        # Control, matching the review's reproducer: the *analyzed* side's
        # line was already guarded before this round (by the old single
        # _username_for_role(log_text, perspective) call) - confirms that
        # guard still holds now that both roles are validated up front.
        payload = _payload()
        payload["log"] = "\n".join(
            line for line in payload["log"].split("\n") if not line.startswith("|player|p1|")
        )
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_DW_2_10_rating_beyond_swift_int_magnitude_is_normalized_to_null(self) -> None:
        # Well-formed as a Python int (so _validate_payload_fields has
        # nothing to reject), but large enough to overflow Phase 3's Swift
        # Int decoder - the same class of defect MAX_TURN_NUMBER already
        # guards for turn numbers, now applied to rating's own declared
        # int|null type.
        payload = _payload()
        payload["rating"] = 10**30
        doc = analyze_replay(payload, perspective="p1", **FAST_KWARGS)
        assert doc["rating"] is None

    def test_DW_2_10_rating_within_bound_is_preserved(self) -> None:
        payload = _payload()
        payload["rating"] = 1500
        doc = analyze_replay(payload, perspective="p1", **FAST_KWARGS)
        assert doc["rating"] == 1500

    def test_DW_2_10_undeterminable_format_is_rejected_not_emitted_as_null(self) -> None:
        payload = _payload()
        del payload["formatid"]
        payload["log"] = "\n".join(line for line in payload["log"].split("\n") if not line.startswith("|tier|"))
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)

    def test_replay_id_with_unsafe_characters_is_rejected(self) -> None:
        payload = _payload()
        payload["id"] = "../../../etc/passwd"
        with pytest.raises(ReplayAnalysisError):
            analyze_replay(payload, perspective="p1", **FAST_KWARGS)


class _AlwaysPanics(BaseException):
    """Stands in for pyo3's real `PanicException`: derives from
    `BaseException`, not `Exception`, so this only stays caught if the
    implementation's `except BaseException` actually is one."""


@pytest.mark.skipif(not SHORT_REPLAY.exists(), reason="replay corpus not fetched")
def test_DW_2_4_turn_with_every_sample_panicking_reports_null_not_dropped(monkeypatch) -> None:
    def _panic(*args, **kwargs):
        raise _AlwaysPanics("simulated poke-engine NonFinite panic")

    monkeypatch.setattr("poke_engine.monte_carlo_tree_search", _panic)

    doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)

    # Every turn the driver reached is still present - none silently dropped -
    # and every one has a null winProbability, since every sample failed.
    assert doc["totalTurns"] > 0
    assert len(doc["turns"]) == doc["totalTurns"]
    assert all(t["winProbability"] is None for t in doc["turns"])
    # No search result ever existed, so nothing could be graded either.
    assert doc["gradableTurns"] == 0
    assert all(t["playedActionValue"] is None and t["costOfPlayed"] is None for t in doc["turns"])
    # DW-2.9: zero samples survived, so samplesUsed says so on every turn.
    assert all(t["samplesUsed"] == 0 for t in doc["turns"])
    # DW-2.13: every turn is ungradable, so every turn carries a reason.
    assert all(t["ungradableReason"] is not None for t in doc["turns"])


@pytest.mark.skipif(not SHORT_REPLAY.exists(), reason="replay corpus not fetched")
def test_DW_2_9_samples_used_reflects_surviving_samples_not_configured_count(monkeypatch) -> None:
    real_search = poke_engine.monte_carlo_tree_search
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        # Deterministic alternation: any four consecutive calls split 2
        # surviving / 2 failing, regardless of where a turn's four samples
        # start in the overall call sequence.
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise _AlwaysPanics("simulated poke-engine NonFinite panic")
        return real_search(*args, **kwargs)

    monkeypatch.setattr("poke_engine.monte_carlo_tree_search", _flaky)

    doc = analyze_replay(_payload(), perspective="p1", search_time_ms=100, n_opponent_samples=4, threads=2)

    # The configured count is unchanged - it describes the request, not the
    # outcome - while samplesUsed reports what actually survived per turn.
    assert doc["engine"]["opponentSamples"] == 4
    assert all(t["samplesUsed"] == 2 for t in doc["turns"])
    assert all(t["winProbability"] is not None for t in doc["turns"])
    # A turn backed by 2 of 4 samples is now distinguishable in the document
    # itself from one backed by all 4 - before this fix the two were
    # byte-shaped identically.
    assert all(t["samplesUsed"] < doc["engine"]["opponentSamples"] for t in doc["turns"])


class _ZeroVisitOption:
    """Stands in for one of poke-engine's `result.side_one` entries -
    `set_search.aggregate` reads `.move_choice`/`.visits`/`.total_score` off
    it. Zero visits, not a panic: every sample completes cleanly, the search
    itself just explores nothing."""

    def __init__(self, move_choice: str) -> None:
        self.move_choice = move_choice
        self.visits = 0
        self.total_score = 0.0


class _ZeroVisitResult:
    def __init__(self) -> None:
        self.side_one = [_ZeroVisitOption("tackle")]


@pytest.mark.skipif(not SHORT_REPLAY.exists(), reason="replay corpus not fetched")
def test_DW_2_14_zero_visit_search_emits_null_not_a_confident_loss(monkeypatch) -> None:
    def _zero_visits(*args, **kwargs):
        return _ZeroVisitResult()

    monkeypatch.setattr("poke_engine.monte_carlo_tree_search", _zero_visits)

    doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)

    # Every sample "succeeded" (no panic, no exception), so this is a
    # different path than DW-2.4/DW-2.9's total-panic case - and must land
    # in the same place: null, not 0.0.
    assert doc["totalTurns"] > 0
    assert all(t["winProbability"] is None for t in doc["turns"])
    assert all(t["topActions"] == [] for t in doc["turns"])
    assert doc["gradableTurns"] == 0
    assert all(t["playedActionValue"] is None and t["costOfPlayed"] is None for t in doc["turns"])
    # samplesUsed still reports every sample as having contributed - a
    # zero-visit search is not a *failed* sample, which is exactly why the
    # old `or 1` substitution silently turned "no data" into "certain loss"
    # instead of surfacing it the way a real failure already was.
    assert all(t["samplesUsed"] == 2 for t in doc["turns"])
    # DW-2.14's coverage clause: a zero-visit turn is excluded from eval-bar
    # coverage, not counted as a (wrong) evaluation.
    eval_bar, _grading = coverage(doc)
    assert eval_bar == 0.0


class _NonFiniteOption:
    """Same shape as `_ZeroVisitOption`, but with real visits and a
    non-finite score - poke-engine's own documented failure mode is
    literally named `NonFinite`."""

    def __init__(self, move_choice: str, visits: int, total_score: float) -> None:
        self.move_choice = move_choice
        self.visits = visits
        self.total_score = total_score


class _NonFiniteResult:
    def __init__(self) -> None:
        self.side_one = [_NonFiniteOption("tackle", 5, float("nan"))]


@pytest.mark.skipif(not SHORT_REPLAY.exists(), reason="replay corpus not fetched")
def test_non_finite_score_emits_null_not_invalid_json(monkeypatch) -> None:
    def _non_finite(*args, **kwargs):
        return _NonFiniteResult()

    monkeypatch.setattr("poke_engine.monte_carlo_tree_search", _non_finite)

    doc = analyze_replay(_payload(), perspective="p1", **FAST_KWARGS)

    assert doc["totalTurns"] > 0
    assert all(t["winProbability"] is None for t in doc["turns"])
    assert all(t["topActions"] == [] for t in doc["turns"])
    assert doc["gradableTurns"] == 0
    # The demonstrated consequence this guards against: json.dumps defaults
    # to allow_nan=True, so a NaN reaching the document would still
    # serialize "successfully" here and only fail a downstream consumer
    # (Swift's JSONDecoder, confirmed separately) - allow_nan=False makes
    # the guarantee explicit and would fail this test loudly if the
    # upstream null-routing regressed.
    json.dumps(doc, allow_nan=False)


@pytest.mark.skipif(not SHORT_REPLAY.exists(), reason="replay corpus not fetched")
class TestDW_2_15_MainExitCodes:
    """DW-2.10 and DW-2.11 are both claims about `main`'s exit status - cover
    `main` directly rather than only the library functions it wraps, since
    its own `except` tuple is exactly the line that let a raw traceback
    through in the round-3 review."""

    @staticmethod
    def _write_payload(tmp_path: Path, mutate=None) -> Path:
        payload = _payload()
        if mutate is not None:
            mutate(payload)
        path = tmp_path / "replay.json"
        path.write_text(json.dumps(payload))
        return path

    def test_success_exits_zero_and_writes_a_document(self, tmp_path: Path) -> None:
        from scripts.analyze_replay import main

        replay_path = self._write_payload(tmp_path)
        out_path = tmp_path / "out.json"
        exit_code = main(
            [
                "--replay",
                str(replay_path),
                "--perspective",
                "p1",
                "--out",
                str(out_path),
                "--search-time-ms",
                "100",
                "--opponent-samples",
                "2",
                "--threads",
                "2",
                "--quiet",
            ]
        )
        assert exit_code == 0
        assert out_path.exists()
        assert json.loads(out_path.read_text())["schemaVersion"] == 1

    def test_invalid_threads_exits_one_and_writes_nothing(self, tmp_path: Path) -> None:
        from scripts.analyze_replay import main

        replay_path = self._write_payload(tmp_path)
        out_path = tmp_path / "out.json"
        exit_code = main(
            [
                "--replay",
                str(replay_path),
                "--perspective",
                "p1",
                "--out",
                str(out_path),
                "--threads",
                "-1",
                "--quiet",
            ]
        )
        assert exit_code == 1
        assert not out_path.exists()

    def test_malformed_players_exits_one_and_writes_nothing(self, tmp_path: Path) -> None:
        from scripts.analyze_replay import main

        replay_path = self._write_payload(tmp_path, mutate=lambda p: p.__setitem__("players", [1, 2]))
        out_path = tmp_path / "out.json"
        exit_code = main(
            [
                "--replay",
                str(replay_path),
                "--perspective",
                "p1",
                "--out",
                str(out_path),
                "--search-time-ms",
                "100",
                "--opponent-samples",
                "2",
                "--threads",
                "2",
                "--quiet",
            ]
        )
        assert exit_code == 1
        assert not out_path.exists()

    def test_malformed_log_content_exits_one_without_raising_out_of_main(self, tmp_path: Path) -> None:
        # The exact regression this test guards: main()'s `except` tuple
        # must cover everything analyze_replay can now raise for malformed
        # log content, or this call raises out of main() and pytest reports
        # an error here rather than a clean assertion.
        from scripts.analyze_replay import main

        def _corrupt(payload: dict) -> None:
            target = "|switch|p1a: Meowscarada|Meowscarada, F|100/100"
            assert target in payload["log"]
            payload["log"] = payload["log"].replace(
                target, "|switch|p1a: Meowscarada|NotARealPokemon, L50, M|100/100", 1
            )

        replay_path = self._write_payload(tmp_path, mutate=_corrupt)
        out_path = tmp_path / "out.json"
        exit_code = main(
            [
                "--replay",
                str(replay_path),
                "--perspective",
                "p1",
                "--out",
                str(out_path),
                "--search-time-ms",
                "100",
                "--opponent-samples",
                "2",
                "--threads",
                "2",
                "--quiet",
            ]
        )
        assert exit_code == 1
        assert not out_path.exists()

    def test_deeply_nested_json_exits_one_without_raising_out_of_main(self, tmp_path: Path) -> None:
        # RecursionError, not OSError/JSONDecodeError - json.loads' own
        # exception for pathologically deep input, and the one the read
        # path's except tuple previously did not cover.
        from scripts.analyze_replay import main

        replay_path = tmp_path / "deep.json"
        replay_path.write_text("[" * 200_000)
        out_path = tmp_path / "out.json"
        exit_code = main(["--replay", str(replay_path), "--perspective", "p1", "--out", str(out_path), "--quiet"])
        assert exit_code == 1
        assert not out_path.exists()

    def test_unwritable_output_path_exits_one_without_raising_out_of_main(self, tmp_path: Path) -> None:
        # The read path was already wrapped in try/except OSError; the write
        # path was not, despite being just as capable of hitting a
        # permissions or disk error.
        from scripts.analyze_replay import main

        replay_path = self._write_payload(tmp_path)
        exit_code = main(
            [
                "--replay",
                str(replay_path),
                "--perspective",
                "p1",
                "--out",
                "/System/nope/out.json",
                "--search-time-ms",
                "100",
                "--opponent-samples",
                "2",
                "--threads",
                "2",
                "--quiet",
            ]
        )
        assert exit_code == 1

    def test_calling_main_does_not_leak_global_logging_state(self, tmp_path: Path) -> None:
        # logging.disable(logging.WARNING) previously lived inside main()
        # itself - a permanent, process-global mutation with no restore.
        # Since main() is now a function this same test module imports and
        # calls directly (this class exists for exactly that reason), that
        # mutation would silence WARNING for every test ordered after the
        # first one that called main(), including in a completely unrelated
        # module sharing this pytest process.
        from scripts.analyze_replay import main

        logging.disable(logging.NOTSET)  # start from a known-clean state
        replay_path = self._write_payload(tmp_path)
        out_path = tmp_path / "out.json"
        main(
            [
                "--replay",
                str(replay_path),
                "--perspective",
                "p1",
                "--out",
                str(out_path),
                "--search-time-ms",
                "100",
                "--opponent-samples",
                "2",
                "--threads",
                "2",
                "--quiet",
            ]
        )
        assert logging.getLogger().isEnabledFor(logging.WARNING)
