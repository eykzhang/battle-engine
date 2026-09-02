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
from pathlib import Path

import pytest

poke_engine = pytest.importorskip(
    "poke_engine", reason="poke-engine not built; run scripts/build_poke_engine.sh"
)

from battle_engine.replay_analysis import (  # noqa: E402
    ReplayAnalysisError,
    analyze_replay,
    coverage,
)
from battle_engine.replay_log import ReplayParseError  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "data" / "replays_showdown"
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
                "playedAction",
                "playedActionValue",
                "costOfPlayed",
                "topActions",
            }
            wp = turn["winProbability"]
            assert wp is None or (isinstance(wp, float) and 0.0 <= wp <= 1.0)
            assert isinstance(turn["gradable"], bool)
            if not turn["gradable"]:
                assert turn["playedActionValue"] is None
                assert turn["costOfPlayed"] is None
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
