"""Phase 6 / M7 (BattleBrain): turn a real gen9ou replay into a per-turn analysis
document the app can ship as a fixture.

This is deliberately an assembly, not new engine work: every non-trivial piece
- driving a real `Battle` off a protocol log, translating it into a poke-engine
`State`, sampling the opponent's team from usage statistics, and searching -
already exists (`fidelity.py`, `poke_engine_state.py`, `set_prediction.py`,
`set_search.py`). What is new here is turning per-turn search results into a
win-probability curve plus a graded "how good was the move actually played"
verdict, at a scope a batch script (not a live `Player`) can use.

**Perspective is explicit and controls which side is `side_one`.**
`fidelity.ReplayDriver` builds its `Battle` from whichever `player_username` it
is given, defaulting to p1's own username read out of the log; that username is
what decides which side poke-env (and therefore
`poke_engine_state.state_from_poke_env`) calls "ours" (`side_one`). To analyze
from p2's perspective, the driver has to be built with p2's username instead -
`_username_for_role` below re-derives either username directly from the log's
own `|player|p1|...`/`|player|p2|...` lines (the same source `ReplayDriver`
already trusts for p1), rather than trusting `payload["players"]`'s ordering,
which is unvalidated external input.

**Every `poke_engine` call is wrapped per sample, not per turn.** A pyo3
`PanicException` (a real, measured ~0.2% failure mode of the threaded search)
derives from `BaseException`, not `Exception`, and crosses an `except
Exception` silently - see `set_search.py`'s `_search_each` for the same
pattern this module copies. Scoped per sample, one panic costs one of
`n_opponent_samples` searches; scoped per turn, it would cost the whole turn.
If every sample for a turn fails, the turn is still emitted with
`winProbability: null` - dropping it would look identical to "this turn was
never reached," which is a different and more serious kind of gap.

**Gradability is a real judgment call, recorded in `gradable`.** A turn's
`playedAction` is populated whenever the log's action translates at all
(independent of whether it could be scored); `gradable` is only true when that
action is *also* addressable against the state and its value was actually
found among the turn's aggregated actions, because both `playedActionValue`
and `costOfPlayed` need that lookup to succeed. Collapsing these into one flag
would hide which reason a turn couldn't be scored - which is exactly why an
ungradable turn also carries `ungradableReason`, one of the small closed set
of `REASON_*` constants near `_grade_turn` (plus `fidelity.UnscorableTurn`'s
own reasons, surfaced verbatim rather than re-labeled). A gradable turn's
`ungradableReason` is always `None`.

Addressability is checked against a *separate*, deterministic translation of
`battle` under its own fixed-seed `UsageStatsFiller` - not against one of the
`n_opponent_samples` search states. This matters and is not incidental: a
replay-driven `Battle` carries no `|request|` message, so poke-env has no more
knowledge of the *analyzed side's own* unrevealed moves than it does of the
opponent's - `state_from_poke_env` fills both sides' unrevealed slots from
whatever filler it is given. A `UsageStatsFiller`-sampled search state
therefore guesses at our own unrevealed moves too, and two different samples
can genuinely disagree about whether a given move is even on our active
Pokemon (confirmed directly: turn 1 of a real corpus replay showed sample 0
with `[flowertrick, uturn, tripleaxel, copycat]` and sample 1 with
`[tripleaxel, knockoff, toxicspikes, aurasphere]` for the *same* Pokemon).
Checking against one arbitrary sample would make `gradable` depend on search
RNG. A fixed seed (`_sample_seed(seed, turn, -1)`, outside the range the search
samples use) removes that dependence while keeping the prior.

The prior is kept deliberately, and a plain `RevealedOnlyFiller` was measured
and rejected for this: over 286 translatable played-actions from 12 corpus
replays, revealed-only addressed 49.3% of them and a deterministic
`UsageStatsFiller` addressed 69.6%. That is the same effect M4 measured on
representability (30.4% -> 66.3%), and `fidelity._prepare` asks the question of
the prior for exactly this reason - `RevealedOnlyFiller` is only its default
when no prior is supplied, and is the *baseline* M4 improves on rather than the
condition it ships. Grading on revealed-only would discard roughly a third of
the turns the engine can actually score.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import poke_engine

from battle_engine.fidelity import (
    EngineAction,
    ReplayDriver,
    UnscorableTurn,
    _addressable,
    engine_action,
)
from battle_engine.poke_engine_state import state_from_poke_env
from battle_engine.replay_log import ParsedReplay, ReplayParseError, TurnTransition, parse_replay_json
from battle_engine.set_prediction import UsageStatsFiller
from battle_engine.set_search import SWITCH_PREFIX, aggregate
from battle_engine.usage_stats import default_usage_stats

SCHEMA_VERSION = 1
PERSPECTIVES = ("p1", "p2")

# poke-engine (the compiled extension) exposes no __version__/tag attribute
# (checked directly: dir(poke_engine) carries neither) - this is asserted from
# battle-engine/CLAUDE.md's documented pinned checkout, not introspected:
#   git clone --depth 1 --branch v0.0.48 https://github.com/pmariglia/poke-engine.git
POKE_ENGINE_TAG = "v0.0.48"

DEFAULT_SEARCH_TIME_MS = 1000
DEFAULT_OPPONENT_SAMPLES = 8
DEFAULT_THREADS = 4
DEFAULT_USAGE_STATS_CUTOFF = 1500

# No real Showdown battle runs anywhere near this many turns - a bound well
# inside every integer type this document will ever cross (Swift's `Int`
# included), so a value this module accepts can never be "not representable"
# downstream. Also rejects the obvious malformed cases (negative, zero).
MAX_TURN_NUMBER = 100_000

# Real Showdown replay ids (and every caller-supplied `replay_id` fallback,
# which is always a filename stem) are exactly this shape. `replayId` in the
# emitted document is inert today, but Phase 3 keys the bundled fixture pair
# by it - constraining the charset now closes that off before it becomes a
# path-traversal sink.
_SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

# Same reasoning as `MAX_TURN_NUMBER`: a bound that targets actual downstream
# representability (Swift's `Int` maxes out around 9.2e18), not "what a
# real Showdown ladder rating looks like" (which never comes close). Without
# this, a payload's `rating` - well-formed as a Python `int`, so
# `_validate_payload_fields` has nothing to reject - can overflow Phase 3's
# decoder the same way an unbounded turn number would have.
MAX_RATING_MAGNITUDE = 10**15


class ReplayAnalysisError(ValueError):
    """The replay payload or request could not be analyzed.

    Distinct from `ReplayParseError` (a malformed payload) so callers can
    catch both under one type without also catching engine-internal errors.
    """


def _validate_payload_fields(payload: Dict[str, Any]) -> None:
    """Reject the third-party fields that have no safe normalized fallback,
    before they reach `parse_replay_json` or the emitted document.

    `rating` is deliberately not checked here - the schema declares it
    `int|null`, so a malformed rating is normalized to `None` in
    `analyze_replay` rather than rejected (an already-anchored behavior; see
    its own test). `players`, `id`, and `formatid` are typed `[str,str]`/`str`
    with no null variant, so a malformed value is rejected outright rather
    than silently substituted - "emitted non-verbatim but still wrong" would
    be worse than failing loudly. A falsy `players` (missing, `None`, `[]`,
    `""`) is left alone: `parse_replay_json` already falls back to the log's
    own `|player|` lines for that case, which is more trustworthy than the
    payload's ordering to begin with (see `_username_for_role`).
    """
    players = payload.get("players")
    if players and not (
        isinstance(players, (list, tuple))
        and len(players) == 2
        and all(isinstance(p, str) and p for p in players)
    ):
        raise ReplayAnalysisError(f"players must be a 2-element array of non-empty strings, got {players!r}")

    replay_id = payload.get("id")
    if replay_id is not None and not isinstance(replay_id, str):
        raise ReplayAnalysisError(f"id must be a string, got {replay_id!r}")

    format_id = payload.get("formatid")
    if format_id is not None and not isinstance(format_id, str):
        raise ReplayAnalysisError(f"formatid must be a string, got {format_id!r}")


def _username_for_role(log_text: str, role: str) -> str:
    """The username on `role`'s (`"p1"`/`"p2"`) first `|player|` line.

    Read from the log itself, not `payload["players"]`: the payload is
    external input and its `players` array's ordering is not guaranteed to
    line up with p1/p2 the way the log's own `|player|p1|...`/`|player|p2|...`
    lines are. Mirrors `fidelity.ReplayDriver._p1_username`'s source of truth,
    generalized to either role, but raises instead of falling back to a
    placeholder: a wrong perspective username silently drives the `Battle`
    from the wrong side with no error at all, which is worse than failing loudly.
    """
    for line in log_text.split("\n"):
        fields = line.split("|")
        if len(fields) > 3 and fields[1] == "player" and fields[2] == role and fields[3]:
            return fields[3]
    raise ReplayAnalysisError(f"no |player|{role}|...| line found in the replay log")


def _sample_seed(seed: int, turn: int, sample_index: int) -> int:
    """Deterministic, decorrelated per (turn, sample) - same convention
    `SetSearchPlayer._sampled_states` uses (a base draw plus the sample index),
    so re-running fixture generation with the same `seed` reproduces the same
    document."""
    return seed + turn * 1000 + sample_index


@dataclass
class _TurnSearch:
    """What one turn's sampled-and-searched opponents produced.

    `samples_used` is what makes a turn backed by 1 surviving sample
    distinguishable from one backed by 8 in the emitted document
    (`turns[].samplesUsed`)."""

    ranked: Tuple[Tuple[str, int, float], ...]
    win_probability: Optional[float]
    samples_used: int


def _search_turn(
    battle: Any,
    *,
    stats: Any,
    n_samples: int,
    threads: int,
    per_sample_ms: int,
    turn: int,
    seed: int,
) -> _TurnSearch:
    """Sample `n_samples` opponent teams from `stats`, search each, and
    aggregate. Both translation and search are wrapped per sample with
    `BaseException` - see the module docstring for why `Exception` is not
    enough."""
    results: List[Any] = []
    for index in range(n_samples):
        filler = UsageStatsFiller(stats=stats, rng=random.Random(_sample_seed(seed, turn, index)))
        try:
            state = state_from_poke_env(battle, filler=filler).state
        except BaseException as exc:  # noqa: BLE001 - see module docstring
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            continue
        try:
            results.append(
                poke_engine.monte_carlo_tree_search(state, duration_ms=per_sample_ms, iterations=0, threads=threads)
            )
        except BaseException as exc:  # noqa: BLE001 - see module docstring
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            continue

    if not results:
        return _TurnSearch(ranked=(), win_probability=None, samples_used=len(results))

    ranked = aggregate(results)
    total_visits = sum(visits for _, visits, _ in ranked)
    # poke-engine's own documented failure mode is literally named
    # `NonFinite` (module docstring). A non-finite `total_score` on even one
    # option poisons the aggregate - `0 * inf` is `nan` in IEEE754, so a
    # zero-visit option with an infinite score corrupts the weighted mean
    # even though it never gets a chance to matter by visit share. Checked
    # per-entry (not just on the final mean) so the cause is caught before it
    # has a chance to cancel out or dilute into something that looks finite.
    # `json.dumps` defaults to `allow_nan=True`, so without this a `NaN`/
    # `Infinity` token would reach the file verbatim - literally invalid
    # JSON, which fails the *entire* document for every consumer, not just
    # this turn. Treated as "no data", the same as the zero-visit case below:
    # this keeps the document loadable rather than merely failing louder at
    # write time (see also `scripts/analyze_replay.py`'s `allow_nan=False`,
    # a backstop for whatever this per-entry check doesn't anticipate).
    if total_visits == 0 or not all(math.isfinite(score) for _, _, score in ranked):
        # DW-2.14: every sample completed without panicking, but the search
        # itself explored nothing (or returned unusable scores). The old
        # `or 1` denominator substitution turned the zero-visit case into
        # winProbability: 0.0 - a *confident loss* in this domain, not "no
        # data". Treat both cases the same (ranked forced empty too), so
        # _grade_turn's existing "not search.ranked" gate reports
        # REASON_NO_SEARCH_RESULTS here as well, with no new reason needed,
        # and coverage()'s eval-bar figure excludes it via the same null
        # winProbability every other no-data turn already produces.
        return _TurnSearch(ranked=(), win_probability=None, samples_used=len(results))
    win_probability = sum(visits * score for _, visits, score in ranked) / total_visits
    if not math.isfinite(win_probability):
        return _TurnSearch(ranked=(), win_probability=None, samples_used=len(results))
    return _TurnSearch(ranked=ranked, win_probability=win_probability, samples_used=len(results))


# Stable, closed-vocabulary reasons for `turns[].ungradableReason` - one per
# early return in `_grade_turn` below that is not itself an `UnscorableTurn`
# reason (those already have their own small closed vocabulary - see
# `fidelity.UnscorableTurn`'s docstring - and are surfaced verbatim rather
# than collapsed into one generic label, so a consumer gets the same
# specificity `engine_action` itself computed). Two reasons, not one, for
# "search produced nothing to compare against" versus "search ran but never
# considered this specific action": the first is the same root cause as a
# null `winProbability` (every sample panicked, or - rarer - a sample
# returned zero ranked actions); the second means the search itself
# succeeded and the action passed the baseline addressability check, but its
# canonical name was never among the aggregated results MCTS actually
# explored - a materially different, more surprising condition worth telling
# apart.
REASON_NO_TRANSITION = "no_transition"
REASON_BASELINE_TRANSLATION_FAILED = "baseline_translation_failed"
REASON_NOT_ADDRESSABLE = "not_addressable"
REASON_NO_SEARCH_RESULTS = "no_search_results"
REASON_ACTION_NOT_IN_SEARCH_RESULTS = "action_not_in_search_results"


def _grade_turn(
    transition: Optional[TurnTransition],
    perspective: str,
    battle: Any,
    search: _TurnSearch,
    stats: Any,
    baseline_seed: int,
) -> Tuple[Optional[str], Optional[float], Optional[float], bool, Optional[str]]:
    """(playedAction, playedActionValue, costOfPlayed, gradable, ungradableReason)
    for one turn.

    `playedAction` is set whenever the log's action translates, independent of
    `gradable` - see the module docstring for why the two are kept separate,
    and for why addressability is checked against its own deterministic
    baseline state rather than against one of `search`'s sampled states.

    `ungradableReason` is `None` exactly when `gradable` is `True`; otherwise
    it names which of the five early-return branches produced the `False` -
    see the `REASON_*` constants above.
    """
    if transition is None:
        return None, None, None, False, REASON_NO_TRANSITION

    action = transition.action(perspective)
    try:
        engine_act: EngineAction = engine_action(action, transition)
    except UnscorableTurn as exc:
        return None, None, None, False, exc.reason

    played_action = engine_act.text

    try:
        baseline = state_from_poke_env(
            battle, filler=UsageStatsFiller(stats=stats, rng=random.Random(baseline_seed))
        ).state
    except BaseException as exc:  # noqa: BLE001 - see module docstring
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return played_action, None, None, False, REASON_BASELINE_TRANSLATION_FAILED

    if not _addressable(baseline, "side_one", engine_act):
        return played_action, None, None, False, REASON_NOT_ADDRESSABLE
    if not search.ranked:
        return played_action, None, None, False, REASON_NO_SEARCH_RESULTS

    # `EngineAction.text` names a switch by bare species id (fidelity.py's own
    # addressing convention, which `_addressable` above depends on and this
    # function must not change). `set_search.aggregate` keys its ranked
    # actions the way poke-engine's own `move_choice` renders one -
    # `"switch <species>"` (`SWITCH_PREFIX`). The two meet only here; without
    # this translation every switch looks ungradable even when its value was
    # actually searched.
    key = played_action.strip().lower()
    if engine_act.kind == "switch":
        key = f"{SWITCH_PREFIX}{key}"
    played_value = next((score for name, _, score in search.ranked if name == key), None)
    if played_value is None:
        return played_action, None, None, False, REASON_ACTION_NOT_IN_SEARCH_RESULTS

    # The maximum *value* across ranked actions, not `ranked[0]` (which is
    # ordered by visit share - MCTS's revealed preference, not necessarily
    # the highest-scoring option). `played_value` is itself one of the values
    # `best_value` maxes over, so `costOfPlayed` can never be negative.
    best_value = max(score for _, _, score in search.ranked)
    return played_action, played_value, best_value - played_value, True, None


def analyze_replay(
    payload: Dict[str, Any],
    *,
    perspective: str,
    replay_id: Optional[str] = None,
    search_time_ms: int = DEFAULT_SEARCH_TIME_MS,
    n_opponent_samples: int = DEFAULT_OPPONENT_SAMPLES,
    threads: int = DEFAULT_THREADS,
    usage_stats_cutoff: int = DEFAULT_USAGE_STATS_CUTOFF,
    seed: int = 0,
) -> Dict[str, Any]:
    """A schema-v1 per-turn analysis document for `payload`, from `perspective`.

    `payload` is a Showdown replay-API JSON document (external input - never
    assumed to be well-formed, in *type* or in *content*). Raises
    `ReplayAnalysisError` for an invalid `perspective`, a non-positive search
    parameter, a malformed `players`/`id`/`formatid` field (see
    `_validate_payload_fields` for which fields are rejected outright versus
    normalized to `None`), an unsafe-charset replay id, an undeterminable
    `format`, fewer than two non-empty player usernames (payload- or
    log-derived), or malformed log *content* - an unknown species/type/move,
    a truncated protocol line, an out-of-range/negative/duplicated turn
    number, or a log that does not name `perspective`'s player. Raises
    `ReplayParseError` via `parse_replay_json` for a missing/non-string `log`.
    `replay_id` is a caller-supplied fallback (e.g. the source filename's
    stem) used only when the payload itself carries no `"id"`.

    A turn is always emitted once `ReplayDriver` reaches it, even when every
    sample fails (`winProbability: null`) or no replay transition matches it
    (ungradable) - see the module docstring for why dropping either would be
    worse than reporting them as such.
    """
    if perspective not in PERSPECTIVES:
        raise ReplayAnalysisError(f"perspective must be one of {PERSPECTIVES}, got {perspective!r}")
    if n_opponent_samples < 1:
        raise ReplayAnalysisError("n_opponent_samples must be at least 1")
    if search_time_ms < 1:
        raise ReplayAnalysisError("search_time_ms must be at least 1")
    if threads < 1:
        raise ReplayAnalysisError("threads must be at least 1")
    _validate_payload_fields(payload)

    try:
        parsed: ParsedReplay = parse_replay_json(payload)
    except ReplayParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - poke-env's parser raises whatever
        # type the malformed content happens to trip (KeyError, IndexError,
        # ValueError, ...), not just ValueError - see module docstring and
        # the ReplayDriver loop below for the same pattern. `except
        # Exception` (not `BaseException`) already leaves
        # `KeyboardInterrupt`/`SystemExit` propagating on their own - no
        # manual re-raise needed here, unlike the per-sample `BaseException`
        # catches around the real `poke_engine` panic. ReplayParseError is
        # excluded above so it still propagates as itself, matching this
        # module's documented contract and the anchored missing-log test.
        raise ReplayAnalysisError(f"malformed replay log: {exc}") from exc
    by_turn = {t.turn: t for t in parsed.transitions}
    stats = default_usage_stats(format_id="gen9ou", cutoff=usage_stats_cutoff)

    battle_id = payload.get("id") or replay_id or "unknown"
    if not _SAFE_ID_PATTERN.fullmatch(battle_id):
        raise ReplayAnalysisError(f"replay id contains unsafe characters: {battle_id!r}")

    if not parsed.format_id:
        raise ReplayAnalysisError("could not determine format: no 'formatid' in payload and no |tier| line in log")

    # `ReplayDriver.__iter__` splits the log on "\n" only and does not strip
    # a trailing "\r", unlike `parse_replay_json`'s tracker
    # (`replay_log.py`'s `line.rstrip("\r")` per line, already applied above).
    # A CRLF-terminated payload - ordinary third-party network output, not
    # malformed data - then desyncs poke-env's own paired line matching (e.g.
    # a "-sidestart"/"-sideend" pair no longer agrees) and raises deep inside
    # `Battle.parse_message`. Normalized once here, for every remaining use
    # of the log text, so CRLF input analyzes correctly instead of merely
    # failing more cleanly.
    log_text = payload["log"].replace("\r\n", "\n").replace("\r", "\n")

    # Validate *both* roles' `|player|` lines against the log itself, not
    # against `parsed.players`. `replay_log.py`'s own fallback
    # (`players or (tracker.p1.username, tracker.p2.username)`) means
    # `payload["players"]` wins whenever it is truthy - which it always is on
    # a real replay-API payload - so `parsed.players` reflects the *payload*
    # array in the common case, never the log. A malformed or missing
    # `|player|` line for the side *not* being analyzed was therefore
    # invisible: nothing inspected the log's own player lines for that side.
    # `_username_for_role` raises a typed error per role; calling it for both
    # closes the gap regardless of what the payload claims, and doubles as
    # the source of `username` below (no separate call needed).
    usernames = {role: _username_for_role(log_text, role) for role in PERSPECTIVES}
    username = usernames[perspective]
    per_sample_ms = max(1, search_time_ms // n_opponent_samples)

    turns: List[Dict[str, Any]] = []
    seen_turn_numbers = set()
    driver_iter = iter(ReplayDriver(log_text, battle_id, player_username=username))
    while True:
        try:
            marker, turn, battle = next(driver_iter)
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001 - poke-env's parser can raise
            # KeyError, IndexError, or ValueError from a malformed protocol
            # line (an unknown species/type/move, a truncated line, ...), not
            # just ValueError - the one shape the previous fix-forward
            # round's narrower catch handled. `except Exception` already
            # leaves `KeyboardInterrupt`/`SystemExit` propagating. Scoped to
            # just the `next()` call, not the rest of the loop body, so this
            # can't re-wrap a `ReplayAnalysisError` raised below.
            raise ReplayAnalysisError(f"malformed replay log: {exc}") from exc
        if marker != "turn":
            continue
        if not (1 <= turn <= MAX_TURN_NUMBER):
            raise ReplayAnalysisError(f"malformed replay log: turn number {turn} is out of range")
        if turn in seen_turn_numbers:
            raise ReplayAnalysisError(f"malformed replay log: duplicate turn number {turn}")
        seen_turn_numbers.add(turn)
        search = _search_turn(
            battle,
            stats=stats,
            n_samples=n_opponent_samples,
            threads=threads,
            per_sample_ms=per_sample_ms,
            turn=turn,
            seed=seed,
        )
        played_action, played_value, cost, gradable, ungradable_reason = _grade_turn(
            by_turn.get(turn), perspective, battle, search, stats, _sample_seed(seed, turn, -1)
        )
        turns.append(
            {
                "turn": turn,
                "winProbability": None if search.win_probability is None else round(search.win_probability, 4),
                "gradable": gradable,
                "ungradableReason": ungradable_reason,
                "samplesUsed": search.samples_used,
                "playedAction": played_action,
                "playedActionValue": None if played_value is None else round(played_value, 4),
                "costOfPlayed": None if cost is None else round(cost, 4),
                "topActions": [
                    {
                        "action": name,
                        "visitShare": round(visits / max(1, sum(v for _, v, _ in search.ranked)), 4),
                        "value": round(score, 4),
                    }
                    for name, visits, score in search.ranked
                ],
            }
        )

    rating = payload.get("rating")
    rating_is_usable = (
        isinstance(rating, int) and not isinstance(rating, bool) and abs(rating) <= MAX_RATING_MAGNITUDE
    )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "replayId": battle_id,
        "format": parsed.format_id,
        "rating": rating if rating_is_usable else None,
        "players": list(parsed.players),
        "perspective": perspective,
        "engine": {
            "searchBudgetMsPerTurn": search_time_ms,
            "opponentSamples": n_opponent_samples,
            "threads": threads,
            "usageStatsCutoff": usage_stats_cutoff,
            "pokeEngineTag": POKE_ENGINE_TAG,
        },
        "totalTurns": len(turns),
        "gradableTurns": sum(1 for t in turns if t["gradable"]),
        "turns": turns,
    }


def coverage(document: Dict[str, Any]) -> Tuple[float, float]:
    """(evalBarCoverage, gradingCoverage) as fractions in [0, 1], both 0.0 for
    a document with no turns."""
    turns = document["turns"]
    total = document["totalTurns"]
    if total == 0:
        return 0.0, 0.0
    eval_bar = sum(1 for t in turns if t["winProbability"] is not None) / total
    grading = document["gradableTurns"] / total
    return eval_bar, grading
