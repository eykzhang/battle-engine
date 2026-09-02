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
would hide which reason a turn couldn't be scored.

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

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from battle_engine.set_search import aggregate
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


class ReplayAnalysisError(ValueError):
    """The replay payload or request could not be analyzed.

    Distinct from `ReplayParseError` (a malformed payload) so callers can
    catch both under one type without also catching engine-internal errors.
    """


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
    """What one turn's sampled-and-searched opponents produced."""

    ranked: Tuple[Tuple[str, int, float], ...]
    win_probability: Optional[float]
    sample_failures: int


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
    sample_failures = 0
    for index in range(n_samples):
        filler = UsageStatsFiller(stats=stats, rng=random.Random(_sample_seed(seed, turn, index)))
        try:
            state = state_from_poke_env(battle, filler=filler).state
        except BaseException as exc:  # noqa: BLE001 - see module docstring
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            sample_failures += 1
            continue
        try:
            results.append(
                poke_engine.monte_carlo_tree_search(state, duration_ms=per_sample_ms, iterations=0, threads=threads)
            )
        except BaseException as exc:  # noqa: BLE001 - see module docstring
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            sample_failures += 1
            continue

    if not results:
        return _TurnSearch(ranked=(), win_probability=None, sample_failures=sample_failures)

    ranked = aggregate(results)
    total_visits = sum(visits for _, visits, _ in ranked) or 1
    win_probability = sum(visits * score for _, visits, score in ranked) / total_visits
    return _TurnSearch(ranked=ranked, win_probability=win_probability, sample_failures=sample_failures)


def _grade_turn(
    transition: Optional[TurnTransition],
    perspective: str,
    battle: Any,
    search: _TurnSearch,
    stats: Any,
    baseline_seed: int,
) -> Tuple[Optional[str], Optional[float], Optional[float], bool]:
    """(playedAction, playedActionValue, costOfPlayed, gradable) for one turn.

    `playedAction` is set whenever the log's action translates, independent of
    `gradable` - see the module docstring for why the two are kept separate,
    and for why addressability is checked against its own deterministic
    baseline state rather than against one of `search`'s sampled states.
    """
    if transition is None:
        return None, None, None, False

    action = transition.action(perspective)
    try:
        engine_act: EngineAction = engine_action(action, transition)
    except UnscorableTurn:
        return None, None, None, False

    played_action = engine_act.text

    try:
        baseline = state_from_poke_env(
            battle, filler=UsageStatsFiller(stats=stats, rng=random.Random(baseline_seed))
        ).state
    except BaseException as exc:  # noqa: BLE001 - see module docstring
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return played_action, None, None, False

    if not _addressable(baseline, "side_one", engine_act):
        return played_action, None, None, False
    if not search.ranked:
        return played_action, None, None, False

    key = played_action.strip().lower()
    played_value = next((score for name, _, score in search.ranked if name == key), None)
    if played_value is None:
        return played_action, None, None, False

    best_value = search.ranked[0][2]
    return played_action, played_value, best_value - played_value, True


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
    assumed to be well-formed; a missing/malformed `log` raises
    `ReplayParseError` via `parse_replay_json`). Raises `ReplayAnalysisError`
    for an invalid `perspective` or a log that does not name that perspective's
    player. `replay_id` is a caller-supplied fallback (e.g. the source
    filename's stem) used only when the payload itself carries no `"id"`.

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

    parsed: ParsedReplay = parse_replay_json(payload)
    by_turn = {t.turn: t for t in parsed.transitions}
    stats = default_usage_stats(format_id="gen9ou", cutoff=usage_stats_cutoff)

    battle_id = payload.get("id") or replay_id or "unknown"
    username = _username_for_role(payload["log"], perspective)
    per_sample_ms = max(1, search_time_ms // n_opponent_samples)

    turns: List[Dict[str, Any]] = []
    for marker, turn, battle in ReplayDriver(payload["log"], battle_id, player_username=username):
        if marker != "turn":
            continue
        search = _search_turn(
            battle,
            stats=stats,
            n_samples=n_opponent_samples,
            threads=threads,
            per_sample_ms=per_sample_ms,
            turn=turn,
            seed=seed,
        )
        played_action, played_value, cost, gradable = _grade_turn(
            by_turn.get(turn), perspective, battle, search, stats, _sample_seed(seed, turn, -1)
        )
        turns.append(
            {
                "turn": turn,
                "winProbability": None if search.win_probability is None else round(search.win_probability, 4),
                "gradable": gradable,
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

    return {
        "schemaVersion": SCHEMA_VERSION,
        "replayId": battle_id,
        "format": parsed.format_id,
        "rating": rating if isinstance(rating, int) else None,
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
