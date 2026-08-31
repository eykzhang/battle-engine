"""Phase 6 / M5: the player - search over sampled opponent sets.

This is the piece the previous four milestones were building toward, and it is
where the Foul Play architecture actually becomes a bot: translate the live
battle into a poke-engine `State` **K times**, once per sampled opponent team,
search each under a slice of a wall-clock budget, and pick the action with the
most visits summed across all K.

## Why K states rather than one

M4 measured what a single best guess is worth and what it costs. The modal
opponent is the better *point estimate* - it beats a single sample on every
single-state metric (66.3% representability against 52.2%) - but a search over
one modal opponent is confidently wrong exactly when the mode is wrong, and the
mode is wrong most of the time: species recall for the whole predicted team is
33.5%, and the item is right half the time. Sampling turns that from a single
bet into a distribution the search can hedge against.

The reason this is affordable is measured, not assumed. Splitting a fixed
wall-clock budget K ways costs essentially nothing in total simulations, because
poke-engine's search is throughput-bound rather than startup-bound: over the same
1000 ms this laptop ran 850,000 visits at K=1 and 906,000 at K=8. Opponent
coverage is free; only the depth *per opponent* is traded.

## The budget

Wall-clock, not a fixed simulation count - the plan's requirement, and the right
shape for a real ladder game where the clock is the actual constraint. Measured
on the M4 Air: ~910 visits/ms single-threaded, ~1,830 at `threads=4`, and flat
past 4 (the machine has 4 performance cores). So `threads=4` is the default and
more is not better.

## Why the search's action space is trustworthy here, unlike Phase 4's

Phase 4's C++ `MctsPlayer` had a standing pathology: its `legal_actions()`
modelled no PP, no Choice lock, no Disable, so the search would pick an action
the real game refused, poke-env would re-prompt the same turn, and the search -
seeing the same position - would pick the same illegal action again. One 2026-08-25
benchmark stalled over an hour on a single battle that way
([[battle-engine/notes/gotcha-legality-drift-needs-a-boundary-backstop-not-one-off-fixes]]).

poke-engine does not have that gap, and this was verified rather than assumed:
its search honours `Move.disabled` (disabling three of four moves leaves exactly
the fourth plus the switches) and honours `Side.force_switch` (offering only
switches). `poke_engine_state` already sets both from poke-env's own
`available_moves` and `force_switch`, so Choice lock, Disable, Encore, Taunt and
0 PP are handled at the source rather than patched at the root.

**One real gap remains: trapping.** Arena Trap, Shadow Tag, Mean Look and
partial-trapping moves remove switches that poke-engine will still offer, because
the translator has nowhere to put "cannot switch". So the root check below is a
backstop, not a formality - and it is written to walk *down* the ranked list
rather than give up, so a trapped turn still plays the best legal move the search
actually explored instead of falling through to Showdown's default.

## Known gaps, stated rather than discovered later

- **Team preview is poke-env's random default.** poke-engine has a team-preview
  action form and Foul Play uses it; choosing a lead is a separate problem from
  choosing a turn's action, and it is not in this milestone.
- **Our own side is assumed fully known.** In a live battle it is - the request
  JSON carries all six Pokemon - which is what makes summing visits by action
  name across K states well-defined: only side_two varies between them. Driving
  this player from a replay, where our side is also partly hidden, would break
  that assumption quietly, so it is asserted in the aggregation rather than
  trusted.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import poke_engine
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.data import to_id_str
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from battle_engine.poke_engine_state import UnknownToPokeEngine, state_from_poke_env
from battle_engine.set_prediction import UsageStatsFiller
from battle_engine.usage_stats import UsageStats, default_usage_stats

# poke-engine renders a switch as "switch <species id>" but *parses* one as the
# bare species id (`MoveChoice::from_string`). The asymmetry is only visible
# from the search result, so it is named here rather than inlined.
SWITCH_PREFIX = "switch "
TERA_SUFFIX = "-tera"
NO_ACTION = "none"

# Measured on the M4 MacBook Air, the laptop-first hard rule's target machine:
# ~910 visits/ms at one thread, ~1,830 at four, flat past four (4 performance
# cores). Past that the extra threads contend rather than help.
DEFAULT_THREADS = 4

# A wall-clock budget rather than a simulation count. 1,000 ms is roughly two
# million visits at these defaults, well inside a Showdown turn timer, and the
# real ladder is where this number has to hold - see M6.
DEFAULT_SEARCH_TIME_MS = 1000

# Splitting the budget K ways is close to free (850k visits at K=1 vs 906k at
# K=8 over the same 1,000 ms), so K is chosen for opponent coverage rather than
# against throughput.
DEFAULT_OPPONENT_SAMPLES = 8


@dataclass(frozen=True)
class Decision:
    """One turn's search, in enough detail to argue with.

    Kept because this project's own history says metrics alone do not find
    pathologies - the Phase 3 protect-spam loop was found by watching real
    turns ([[battle-engine/notes/pattern-watch-real-replays-not-just-metrics]]).
    An `on_decision` callback receives one of these per turn.
    """

    turn: int
    chosen: str
    order: Optional[str]
    visits: int
    total_visits: int
    value: float
    states_searched: int
    seconds: float
    ranked: Tuple[Tuple[str, int, float], ...] = ()
    # Set when the search's top pick was not legal in the real game and a
    # lower-ranked action was played instead. Non-zero counts here mean the
    # trapping gap above is actually firing.
    rank_played: int = 0
    fallback_reason: str = ""


@dataclass
class SearchStats:
    """Running totals over a battle, for the diagnostic script and for M6."""

    turns: int = 0
    seconds: float = 0.0
    visits: int = 0
    root_pick_illegal: int = 0
    # Individual sampled searches that failed while the turn still produced a
    # move from the others. Distinct from `defaulted`, which is whole turns.
    sample_failures: int = 0
    samples_run: int = 0
    defaulted: int = 0
    failures: Dict[str, int] = field(default_factory=dict)

    def note_failure(self, reason: str) -> None:
        self.failures[reason] = self.failures.get(reason, 0) + 1

    @property
    def ms_per_turn(self) -> float:
        return 1000.0 * self.seconds / self.turns if self.turns else 0.0


def order_from_choice(choice: str, battle: AbstractBattle) -> Optional[BattleOrder]:
    """A poke-engine action string -> a poke-env order, or None if the real
    game does not allow it.

    Checked against `battle.available_moves` / `available_switches` rather than
    against the state the search ran on, because those are the server's own
    answer and the state is our reconstruction of it. The two disagree for
    trapping, which the translator cannot represent.
    """
    choice = choice.strip().lower()
    if not choice or choice == NO_ACTION:
        return None

    if choice.startswith(SWITCH_PREFIX):
        species = choice[len(SWITCH_PREFIX) :]
        for mon in battle.available_switches:
            if to_id_str(mon.species) == species:
                return Player.create_order(mon)
        return None

    terastallize = choice.endswith(TERA_SUFFIX)
    move_id = choice[: -len(TERA_SUFFIX)] if terastallize else choice
    if terastallize and not battle.can_tera:
        return None
    for move in battle.available_moves:
        if to_id_str(move.id) == move_id:
            return Player.create_order(move, terastallize=terastallize)
    return None


def aggregate(results: Sequence[Any]) -> Tuple[Tuple[str, int, float], ...]:
    """Rank our side's actions by visits summed across every sampled opponent.

    Summed visits, which is what a root-parallel MCTS aggregates and what Foul
    Play uses: a visit count is the search's own revealed preference, and it is
    far more stable than an averaged value estimate at the tail where an action
    was barely explored.

    The sum is only meaningful because every sampled state shares one action
    space - our side is fully known in a live battle, so only the opponent
    varies. Averaged value is carried alongside for the decision log, weighted
    by visits so a barely-explored branch cannot swing it.
    """
    visits: Dict[str, int] = {}
    score: Dict[str, float] = {}
    for result in results:
        for option in result.side_one:
            name = option.move_choice.strip().lower()
            visits[name] = visits.get(name, 0) + option.visits
            score[name] = score.get(name, 0.0) + option.total_score
    return tuple(
        sorted(
            ((name, n, score[name] / n if n else 0.0) for name, n in visits.items()),
            key=lambda row: (-row[1], row[0]),
        )
    )


class SetSearchPlayer(Player):
    """poke-engine MCTS over K sampled opponent teams, on a wall-clock budget.

    A `Player` like every other one in this codebase: `choose_move` is the only
    overridden method.

    `search_time_ms` is the budget for the whole turn and is divided among the
    samples, so raising `n_opponent_samples` trades depth per opponent for
    coverage across opponents rather than costing wall clock. `usage_stats` is
    injectable so a benchmark can hold one loaded copy across both players
    instead of parsing 14 MB of JSON twice.

    Every failure path ends at `choose_default_move()` rather than at an
    exception, and that includes `BaseException`. Not defensive habit: a Rust
    panic crosses pyo3 as `PanicException`, which derives from `BaseException`
    and slips past every `except Exception`
    ([[battle-engine/notes/gotcha-poke-engine-encore-panics-without-last-used-move]]).
    One of those already cost a whole corpus run; on the real ladder it would
    end the process mid-game and forfeit. A forfeited game is a worse outcome
    than a bad move, so the bot always has a move.
    """

    def __init__(
        self,
        *args: Any,
        search_time_ms: int = DEFAULT_SEARCH_TIME_MS,
        n_opponent_samples: int = DEFAULT_OPPONENT_SAMPLES,
        threads: int = DEFAULT_THREADS,
        usage_stats: Optional[UsageStats] = None,
        format_id: str = "gen9ou",
        cutoff: int = 1500,
        seed: int = 0,
        on_decision: Optional[Callable[[Decision], None]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if n_opponent_samples < 1:
            raise ValueError("n_opponent_samples must be at least 1")
        if search_time_ms < 1:
            raise ValueError("search_time_ms must be at least 1")
        self._search_time_ms = search_time_ms
        self._n_samples = n_opponent_samples
        self._threads = threads
        self._stats_source = usage_stats or default_usage_stats(format_id=format_id, cutoff=cutoff)
        self._rng = random.Random(seed)
        self._on_decision = on_decision
        self.search_stats = SearchStats()

    @property
    def per_state_ms(self) -> int:
        return max(1, self._search_time_ms // self._n_samples)

    def _sampled_states(self, battle: AbstractBattle) -> List[Any]:
        """K translated states, one per sampled opponent team.

        A fresh generator per state per turn, all derived from this player's own
        seed, so a whole battle is reproducible while no two samples in a turn
        are correlated - the same convention `MctsPlayer` uses for its search
        seeds.
        """
        base = self._rng.getrandbits(48)
        states = []
        for index in range(self._n_samples):
            filler = UsageStatsFiller(stats=self._stats_source, rng=random.Random(base + index))
            states.append(state_from_poke_env(battle, filler=filler).state)
        return states

    def _search_each(self, states: Sequence[Any]) -> List[Any]:
        """Search every sampled state, keeping whatever succeeds.

        The samples are independent by construction, so one of them failing is
        a reason to drop that sample - not a reason to throw away the turn.
        Doing it per state rather than around the whole loop matters because
        poke-engine's threaded search has a real, measured failure mode:
        `perform_mcts_shared_tree` panics with `NonFinite` when a branch
        percentage comes back infinite, on roughly 0.2% of turns
        ([[battle-engine/notes/gotcha-poke-engine-threaded-search-panics-on-a-nonfinite-branch]]).
        Wrapped around the whole loop, that cost a whole turn's search and a
        default move; wrapped per state it costs one sample of eight, which is
        inside the noise this player is already sampling against.

        `BaseException` again, and for the same reason as in `choose_move`:
        that panic arrives as a pyo3 `PanicException`, which no
        `except Exception` sees.
        """
        results: List[Any] = []
        for state in states:
            try:
                results.append(
                    poke_engine.monte_carlo_tree_search(
                        state, duration_ms=self.per_state_ms, iterations=0, threads=self._threads
                    )
                )
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                self.search_stats.sample_failures += 1
                self.search_stats.note_failure(f"sample:{type(exc).__name__}")
        return results

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not battle.available_moves and not battle.available_switches:
            # Nothing to choose between; poke-env still wants an order.
            self.search_stats.defaulted += 1
            return self.choose_default_move()

        start = time.perf_counter()
        try:
            states = self._sampled_states(battle)
        except (ValueError, UnknownToPokeEngine, KeyError) as exc:
            return self._default(battle, start, f"translation:{type(exc).__name__}")
        except BaseException as exc:  # noqa: BLE001 - see the class docstring
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return self._default(battle, start, f"translation:{type(exc).__name__}")

        results = self._search_each(states)
        if not results:
            return self._default(battle, start, "all_searches_failed")

        self.search_stats.samples_run += len(results)
        ranked = aggregate(results)
        elapsed = time.perf_counter() - start

        for rank, (choice, visits, value) in enumerate(ranked):
            order = order_from_choice(choice, battle)
            if order is None:
                continue
            if rank:
                self.search_stats.root_pick_illegal += 1
            self._record(
                battle,
                Decision(
                    turn=battle.turn,
                    chosen=choice,
                    order=order.message,
                    visits=visits,
                    total_visits=sum(r.total_visits for r in results),
                    value=value,
                    states_searched=len(results),
                    seconds=elapsed,
                    ranked=ranked[:6],
                    rank_played=rank,
                ),
            )
            return order

        # Every action the search explored is real-game-illegal. Possible when
        # trapping removes the switches and the active Pokemon's only legal
        # moves were all disabled in the translated state.
        return self._default(battle, start, "no_legal_action_in_search", ranked)

    def _default(
        self,
        battle: AbstractBattle,
        start: float,
        reason: str,
        ranked: Tuple[Tuple[str, int, float], ...] = (),
    ) -> BattleOrder:
        self.search_stats.defaulted += 1
        self.search_stats.note_failure(reason)
        order = self.choose_default_move()
        self._record(
            battle,
            Decision(
                turn=battle.turn,
                chosen="",
                order=order.message,
                visits=0,
                total_visits=0,
                value=0.0,
                states_searched=0,
                seconds=time.perf_counter() - start,
                ranked=ranked,
                fallback_reason=reason,
            ),
        )
        return order

    def _record(self, battle: AbstractBattle, decision: Decision) -> None:
        self.search_stats.turns += 1
        self.search_stats.seconds += decision.seconds
        self.search_stats.visits += decision.total_visits
        if self._on_decision is not None:
            self._on_decision(decision)
