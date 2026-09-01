"""Tests for scripts/ladder_setsearch.py's pause/resume/concurrency control
flow (_play_with_pause_support, _wait_for_pause_file) - the fix that replaced
a sequential player.ladder(1)-per-game loop (which defeated
max_concurrent_battles entirely, the same bug battle_engine/benchmark.py's
run_benchmark had before its own fix) with one bulk player.ladder(remaining)
call per pause/resume segment.

Uses a fake player rather than a real poke_env Player - these tests exercise
this script's own control flow (bulk-vs-sequential calling, the pause/resume
state machine, callback restoration), not poke-env's internals. The
cross-event-loop bridge (handle_threaded_coroutines(player._battle_count_queue
.join(), player.ps_client.loop)) reuses the exact mechanism
battle_engine/benchmark.py's run_benchmark already proved correct against
real poke-env source and a real local-server run (see that file's own tests
and notes/benchmark-concurrency-build.md) - not re-derived here, since the
fake's ps_client.loop is the same loop the test itself runs on and can't
distinguish a same-loop call from a cross-loop one.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.ladder_setsearch import _play_with_pause_support, _wait_for_pause_file  # noqa: E402


class _FakeLadderPlayer:
    """Mimics exactly the poke_env.player.Player surface
    _play_with_pause_support touches: _battle_finished_callback,
    ladder(n) (async), _battle_count_queue (bounded, put/get/task_done/join),
    ps_client.loop. ladder() models poke-env's own real behavior (verified
    against poke_env/player/player.py's _ladder): each game's queue slot is
    claimed via put() (blocks if the bound is already full - real bounded
    concurrency), and _battle_finished_callback fires only once that game's
    slot is released via task_done()."""

    def __init__(self, max_concurrent: int = 2, game_duration: float = 0.02) -> None:
        self._battle_finished_callback = lambda battle: None
        self._battle_count_queue: asyncio.Queue = asyncio.Queue(maxsize=max_concurrent)
        self.ps_client = SimpleNamespace(loop=asyncio.get_event_loop())
        self._game_duration = game_duration
        self.ladder_call_sizes: list[int] = []

    async def ladder(self, n_games: int) -> None:
        self.ladder_call_sizes.append(n_games)
        for _ in range(n_games):
            await self._battle_count_queue.put(None)
            asyncio.create_task(self._play_one_game())
        await self._battle_count_queue.join()

    async def _play_one_game(self) -> None:
        await asyncio.sleep(self._game_duration)
        await self._battle_count_queue.get()
        self._battle_count_queue.task_done()
        self._battle_finished_callback(object())


class _RaisingLadderPlayer(_FakeLadderPlayer):
    async def ladder(self, n_games: int) -> None:
        self.ladder_call_sizes.append(n_games)
        raise RuntimeError("simulated ladder() failure")


def test_no_pause_file_plays_all_games_via_one_bulk_ladder_call():
    async def run():
        player = _FakeLadderPlayer(max_concurrent=2, game_duration=0.01)
        completed = []
        player._battle_finished_callback = lambda battle: completed.append(battle)

        await _play_with_pause_support(player, n_games=5, pause_file=None, pause_poll_seconds=0.01)

        assert player.ladder_call_sizes == [5]
        assert len(completed) == 5
        # Restored to the caller's own callback (identity, not just non-None),
        # confirmed by calling it once more and observing the append happen.
        completed.clear()
        player._battle_finished_callback(object())
        assert len(completed) == 1

    asyncio.run(run())


def test_callback_restored_after_normal_completion():
    async def run():
        player = _FakeLadderPlayer(max_concurrent=1, game_duration=0.01)
        sentinel_calls = []

        def original(battle):
            sentinel_calls.append(battle)

        player._battle_finished_callback = original
        await _play_with_pause_support(player, n_games=3, pause_file=None, pause_poll_seconds=0.01)

        # The original (pre-wrap) callback is exactly what's restored, and it
        # was called through by the wrapper for every game (not bypassed).
        assert player._battle_finished_callback is original
        assert len(sentinel_calls) == 3

    asyncio.run(run())


def test_callback_restored_even_when_ladder_raises():
    async def run():
        player = _RaisingLadderPlayer(max_concurrent=1)
        original = player._battle_finished_callback

        raised = False
        try:
            await _play_with_pause_support(player, n_games=3, pause_file=None, pause_poll_seconds=0.01)
        except RuntimeError:
            raised = True

        assert raised, "the real ladder() exception must propagate, not be swallowed"
        assert player._battle_finished_callback is original

    asyncio.run(run())


def test_pause_file_stops_launching_new_games_but_drains_in_flight(tmp_path):
    async def run():
        pause_file = tmp_path / "PAUSE"
        player = _FakeLadderPlayer(max_concurrent=2, game_duration=0.05)
        completed = []
        player._battle_finished_callback = lambda battle: completed.append(battle)

        async def touch_pause_after_first_game():
            # Let the bulk ladder() call get through roughly its first
            # in-flight batch, then request a pause - the real-world shape
            # (pause requested mid-run, not before it starts).
            await asyncio.sleep(0.06)
            pause_file.touch()

        async def remove_pause_after_a_bit():
            await asyncio.sleep(0.2)
            pause_file.unlink()

        toucher = asyncio.create_task(touch_pause_after_first_game())
        remover = asyncio.create_task(remove_pause_after_a_bit())

        await _play_with_pause_support(player, n_games=6, pause_file=pause_file, pause_poll_seconds=0.02)
        await toucher
        await remover

        # Paused mid-run means more than one ladder() call: an interrupted
        # bulk call for the first segment, and at least one more for the
        # remainder after resuming - never a single ladder(6) covering the
        # whole run uninterrupted.
        assert len(player.ladder_call_sizes) >= 2
        assert sum(player.ladder_call_sizes) >= 6, (
            "each resumed segment must request exactly the remaining games, "
            "summing to at least the original n_games (a segment can request "
            "more than what's strictly left to complete if games finished "
            "between the check and the next launch, but never fewer overall)"
        )
        # All 6 games actually completed once resumed - pausing must not
        # lose track of the target count.
        assert len(completed) == 6

    asyncio.run(run())


def test_run_completing_exactly_when_pause_file_exists_does_not_hang(tmp_path):
    """Regression test for a confirmed hang: if the pause file is still
    present the instant the final segment's ladder(remaining) call finishes
    ALL requested games on its own (never cancelled), there is nothing left
    to pause before - the function must return, not enter the paused-wait
    loop and block forever on a file that was never going to be removed by
    anything in this scenario.

    Timing is deliberately structured so the file appears well AFTER
    ladder() has already started (not present at the initial race against
    _wait_for_pause_file) but BEFORE the natural-completion trailing check -
    pause_poll_seconds is set long enough that the pause-file watcher never
    gets a chance to win the initial race and trigger a cancel, isolating
    the specific branch this regression targets: natural completion with
    the file coincidentally already sitting there.
    """

    async def run():
        pause_file = tmp_path / "PAUSE"
        player = _FakeLadderPlayer(max_concurrent=2, game_duration=0.03)

        async def touch_pause_after_games_are_well_underway():
            # All 3 games (2 concurrent, 0.03s each) naturally finish around
            # ~0.06s. Touching at 0.04s lands after they've started but
            # before the run's trailing pause-check runs - it lands, but too
            # late to cause a cancellation (pause_poll_seconds below is
            # longer than the whole run).
            await asyncio.sleep(0.04)
            pause_file.touch()

        toucher = asyncio.create_task(touch_pause_after_games_are_well_underway())

        await asyncio.wait_for(
            _play_with_pause_support(player, n_games=3, pause_file=pause_file, pause_poll_seconds=0.5),
            timeout=2.0,
        )
        await toucher

        # Reached the timeout's inner await without ever cancelling - one
        # uncancelled ladder(3) call, not an interrupted/resumed sequence.
        assert player.ladder_call_sizes == [3]
        assert pause_file.exists()

    asyncio.run(run())


def test_wait_for_pause_file_completes_once_file_exists(tmp_path):
    async def run():
        pause_file = tmp_path / "PAUSE"
        assert not pause_file.exists()

        async def touch_it():
            await asyncio.sleep(0.03)
            pause_file.touch()

        toucher = asyncio.create_task(touch_it())
        await asyncio.wait_for(_wait_for_pause_file(pause_file, poll_seconds=0.01), timeout=1.0)
        await toucher
        assert pause_file.exists()

    asyncio.run(run())
