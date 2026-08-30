"""Tests for scripts/fetch_showdown_replays.py (Phase 6 M2).

Only the parts that do not touch the network are unit-tested here: the entry
filter and the resumable top-up. Those are where a silent bug costs real
bandwidth or, worse, a corpus quietly filtered down to nothing - the download
loop itself is exercised by running the script, and its live behavior is
recorded in the M2 build notes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_showdown_replays import fetch_replays, is_usable  # noqa: E402


def _entry(**overrides) -> dict:
    # Shape verified against a live search.json response on 2026-08-30, not
    # assumed: every key here is present on every real entry.
    base = {
        "uploadtime": 1788114045,
        "id": "gen9ou-2672946156",
        "format": "[Gen 9] OU",
        "players": ["alice", "bob"],
        "rating": 1472,
        "private": 0,
        "password": None,
    }
    base.update(overrides)
    return base


def test_DW_M2_1_private_and_password_protected_replays_are_skipped():
    # `<id>.json` will not serve either one without the password, so downloading
    # them burns requests and writes nothing usable.
    assert is_usable(_entry(), min_rating=1200) is True
    assert is_usable(_entry(private=1), min_rating=1200) is False
    assert is_usable(_entry(password="hunter2"), min_rating=1200) is False


def test_DW_M2_1_an_unrated_game_fails_a_positive_rating_threshold():
    # `rating` is null on unrated ladder games and on tournament games. Treating
    # null as 0 would be harmless; treating it as passing would quietly fill the
    # corpus with unrated games while the run reports a rating filter. Neither is
    # left to chance.
    assert is_usable(_entry(rating=None), min_rating=1200) is False
    assert is_usable(_entry(rating=1100), min_rating=1200) is False
    assert is_usable(_entry(rating=1200), min_rating=1200) is True
    # A zero threshold means "no rating filter", so unrated games are allowed.
    assert is_usable(_entry(rating=None), min_rating=0) is True


def test_DW_M2_1_an_already_full_directory_is_a_no_op_with_no_requests(tmp_path):
    # The bug this guards against is the one review already found in
    # fetch_replay_sample.py: counting only new files toward --n, so a re-run
    # fetches n MORE on top of what is there. This asserts the top-up reading -
    # and it needs no network mock, because a correct implementation never opens
    # a socket when the target is already met.
    for i in range(3):
        (tmp_path / f"gen9ou-{i}.json").write_text("{}")
    assert fetch_replays("gen9ou", n=3, min_rating=1200, out_dir=tmp_path) == 3
    assert len(list(tmp_path.glob("*.json"))) == 3
