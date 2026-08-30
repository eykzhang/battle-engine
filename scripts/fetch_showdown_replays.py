"""Fetch and cache real gen9ou replays from Pokemon Showdown's own replay API.

Phase 6 M2 needs battles where *both* players' chosen actions are recoverable per
turn. Metamon's parsed-replay archive (`scripts/fetch_replay_sample.py`) cannot
supply that: its states are POV-flattened, only the POV player's action is
recorded, and `opponent_prev_move` means "last move ever used" rather than "move
used this turn" (notes/gotcha-prev-move-means-last-ever-used-not-last-turn.md).
Showdown's raw `|`-delimited protocol log has both sides' moves, damage, Tera,
status and item procs, so this script goes to the source instead.

    .venv/bin/python scripts/fetch_showdown_replays.py --n 50 --min-rating 1400

Two endpoints, both verified against live responses on 2026-08-30 rather than
from documentation (there is none):

- `search.json?format=gen9ou[&before=UPLOADTIME]` returns a JSON *list* of 51
  entries, newest first, each with `id`, `format`, `players`, `rating`,
  `uploadtime`, `private`, `password`. `rating` is `null` for unrated games.
- `<id>.json` returns the replay itself, including `log` - the full protocol text.

Pagination uses `before=UPLOADTIME` rather than `page=N`. Both work (page was
tested out to 100), but `page` overlaps by one entry between consecutive pages
while `before` is strictly less-than, which makes it a clean cursor. The one cost
is that replays sharing the oldest `uploadtime` of a batch are skipped along with
it. A measured search page of 51 entries spanned 1,191 seconds of gen9ou uploads,
so two replays landing on the same second is uncommon and the loss is small;
re-fetching an overlapping page on every cursor step would cost more.

Runs are resumable and idempotent: anything already in `--out-dir` counts toward
`--n` and is never re-downloaded. Progress is printed as it goes, per
notes/gotcha-benchmark-runs-need-empirical-timing-and-progress-visibility.md -
a script that prints nothing for minutes is indistinguishable from a hung one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"

# (connect timeout, read timeout). requests defaults to unbounded, which turns a
# stalled socket into a hung overnight run.
_REQUEST_TIMEOUT = (10, 30)

# Politeness floor between requests. Showdown's replay API is a free public
# service with no published rate limit; ~2 requests/second is well under what a
# single browser session generates while clicking through replays.
_DEFAULT_DELAY_SECONDS = 0.5

# Retried transient failures. 429 and 5xx are the server asking us to back off;
# a connection error is usually a flaky link. A 404 is not retried - that replay
# simply does not exist.
_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--n", type=int, default=50, help="replays to collect")
    parser.add_argument(
        "--min-rating", type=int, default=1200,
        help="skip replays below this Elo; unrated games (rating null) are always "
             "skipped when this is above 0",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/replays_showdown"))
    parser.add_argument(
        "--delay", type=float, default=_DEFAULT_DELAY_SECONDS,
        help="seconds to wait between API requests",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--max-pages", type=int, default=200,
        help="safety stop on how far back through the search index to scan",
    )
    return parser.parse_args()


def is_usable(entry: dict, min_rating: int) -> bool:
    """Whether a `search.json` entry is worth downloading.

    Private and password-protected replays are excluded because `<id>.json` will
    not serve them without the password, and a rating filter is the project's
    standing proxy for game quality (same role `--min-elo` plays in
    `fetch_replay_sample.py`). `rating` is `null` on unrated ladder games and on
    tournament games, so a null rating fails any positive threshold rather than
    being silently treated as 0 or as passing.
    """
    if entry.get("private"):
        return False
    if entry.get("password"):
        return False
    rating = entry.get("rating")
    if min_rating > 0 and (rating is None or rating < min_rating):
        return False
    return True


def _get_json(url: str, params: dict | None, max_retries: int, delay: float):
    """GET with backoff on the transient failures listed in `_RETRY_STATUS_CODES`.

    Returns the decoded JSON, or None if every attempt failed. Callers treat None
    as "skip this one and keep going" rather than aborting the run - a single bad
    replay should not throw away the ones already on disk.
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            if response.status_code in _RETRY_STATUS_CODES:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt == max_retries:
                print(f"  giving up on {url} after {max_retries + 1} attempts: {exc!r}",
                      file=sys.stderr)
                return None
            # Exponential backoff on top of the politeness delay: 1x, 2x, 4x.
            time.sleep(delay * (2 ** attempt))
    return None


def fetch_replays(
    fmt: str, n: int, min_rating: int, out_dir: Path,
    delay: float = _DEFAULT_DELAY_SECONDS,
    max_retries: int = 3,
    max_pages: int = 200,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-up semantics, matching fetch_replay_sample.py: an existing file counts
    # toward --n, so re-running with the same --n is a no-op rather than fetching
    # n more on top of what is already there.
    on_disk = {path.stem for path in out_dir.glob("*.json")}
    collected = len(on_disk)
    if collected >= n:
        print(f"Already have {collected} >= {n} replays in {out_dir}, nothing to fetch.")
        return collected

    started = time.monotonic()
    before: int | None = None
    scanned = 0
    skipped_filtered = 0
    ratings: list[int] = []

    for page in range(max_pages):
        params = {"format": fmt}
        if before is not None:
            params["before"] = before
        entries = _get_json(SEARCH_URL, params, max_retries, delay)
        time.sleep(delay)
        if not entries:
            print(f"Search returned nothing on page {page + 1}; stopping.", file=sys.stderr)
            break

        oldest = min(entry["uploadtime"] for entry in entries)
        if before is not None and oldest >= before:
            # No forward progress: the cursor would repeat this page forever.
            print("Search cursor stopped advancing; stopping.", file=sys.stderr)
            break
        before = oldest

        for entry in entries:
            scanned += 1
            if not is_usable(entry, min_rating):
                skipped_filtered += 1
                continue
            replay_id = entry["id"]
            if replay_id in on_disk:
                continue

            payload = _get_json(REPLAY_URL.format(replay_id=replay_id), None,
                                max_retries, delay)
            time.sleep(delay)
            if payload is None or not payload.get("log"):
                continue

            (out_dir / f"{replay_id}.json").write_text(json.dumps(payload))
            on_disk.add(replay_id)
            collected += 1
            ratings.append(entry["rating"])
            if collected % 10 == 0 or collected == n:
                elapsed = time.monotonic() - started
                rate = collected / elapsed if elapsed else 0.0
                print(f"  {collected}/{n} replays ({scanned} scanned, "
                      f"{elapsed:.0f}s elapsed, {rate:.2f}/s)")
            if collected >= n:
                break
        if collected >= n:
            break

    rating_range = f"{min(ratings)}-{max(ratings)}" if ratings else "n/a"
    print(f"Done: {collected} replays in {out_dir} (scanned {scanned} search results, "
          f"{skipped_filtered} filtered out, ratings this run {rating_range}, "
          f"{time.monotonic() - started:.0f}s)")
    return collected


def main() -> None:
    args = parse_args()
    fetch_replays(
        args.format, args.n, args.min_rating, args.out_dir,
        args.delay, args.max_retries, args.max_pages,
    )


if __name__ == "__main__":
    main()
