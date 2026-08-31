"""Fetch and cache Smogon's usage statistics for a format, as Phase 6 M4's prior.

M3 measured why this milestone exists. Over 5,846 scored turns of real gen9ou,
only 30.4% had both players' actions addressable from revealed information
alone, and a battle eventually reveals just 40.5% of abilities and 26.8% of
items. In-battle inference therefore cannot be the primary source of set
knowledge - most of the information is never shown at all. Usage statistics are.

Smogon publishes two forms of the same month's data. The plain `.txt` tables are
human-readable and lossy; the **chaos** JSON under `stats/<month>/chaos/` is the
machine-readable one, and it carries per-species distributions over abilities,
items, moves, Tera types, spreads (nature + EVs) and teammates. The spreads are
the part M3 says matters most and the part no replay can ever supply.

    .venv/bin/python scripts/fetch_usage_stats.py
    .venv/bin/python scripts/fetch_usage_stats.py --format gen9ou --cutoff 1500

Verified against live responses on 2026-08-30 rather than from documentation
(there is none):

- `https://www.smogon.com/stats/` is an Apache directory index of `YYYY-MM/`
  months. The newest month is published a few days into the following month, so
  "this month" usually 404s - the default resolves the newest month that
  actually has the requested file rather than computing one from the clock.
- `https://www.smogon.com/stats/<month>/chaos/<format>-<cutoff>.json` is ~9.7 MB
  for gen9ou-1695. Available cutoffs for gen9ou are 0, 1500, 1695 and 1825.
- The payload is `{"info": {...}, "data": {"<Display Name>": {...}}}`. Species
  keys are display names ("Great Tusk", "Slowking-Galar"), not ids.

The cutoff is a rating floor on the games counted, and it is a real modelling
choice rather than "higher is better": it should match the population the bot
will actually play. 1500 is the default here because Phase 6's gate is the open
gen9ou ladder, where Phase 3 measured a 1305 Glicko. `battle_engine/usage_stats.py`
loads whatever cutoffs are on disk, so fetching several and comparing them in the
M4 evaluation is the cheap way to settle it with a number instead of an opinion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

STATS_INDEX_URL = "https://www.smogon.com/stats/"
CHAOS_URL = "https://www.smogon.com/stats/{month}/chaos/{format_id}-{cutoff}.json"

# (connect timeout, read timeout). requests defaults to unbounded, which turns a
# stalled socket into a hung run. The read timeout is generous because the chaos
# payload is ~10 MB.
_REQUEST_TIMEOUT = (10, 120)

_MONTH_RE = re.compile(r"\b(20\d\d-\d\d)/")

# How far back to walk the month index before giving up. The newest month is
# normally 1 back; 6 covers a format that has stopped being reported without
# turning a typo into 100 requests.
_MAX_MONTHS_BACK = 6

DEFAULT_FORMAT = "gen9ou"
DEFAULT_CUTOFF = 1500
DEFAULT_OUT_DIR = Path("data/usage_stats")


def available_months(session: requests.Session) -> list[str]:
    """Every `YYYY-MM` month in Smogon's stats index, newest first."""
    response = session.get(STATS_INDEX_URL, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    return sorted(set(_MONTH_RE.findall(response.text)), reverse=True)


def cache_path(out_dir: Path, month: str, format_id: str, cutoff: int) -> Path:
    return out_dir / f"{month}_{format_id}-{cutoff}.json"


def fetch_chaos(
    session: requests.Session, month: str, format_id: str, cutoff: int
) -> dict | None:
    """The chaos payload for one month, or None if that month has no such file.

    A missing month/format combination is a 404 and is a normal outcome while
    resolving "the newest month that has this format" - it is not an error.
    """
    url = CHAOS_URL.format(month=month, format_id=format_id, cutoff=cutoff)
    response = session.get(url, timeout=_REQUEST_TIMEOUT)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--format", dest="format_id", default=DEFAULT_FORMAT)
    parser.add_argument(
        "--cutoff",
        type=int,
        default=DEFAULT_CUTOFF,
        help="rating floor on the games counted (gen9ou: 0, 1500, 1695, 1825)",
    )
    parser.add_argument(
        "--month",
        default=None,
        help="YYYY-MM; default is the newest month that actually has this file",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the file is already cached",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "battle-engine/phase-6 (usage-stats fetch)"

    months = [args.month] if args.month else available_months(session)[:_MAX_MONTHS_BACK]
    if not months:
        print("no months found in the stats index", file=sys.stderr)
        return 1

    for month in months:
        destination = cache_path(args.out_dir, month, args.format_id, args.cutoff)
        if destination.exists() and not args.force:
            print(f"already cached: {destination}")
            return 0
        print(f"trying {month} ...", flush=True)
        payload = fetch_chaos(session, month, args.format_id, args.cutoff)
        if payload is None:
            continue
        info = payload.get("info", {})
        destination.write_text(json.dumps(payload))
        print(
            f"wrote {destination} "
            f"({destination.stat().st_size / 1e6:.1f} MB, "
            f"{len(payload.get('data', {}))} species, "
            f"{info.get('number of battles', '?')} battles, "
            f"cutoff {info.get('cutoff', '?')})"
        )
        return 0

    print(
        f"no chaos file for {args.format_id}-{args.cutoff} in the last "
        f"{len(months)} months ({', '.join(months)})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
