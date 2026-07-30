"""Pull a small, ELO-filtered sample of Metamon's gen9ou parsed-replay dataset
without downloading the full archive.

The source archive (jakegrigsby/metamon-parsed-replays on Hugging Face,
gen9ou.tar.gz) is 20+ GB, which violates this project's laptop-first rule to
download wholesale. Its members are named
`{battle_id}_{elo}_{pov}_vs_{opponent}_{date}_{WIN|LOSS}.json.lz4`, so ELO and
outcome are readable from the filename alone. This script streams the
.tar.gz over HTTP and decompresses it member-by-member (tarfile's streaming
mode never seeks), stopping as soon as it has collected `--n` replays at or
above `--min-elo` — only the bytes up to that point are ever downloaded.

    .venv/bin/python scripts/fetch_replay_sample.py --n 200 --min-elo 1200

Caveat found by review (2026-07-30): the archive is stored path-sorted by
year/month, and this script always takes the *first* N qualifying members it
scans — so a default run is temporally biased (an actual run collected 30/30
files from a single month). `--accept-probability` trades more bandwidth for
less bias: thinning acceptances forces the scan to run deeper into the
archive (further into later dates) to still reach `--n`. Streaming mode
can't skip a member's body without reading it (no seeking), so this really
does cost proportionally more download/decompute — left off by default (1.0,
old behavior) rather than silently made slower. The achieved date range of
whatever got collected is always printed at the end either way, so the bias
is visible instead of silently opaque.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import requests

DATASET_URL = (
    "https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays/"
    "resolve/main/{format}.tar.gz"
)

# Verified against real archive members (2026-07-30): paths look like
# gen9ou/2023/11/smogtours-gen9ou-731515_Unrated_lockon62163_vs_icebeam46118_11-19-2023_LOSS.json.lz4
# i.e. nested under year/month dirs, date is MM-DD-YYYY, and ladder games use a
# numeric ELO while tournament ("smogtours") games are literally "Unrated" —
# those are skipped here since this filter is a simple numeric ELO threshold,
# not a full quality signal. Battle id has no underscores, so the first
# underscore-delimited field is ELO; date + WIN/LOSS are pinned to the end.
_NAME_RE = re.compile(
    r"^[^_]+_(?P<elo>\d+|Unrated)_.+_(?P<date>\d{2}-\d{2}-\d{4})_(?P<result>WIN|LOSS)\.json\.lz4$"
)

# (connect timeout, read timeout) - unbounded by default in requests, and a
# multi-GB HTTP stream that runs for minutes can otherwise hang forever on a
# stalled socket.
_REQUEST_TIMEOUT = (10, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--n", type=int, default=200, help="replays to collect")
    parser.add_argument("--min-elo", type=int, default=1200)
    parser.add_argument(
        "--accept-probability", type=float, default=1.0,
        help="probability of keeping a qualifying replay, to spread the sample "
             "across more of the archive at the cost of scanning (and downloading) "
             "further into it; 1.0 = take every qualifying match (default, fastest)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/replays_raw"),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def fetch_sample(
    fmt: str, n: int, min_elo: int, out_dir: Path,
    accept_probability: float = 1.0, seed: int = 0,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = DATASET_URL.format(format=fmt)
    rng = random.Random(seed)

    # Top-up semantics: count what's already on disk toward the target.
    # Real bug caught by review: the exists-check further down used to skip
    # a write without incrementing `collected`, which meant a re-run with
    # the same --n never actually topped up to n - it always added n *more*
    # brand-new files on top of whatever was already there, scanning
    # substantially further into the archive each time, the exact bandwidth
    # cost this script exists to avoid.
    collected = len(list(out_dir.glob("*.json.lz4")))
    if collected >= n:
        print(f"Already have {collected} >= {n} replays in {out_dir}, nothing to fetch.")
        return collected

    seen = 0
    unparsed_shown = 0
    dates_collected: list[str] = []
    try:
        with requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    seen += 1
                    name = Path(member.name).name
                    match = _NAME_RE.match(name)
                    if match is None:
                        if unparsed_shown < 3:
                            print(f"  (skipping unparsed filename: {name})", file=sys.stderr)
                            unparsed_shown += 1
                        continue
                    if match["elo"] == "Unrated" or int(match["elo"]) < min_elo:
                        continue
                    if accept_probability < 1.0 and rng.random() > accept_probability:
                        continue
                    if (out_dir / name).exists():  # idempotent re-runs
                        continue

                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    (out_dir / name).write_bytes(fh.read())
                    collected += 1
                    dates_collected.append(match["date"])
                    if collected % 25 == 0:
                        print(f"  collected {collected}/{n} (scanned {seen})")
                    if collected >= n:
                        break
    except (tarfile.TarError, requests.RequestException, OSError) as exc:
        print(f"Stopped early after an error ({exc!r}) — {collected} replays "
              f"collected before the failure are still on disk.", file=sys.stderr)

    if dates_collected:
        # MM-DD-YYYY doesn't sort correctly as a plain string - parse first.
        parsed = [datetime.strptime(d, "%m-%d-%Y") for d in dates_collected]
        date_range = f"{min(parsed):%m-%d-%Y} to {max(parsed):%m-%d-%Y}"
    else:
        date_range = "n/a"
    print(f"Done: {collected} replays >= {min_elo} ELO written to {out_dir} "
          f"(scanned {seen} archive members, dates {date_range})")
    return collected


def main() -> None:
    args = parse_args()
    fetch_sample(
        args.format, args.n, args.min_elo, args.out_dir,
        args.accept_probability, args.seed,
    )


if __name__ == "__main__":
    main()
