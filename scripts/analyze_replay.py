#!/usr/bin/env python
"""BattleBrain fixture generation: turn one Showdown replay into a per-turn
win-probability/grading analysis, schema v1.

    # ladder-parity defaults (1000ms/turn, 8 opponent samples, 4 threads,
    # usage-stats cutoff 1500) - a few minutes for a real 50-100 turn replay
    .venv/bin/python scripts/analyze_replay.py --replay gen9ou-2672899958 \\
        --perspective p1 --out data/analysis/gen9ou-2672899958.json

    # a raw path also works, and a smaller budget for a quick smoke test
    .venv/bin/python scripts/analyze_replay.py \\
        --replay data/replays_showdown/gen9ou-2672927429.json --perspective p2 \\
        --out /tmp/out.json --search-time-ms 200 --opponent-samples 2

Needs the gen9 poke-engine extension built (scripts/build_poke_engine.sh) and
a cached usage-stats file for the requested --cutoff
(scripts/fetch_usage_stats.py). No Showdown server, no network - it reads
cached replay JSON off disk, same as scripts/fidelity_harness.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from battle_engine.replay_analysis import (
    DEFAULT_OPPONENT_SAMPLES,
    DEFAULT_SEARCH_TIME_MS,
    DEFAULT_THREADS,
    DEFAULT_USAGE_STATS_CUTOFF,
    ReplayAnalysisError,
    analyze_replay,
    coverage,
)
from battle_engine.replay_log import ReplayParseError

DEFAULT_CORPUS = Path("data/replays_showdown")


def _resolve_replay_path(replay: str, corpus: Path) -> Path:
    """`--replay` accepts a path to a JSON file or a bare replay id, resolved
    against `corpus` (Showdown replay ids never contain a path separator or a
    literal ".json", so this order can't misfire on a real id)."""
    as_path = Path(replay)
    if as_path.suffix == ".json" and as_path.exists():
        return as_path
    candidate = corpus / f"{replay}.json"
    if candidate.exists():
        return candidate
    if as_path.exists():
        return as_path
    raise FileNotFoundError(f"no replay found at {as_path} or {candidate}")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", required=True, help="a replay id (looked up in --corpus) or a path to its JSON")
    parser.add_argument("--perspective", required=True, choices=("p1", "p2"), help="whose win probability to report")
    parser.add_argument("--out", required=True, type=Path, help="where to write the schema-v1 JSON document")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help=f"default: {DEFAULT_CORPUS}")
    parser.add_argument("--search-time-ms", type=int, default=DEFAULT_SEARCH_TIME_MS, help="budget per turn, ms")
    parser.add_argument("--opponent-samples", type=int, default=DEFAULT_OPPONENT_SAMPLES)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cutoff", type=int, default=DEFAULT_USAGE_STATS_CUTOFF, help="usage-stats rating cutoff")
    parser.add_argument("--seed", type=int, default=0, help="opponent-sampling seed, for reproducible regeneration")
    parser.add_argument("--quiet", action="store_true", help="suppress the coverage summary on stderr")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    # poke-env logs a warning per unrecognized protocol effect; not useful
    # noise for a single-replay run.
    logging.disable(logging.WARNING)

    try:
        path = _resolve_replay_path(args.replay, args.corpus)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read {path}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict):
        print(f"error: {path} is not a JSON object", file=sys.stderr)
        return 1

    try:
        document = analyze_replay(
            payload,
            perspective=args.perspective,
            replay_id=path.stem,
            search_time_ms=args.search_time_ms,
            n_opponent_samples=args.opponent_samples,
            threads=args.threads,
            usage_stats_cutoff=args.cutoff,
            seed=args.seed,
        )
    except (ReplayAnalysisError, ReplayParseError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")

    if not args.quiet:
        eval_bar, grading = coverage(document)
        print(
            f"{document['replayId']}: {document['totalTurns']} turns, "
            f"eval-bar coverage {eval_bar:.1%}, grading coverage {grading:.1%} "
            f"({document['gradableTurns']}/{document['totalTurns']}) -> {args.out}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
