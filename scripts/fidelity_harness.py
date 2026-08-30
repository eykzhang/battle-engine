#!/usr/bin/env python
"""Phase 6 / M3: score a forward model against a real gen9ou replay corpus.

Runs both scoring conditions by default and prints the delta between them,
because that delta is the number the milestone exists to produce: how much of
the model's error is missing *knowledge* (which M4's set prediction can close)
rather than missing *mechanics* (which it cannot).

    # the full corpus, both conditions
    .venv/bin/python scripts/fidelity_harness.py

    # one condition, a subset, machine-readable output
    .venv/bin/python scripts/fidelity_harness.py --limit 50 --condition action \\
        --json data/fidelity.json

Needs the corpus fetched (scripts/fetch_showdown_replays.py) and the gen9
poke-engine extension built (scripts/build_poke_engine.sh). No Showdown server
and no network - it reads cached replay JSON off disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from battle_engine.fidelity import CONDITIONS, FidelityReport, PokeEngineBackend, score_corpus

DEFAULT_CORPUS = Path("data/replays_showdown")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"directory of Showdown replay JSON (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument("--limit", type=int, default=None, help="score only the first N replays")
    parser.add_argument(
        "--condition",
        choices=("action", "hindsight", "both"),
        default="both",
        help=(
            "action: the oracle supplies only the move/switch/Tera the turn needs. "
            "hindsight: it also supplies every ability, Tera type and move the battle "
            "will eventually reveal. both (default): run each and print the delta."
        ),
    )
    parser.add_argument("--json", type=Path, default=None, help="also write per-turn scores here")
    parser.add_argument("--quiet", action="store_true", help="no per-replay progress")
    return parser.parse_args(argv)


def _progress(total: int, quiet: bool):
    if quiet:
        return None

    def report(index: int, path: Path, report: FidelityReport) -> None:
        # Every 25 files rather than every file: a 300-replay run takes a few
        # seconds, so per-file output is noise, but a silent run that turns
        # out to be a slow one is the failure mode
        # notes/gotcha-benchmark-runs-need-empirical-timing-and-progress-visibility.md
        # already recorded once.
        if (index + 1) % 25 == 0 or index + 1 == total:
            print(f"  [{index + 1}/{total}] scored {report.scored} turns", file=sys.stderr)

    return report


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    # poke-env logs a warning for every protocol effect it does not recognize.
    # Over 300 replays that is tens of thousands of lines, and it would bury
    # the report this script exists to print.
    logging.disable(logging.WARNING)

    paths = sorted(args.corpus.glob("*.json"))
    if not paths:
        print(
            f"No replays in {args.corpus}. Fetch some first:\n"
            "  .venv/bin/python scripts/fetch_showdown_replays.py --n 300 --min-rating 1300",
            file=sys.stderr,
        )
        return 1
    if args.limit is not None:
        paths = paths[: args.limit]

    conditions = [False, True] if args.condition == "both" else [args.condition == "hindsight"]
    backend = PokeEngineBackend()
    reports: List[FidelityReport] = []
    for hindsight in conditions:
        if not args.quiet:
            print(f"Scoring {len(paths)} replays - condition: {CONDITIONS[hindsight]}", file=sys.stderr)
        reports.append(
            score_corpus(paths, backend=backend, hindsight=hindsight, on_replay=_progress(len(paths), args.quiet))
        )

    for report in reports:
        print()
        print(report.render())

    if len(reports) == 2:
        print()
        print(_delta(*reports))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    report.condition: {
                        "backend": report.backend,
                        "replays": report.replays,
                        "turns_seen": report.turns_seen,
                        "skipped": dict(report.skipped),
                        "scores": [asdict(score) for score in report.scores],
                    }
                    for report in reports
                },
                indent=1,
            )
        )
        print(f"\nWrote per-turn scores to {args.json}", file=sys.stderr)
    return 0


def _delta(action: FidelityReport, hindsight: FidelityReport) -> str:
    """What perfect in-battle set knowledge is worth, in this model's accuracy.

    A lower bound on set prediction's headroom, not an upper one: the
    hindsight condition can only supply what the battle eventually shows, and
    over this corpus that is 40.5% of abilities and 26.8% of items. It also
    cannot supply EV spreads, which no replay ever reveals.
    """
    lines = ["What set knowledge buys (hindsight-oracle minus action-oracle):"]
    for label, attribute in (("exactly right (best branch)", "best_exact"), ("exactly right (modal)", "modal_exact")):
        before = sum(1 for s in action.scores if getattr(s, attribute))
        after = sum(1 for s in hindsight.scores if getattr(s, attribute))
        total = max(action.scored, 1)
        lines.append(
            f"  {label:<34}{100.0 * before / total:.1f}% -> {100.0 * after / total:.1f}%"
            f"  ({100.0 * (after - before) / total:+.1f} pts)"
        )
    for cause in ("hp", "weather", "boost", "fainted", "status"):
        before = sum(1 for s in action.scores if any(d.category == cause for d in s.best_divergences))
        after = sum(1 for s in hindsight.scores if any(d.category == cause for d in s.best_divergences))
        lines.append(f"  turns with a {cause + ' divergence':<21}{before} -> {after}  ({after - before:+d})")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
