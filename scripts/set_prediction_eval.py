#!/usr/bin/env python
"""Phase 6 / M4: score a set-prediction filler against a real gen9ou corpus.

Asks a filler what it thinks of each side at every turn boundary and scores it
against what that battle eventually revealed. This is one half of M4's evidence;
the other half is

    .venv/bin/python scripts/fidelity_harness.py --condition action --prior usage-stats

which asks whether the prediction actually moves the forward model's accuracy.
Neither number is sufficient alone - see `battle_engine/set_eval.py`'s docstring.

    # every condition, the full corpus
    .venv/bin/python scripts/set_prediction_eval.py

    # one condition
    .venv/bin/python scripts/set_prediction_eval.py --condition usage-1500 --limit 50

Reads cached replay JSON off disk: no Showdown server, no network.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List

from battle_engine.poke_engine_state import RevealedOnlyFiller
from battle_engine.set_eval import SetEvalReport, score_corpus
from battle_engine.set_prediction import UsageStatsFiller

DEFAULT_CORPUS = Path("data/replays_showdown")

# Each condition isolates one modelling choice, so a difference between two rows
# has exactly one cause. `revealed-only` is the baseline M4 has to beat and
# scores zero on every line by construction - it is in the table because a
# baseline that is zero *by construction* should be visible as such rather than
# asserted in prose.
CONDITIONS: Dict[str, Callable[[], object]] = {
    "revealed-only": RevealedOnlyFiller,
    "usage-0": lambda: UsageStatsFiller.from_cache(cutoff=0),
    "usage-1500": lambda: UsageStatsFiller.from_cache(cutoff=1500),
    "usage-1695": lambda: UsageStatsFiller.from_cache(cutoff=1695),
    "usage-1825": lambda: UsageStatsFiller.from_cache(cutoff=1825),
    "usage-1500-no-teammates": lambda: UsageStatsFiller.from_cache(
        cutoff=1500, condition_on_teammates=False
    ),
    "usage-1500-sampled": lambda: UsageStatsFiller.from_cache(
        cutoff=1500, rng=random.Random(1)
    ),
}


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N replays")
    parser.add_argument(
        "--condition",
        action="append",
        choices=sorted(CONDITIONS),
        default=None,
        help="repeatable; default runs every condition",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _progress(total: int, quiet: bool):
    if quiet:
        return None

    def report(index: int, path: Path, report: SetEvalReport) -> None:
        if (index + 1) % 50 == 0 or index + 1 == total:
            print(f"  [{index + 1}/{total}]", file=sys.stderr)

    return report


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    # poke-env warns on every protocol effect it has no handler for; over 300
    # replays that would bury the report.
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

    names = args.condition or list(CONDITIONS)
    reports: List[SetEvalReport] = []
    for name in names:
        if not args.quiet:
            print(f"Scoring {len(paths)} replays - condition: {name}", file=sys.stderr)
        try:
            filler = CONDITIONS[name]()
        except FileNotFoundError as exc:
            print(f"  skipped {name}: {exc}", file=sys.stderr)
            continue
        reports.append(score_corpus(paths, filler, on_replay=_progress(len(paths), args.quiet)))

    for report in reports:
        print()
        print(report.render())

    if len(reports) > 1:
        print()
        print(_table(names, reports))
    return 0


def _table(names: List[str], reports: List[SetEvalReport]) -> str:
    columns = ("species recall", "species precision", "ability", "item", "tera type", "moves recall")
    lines = ["Conditions side by side (turn-weighted; see set_eval.py on denominators):"]
    lines.append("  " + f"{'condition':<26}" + "".join(f"{c:>19}" for c in columns))
    for name, report in zip(names, reports):
        rates = report.overall.rates()
        row = "".join(
            f"{(f'{100.0 * h / a:.1f}%' if a else 'n/a'):>19}" for h, a in (rates[c] for c in columns)
        )
        lines.append(f"  {name:<26}{row}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
