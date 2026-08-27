"""Tests for battle_engine.teams - the gen9ou team pool used identically for
both self-play sides during PPO training and for real ladder play (see
teams.py's own module docstring for why pool size/diversity matters).

The real legality gate (`node pokemon-showdown validate-team gen9ou`) is an
integration test against this checkout's local simulator, same
skip-if-unavailable convention as tests/test_rl_env.py's server-dependent
test - not a hard dependency for the rest of the suite.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from poke_env.teambuilder.teambuilder import Teambuilder

from battle_engine.teams import GEN9OU_SAMPLE_TEAMS, RandomTeamFromPool

_POKEMON_SHOWDOWN_DIR = Path(__file__).resolve().parent.parent / "pokemon-showdown"


def _validator_available() -> bool:
    return shutil.which("node") is not None and _POKEMON_SHOWDOWN_DIR.is_dir()


class _PassthroughTeambuilder(Teambuilder):
    """Teambuilder is an ABC (yield_team is abstract) - this exists only to
    get an instance so parse_showdown_team/join_team (both real, concrete
    methods) can be called directly, matching teams.py's own docstring
    instructions for re-validating the pool by hand.
    """

    def yield_team(self) -> str:
        return ""


def test_pool_has_grown_well_past_the_original_five():
    # The original pool was 5 teams / 24 distinct species - diagnosed as a
    # real, compounding cause of the local-vs-ladder gap alongside the
    # encoding gap (see module docstring). This just guards against a future
    # edit silently shrinking the pool back down.
    assert len(GEN9OU_SAMPLE_TEAMS) >= 20


def test_pool_has_no_duplicate_teams():
    # Whitespace-normalized so a trivial formatting difference doesn't mask
    # a real duplicate, and doesn't false-positive one either.
    normalized = ["\n".join(line.strip() for line in t.strip().splitlines()) for t in GEN9OU_SAMPLE_TEAMS]
    assert len(normalized) == len(set(normalized))


def test_every_team_parses_to_exactly_six_pokemon():
    tb = _PassthroughTeambuilder()
    for i, raw in enumerate(GEN9OU_SAMPLE_TEAMS):
        parsed = tb.parse_showdown_team(raw)
        assert len(parsed) == 6, f"team {i} parsed to {len(parsed)} Pokemon, not 6"


def test_random_team_from_pool_yields_a_packed_string_from_the_pool():
    builder = RandomTeamFromPool()
    tb = _PassthroughTeambuilder()
    packed_pool = {tb.join_team(tb.parse_showdown_team(t)) for t in GEN9OU_SAMPLE_TEAMS}
    for _ in range(10):
        assert builder.yield_team() in packed_pool


def test_random_team_from_pool_samples_more_than_one_team_across_many_calls():
    # Mutation-testable-in-spirit: if yield_team() were secretly pinned to
    # always return team 0, this would fail almost certainly (26 teams,
    # uniform random choice, 40 draws).
    builder = RandomTeamFromPool()
    seen = {builder.yield_team() for _ in range(40)}
    assert len(seen) > 1


@pytest.mark.skipif(not _validator_available(), reason="local pokemon-showdown checkout or node not available")
@pytest.mark.parametrize("index", range(len(GEN9OU_SAMPLE_TEAMS)))
def test_team_is_legal_under_the_local_checkouts_current_gen9ou_ruleset(index):
    """The real gate this pool's own docstring insists on: don't trust a
    team's legality from its source, the banlist isn't static. Parametrized
    per-team (not one loop) so a single illegal team reports as one clear
    failing test, not a buried assertion inside a loop.
    """
    tb = _PassthroughTeambuilder()
    raw = GEN9OU_SAMPLE_TEAMS[index]
    packed = tb.join_team(tb.parse_showdown_team(raw))
    result = subprocess.run(
        ["node", "pokemon-showdown", "validate-team", "gen9ou"],
        input=packed,
        capture_output=True,
        text=True,
        cwd=_POKEMON_SHOWDOWN_DIR,
    )
    output = (result.stdout + result.stderr).strip()
    assert not output, f"team {index} failed gen9ou validation:\n{output}"
