from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

GEN = 9
GEN_DATA = GenData.from_gen(GEN)

_STAT_NAMES = ["hp", "atk", "def", "spa", "spd", "spe"]


def make_mon(
    species: str,
    evs=(0, 0, 0, 0, 0, 0),
    nature: str = "serious",
    level: int = 100,
    current_hp_fraction: float = 1.0,
    status: Status | None = None,
) -> Pokemon:
    """Build a standalone Pokemon with real computed stats and max_hp, for
    tests that need concrete numbers without a live battle connection.
    """
    mon = Pokemon(gen=GEN, species=species, details=f"{species}, L{level}")
    stats = compute_raw_stats(species, list(evs), [31] * 6, level, nature, GEN_DATA)
    for name, value in zip(_STAT_NAMES, stats):
        mon._stats[name] = value
    mon._max_hp = stats[0]
    mon._current_hp = round(current_hp_fraction * stats[0])
    mon._status = status
    return mon
