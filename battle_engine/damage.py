"""Expected-damage calculator: the gen-9 damage formula, with every stochastic
element (damage roll, crit chance, accuracy, multi-hit count) folded into a
single expected value rather than enumerated as a full outcome distribution.

That's a deliberate Phase-1 simplification (see project roadmap): it keeps the
search's branching factor small at the cost of hiding rare-but-high-impact
outcomes, most notably "this move has a small chance to crit for a KO." Real
outcome distributions matter more once Phase 2/3 need calibrated win
probabilities — this module intentionally only serves move-ranking in search.

Also deliberately out of scope for now, each a separate named simplification
rather than an oversight: weather, screens, held items, abilities besides
Adaptability (handled for free via Pokemon.stab_multiplier), burn's physical
-damage halving, and non-damaging (status) move effects.
"""

from __future__ import annotations

from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon

# Average of the 16 equally-likely damage rolls (85-100, uniform).
_AVERAGE_ROLL = 0.925

# gen7+ crit-stage -> chance table, straight from Showdown's own simulator
# (sim/battle-actions.ts): critMult = [0, 24, 8, 2, 1], chance = 1 / critMult[stage].
# poke-env's Move.crit_ratio is the *raw* data field (0 when absent, meaning
# "use the default stage of 1"), not the in-battle stage directly, so 0 and 1
# both map to the base 1/24 rate.
_CRIT_CHANCE_BY_RATIO = {0: 1 / 24, 1: 1 / 24, 2: 1 / 8, 3: 1 / 2, 4: 1.0}


def _crit_chance(move: Move) -> float:
    if move.crit_ratio == 6:  # poke-env's marker for "always crits" (e.g. Frost Breath)
        return 1.0
    return _CRIT_CHANCE_BY_RATIO.get(move.crit_ratio, 1 / 24)


def _expected_crit_multiplier(move: Move) -> float:
    p_crit = _crit_chance(move)
    return (1 - p_crit) * 1.0 + p_crit * 1.5


def _attack_defense_stats(move: Move) -> tuple[str, str]:
    if move.category == MoveCategory.SPECIAL:
        return "spa", "spd"
    return "atk", "def"


def estimate_stat(mon: Pokemon, stat: str) -> float:
    """mon's stat, falling back to a base-stat estimate when the exact value
    isn't known — true for the opponent's Pokemon in a real battle, whose EVs
    and nature aren't revealed. Assumes max IVs, 0 EVs, neutral nature (same
    convention poke-env's own SimpleHeuristicsPlayer uses for this), so this
    is a rough estimate, not the opponent's real stat.

    Also the right way to get a real (non-percent-scale) max HP for a
    defender: mon.max_hp is 0-100 percent-scale for the opponent's Pokemon
    (poke-env doesn't know their real HP pool either), not real HP points —
    estimate_stat(mon, "hp") falls back to the same known-vs-estimated logic
    as every other stat instead.
    """
    known = mon.stats[stat]
    if known is not None:
        return known
    if stat == "hp":
        return int(int(2 * mon.base_stats["hp"] + 31) * mon.level / 100) + mon.level + 10
    return int(int(2 * mon.base_stats[stat] + 31) * mon.level / 100) + 5


def _fixed_damage(attacker: Pokemon, defender: Pokemon, move: Move) -> float | None:
    """Raw (pre-accuracy) damage for moves with a fixed-damage formula
    (Seismic Toss, Night Shade: attacker's level; Dragon Rage, Sonic Boom: a
    flat number) instead of the base_power/stat formula - poke-env exposes
    this via Move.damage ('level', an int, or 0 for normal moves). These
    moves bypass STAB, crit, and the damage roll entirely (real Showdown
    mechanics), but type immunity (not resistance/weakness scaling) still
    applies. None if `move` isn't one of these - not a fixed-damage move.

    HP-dependent variants (Super Fang, Final Gambit, Endeavor - poke-env
    marks these with a damageCallback, not a plain `damage` value) aren't
    handled: they need the *actual* defender HP, not an estimate, and their
    formulas aren't simple scalars. Left out deliberately, same spirit as
    this module's other named simplifications.
    """
    if move.damage == "level":
        return attacker.level
    if isinstance(move.damage, int) and move.damage > 0:
        return move.damage
    return None


def expected_damage(attacker: Pokemon, defender: Pokemon, move: Move) -> float:
    """Expected damage dealt by `move`, as a fraction of defender's real max
    HP (via estimate_stat, not defender.max_hp directly — see its docstring:
    for the opponent's Pokemon mid-battle, .max_hp is on a 0-100 percent
    scale, not real HP points, and dividing real damage by that silently
    inflates the fraction by roughly real_max_hp/100).

    Folds in accuracy and expected hit count, so a 50-base-power move at 50%
    accuracy contributes half the expected damage of the same move at 100%.
    Returns 0.0 for status moves (no base power) — their non-damage effects
    (stat boosts, status conditions, hazards) aren't modeled by this function.
    """
    if move.category == MoveCategory.STATUS:
        return 0.0

    fixed = _fixed_damage(attacker, defender, move)
    if fixed is not None:
        if defender.damage_multiplier(move.type) == 0:  # type immunity still applies
            return 0.0
        return fixed * move.accuracy / estimate_stat(defender, "hp")

    if not move.base_power:
        return 0.0

    atk_stat, def_stat = _attack_defense_stats(move)
    atk = estimate_stat(attacker, atk_stat)
    defense = estimate_stat(defender, def_stat)

    base = (
        int(int(int(2 * attacker.level / 5 + 2) * move.base_power * atk / defense) / 50)
        + 2
    )

    stab = attacker.stab_multiplier if move.type in attacker.types else 1.0
    type_effectiveness = defender.damage_multiplier(move.type)

    expected = (
        base
        * stab
        * type_effectiveness
        * _AVERAGE_ROLL
        * _expected_crit_multiplier(move)
        * move.accuracy
        * move.expected_hits
    )
    return expected / estimate_stat(defender, "hp")
