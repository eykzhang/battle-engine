import pytest

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

from battle_engine.damage import _crit_chance, estimate_stat, expected_damage
from conftest import GEN, make_mon as _make_mon


def test_expected_damage_matches_hand_computed_value():
    # 252 Atk Jolly Garchomp Earthquake vs 252 HP / 252 Def Bold Blissey.
    # Independently hand-derived (gen7+ formula, base ((2*100/5+2)*100*359/130)/50
    # +2 = 233, then *1.5 STAB *1.0 neutral type *0.925 avg roll *1.02083 crit EV
    # = ~330.02 damage, i.e. ~46.22% of Blissey's 714 max HP.
    garchomp = _make_mon("garchomp", [0, 252, 4, 0, 0, 252], "jolly")
    blissey = _make_mon("blissey", [252, 0, 252, 0, 4, 0], "bold")
    earthquake = Move("earthquake", gen=GEN)

    result = expected_damage(garchomp, blissey, earthquake)

    assert result == pytest.approx(0.4622, abs=1e-3)


def test_expected_damage_status_move_is_zero():
    garchomp = _make_mon("garchomp", [0, 252, 4, 0, 0, 252], "jolly")
    blissey = _make_mon("blissey", [252, 0, 252, 0, 4, 0], "bold")
    toxic = Move("toxic", gen=GEN)

    assert expected_damage(garchomp, blissey, toxic) == 0.0


def test_expected_damage_type_immunity_is_zero():
    # Ground moves never hit Flying-types.
    garchomp = _make_mon("garchomp", [0, 252, 4, 0, 0, 252], "jolly")
    skarmory = _make_mon("skarmory", [252, 0, 252, 0, 4, 0], "bold")
    earthquake = Move("earthquake", gen=GEN)

    assert expected_damage(garchomp, skarmory, earthquake) == 0.0


def test_expected_damage_scales_with_type_effectiveness(monkeypatch):
    garchomp = _make_mon("garchomp", [0, 252, 4, 0, 0, 252], "jolly")
    blissey = _make_mon("blissey", [252, 0, 252, 0, 4, 0], "bold")
    earthquake = Move("earthquake", gen=GEN)

    results = {}
    for multiplier in (0.5, 1.0, 2.0):
        monkeypatch.setattr(
            Pokemon, "damage_multiplier", lambda self, t, m=multiplier: m
        )
        results[multiplier] = expected_damage(garchomp, blissey, earthquake)

    assert results[0.5] < results[1.0] < results[2.0]
    assert results[2.0] == pytest.approx(4 * results[0.5], rel=1e-6)


def test_estimate_stat_falls_back_to_base_stat_when_unknown():
    # A bare-constructed Pokemon has no revealed EVs/nature (this is the real
    # state of an opponent's Pokemon mid-battle) — .stats[stat] is None until
    # something reveals it.
    mon = Pokemon(gen=GEN, species="garchomp", details="Garchomp, L100")
    assert mon.stats["spe"] is None

    assert estimate_stat(mon, "spe") > 0


def test_expected_damage_works_when_attacker_stats_are_unknown():
    # Regression test: expected_damage must not silently return 0 (or crash)
    # just because the attacker is an opponent whose exact stats aren't known
    # — this is exactly the case _opponent_best_reply() depends on.
    attacker = Pokemon(gen=GEN, species="garchomp", details="Garchomp, L100")
    defender = _make_mon("blissey", [252, 0, 252, 0, 4, 0], "bold")
    earthquake = Move("earthquake", gen=GEN)

    assert expected_damage(attacker, defender, earthquake) > 0.0


def test_expected_damage_uses_real_hp_not_percent_scale_for_opponent():
    # Regression test for a real bug found via independent review: for the
    # opponent's Pokemon mid-battle, .max_hp is 0-100 percent-scale (poke-env
    # doesn't know their real HP pool either), not real HP points. Dividing
    # real damage by percent-scale max_hp used to inflate the result several-
    # fold (a super-effective STAB hit came out to ~765% instead of ~99%).
    garchomp = _make_mon("garchomp", [0, 252, 4, 0, 0, 252], "jolly")
    tyranitar_opp = Pokemon(gen=GEN, species="tyranitar", details="Tyranitar, L100, M")
    tyranitar_opp._max_hp = 100  # percent-scale, as poke-env reports for opponents
    tyranitar_opp._current_hp = 100
    earthquake = Move("earthquake", gen=GEN)  # 2x super effective vs Rock/Dark

    result = expected_damage(garchomp, tyranitar_opp, earthquake)

    assert result < 1.5  # sane fraction; the old bug produced ~7.65 here


@pytest.mark.parametrize(
    "move_id, expected_chance",
    [
        ("tackle", 1 / 24),  # no crit boost
        ("stoneedge", 1 / 8),  # high-crit-ratio move
        ("frostbreath", 1.0),  # always crits
    ],
)
def test_crit_chance_matches_showdown_crit_mult_table(move_id, expected_chance):
    move = Move(move_id, gen=GEN)
    assert _crit_chance(move) == pytest.approx(expected_chance)
