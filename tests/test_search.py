from types import SimpleNamespace

from poke_env.battle.move import Move

from battle_engine.search import (
    TwoPlySearchPlayer,
    _apply_damage,
    _opponent_best_reply,
    _project_after_action,
    _switch_urgency_bonus,
)
from conftest import make_mon


def _battle(my_team, my_active, opp_team, opp_active, available_moves=(), available_switches=()):
    return SimpleNamespace(
        team={mon.species: mon for mon in my_team},
        opponent_team={mon.species: mon for mon in opp_team},
        active_pokemon=my_active,
        opponent_active_pokemon=opp_active,
        side_conditions={},
        opponent_side_conditions={},
        available_moves=list(available_moves),
        available_switches=list(available_switches),
    )


def test_apply_damage_reduces_hp_without_mutating_original():
    mon = make_mon("garchomp")
    original_hp = mon.current_hp

    damaged = _apply_damage(mon, 0.3)

    assert damaged.current_hp < original_hp
    assert mon.current_hp == original_hp  # the real Pokemon object is untouched


def test_apply_damage_faints_at_zero_hp():
    mon = make_mon("garchomp")
    fainted = _apply_damage(mon, 1.5)  # more than 100% of max HP
    assert fainted.fainted


def test_opponent_best_reply_picks_highest_expected_damage():
    attacker = make_mon("blissey")
    attacker._moves["tackle"] = Move("tackle", gen=9)
    attacker._moves["icebeam"] = Move("icebeam", gen=9)
    defender = make_mon("garchomp")  # Ice is 4x vs Ground/Dragon

    reply = _opponent_best_reply(attacker, defender)

    assert reply.id == "icebeam"


def test_opponent_best_reply_is_none_when_nothing_revealed():
    attacker = make_mon("blissey")  # no moves revealed
    defender = make_mon("garchomp")

    assert _opponent_best_reply(attacker, defender) is None


def test_project_after_action_damages_both_actives():
    my_active = make_mon("garchomp")
    opp_active = make_mon("blissey")
    opp_active._moves["icebeam"] = Move("icebeam", gen=9)
    battle = _battle([my_active], my_active, [opp_active], opp_active)

    projected = _project_after_action(battle, my_active, Move("earthquake", gen=9))

    assert projected.opponent_active_pokemon.current_hp < opp_active.current_hp
    assert projected.active_pokemon.current_hp < my_active.current_hp
    # originals untouched
    assert opp_active.current_hp == opp_active.max_hp
    assert my_active.current_hp == my_active.max_hp


def test_project_after_action_no_retaliation_if_my_move_faints_opponent():
    # Regression test: a fainted Pokemon can't move. Before this fix, the
    # projection always charged the opponent's full retaliation even when my
    # own move already knocked them out.
    my_active = make_mon("garchomp")
    opp_active = make_mon("blissey", current_hp_fraction=0.01)  # one hit from fainting
    opp_active._moves["icebeam"] = Move("icebeam", gen=9)
    battle = _battle([my_active], my_active, [opp_active], opp_active)

    projected = _project_after_action(battle, my_active, Move("earthquake", gen=9))

    assert projected.opponent_active_pokemon.fainted
    assert projected.active_pokemon.current_hp == my_active.current_hp


def test_search_player_prefers_higher_damage_move_over_weak_move():
    my_active = make_mon("garchomp")
    my_bench = make_mon("blissey")
    opp_active = make_mon("blissey")
    opp_active._moves["icebeam"] = Move("icebeam", gen=9)
    battle = _battle(
        [my_active, my_bench],
        my_active,
        [opp_active],
        opp_active,
        available_moves=[Move("earthquake", gen=9), Move("tackle", gen=9)],
        available_switches=[my_bench],
    )

    player = TwoPlySearchPlayer(battle_format="gen9randombattle", start_listening=False)
    order = player.choose_move(battle)

    assert order.order.id == "earthquake"


def test_switch_urgency_bonus_is_zero_in_a_fine_matchup():
    my_active = make_mon("garchomp")
    opp_active = make_mon("blissey")  # neutral matchup either way
    battle = _battle([my_active], my_active, [opp_active], opp_active)

    assert _switch_urgency_bonus(battle) == 0.0


def test_switch_urgency_bonus_is_positive_in_a_bad_matchup():
    # Electric vs Ground is a 0x immunity: about as bad a matchup as exists.
    my_active = make_mon("electrode")
    opp_active = make_mon("donphan")
    battle = _battle([my_active], my_active, [opp_active], opp_active)

    assert _switch_urgency_bonus(battle) > 0.0


def test_search_player_switches_out_of_a_bad_matchup_even_with_a_real_attack_available():
    # Charizard (Fire/Flying) vs Gastrodon (Water/Ground): Charizard's own
    # Flamethrower is resisted (0.5x, still real chip damage) but Gastrodon's
    # Water STAB hits Charizard for 2x — a bad, replay-observed-style
    # matchup. Tangela (pure Grass) counters Gastrodon hard (4x offense,
    # resists its Water STAB). The bot should pivot rather than keep
    # trading, even though Flamethrower isn't a useless move.
    my_active = make_mon("charizard")
    my_bench = make_mon("tangela")
    opp_active = make_mon("gastrodon")
    opp_active._moves["surf"] = Move("surf", gen=9)
    battle = _battle(
        [my_active, my_bench],
        my_active,
        [opp_active],
        opp_active,
        available_moves=[Move("flamethrower", gen=9)],
        available_switches=[my_bench],
    )

    player = TwoPlySearchPlayer(battle_format="gen9randombattle", start_listening=False)
    order = player.choose_move(battle)

    assert order.order.species == "tangela"
