from types import SimpleNamespace

from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status

from battle_engine.evaluation import evaluate
from conftest import make_mon


def _battle(my_team, my_active, opp_team, opp_active, my_hazards=None, opp_hazards=None):
    return SimpleNamespace(
        team={mon.species: mon for mon in my_team},
        opponent_team={mon.species: mon for mon in opp_team},
        active_pokemon=my_active,
        opponent_active_pokemon=opp_active,
        side_conditions=my_hazards or {},
        opponent_side_conditions=opp_hazards or {},
    )


def test_identical_teams_score_zero():
    mine = make_mon("garchomp")
    theirs = make_mon("garchomp")
    battle = _battle([mine], mine, [theirs], theirs)

    assert evaluate(battle) == 0.0


def test_hp_advantage_scores_positive():
    mine = make_mon("garchomp", current_hp_fraction=1.0)
    theirs = make_mon("garchomp", current_hp_fraction=0.3)
    battle = _battle([mine], mine, [theirs], theirs)

    assert evaluate(battle) > 0.0


def test_fainted_opponent_mon_scores_positive():
    mine_active = make_mon("garchomp")
    mine_bench = make_mon("blissey")
    their_active = make_mon("garchomp")
    their_fainted = make_mon("blissey", current_hp_fraction=0.0, status=Status.FNT)
    battle = _battle(
        [mine_active, mine_bench], mine_active, [their_active, their_fainted], their_active
    )

    assert evaluate(battle) > 0.0


def test_favorable_type_matchup_scores_positive():
    # Excadrill (Ground/Steel) is immune to Electric and resists nothing from
    # a Pikachu with no super-effective STAB options; Excadrill's Ground/Steel
    # typing hits back hard. Clear, unambiguous type matchup in Excadrill's favor.
    excadrill = make_mon("excadrill")
    pikachu = make_mon("pikachu")
    battle = _battle([excadrill], excadrill, [pikachu], pikachu)

    assert evaluate(battle) > 0.0


def test_status_on_opponent_scores_positive():
    mine = make_mon("garchomp")
    theirs_healthy = make_mon("garchomp")
    theirs_paralyzed = make_mon("garchomp", status=Status.PAR)

    healthy_score = evaluate(_battle([mine], mine, [theirs_healthy], theirs_healthy))
    paralyzed_score = evaluate(_battle([mine], mine, [theirs_paralyzed], theirs_paralyzed))

    assert paralyzed_score > healthy_score


def test_hazards_on_opponent_side_score_positive():
    mine = make_mon("garchomp")
    theirs = make_mon("garchomp")
    battle = _battle(
        [mine],
        mine,
        [theirs],
        theirs,
        opp_hazards={SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2},
    )

    assert evaluate(battle) > 0.0


def test_more_hazard_layers_score_worse_for_the_side_carrying_them():
    mine = make_mon("garchomp")
    theirs = make_mon("garchomp")

    one_layer = evaluate(
        _battle([mine], mine, [theirs], theirs, my_hazards={SideCondition.SPIKES: 1})
    )
    three_layers = evaluate(
        _battle([mine], mine, [theirs], theirs, my_hazards={SideCondition.SPIKES: 3})
    )

    assert three_layers < one_layer < 0.0
