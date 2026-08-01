from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.player import Player

from battle_engine.action_space import (
    metamon_action_to_poke_env_action,
    poke_env_action_to_metamon_action,
)
from conftest import make_mon


def _battle(team, active, available_moves=None):
    return SimpleNamespace(
        team={mon.species: mon for mon in team},
        active_pokemon=active,
        available_moves=available_moves if available_moves is not None else [],
    )


def test_plain_move_actions_map_to_poke_envs_move_block():
    # poke-env's plain-move block is 6-9 (SinglesEnv.action_to_order) -
    # metamon's move slots 0-3 pass straight through with that offset.
    for slot in range(4):
        assert metamon_action_to_poke_env_action(slot, battle=None) == 6 + slot


def test_tera_move_actions_map_to_poke_envs_terastallize_block():
    # poke-env's move+terastallize block is 22-25, the last of its four
    # gimmick blocks (mega/z-move/dynamax don't exist in gen9 OU and have
    # no metamon-side equivalent).
    for slot in range(4):
        assert metamon_action_to_poke_env_action(9 + slot, battle=None) == 22 + slot


def test_missing_action_sentinel_raises():
    with pytest.raises(ValueError):
        metamon_action_to_poke_env_action(-1, battle=None)


def test_out_of_range_action_raises():
    with pytest.raises(ValueError):
        metamon_action_to_poke_env_action(13, battle=None)


def test_switch_action_resolves_species_sorted_slot_to_raw_team_index():
    # Team order (raw, insertion/team-list order) deliberately NOT
    # alphabetical, so a bug that used the raw index directly instead of
    # resolving through species identity would be caught.
    zapdos = make_mon("zapdos")
    active = make_mon("garchomp")
    alakazam = make_mon("alakazam")
    bronzong = make_mon("bronzong")
    team = [zapdos, active, alakazam, bronzong]
    battle = _battle(team, active)

    # Species-sorted non-active bench: alakazam, bronzong, zapdos (slots 0,1,2).
    assert metamon_action_to_poke_env_action(4, battle) == team.index(alakazam)
    assert metamon_action_to_poke_env_action(5, battle) == team.index(bronzong)
    assert metamon_action_to_poke_env_action(6, battle) == team.index(zapdos)


def test_switch_action_beyond_bench_size_raises():
    active = make_mon("garchomp")
    team = [active, make_mon("zapdos")]
    battle = _battle(team, active)

    with pytest.raises(ValueError):
        metamon_action_to_poke_env_action(8, battle)  # bench slot 4, only 1 exists


def test_switch_action_round_trips_through_poke_envs_real_action_to_order():
    """Feeds the converted action back into poke-env's own
    SinglesEnv.action_to_order (fake=True to skip the valid_orders legality
    check, which needs a fuller live-battle mock than this test builds) and
    confirms it targets the intended species - verifying the offset/index
    convention against poke-env's real conversion code, not just this
    module's own mirrored understanding of it.
    """
    zapdos = make_mon("zapdos")
    active = make_mon("garchomp")
    alakazam = make_mon("alakazam")
    team = [zapdos, active, alakazam]
    battle = _battle(team, active)

    # Species-sorted non-active bench: alakazam (slot 0, action 4), zapdos (slot 1, action 5).
    poke_env_action = metamon_action_to_poke_env_action(4, battle)  # -> alakazam
    order = SinglesEnv.action_to_order(poke_env_action, battle, fake=True)
    assert order.order is alakazam


def test_switch_action_raises_without_an_active_pokemon():
    # Matches encoding.py's battle_view_from_poke_env, which raises for the
    # same state (team preview / a queued forced-switch-after-faint
    # request) instead of guessing at a bench.
    team = [make_mon("garchomp"), make_mon("zapdos")]
    battle = _battle(team, active=None)

    with pytest.raises(ValueError):
        metamon_action_to_poke_env_action(4, battle)


def test_move_action_round_trips_through_poke_envs_real_action_to_order():
    """Covers the .item() code path test_switch_action_round_trips... does
    NOT reach (the switch branch in SinglesEnv.action_to_order returns
    before touching .item(), so that test alone wouldn't have caught the
    plain-int-vs-np.int64 bug an earlier session fixed in rl_env.py) - a
    real Move-carrying battle, fed through poke-env's actual move branch
    with a genuine np.int64 action, the same type Gymnasium hands back.
    """
    active = make_mon("garchomp")
    for move_id in ["dragonclaw", "earthquake", "stealthrock", "swordsdance"]:
        active._add_move(move_id)
    team = [active, make_mon("zapdos")]
    battle = _battle(team, active, available_moves=list(active.moves.values()))

    poke_env_action = metamon_action_to_poke_env_action(1, battle)  # earthquake
    order = SinglesEnv.action_to_order(np.int64(poke_env_action), battle, fake=True)
    assert order.order.id == "earthquake"


def test_poke_env_move_actions_map_to_metamon_move_slots():
    for slot in range(4):
        assert poke_env_action_to_metamon_action(6 + slot, battle=None) == slot


def test_poke_env_tera_move_actions_map_to_metamon_tera_slots():
    for slot in range(4):
        assert poke_env_action_to_metamon_action(22 + slot, battle=None) == 9 + slot


def test_poke_env_gen9_unused_gimmick_actions_have_no_metamon_equivalent():
    # mega (10-13), z-move (14-17), dynamax (18-21) - none exist in gen9 OU,
    # Metamon's replay data never recorded them, so there's nothing to map to.
    for action in list(range(10, 22)):
        with pytest.raises(ValueError):
            poke_env_action_to_metamon_action(action, battle=None)


def test_poke_env_switch_action_resolves_raw_team_index_to_species_sorted_slot():
    """The inverse of test_switch_action_resolves_species_sorted_slot_to_raw_team_index."""
    zapdos = make_mon("zapdos")
    active = make_mon("garchomp")
    alakazam = make_mon("alakazam")
    bronzong = make_mon("bronzong")
    team = [zapdos, active, alakazam, bronzong]
    battle = _battle(team, active)

    # Species-sorted non-active bench: alakazam, bronzong, zapdos (slots 0,1,2).
    assert poke_env_action_to_metamon_action(team.index(alakazam), battle) == 4
    assert poke_env_action_to_metamon_action(team.index(bronzong), battle) == 5
    assert poke_env_action_to_metamon_action(team.index(zapdos), battle) == 6


def test_poke_env_switch_action_targeting_active_pokemon_raises():
    active = make_mon("garchomp")
    team = [active, make_mon("zapdos")]
    battle = _battle(team, active)

    with pytest.raises(ValueError):
        poke_env_action_to_metamon_action(team.index(active), battle)


def test_metamon_and_poke_env_switch_translations_round_trip():
    zapdos = make_mon("zapdos")
    active = make_mon("garchomp")
    alakazam = make_mon("alakazam")
    bronzong = make_mon("bronzong")
    team = [zapdos, active, alakazam, bronzong]
    battle = _battle(team, active)

    for metamon_action in range(4, 7):  # 3 real bench members here (slots 0,1,2)
        poke_env_action = metamon_action_to_poke_env_action(metamon_action, battle)
        assert poke_env_action_to_metamon_action(poke_env_action, battle) == metamon_action


def test_poke_env_switch_action_raises_without_an_active_pokemon():
    team = [make_mon("garchomp"), make_mon("zapdos")]
    battle = _battle(team, active=None)

    with pytest.raises(ValueError):
        poke_env_action_to_metamon_action(0, battle)
