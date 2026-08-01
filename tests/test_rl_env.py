import random
import socket
from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.battle.status import Status
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import DefaultBattleOrder, ForfeitBattleOrder
from poke_env.player.player import Player

from battle_engine.action_space import metamon_action_to_poke_env_action
from battle_engine.dataset import ACTION_SPACE_SIZE
from battle_engine.encoding import VECTOR_LEN, battle_view_from_poke_env, encode
from battle_engine.rl_env import _HP_VALUE, MetamonActionSinglesEnv, sb3_contrib_action_mask_fn
from conftest import make_mon


def _server_running(host: str = "localhost", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _mon_with_moves(species: str, move_ids: list[str]):
    mon = make_mon(species)
    for move_id in move_ids:
        mon._add_move(move_id)
    return mon


def _battle(
    team,
    active,
    available_switches,
    available_moves=None,
    trapped=False,
    can_tera=True,
    can_mega_evolve=False,
    can_z_move=False,
    can_dynamax=False,
    wait=False,
    gen=9,
    valid_orders=None,
):
    return SimpleNamespace(
        team={mon.species: mon for mon in team},
        active_pokemon=active,
        available_switches=available_switches,
        available_moves=(
            available_moves
            if available_moves is not None
            else (list(active.moves.values()) if active is not None else [])
        ),
        trapped=trapped,
        can_tera=can_tera,
        can_mega_evolve=can_mega_evolve,
        can_z_move=can_z_move,
        can_dynamax=can_dynamax,
        _wait=wait,
        gen=gen,
        valid_orders=valid_orders if valid_orders is not None else [],
        logger=None,
    )


def _standard_battle():
    active = _mon_with_moves("garchomp", ["dragonclaw", "earthquake", "stealthrock", "swordsdance"])
    zapdos = make_mon("zapdos")
    alakazam = make_mon("alakazam")
    bronzong = make_mon("bronzong")
    tyranitar = make_mon("tyranitar")
    ferrothorn = make_mon("ferrothorn")
    team = [active, zapdos, alakazam, bronzong, tyranitar, ferrothorn]
    # Only zapdos and alakazam are legal switch targets right now (e.g. the
    # rest are hypothetically scouted-as-bad, doesn't matter why for this
    # test - only that available_switches is a strict subset of the team).
    return _battle(team, active, available_switches=[zapdos, alakazam])


def test_action_spaces_are_discrete_13():
    env = MetamonActionSinglesEnv(
        battle_format="gen9ou", start_listening=False, fake=True
    )
    assert all(space.n == 13 for space in env.action_spaces.values())


def test_action_mask_matches_hand_derived_legality():
    battle = _standard_battle()
    mask = MetamonActionSinglesEnv.get_action_mask(battle)

    # 0-3 plain moves: all 4 known moves currently available -> all legal.
    # 4-8 switches, species-sorted bench (alakazam, bronzong, ferrothorn,
    # tyranitar, zapdos): only alakazam and zapdos are in available_switches.
    # 9-12 tera moves: can_tera=True, same move legality as 0-3 -> all legal.
    assert mask == [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]


def test_action_mask_agrees_with_native_mask_on_legal_positions():
    """Cross-checks against poke-env's own (real, un-mocked) get_action_mask
    rather than only against a hand-derived expectation, so a
    misunderstanding of poke-env's own legality logic would still be caught.
    """
    battle = _standard_battle()
    native_mask = SinglesEnv.get_action_mask(battle)
    mask = MetamonActionSinglesEnv.get_action_mask(battle)

    for metamon_action in range(13):
        poke_env_action = metamon_action_to_poke_env_action(metamon_action, battle)
        assert mask[metamon_action] == native_mask[poke_env_action]


def test_sb3_contrib_action_mask_fn_reads_agent1s_battle_through_the_wrapper_shape():
    """sb3_contrib_action_mask_fn is called by ActionMasker with the
    SingleAgentWrapper instance it wraps, and reaches the real battle via
    wrapped_env.env.battle1 - this mocks that exact attribute shape
    (.env.battle1) rather than constructing a real SingleAgentWrapper, so
    the test stays fast/server-free while still catching a wrong attribute
    path (e.g. .battle1 directly, or .env.battle2 - the opponent's battle,
    not the RL agent's).
    """
    battle = _standard_battle()
    wrapped_env = SimpleNamespace(env=SimpleNamespace(battle1=battle))

    mask = sb3_contrib_action_mask_fn(wrapped_env)

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, MetamonActionSinglesEnv.get_action_mask(battle))


def test_action_mask_during_wait_turn_is_the_explicit_placeholder():
    # battle._wait means poke-env doesn't actually consume the submitted
    # order (Player._handle_battle_request returns early) - regardless of
    # what the *real* legal switches/moves are, our mask should return the
    # fixed [1, 0, ..., 0] placeholder rather than deriving anything from
    # native_mask. Deliberately built with the active Pokemon NOT at raw
    # team index 0 (zapdos, not garchomp) - an earlier version of this test
    # put the active mon at index 0, which accidentally made "every metamon
    # action mapped to native index 0" true and hid a real bug where the
    # old (inference-based) fallback logic wouldn't have fired at all here.
    zapdos = _mon_with_moves("zapdos", ["thunderbolt"])
    active = _mon_with_moves("kingambit", ["suckerpunch"])
    team = [zapdos, active]
    battle = _battle(team, active, available_switches=[zapdos], wait=True)

    mask = MetamonActionSinglesEnv.get_action_mask(battle)
    assert mask == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_get_action_space_size_reports_13_not_the_native_26():
    assert MetamonActionSinglesEnv.get_action_space_size(9) == 13


def test_action_to_order_forfeit_and_default_sentinels_pass_through():
    battle = _standard_battle()
    assert isinstance(
        MetamonActionSinglesEnv.action_to_order(-1, battle, fake=True), ForfeitBattleOrder
    )
    assert isinstance(
        MetamonActionSinglesEnv.action_to_order(-2, battle, fake=True), DefaultBattleOrder
    )


def test_order_to_action_forfeit_and_default_sentinels_pass_through():
    battle = _standard_battle()
    assert MetamonActionSinglesEnv.order_to_action(ForfeitBattleOrder(), battle, fake=True) == -1
    assert MetamonActionSinglesEnv.order_to_action(DefaultBattleOrder(), battle, fake=True) == -2


def test_order_to_action_round_trips_a_move_order():
    """This is the scenario an independent review found broken: without an
    order_to_action override, poke-env's SingleAgentWrapper feeds the
    opponent's real order through the native (26-way) order_to_action, and
    the result gets fed straight back into this env's own (13-way)
    action_to_order one step later - so the round trip through both of this
    env's methods has to reproduce the original order, not just each
    direction in isolation.
    """
    battle = _standard_battle()
    order = Player.create_order(battle.active_pokemon.moves["earthquake"])

    action = MetamonActionSinglesEnv.order_to_action(order, battle, fake=True)
    assert action == 1  # metamon move slot 1

    reconstructed = MetamonActionSinglesEnv.action_to_order(action, battle, fake=True)
    assert reconstructed.order.id == "earthquake"
    assert not reconstructed.terastallize


def test_order_to_action_round_trips_a_tera_move_order():
    battle = _standard_battle()
    order = Player.create_order(battle.active_pokemon.moves["earthquake"], terastallize=True)

    action = MetamonActionSinglesEnv.order_to_action(order, battle, fake=True)
    assert action == 9 + 1  # metamon tera-move slot 1

    reconstructed = MetamonActionSinglesEnv.action_to_order(action, battle, fake=True)
    assert reconstructed.order.id == "earthquake"
    assert reconstructed.terastallize


def test_order_to_action_round_trips_a_switch_order():
    battle = _standard_battle()
    alakazam = next(mon for mon in battle.team.values() if mon.base_species == "alakazam")
    order = Player.create_order(alakazam)

    action = MetamonActionSinglesEnv.order_to_action(order, battle, fake=True)
    assert action == 4  # species-sorted bench slot 0

    reconstructed = MetamonActionSinglesEnv.action_to_order(action, battle, fake=True)
    assert reconstructed.order.base_species == "alakazam"


def test_action_translation_raises_a_clear_error_without_an_active_pokemon():
    # Matches encoding.py's own battle_view_from_poke_env, which raises for
    # the same state instead of guessing at a bench - see action_space.py's
    # _require_active_pokemon.
    team = [make_mon("garchomp"), make_mon("zapdos")]
    battle = _battle(team, active=None, available_switches=[])

    with pytest.raises(ValueError):
        MetamonActionSinglesEnv.action_to_order(4, battle, fake=True)


def test_action_to_order_falls_back_to_a_random_legal_move_when_not_strict():
    """An independent review found the translation layer's own ValueErrors
    (this scenario: no active_pokemon) previously bypassed strict/non-strict
    entirely and always raised - unlike every other illegal-action case
    SinglesEnv.action_to_order itself handles, and unlike what
    scripts/train_ppo.py actually relies on (strict=False, so an
    undertrained policy's out-of-range picks don't crash training). This is
    the first test that actually exercises that fallback path - the
    real-server integration test above only ever samples from the legal
    mask, so it never touched this code.
    """
    zapdos = make_mon("zapdos")
    team = [make_mon("garchomp"), zapdos]
    battle = _battle(
        team, active=None, available_switches=[], valid_orders=[Player.create_order(zapdos)]
    )

    order = MetamonActionSinglesEnv.action_to_order(4, battle, fake=True, strict=False)

    assert order.order.base_species == "zapdos"  # battle's only valid_orders entry


def test_order_to_action_falls_back_to_a_random_legal_action_when_not_strict():
    # A mega-evolve order has no metamon-space equivalent (gen9 OU doesn't
    # have mega evolution) - SinglesEnv.order_to_action itself translates it
    # fine (mega is a real, if gen9-unused, branch of its own 26-way
    # scheme), so this exercises poke_env_action_to_metamon_action's own
    # ValueError specifically, not SinglesEnv's.
    battle = _standard_battle()
    only_valid_order = Player.create_order(battle.active_pokemon.moves["earthquake"])
    battle.valid_orders = [only_valid_order]
    mega_order = Player.create_order(battle.active_pokemon.moves["earthquake"], mega=True)

    action = MetamonActionSinglesEnv.order_to_action(mega_order, battle, fake=True, strict=False)

    assert action == 1  # earthquake, plain move slot - the fallback's only legal option


def _encodable_battle():
    active = _mon_with_moves("garchomp", ["earthquake"])
    zapdos = make_mon("zapdos")
    opp_active = make_mon("dragonite")
    return SimpleNamespace(
        team={mon.species: mon for mon in [active, zapdos]},
        opponent_team={opp_active.species: opp_active},
        active_pokemon=active,
        opponent_active_pokemon=opp_active,
        side_conditions={},
        opponent_side_conditions={},
        weather={},
        fields={},
    )


def test_embed_battle_matches_the_encoding_pipeline_directly():
    """embed_battle should be a pure pass-through to this project's existing
    encode(battle_view_from_poke_env(...)) pipeline - the same one
    win_prob.py/imitation.py already train on - not a reimplementation.
    """
    battle = _encodable_battle()
    env = MetamonActionSinglesEnv(battle_format="gen9ou", start_listening=False, fake=True)

    vector = env.embed_battle(battle)

    np.testing.assert_array_equal(vector, encode(battle_view_from_poke_env(battle)))


class _RewardBattle:
    """reward_computing_helper keys its per-battle state buffer with a
    WeakKeyDictionary, which requires a weak-referenceable key -
    types.SimpleNamespace instances don't support weak references, so
    these calc_reward tests need a plain object instead.
    """

    def __init__(self, team, opponent_team, won=False, lost=False):
        self.team = team
        self.opponent_team = opponent_team
        self.won = won
        self.lost = lost


def test_calc_reward_reflects_the_configured_hp_weight():
    # poke-env's reward_computing_helper defaults hp_value to 0.0 - a
    # nonzero, weight-matching delta here confirms calc_reward actually
    # threads _HP_VALUE through rather than silently using that default.
    env = MetamonActionSinglesEnv(battle_format="gen9ou", start_listening=False, fake=True)
    opp_mon = make_mon("dragonite")
    battle = _RewardBattle(team={}, opponent_team={opp_mon.species: opp_mon})

    env.calc_reward(battle)  # establishes the per-battle reward buffer
    opp_mon._current_hp = round(opp_mon.max_hp * 0.5)  # damage the opponent by ~half
    reward = env.calc_reward(battle)

    # reward_computing_helper's opponent term is -hp_fraction * hp_value, so
    # damaging the opponent from full HP to a lower fraction raises that
    # term by hp_value * (1 - new_fraction) - not an exact 0.5 multiple
    # since max_hp may be odd (rounding to an integer HP value doesn't land
    # on an exact 0.5 fraction).
    assert reward == pytest.approx(_HP_VALUE * (1 - opp_mon.current_hp_fraction))


def test_calc_reward_rewards_opponent_faints_and_penalizes_own_faints():
    env = MetamonActionSinglesEnv(battle_format="gen9ou", start_listening=False, fake=True)
    my_mon = make_mon("garchomp")
    opp_mon = make_mon("dragonite")
    battle = _RewardBattle(
        team={my_mon.species: my_mon}, opponent_team={opp_mon.species: opp_mon}
    )
    env.calc_reward(battle)  # establishes the per-battle reward buffer

    opp_mon._status = Status.FNT
    opp_mon._current_hp = 0
    after_opponent_faints = env.calc_reward(battle)
    assert after_opponent_faints > 0

    my_mon._status = Status.FNT
    my_mon._current_hp = 0
    after_own_faint_too = env.calc_reward(battle)
    assert after_own_faint_too < 0


def test_calc_reward_reflects_victory():
    env = MetamonActionSinglesEnv(battle_format="gen9ou", start_listening=False, fake=True)
    battle = _RewardBattle(team={}, opponent_team={})

    env.calc_reward(battle)  # establishes the per-battle reward buffer
    battle.won = True
    reward = env.calc_reward(battle)

    assert reward == pytest.approx(1.0)  # _VICTORY_VALUE


def test_action_to_order_plain_move():
    battle = _standard_battle()
    order = MetamonActionSinglesEnv.action_to_order(1, battle, fake=True)  # earthquake
    assert order.order.id == "earthquake"
    assert not order.terastallize


def test_action_to_order_tera_move():
    battle = _standard_battle()
    order = MetamonActionSinglesEnv.action_to_order(9 + 1, battle, fake=True)  # earthquake, tera
    assert order.order.id == "earthquake"
    assert order.terastallize


def test_action_to_order_switch():
    battle = _standard_battle()
    order = MetamonActionSinglesEnv.action_to_order(4, battle, fake=True)  # -> alakazam
    assert order.order.base_species == "alakazam"


@pytest.mark.skipif(not _server_running(), reason="local Showdown server not running")
def test_metamon_action_env_plays_real_battle_steps_against_local_server():
    """The only test in this file against a real, connected battle rather
    than a hand-built mock - proves team preview (gen9ou singles auto-picks
    a lead via poke-env's own _EnvPlayer.teampreview, independent of
    SingleAgentWrapper - see rl_env.py's module docstring for why that
    matters for order_to_action), embed_battle, calc_reward, and the action
    translation all actually work together end to end, not just in
    isolation against mocks.
    """
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
    from poke_env.player import RandomPlayer

    from battle_engine.teams import RandomTeamFromPool

    env = MetamonActionSinglesEnv(battle_format="gen9ou", team=RandomTeamFromPool())
    # start_listening=False, no team= - SingleAgentWrapper never actually
    # connects `opponent` to the server or has it play a real match, only
    # calls its choose_move/teampreview as pure decision functions (see
    # scripts/train_ppo.py's build_env docstring, from the review that
    # caught this same pattern leaking a live, unclosed websocket connection
    # for no purpose - this test predated that fix and had the same leak).
    opponent = RandomPlayer(battle_format="gen9ou", start_listening=False)
    wrapped = SingleAgentWrapper(env, opponent)

    obs, info = wrapped.reset()
    assert obs["observation"].shape == (VECTOR_LEN,)
    assert obs["action_mask"].shape == (ACTION_SPACE_SIZE,)
    assert obs["action_mask"].sum() > 0  # at least one legal action at battle start

    terminated = truncated = False
    steps = 0
    rng = random.Random(0)
    # 1050, not 200: an independent review (2026-08-01) measured this test
    # failing 7/40 (17.5%) runs at a 200-step cap - a genuinely uniform-random
    # policy's gen9ou battles regularly run long (this project's own Phase 3
    # diagnosis separately found real multi-dozen-turn stalling switch-loops),
    # and Showdown itself only force-ends a battle via auto-tie at turn 1000
    # (confirmed via the server's own "bigerror" warning messages). 1050 is
    # safely past that hard server-enforced limit, so every real battle
    # deterministically terminates or truncates well before the cap - not
    # just statistically likely to, removing the flakiness rather than just
    # reducing it.
    while not (terminated or truncated) and steps < 1050:
        legal_actions = [i for i, legal in enumerate(obs["action_mask"]) if legal]
        action = rng.choice(legal_actions)
        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert obs["observation"].shape == (VECTOR_LEN,)
        steps += 1

    assert steps > 0
    assert terminated or truncated  # the battle actually reached an end state
    wrapped.close()
