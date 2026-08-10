from types import SimpleNamespace

import pytest
import torch
from poke_env.player.player import Player

from battle_engine.self_play import (
    FrozenPolicyPlayer,
    MixedOpponentPlayer,
    SelfPlaySnapshotCallback,
    build_inference_policy,
)
from conftest import make_mon


def _mon_with_moves(species: str, move_ids: list[str]):
    mon = make_mon(species)
    for move_id in move_ids:
        mon._add_move(move_id)
    return mon


def _full_battle(battle_tag: str = "battle-1"):
    """A battle mock satisfying both battle_view_from_poke_env's needs
    (encoding.py) and MetamonActionSinglesEnv.get_action_mask's needs
    (rl_env.py/action_space.py) - FrozenPolicyPlayer.choose_move exercises
    both in one call, unlike any single existing test helper.
    """
    active = _mon_with_moves("garchomp", ["dragonclaw", "earthquake", "stealthrock", "swordsdance"])
    zapdos = make_mon("zapdos")
    opp_active = make_mon("dragonite")
    team = [active, zapdos]
    known_moves = list(active.moves.values())
    # The full legal-order set for this state (4 plain moves, 4 tera moves,
    # 1 switch - matching the 9 actions get_action_mask marks legal below),
    # since choose_move calls action_to_order with fake=False, which checks
    # membership in battle.valid_orders regardless of which legal action
    # the (masked, so always-legal) policy ends up sampling.
    valid_orders = (
        [Player.create_order(m) for m in known_moves]
        + [Player.create_order(m, terastallize=True) for m in known_moves]
        + [Player.create_order(zapdos)]
    )
    return SimpleNamespace(
        battle_tag=battle_tag,
        player_username="p1",
        team={mon.species: mon for mon in team},
        opponent_team={opp_active.species: opp_active},
        active_pokemon=active,
        opponent_active_pokemon=opp_active,
        side_conditions={},
        opponent_side_conditions={},
        weather={},
        fields={},
        available_switches=[zapdos],
        available_moves=known_moves,
        trapped=False,
        can_tera=True,
        can_mega_evolve=False,
        can_z_move=False,
        can_dynamax=False,
        _wait=False,
        gen=9,
        valid_orders=valid_orders,
        logger=None,
    )


def test_deterministic_flag_defaults_false_and_is_threaded_to_policy_predict():
    # Default False matches training-time self-play's need for response
    # variety across the snapshot pool; load_ppo_player (ppo_eval.py)
    # overrides this to True for evaluating one trained model's actual
    # best-guess behavior instead.
    player = FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False)
    assert player._deterministic is False

    player.add_snapshot(build_inference_policy().state_dict())
    calls = []
    original_predict = player._policy.predict

    def spy_predict(*args, **kwargs):
        calls.append(kwargs.get("deterministic"))
        return original_predict(*args, **kwargs)

    player._policy.predict = spy_predict
    player.choose_move(_full_battle())

    assert calls == [False]


def test_deterministic_true_is_threaded_to_policy_predict():
    player = FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False, deterministic=True)
    player.add_snapshot(build_inference_policy().state_dict())
    calls = []
    original_predict = player._policy.predict

    def spy_predict(*args, **kwargs):
        calls.append(kwargs.get("deterministic"))
        return original_predict(*args, **kwargs)

    player._policy.predict = spy_predict
    player.choose_move(_full_battle())

    assert calls == [True]


def test_choose_move_raises_without_a_seeded_snapshot():
    player = FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False)
    battle = _full_battle()

    with pytest.raises(RuntimeError):
        player.choose_move(battle)


def test_choose_move_returns_a_real_order_once_seeded():
    player = FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False)
    player.add_snapshot(build_inference_policy().state_dict())

    order = player.choose_move(_full_battle())

    assert order is not None


def test_choose_move_resamples_between_battles_but_not_within_one():
    """An earlier version of this test seeded only ONE snapshot, so
    "resampling didn't change anything" was true whether or not resampling
    actually happened between calls - proven vacuous by mutation testing
    (swapping the battle_tag check for an unconditional resample still
    passed it). Seeding two clearly distinguishable snapshots (b is a's
    weights shifted by a constant) makes both directions of the claim
    checkable: repeated calls within the same battle_tag must never change
    the loaded weights, and a real weight change must be observable across
    enough distinct battle_tags for resampling to be proven capable of
    actually picking the other snapshot, not just capable of doing nothing.
    """
    player = FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False, seed=0)
    snapshot_a = build_inference_policy().state_dict()
    snapshot_b = {key: value + 1.0 for key, value in snapshot_a.items()}
    player.add_snapshot(snapshot_a)
    player.add_snapshot(snapshot_b)

    battle = _full_battle(battle_tag="battle-1")
    player.choose_move(battle)
    weights_after_first_call = {k: v.clone() for k, v in player._policy.state_dict().items()}

    for _ in range(5):
        player.choose_move(battle)  # same battle_tag - must not resample
        current = player._policy.state_dict()
        for key in weights_after_first_call:
            torch.testing.assert_close(weights_after_first_call[key], current[key])

    saw_a_different_snapshot_get_picked = False
    for i in range(10):
        player.choose_move(_full_battle(battle_tag=f"battle-{i + 2}"))
        current = player._policy.state_dict()
        if any(not torch.equal(weights_after_first_call[k], current[k]) for k in current):
            saw_a_different_snapshot_get_picked = True
            break
    assert saw_a_different_snapshot_get_picked


def test_add_snapshot_stores_an_independent_copy_not_a_live_view():
    player = FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False)
    source_policy = build_inference_policy()
    player.add_snapshot(source_policy.state_dict())

    with torch.no_grad():
        for param in source_policy.parameters():
            param.add_(1.0)  # mutate the source after snapshotting

    stored = player._snapshots[0]
    for key, tensor in source_policy.state_dict().items():
        assert not torch.equal(tensor, stored[key])


def test_snapshot_pool_evicts_oldest_beyond_max_snapshots():
    player = FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False, max_snapshots=2)
    for _ in range(3):
        player.add_snapshot(build_inference_policy().state_dict())

    assert len(player._snapshots) == 2


class _TaggedPlayer(Player):
    """A minimal Player stub whose choose_move just reports its own
    identity - lets MixedOpponentPlayer tests check *which* sub-player
    handled a battle without depending on FrozenPolicyPlayer/
    TwoPlySearchPlayer's real (and heavier) decision logic.
    """

    def __init__(self, *args, tag: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = tag
        self.calls = 0

    def choose_move(self, battle):
        self.calls += 1
        return self.tag


def test_mixed_opponent_player_requires_at_least_one_sub_player():
    with pytest.raises(ValueError):
        MixedOpponentPlayer(battle_format="gen9ou", start_listening=False, players=[])


def test_mixed_opponent_player_delegates_to_the_only_sub_player():
    sub = _TaggedPlayer(battle_format="gen9ou", start_listening=False, tag="only")
    mixed = MixedOpponentPlayer(
        battle_format="gen9ou", start_listening=False, players=[(1.0, sub)]
    )

    result = mixed.choose_move(_full_battle())

    assert result == "only"
    assert sub.calls == 1


def test_mixed_opponent_player_zero_weight_sub_player_is_never_selected():
    always = _TaggedPlayer(battle_format="gen9ou", start_listening=False, tag="always")
    never = _TaggedPlayer(battle_format="gen9ou", start_listening=False, tag="never")
    mixed = MixedOpponentPlayer(
        battle_format="gen9ou",
        start_listening=False,
        players=[(1.0, always), (0.0, never)],
        seed=0,
    )

    for i in range(20):
        result = mixed.choose_move(_full_battle(battle_tag=f"battle-{i}"))
        assert result == "always"
    assert never.calls == 0


def test_mixed_opponent_player_resamples_between_battles_but_not_within_one():
    """Same mutation-tested shape as FrozenPolicyPlayer's own resampling
    test (test_choose_move_resamples_between_battles_but_not_within_one) -
    seeding two distinguishable, non-degenerate sub-players (both nonzero
    weight) so the assertion actually exercises resampling logic rather than
    passing regardless of whether it fires.
    """
    a = _TaggedPlayer(battle_format="gen9ou", start_listening=False, tag="a")
    b = _TaggedPlayer(battle_format="gen9ou", start_listening=False, tag="b")
    mixed = MixedOpponentPlayer(
        battle_format="gen9ou", start_listening=False, players=[(0.5, a), (0.5, b)], seed=0
    )

    battle = _full_battle(battle_tag="battle-1")
    first_result = mixed.choose_move(battle)
    for _ in range(5):
        assert mixed.choose_move(battle) == first_result  # same battle_tag - must not resample

    saw_the_other_sub_player_get_picked = False
    for i in range(20):
        result = mixed.choose_move(_full_battle(battle_tag=f"battle-{i + 2}"))
        if result != first_result:
            saw_the_other_sub_player_get_picked = True
            break
    assert saw_the_other_sub_player_get_picked


def test_self_play_snapshot_callback_pushes_on_the_configured_interval():
    player = FrozenPolicyPlayer(battle_format="gen9ou", start_listening=False)
    callback = SelfPlaySnapshotCallback(player, snapshot_interval=10)
    callback.model = SimpleNamespace(policy=build_inference_policy())

    callback.num_timesteps = 5
    callback._on_step()
    assert len(player._snapshots) == 0

    callback.num_timesteps = 10
    callback._on_step()
    assert len(player._snapshots) == 1
