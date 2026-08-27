from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from battle_engine import encoding as enc
from battle_engine.encoding import (
    MAX_BENCH,
    VECTOR_LEN,
    BattleView,
    PokemonView,
    _PROTECT_COUNTER_SCALE,
    battle_view_from_poke_env,
    battle_view_from_replay_state,
    battle_views_from_replay,
    encode,
)
from conftest import make_mon


def _battle(
    my_team, my_active, opp_team, opp_active,
    my_hazards=None, opp_hazards=None, weather=None, fields=None,
):
    return SimpleNamespace(
        team={mon.species: mon for mon in my_team},
        opponent_team={mon.species: mon for mon in opp_team},
        active_pokemon=my_active,
        opponent_active_pokemon=opp_active,
        side_conditions=my_hazards or {},
        opponent_side_conditions=opp_hazards or {},
        weather=weather or {},
        fields=fields or {},
    )


def _replay_pokemon(
    name="garchomp", hp_pct=1.0, status="nostatus", types="dragon ground",
    boosts=None, base_stats=None, ability="unknownability", base_species=None,
    item="unknownitem", moves=None, effect="noeffect",
):
    boosts = boosts or {}
    base_stats = base_stats or {
        "hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102,
    }
    return {
        "name": name,
        "base_species": base_species or name,
        "hp_pct": hp_pct,
        "types": types,
        "status": status,
        "ability": ability,
        "item": item,
        "moves": moves or [],
        "effect": effect,
        **{f"{stat}_boost": boosts.get(stat, 0) for stat in
           ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
        **{f"base_{stat}": base_stats[stat] for stat in
           ("hp", "atk", "def", "spa", "spd", "spe")},
    }


def _replay_move(name="nomove", current_pp=0):
    # current_pp must vary between two calls with the same name that are
    # meant to represent two genuinely SEPARATE real uses of that move -
    # real replay data's prev_move dict is a full snapshot (byte-identical
    # across states where nothing happened for that side, see
    # encoding.py's _replay_protect_streaks docstring), so two calls with
    # identical args here look, to that function, exactly like a real
    # replay's "nothing happened" carry-over rather than two distinct uses.
    return {
        "name": name, "move_type": "nomove", "category": "nomove",
        "base_power": 0, "accuracy": 1.0, "priority": 0, "current_pp": current_pp, "max_pp": 16,
    }


def _replay_state(
    player_active, opponent_active, available_switches=None,
    opponents_remaining=6, player_conditions="noconditions",
    opponent_conditions="noconditions", weather="noweather", battle_field="nofield",
    player_prev_move=None, opponent_prev_move=None, can_tera=True,
):
    return {
        "player_active_pokemon": player_active,
        "opponent_active_pokemon": opponent_active,
        "available_switches": available_switches or [],
        "opponents_remaining": opponents_remaining,
        "player_conditions": player_conditions,
        "opponent_conditions": opponent_conditions,
        "weather": weather,
        "battle_field": battle_field,
        "player_prev_move": player_prev_move or _replay_move(),
        "opponent_prev_move": opponent_prev_move or _replay_move(),
        "can_tera": can_tera,
    }


def test_vector_has_fixed_shape():
    mine = make_mon("garchomp")
    theirs = make_mon("dragapult")
    view = battle_view_from_poke_env(_battle([mine], mine, [theirs], theirs))

    assert encode(view).shape == (VECTOR_LEN,)


def test_unknown_pokemon_view_encodes_as_an_all_zero_block():
    battle_view = BattleView(
        my_active=PokemonView.unknown(),
        my_bench=[PokemonView.unknown()] * MAX_BENCH,
        opp_active=PokemonView.unknown(),
        opp_remaining_fraction=1.0,
        my_hazards=set(),
        opp_hazards=set(),
        weather=None,
        terrain=None,
    )

    vec = encode(battle_view)
    my_active_block = vec[: enc._POKEMON_VEC_LEN]
    # my_active (block 0), then MAX_BENCH bench blocks follow immediately.
    first_bench_block = vec[enc._POKEMON_VEC_LEN : 2 * enc._POKEMON_VEC_LEN]

    assert (my_active_block == 0.0).all()  # known=False lands at index 0, also 0
    assert (first_bench_block == 0.0).all()  # padding, not just a real empty slot


def test_hp_fraction_position_reflects_current_hp():
    full_hp = make_mon("garchomp", current_hp_fraction=1.0)
    half_hp = make_mon("garchomp", current_hp_fraction=0.5)
    opp = make_mon("dragapult")

    full_vec = encode(battle_view_from_poke_env(_battle([full_hp], full_hp, [opp], opp)))
    half_vec = encode(battle_view_from_poke_env(_battle([half_hp], half_hp, [opp], opp)))

    # index 0 = "known" flag, index 1 = hp_fraction, for the my_active block.
    # half_hp.current_hp_fraction isn't exactly 0.5: poke-env rounds the
    # requested fraction to a whole HP point first (make_mon's own doing).
    assert full_vec[1] == 1.0
    assert half_vec[1] == half_hp.current_hp_fraction
    assert 0.49 < half_vec[1] < 0.51


def test_fainted_pokemon_sets_fainted_flag_not_status():
    fainted = make_mon("garchomp", current_hp_fraction=0.0, status=Status.FNT)
    opp = make_mon("dragapult")
    view = battle_view_from_poke_env(_battle([fainted], fainted, [opp], opp))

    assert view.my_active.fainted is True
    assert view.my_active.status is None  # FNT isn't a "status" for encoding purposes


def test_bench_padding_marks_missing_slots_unknown():
    mine = make_mon("garchomp")
    bench_mon = make_mon("blissey")
    opp = make_mon("dragapult")
    view = battle_view_from_poke_env(_battle([mine, bench_mon], mine, [opp], opp))

    assert len(view.my_bench) == MAX_BENCH
    assert view.my_bench[0].known is True  # blissey
    assert all(slot.known is False for slot in view.my_bench[1:])


def test_no_active_pokemon_raises_a_clear_error_not_a_bench_overflow_assert():
    # Real gap caught by review: at team preview, poke-env's battle.team
    # already holds all 6 mons but active_pokemon is still None. Before this
    # guard, "bench = everything except active" kept all 6 and tripped
    # _pad_bench's overflow assert with a confusing, unrelated message.
    import pytest

    six_mons = [
        make_mon(s) for s in
        ("garchomp", "dragapult", "tinkaton", "blissey", "excadrill", "pikachu")
    ]
    opp = make_mon("dragapult")
    battle = _battle(six_mons, None, [opp], opp)

    with pytest.raises(ValueError, match="team preview"):
        battle_view_from_poke_env(battle)


def test_hazards_report_only_the_single_most_recent_condition():
    # Matches real replay data (verified in review): the hazard field is
    # single-valued, last-write-wins, not a set of everything active. Spikes
    # here is a stack *count* (2), not a turn - Stealth Rock's turn number
    # (1) is the only real recency signal available, so it wins regardless.
    mine = make_mon("garchomp")
    opp = make_mon("dragapult")
    battle = _battle(
        [mine], mine, [opp], opp,
        my_hazards={SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2},
    )
    view = battle_view_from_poke_env(battle)

    assert view.my_hazards == {"stealthrock"}


def test_hazards_pick_the_later_turn_among_presence_type_conditions():
    mine = make_mon("garchomp")
    opp = make_mon("dragapult")
    battle = _battle(
        [mine], mine, [opp], opp,
        my_hazards={SideCondition.REFLECT: 3, SideCondition.STEALTH_ROCK: 7},
    )
    view = battle_view_from_poke_env(battle)

    assert view.my_hazards == {"stealthrock"}  # turn 7 > turn 3


def test_hazards_fall_back_to_stackable_condition_when_no_presence_type_active():
    mine = make_mon("garchomp")
    opp = make_mon("dragapult")
    battle = _battle(
        [mine], mine, [opp], opp,
        my_hazards={SideCondition.SPIKES: 2},
    )
    view = battle_view_from_poke_env(battle)

    assert view.my_hazards == {"spikes"}


def test_weather_and_terrain_round_trip_from_live_battle():
    mine = make_mon("garchomp")
    opp = make_mon("dragapult")
    battle = _battle(
        [mine], mine, [opp], opp,
        weather={Weather.SANDSTORM: 1},
        fields={Field.GRASSY_TERRAIN: 1},
    )
    view = battle_view_from_poke_env(battle)

    assert view.weather == "sandstorm"
    assert view.terrain == "grassyterrain"


def test_favorable_type_matchup_scores_positive_matchup_dimension():
    # Same fixture as test_evaluation.py's favorable-matchup case: Excadrill
    # (Ground/Steel) is immune to Electric with nothing super-effective back.
    excadrill = make_mon("excadrill")
    pikachu = make_mon("pikachu")
    favorable = encode(battle_view_from_poke_env(_battle([excadrill], excadrill, [pikachu], pikachu)))
    unfavorable = encode(battle_view_from_poke_env(_battle([pikachu], pikachu, [excadrill], excadrill)))

    # matchup score is third-from-last (my_active_hazard_immune/opp_active_hazard_immune
    # were appended after it - see encoding.py's VECTOR_LEN).
    assert favorable[-3] > 0.0
    assert unfavorable[-3] < 0.0


def test_levitate_grants_immunity_to_ground_in_matchup_score():
    # Bronzong (Steel/Psychic) is 2x weak to Ground by raw typing - verified
    # via PokemonType.damage_multiplier before writing this test - and has
    # real ability ambiguity in-game (Levitate/Heatproof/Heavy Metal), so
    # poke-env correctly leaves .ability as None until revealed (confirmed:
    # a single-possible-ability species like Rotom gets auto-filled by
    # poke-env instead, since that's genuinely not ambiguous - a bad choice
    # for isolating "unrevealed vs revealed", caught while writing this test).
    excadrill = make_mon("excadrill")  # Ground/Steel
    bronzong_unrevealed = make_mon("bronzong")
    bronzong_levitate = make_mon("bronzong")
    bronzong_levitate.ability = "levitate"
    assert bronzong_unrevealed.ability is None  # sanity check on the premise above

    unrevealed_vec = encode(
        battle_view_from_poke_env(_battle([excadrill], excadrill, [bronzong_unrevealed], bronzong_unrevealed))
    )
    levitate_vec = encode(
        battle_view_from_poke_env(_battle([excadrill], excadrill, [bronzong_levitate], bronzong_levitate))
    )

    # With Levitate known, Excadrill's Ground-type offense is neutralized
    # (2x -> 0x), so the matchup score should drop relative to the
    # ability-unrevealed case, where the raw type chart alone applies.
    # matchup score is third-from-last (my_active_hazard_immune/opp_active_hazard_immune
    # were appended after it) - [-1] would now also pick up Levitate's
    # separate, real effect on opp_active_hazard_immune, which isn't what this
    # test is about.
    assert levitate_vec[-3] < unrevealed_vec[-3]


def test_wonder_guard_blocks_non_super_effective_hits():
    shedinja = make_mon("shedinja")
    shedinja.ability = "wonderguard"
    garchomp = make_mon("garchomp")  # Dragon/Ground: neutral against Bug/Ghost

    view = battle_view_from_poke_env(_battle([garchomp], garchomp, [shedinja], shedinja))

    from battle_engine.encoding import _type_multiplier
    # Garchomp's types aren't super-effective into Bug/Ghost, so every
    # multiplier should be zeroed out by Wonder Guard.
    for t in garchomp.types:
        if t is not None:
            assert _type_multiplier(t, shedinja.types, "wonderguard") == 0.0

    # But a genuinely super-effective type should pass through unaffected -
    # verified via PokemonType.damage_multiplier before writing this
    # assertion: Fire is 2x into Shedinja's Bug/Ghost typing. Without this
    # check, an implementation that just always returned 0.0 under
    # Wonder Guard would incorrectly pass the assertions above too.
    assert _type_multiplier(PokemonType.FIRE, shedinja.types, "wonderguard") == 2.0


def test_flying_type_is_hazard_immune():
    # Talonflame (Fire/Flying) - immune to Spikes/Toxic Spikes/Sticky Web
    # regardless of ability/item, since Flying itself grants the immunity.
    talonflame = make_mon("talonflame")
    assert enc._is_hazard_immune(battle_view_from_poke_env(
        _battle([talonflame], talonflame, [make_mon("garchomp")], make_mon("garchomp"))
    ).my_active) is True


def test_levitate_ability_is_hazard_immune():
    # Bronzong (Steel/Psychic, no innate Flying) - only immune once Levitate
    # is actually known, same ambiguity rationale as the matchup-score test
    # above (Bronzong's ability is genuinely ambiguous in-game).
    bronzong = make_mon("bronzong")
    bronzong.ability = "levitate"
    view = battle_view_from_poke_env(
        _battle([bronzong], bronzong, [make_mon("garchomp")], make_mon("garchomp"))
    )
    assert enc._is_hazard_immune(view.my_active) is True


def test_heavy_duty_boots_is_hazard_immune():
    # Garchomp (Dragon/Ground, no relevant ability) - would otherwise be
    # squarely hazard-vulnerable; Heavy-Duty Boots overrides that regardless
    # of typing/ability (also the only one of the three immunity sources
    # that blocks Stealth Rock too, not just Spikes/Toxic Spikes/Sticky Web -
    # see _is_hazard_immune's own docstring for why this case was added
    # after the original Flying/Levitate-only version shipped).
    garchomp = make_mon("garchomp")
    garchomp.item = "heavydutyboots"
    view = battle_view_from_poke_env(
        _battle([garchomp], garchomp, [make_mon("dragapult")], make_mon("dragapult"))
    )
    assert enc._is_hazard_immune(view.my_active) is True


def test_ordinary_grounded_type_is_not_hazard_immune():
    garchomp = make_mon("garchomp")  # Dragon/Ground, no relevant ability/item
    view = battle_view_from_poke_env(
        _battle([garchomp], garchomp, [make_mon("dragapult")], make_mon("dragapult"))
    )
    assert enc._is_hazard_immune(view.my_active) is False


def test_is_hazard_immune_defaults_false_when_types_unknown():
    # An information gap (types unknown/empty) must never silently read as a
    # false immunity signal - see _is_hazard_immune's own docstring.
    assert enc._is_hazard_immune(PokemonView.unknown()) is False


def test_hazard_immunity_is_encoded_for_both_active_pokemon():
    # my_active_hazard_immune, opp_active_hazard_immune are the last two
    # dimensions of the vector (see encoding.py's VECTOR_LEN) - Talonflame
    # (Flying, immune) vs. Garchomp (grounded, vulnerable) makes both read
    # distinctly on each side.
    talonflame = make_mon("talonflame")
    garchomp = make_mon("garchomp")
    vec = encode(battle_view_from_poke_env(_battle([talonflame], talonflame, [garchomp], garchomp)))
    assert vec[-2] == 1.0  # my_active_hazard_immune (Talonflame - immune)
    assert vec[-1] == 0.0  # opp_active_hazard_immune (Garchomp - vulnerable)


def test_replay_adapter_maps_unknown_ability_token_to_none():
    mon = _replay_pokemon("dragapult", ability="unknownability")
    revealed = _replay_pokemon("garchomp", ability="roughskin")
    state = _replay_state(revealed, mon)

    view = battle_view_from_replay_state(state)

    assert view.opp_active.ability is None
    assert view.my_active.ability == "roughskin"


def test_replay_adapter_maps_unknown_item_token_to_none():
    mon = _replay_pokemon("dragapult", item="unknownitem")
    revealed = _replay_pokemon("garchomp", item="heavydutyboots")
    no_item = _replay_pokemon("blissey", item="noitem")
    state = _replay_state(revealed, mon, available_switches=[no_item])

    view = battle_view_from_replay_state(state)

    assert view.opp_active.item is None
    assert view.my_active.item == "heavydutyboots"
    assert view.my_bench[0].item is None  # "noitem" folds to None, same as unrevealed


def test_move_summary_detects_recovery_hazard_setup_removal_pivot_priority():
    # One Pokemon whose real moveset exercises every MoveSummary flag at
    # once, checked against Showdown's own movedex (_MOVES_DEX) rather than
    # hand-picked expectations - roost/stealthrock/rapidspin/swordsdance/
    # uturn/suckerpunch are real, stable moves, not synthetic test fixtures.
    mon = _replay_pokemon(
        "garchomp",
        moves=[
            {"name": "roost"},        # flags.heal
            {"name": "stealthrock"},  # sideCondition -> hazard setup
            {"name": "rapidspin"},    # hardcoded hazard-removal set
            {"name": "swordsdance"},  # boosts.atk > 0
        ],
    )
    state = _replay_state(mon, _replay_pokemon("dragapult"))

    view = battle_view_from_replay_state(state)
    moves = view.my_active.moves

    assert moves.has_recovery is True
    assert moves.has_hazard_setup is True
    assert moves.has_hazard_removal is True
    assert moves.has_setup_boost is True
    assert moves.has_pivot is False
    assert moves.has_priority is False
    assert moves.max_base_power == 50  # rapidspin is a 50-BP damaging move, not pure utility

    pivot_and_priority = _replay_pokemon(
        "talonflame",
        moves=[
            {"name": "uturn"},        # selfSwitch
            {"name": "suckerpunch"},  # priority 1
            {"name": "tackle"},       # base_power 40, no flags above
        ],
    )
    state2 = _replay_state(pivot_and_priority, _replay_pokemon("dragapult"))
    moves2 = battle_view_from_replay_state(state2).my_active.moves

    assert moves2.has_pivot is True
    assert moves2.has_priority is True
    assert moves2.has_recovery is False
    assert moves2.max_base_power == 70  # uturn=70, suckerpunch=70, tackle=40 - the max


def test_has_setup_boost_ignores_opponent_targeted_buffs_and_catches_onhit_setup_moves():
    # Regression test for a real bug a review caught (2026-07-31): the
    # original check flagged *any* move with a positive value anywhere in
    # its movedex boosts dict, without checking who the boost applies to.
    # Swagger/Flatter have positive boosts values but target: "normal" -
    # they buff the OPPONENT (while confusing/taunting them), not the user.
    # The Swords Dance case in the test above would have passed even before
    # this fix (it's a genuine self-boost), so it didn't guard against this.
    opponent_buffer = _replay_pokemon("wobbuffet", moves=[{"name": "swagger"}, {"name": "flatter"}])
    state = _replay_state(opponent_buffer, _replay_pokemon("dragapult"))
    assert battle_view_from_replay_state(state).my_active.moves.has_setup_boost is False

    # The other half of the same bug: Belly Drum and Acupressure are real,
    # competitively significant self-buff moves with NO declarative boosts
    # field at all (implemented via onHit simulator logic instead), so the
    # boosts-dict check alone always missed them - _ONHIT_SETUP_MOVES exists
    # specifically to catch these two.
    belly_drummer = _replay_pokemon("azumarill", moves=[{"name": "bellydrum"}])
    state2 = _replay_state(belly_drummer, _replay_pokemon("dragapult"))
    assert battle_view_from_replay_state(state2).my_active.moves.has_setup_boost is True

    acupressure_user = _replay_pokemon("smeargle", moves=[{"name": "acupressure"}])
    state3 = _replay_state(acupressure_user, _replay_pokemon("dragapult"))
    assert battle_view_from_replay_state(state3).my_active.moves.has_setup_boost is True


def test_item_encodes_as_one_hot_with_other_bucket_for_rare_items():
    known_vocab_item = _replay_pokemon("garchomp", item="leftovers")
    rare_item = _replay_pokemon("garchomp", item="mail")  # not in _ITEM_VOCAB
    unrevealed = _replay_pokemon("garchomp", item="unknownitem")

    vocab_vec = enc._item_vector(battle_view_from_replay_state(
        _replay_state(known_vocab_item, _replay_pokemon("dragapult"))
    ).my_active.item)
    rare_vec = enc._item_vector(battle_view_from_replay_state(
        _replay_state(rare_item, _replay_pokemon("dragapult"))
    ).my_active.item)
    unrevealed_vec = enc._item_vector(battle_view_from_replay_state(
        _replay_state(unrevealed, _replay_pokemon("dragapult"))
    ).my_active.item)

    assert vocab_vec.sum() == 1.0
    assert vocab_vec[enc._ITEM_VOCAB.index("leftovers")] == 1.0
    assert rare_vec.sum() == 1.0
    assert rare_vec[-1] == 1.0  # "other known item" bucket
    assert unrevealed_vec.sum() == 0.0


def test_replay_adapter_parses_verified_schema():
    player = _replay_pokemon("garchomp", hp_pct=0.8, types="dragon ground")
    opponent = _replay_pokemon("dragapult", hp_pct=1.0, types="dragon ghost")
    switch = _replay_pokemon("blissey", hp_pct=1.0, types="normal notype")
    state = _replay_state(
        player, opponent, available_switches=[switch], opponents_remaining=4,
        player_conditions="stealthrock spikes", weather="sandstorm", battle_field="grassyterrain",
    )

    view = battle_view_from_replay_state(state)

    assert view.my_active.hp_fraction == 0.8
    assert view.my_active.types == (PokemonType.DRAGON, PokemonType.GROUND)
    assert view.my_bench[0].known is True
    assert view.my_bench[0].types == (PokemonType.NORMAL,)  # "notype" filtered out
    assert view.opp_remaining_fraction == 4 / 6
    assert view.my_hazards == {"stealthrock", "spikes"}
    assert view.weather == "sandstorm"
    assert view.terrain == "grassyterrain"


def test_replay_adapter_parses_leading_notype():
    # test_replay_adapter_parses_verified_schema above only covers trailing
    # "notype" ("normal notype"). Real data puts "notype" first 545/1393
    # times (39%, e.g. "notype water" for Dondozo) - majority-adjacent case,
    # not just an edge case, so it needs its own coverage.
    mon = _replay_pokemon("dondozo", types="notype water")
    state = _replay_state(mon, _replay_pokemon("dragapult"))

    view = battle_view_from_replay_state(state)

    assert view.my_active.types == (PokemonType.WATER,)


def test_replay_adapter_status_and_fainted():
    fainted = _replay_pokemon("garchomp", hp_pct=0.0, status="fnt")
    paralyzed = _replay_pokemon("dragapult", status="par")
    state = _replay_state(fainted, paralyzed)

    view = battle_view_from_replay_state(state)

    assert view.my_active.fainted is True
    assert view.my_active.status is None
    assert view.opp_active.status == Status.PAR


def test_equivalent_state_encodes_the_same_across_both_adapters():
    # A deliberately simple, fully-known position (no boosts, no bench) so
    # the live-battle and replay-state adapters can be driven to describe
    # exactly the same thing and compared value-for-value.
    live_mon = make_mon("garchomp", current_hp_fraction=0.6)
    live_opp = make_mon("dragapult", current_hp_fraction=1.0)
    live_battle = _battle(
        [live_mon], live_mon, [live_opp], live_opp,
        my_hazards={SideCondition.STEALTH_ROCK: 1},
        weather={Weather.SANDSTORM: 1},
    )
    live_vec = encode(battle_view_from_poke_env(live_battle))

    # Source hp_pct from the live mon's own computed fraction, not the
    # requested 0.6: make_mon rounds to a whole HP point first, so the two
    # wouldn't otherwise match exactly.
    replay_state = _replay_state(
        _replay_pokemon(
            "garchomp", hp_pct=live_mon.current_hp_fraction, types="dragon ground",
            base_stats=live_mon.base_stats,
        ),
        _replay_pokemon(
            "dragapult", hp_pct=live_opp.current_hp_fraction, types="dragon ghost",
            base_stats=live_opp.base_stats,
        ),
        player_conditions="stealthrock",
        weather="sandstorm",
    )
    replay_vec = encode(battle_view_from_replay_state(replay_state))

    assert np.allclose(live_vec, replay_vec)


def test_equivalent_state_with_fainted_bench_mon_encodes_the_same_across_both_adapters():
    # Extends the parity check above to exactly the case review found
    # broken before the fix: a fainted bench Pokemon. battle_view_from_replay_state
    # (single-state) can't see this at all - only battle_views_from_replay
    # can, since "fainted" is only inferable by noticing a previously-seen
    # teammate has disappeared from available_switches across turns.
    from battle_engine.encoding import battle_views_from_replay

    live_active = make_mon("garchomp", current_hp_fraction=1.0)
    live_fainted_bench = make_mon("blissey", current_hp_fraction=0.0, status=Status.FNT)
    live_opp = make_mon("dragapult", current_hp_fraction=1.0)
    live_battle = _battle(
        [live_active, live_fainted_bench], live_active, [live_opp], live_opp,
    )
    live_vec = encode(battle_view_from_poke_env(live_battle))

    replay_garchomp = _replay_pokemon(
        "garchomp", hp_pct=1.0, types="dragon ground", base_stats=live_active.base_stats,
    )
    replay_dragapult = _replay_pokemon(
        "dragapult", hp_pct=1.0, types="dragon ghost", base_stats=live_opp.base_stats,
    )
    replay_blissey_alive = _replay_pokemon(
        "blissey", hp_pct=1.0, types="normal notype", base_stats=live_fainted_bench.base_stats,
    )
    # Turn 0: Blissey still alive and switchable. Turn 1: it's gone from
    # available_switches (fainted) - the only signal a replay ever gives.
    state_0 = _replay_state(replay_garchomp, replay_dragapult, available_switches=[replay_blissey_alive])
    state_1 = _replay_state(replay_garchomp, replay_dragapult, available_switches=[])

    replay_views = battle_views_from_replay([state_0, state_1])
    replay_vec = encode(replay_views[1])

    assert np.allclose(live_vec, replay_vec)


def test_battle_views_from_replay_tracks_identity_across_in_battle_form_changes():
    # Real bug caught while building this fix, not in the original review:
    # Metamon renames a Pokemon's "name" field on in-battle form changes
    # (observed in real data: Terapagos -> "terapagosterastal" on
    # Terastallizing, Minior -> "miniormeteor" on its shield-break trigger).
    # Tracking identity by "name" treated one physical teammate as two,
    # overflowing a real replay's bench past MAX_BENCH. base_species stays
    # stable across these (confirmed against the same real data) and is
    # what battle_views_from_replay actually keys its roster on.
    terapagos_normal = _replay_pokemon(
        "terapagos", base_species="terapagos", types="normal", base_stats={
            "hp": 90, "atk": 65, "def": 85, "spa": 65, "spd": 85, "spe": 60,
        },
    )
    terapagos_stellar = _replay_pokemon(
        "terapagosterastal", base_species="terapagos", types="stellar", base_stats={
            "hp": 160, "atk": 105, "def": 110, "spa": 130, "spd": 110, "spe": 85,
        },
    )
    other_active = _replay_pokemon("garchomp")
    state_0 = _replay_state(other_active, _replay_pokemon("dragapult"), available_switches=[terapagos_normal])
    state_1 = _replay_state(other_active, _replay_pokemon("dragapult"), available_switches=[terapagos_stellar])

    from battle_engine.encoding import battle_views_from_replay
    views = battle_views_from_replay([state_0, state_1])

    # Exactly one bench slot used, both states - not two (one stale
    # "terapagos" plus a "new" "terapagosterastal").
    assert sum(1 for slot in views[0].my_bench if slot.known) == 1
    assert sum(1 for slot in views[1].my_bench if slot.known) == 1


def test_encoder_handles_every_state_in_every_real_fetched_replay():
    # Runs the whole pipeline (both adapters, every real downloaded replay)
    # rather than trusting hand-built fixtures alone - this is exactly the
    # kind of check review found missing, and exactly what would have
    # caught the fainted-bench and form-change-identity bugs immediately.
    import glob
    import json
    from pathlib import Path

    import lz4.frame

    from battle_engine.encoding import battle_views_from_replay

    replay_dir = Path("data/replays_raw")
    paths = sorted(glob.glob(str(replay_dir / "*.json.lz4")))
    if not paths:
        import pytest
        pytest.skip("no fetched replay sample at data/replays_raw "
                    "(run scripts/fetch_replay_sample.py first)")

    total_states = 0
    for path in paths:
        with open(path, "rb") as f:
            data = json.loads(lz4.frame.decompress(f.read()))
        for view in battle_views_from_replay(data["states"]):
            vec = encode(view)
            assert vec.shape == (VECTOR_LEN,)
            assert np.isfinite(vec).all()
            total_states += 1
    assert total_states > 0


# --- protect_counter (2026-08-01, Phase 3 win-rate-plateau diagnosis) -------


def test_protect_counter_passes_through_exactly_on_live_adapter():
    # poke-env's own Pokemon.protect_counter is the exact, real mechanic
    # (see module docstring) - the live adapter should read it directly,
    # not reimplement any part of the reset/increment logic itself.
    mine = make_mon("garchomp")
    mine._protect_counter = 2
    theirs = make_mon("dragapult")
    view = battle_view_from_poke_env(_battle([mine], mine, [theirs], theirs))

    assert view.my_active.protect_counter == 2


def test_protect_counter_encodes_as_a_normalized_scalar_and_clamps():
    # _encode_pokemon appends protect_counter as the LAST field of a single
    # Pokemon's block (see encoding.py) - testing that block directly avoids
    # fragile index arithmetic into the full multi-Pokemon concatenated
    # vector encode() returns.
    view_at_0 = PokemonView.unknown()
    view_at_0.known = True  # unknown() zeroes protect_counter too; only care about that field here
    defender = PokemonView.unknown()  # no real matchup needed for this test

    vec_at_0 = enc._encode_pokemon(replace(view_at_0, protect_counter=0), defender)
    vec_at_2 = enc._encode_pokemon(replace(view_at_0, protect_counter=2), defender)
    vec_at_999 = enc._encode_pokemon(replace(view_at_0, protect_counter=999), defender)

    assert vec_at_0[-1] == pytest.approx(0.0)
    assert vec_at_2[-1] == pytest.approx(2.0 / _PROTECT_COUNTER_SCALE)
    assert vec_at_999[-1] == pytest.approx(1.0)  # far beyond any real streak - must clamp


def test_bench_and_single_state_replay_pokemon_always_show_zero_protect_counter():
    # A bench mon's real streak is always 0 (poke-env itself resets on
    # switch-out) - and battle_view_from_replay_state (single-state, no
    # history) has no way to reconstruct a nonzero value at all, same
    # documented gap as fainted-teammate reconstruction.
    zapdos = _replay_pokemon("zapdos")
    state = _replay_state(
        _replay_pokemon("garchomp"), _replay_pokemon("dragapult"),
        available_switches=[zapdos],
        player_prev_move=_replay_move("protect", current_pp=15),
    )
    view = battle_view_from_replay_state(state)

    assert view.my_bench[0].protect_counter == 0
    # Single-state adapter: even the ACTIVE mon's protect_counter can't be
    # reconstructed without history, so it's always 0 here too, regardless
    # of that state's own prev_move.
    assert view.my_active.protect_counter == 0


def test_replay_protect_streak_increments_across_consecutive_same_side_uses():
    # current_pp decreases each real use (16, 15, 14, ...) - real replay
    # data's prev_move dict changes (at minimum, current_pp drops) on every
    # genuine reuse; two states with the byte-identical dict instead mean
    # nothing happened for that side that transition (see
    # _replay_protect_streaks' docstring) and must NOT be read as a second
    # use of the same move.
    active = _replay_pokemon("garchomp")
    opponent = _replay_pokemon("dragapult")
    states = [
        _replay_state(active, opponent, player_prev_move=_replay_move("nomove")),
        _replay_state(active, opponent, player_prev_move=_replay_move("protect", current_pp=15)),
        _replay_state(active, opponent, player_prev_move=_replay_move("protect", current_pp=14)),
    ]

    views = battle_views_from_replay(states)

    assert [v.my_active.protect_counter for v in views] == [0, 1, 2]


def test_replay_protect_streak_carries_forward_unchanged_across_a_phantom_no_action_state():
    """The real bug an independent review found (2026-08-01): Metamon emits
    extra decision states where one side didn't actually act (e.g. the
    opponent choosing a forced switch after a KO) - the prev_move field in
    those states isn't reset to "nomove", it's an exact byte-for-byte copy
    of the last state's prev_move (verified against real replay data: same
    name AND same current_pp). The original version of this function had
    no way to distinguish that from a genuine second consecutive protect
    (both showed the name "protect"), so it kept incrementing an
    already-resolved streak - measured, this made 59.4% of reconstructed
    streak-2 values wrong. The fix: an identical prev_move dict means carry
    the PREVIOUS streak forward as-is, neither incrementing nor resetting.
    """
    active = _replay_pokemon("garchomp")
    opponent = _replay_pokemon("dragapult")
    protect_at_15 = _replay_move("protect", current_pp=15)
    states = [
        _replay_state(active, opponent, player_prev_move=_replay_move("nomove")),
        _replay_state(active, opponent, player_prev_move=protect_at_15),
        # Phantom state: byte-identical prev_move dict, nothing really
        # happened for this side - must NOT be read as a second protect.
        _replay_state(active, opponent, player_prev_move=dict(protect_at_15)),
        # A genuine second use afterwards - pp actually drops now.
        _replay_state(active, opponent, player_prev_move=_replay_move("protect", current_pp=14)),
    ]

    views = battle_views_from_replay(states)

    assert [v.my_active.protect_counter for v in views] == [0, 1, 1, 2]


def test_replay_protect_streak_resets_on_a_different_move():
    # A leading "nomove" state models the real, always-present turn-1 state
    # (see _replay_protect_streaks' docstring: the very first state of any
    # real replay has no prior action to compare against, so its own streak
    # is unconditionally 0 regardless of what its own prev_move claims -
    # untestable/meaningless as a "does this move count" case on its own,
    # since real data never actually puts a protect-counter move there).
    active = _replay_pokemon("garchomp")
    opponent = _replay_pokemon("dragapult")
    states = [
        _replay_state(active, opponent, player_prev_move=_replay_move("nomove")),
        _replay_state(active, opponent, player_prev_move=_replay_move("protect", current_pp=15)),
        _replay_state(active, opponent, player_prev_move=_replay_move("earthquake", current_pp=7)),
        _replay_state(active, opponent, player_prev_move=_replay_move("protect", current_pp=14)),
    ]

    views = battle_views_from_replay(states)

    assert [v.my_active.protect_counter for v in views] == [0, 1, 0, 1]


def test_replay_protect_streak_resets_when_the_active_species_changes():
    garchomp = _replay_pokemon("garchomp")
    zapdos = _replay_pokemon("zapdos")
    opponent = _replay_pokemon("dragapult")
    states = [
        _replay_state(garchomp, opponent, player_prev_move=_replay_move("nomove")),
        _replay_state(garchomp, opponent, player_prev_move=_replay_move("protect", current_pp=15)),
        _replay_state(garchomp, opponent, player_prev_move=_replay_move("protect", current_pp=14)),
        # A switch - real replay data shows the PRIOR mon's prev_move
        # frozen/carried over here just as often as "nomove" (see
        # _replay_protect_streaks' docstring - the field is a per-side
        # "last real move used", not turn-scoped), but the reset is keyed
        # on species identity changing, not on prev_move's value at all -
        # this deliberately reuses the exact same dict to prove that.
        _replay_state(zapdos, opponent, available_switches=[garchomp],
                      player_prev_move=_replay_move("protect", current_pp=14)),
    ]

    views = battle_views_from_replay(states)

    assert [v.my_active.protect_counter for v in views] == [0, 1, 2, 0]


def test_replay_protect_streak_tracks_opponent_side_independently():
    active = _replay_pokemon("garchomp")
    opponent = _replay_pokemon("dragapult")
    states = [
        _replay_state(active, opponent, player_prev_move=_replay_move("nomove"),
                      opponent_prev_move=_replay_move("nomove")),
        _replay_state(active, opponent, player_prev_move=_replay_move("earthquake", current_pp=7),
                      opponent_prev_move=_replay_move("protect", current_pp=15)),
        _replay_state(active, opponent, player_prev_move=_replay_move("protect", current_pp=15),
                      opponent_prev_move=_replay_move("protect", current_pp=14)),
    ]

    views = battle_views_from_replay(states)

    assert [v.my_active.protect_counter for v in views] == [0, 0, 1]
    assert [v.opp_active.protect_counter for v in views] == [0, 1, 2]


# --- MoveView / per-move-slot type effectiveness (2026-08-26, encoding rewrite Phase 1) ---
#
# The real diagnosed bug this phase exists to fix: a trained policy using
# Draco Meteor into Clefable (immune, Fairy-type) four turns straight,
# because nothing in the vector said "this specific move, right now, does
# nothing" - only a moveset-wide type-coverage aggregate. These tests go
# straight through the new MoveView pipeline (_move_view -> _move_slot_vector),
# not just the underlying _type_multiplier primitive (already covered by
# test_wonder_guard_blocks_non_super_effective_hits and the matchup-score
# tests above).

# _move_slot_vector's layout is [type one-hot][category one-hot]
# [secondary-kind one-hot][20 scalars], scalars in this exact order: known,
# stab, base_power, accuracy, priority, targets_opponent, effectiveness,
# secondary_chance, self_boost_chance, self_boost_magnitude, fixed_damage,
# multi_hit, is_contact, is_sound, is_punch, is_bite, is_pulse, is_bullet,
# is_wind, is_protect_counter (see encoding.py's _move_slot_vector).
_SCALARS_OFFSET = len(enc._ALL_TYPES) + len(enc._MOVE_CATEGORIES) + len(enc._SECONDARY_KINDS)
_STAB_IDX = _SCALARS_OFFSET + 1
_EFFECTIVENESS_IDX = _SCALARS_OFFSET + 6


def _move_slot(
    move_id, user_types, defender_types, defending_ability=None, defending_item=None,
    weather=None, terrain=None, user_grounded=True,
):
    move = enc._move_view(move_id)
    assert move is not None, f"{move_id!r} not found in _MOVES_DEX - fix the test fixture"
    return enc._move_slot_vector(
        move, user_types, defender_types, defending_ability, defending_item,
        weather=weather, terrain=terrain, user_grounded=user_grounded,
    )


def test_DW_1_1_fighting_move_is_immune_against_pure_ghost_type():
    dusclops = make_mon("dusclops")  # pure Ghost
    vec = _move_slot("closecombat", (PokemonType.FIGHTING,), dusclops.types)
    assert vec[_EFFECTIVENESS_IDX] == 0.0


def test_DW_1_1_water_move_is_immune_against_water_absorb_holder():
    quagsire = make_mon("quagsire")  # Water/Ground - ambiguous ability, forced known here
    vec = _move_slot(
        "scald", (PokemonType.WATER,), quagsire.types, defending_ability="waterabsorb"
    )
    assert vec[_EFFECTIVENESS_IDX] == 0.0

    # Sanity check on the premise: without Water Absorb known, Scald into a
    # Water/Ground mon is NOT immune (Water resists itself at 0.5x, but
    # Ground is weak to Water at 2x - the two cancel to a neutral 1.0
    # combined, verified via PokemonType.damage_multiplier directly before
    # writing this assertion) - the ability is what's doing the work above.
    vec_no_ability = _move_slot("scald", (PokemonType.WATER,), quagsire.types)
    assert vec_no_ability[_EFFECTIVENESS_IDX] == pytest.approx(1.0)


def test_DW_1_1_ground_move_is_immune_against_air_balloon_holder_while_held():
    garchomp = make_mon("garchomp")  # Dragon/Ground - ordinarily hit normally by Ground
    vec = _move_slot(
        "earthquake", (PokemonType.GROUND,), garchomp.types, defending_item="airballoon"
    )
    assert vec[_EFFECTIVENESS_IDX] == 0.0

    # Sanity check: without the balloon, Earthquake into Garchomp is NOT
    # immune (Ground vs Dragon/Ground is neutral - the item is doing the work).
    vec_no_item = _move_slot("earthquake", (PokemonType.GROUND,), garchomp.types)
    assert vec_no_item[_EFFECTIVENESS_IDX] == pytest.approx(1.0)


def test_DW_1_1_wonder_guard_blocks_a_resisted_not_just_immune_hit():
    # Milotic (pure Water) resists Water at 0.5x - not immune, not the
    # "usual" type Wonder Guard is associated with (it isn't tied to any
    # single type at all) - Wonder Guard blocks ANY non-super-effective hit,
    # which this proves by using a genuinely-resisted matchup, not an
    # already-immune one.
    milotic = make_mon("milotic")
    vec = _move_slot(
        "scald", (PokemonType.WATER,), milotic.types, defending_ability="wonderguard"
    )
    assert vec[_EFFECTIVENESS_IDX] == 0.0


def test_DW_1_1_stab_flagged_only_when_move_type_is_in_the_users_own_types():
    # Same move ("flamethrower", Fire-type), same defender - only the
    # user's own types change, isolating STAB from every other factor
    # (effectiveness, accuracy, ...).
    dusclops = make_mon("dusclops")
    fire_user_stab = _move_slot("flamethrower", (PokemonType.FIRE,), dusclops.types)
    non_fire_user_no_stab = _move_slot(
        "flamethrower", (PokemonType.WATER, PokemonType.GROUND), dusclops.types
    )
    assert fire_user_stab[_STAB_IDX] == 1.0
    assert non_fire_user_no_stab[_STAB_IDX] == 0.0


def test_move_slot_effectiveness_defaults_to_zero_for_a_non_opponent_directed_move():
    # Stealth Rock (target: "foeSide", not a specific Pokemon) shouldn't get
    # a misleading per-move multiplier - see this phase's own Edge Cases
    # note. targets_opponent itself should read False, distinguishing this
    # from a real computed 0.0 (immune).
    dusclops = make_mon("dusclops")
    vec = _move_slot("stealthrock", (PokemonType.ROCK,), dusclops.types)
    targets_opponent_idx = _SCALARS_OFFSET + 5
    assert vec[targets_opponent_idx] == 0.0
    assert vec[_EFFECTIVENESS_IDX] == 0.0


def test_ring_target_cancels_only_the_type_chart_immunity_not_ability_immunity():
    # Real mechanic verified against Showdown's own current sim source (see
    # module docstring) - Ring Target cancels a TYPE-CHART 0x (Gengar's
    # Ghost typing blocking Normal) but never an ability-granted one
    # (Levitate blocking Ground stays blocked even with Ring Target held).
    gengar = make_mon("gengar")  # Ghost/Poison - immune to Normal via Ghost typing alone
    normal_blocked = enc._type_multiplier(PokemonType.NORMAL, gengar.types)
    normal_with_ring_target = enc._type_multiplier(
        PokemonType.NORMAL, gengar.types, defending_item="ringtarget"
    )
    assert normal_blocked == 0.0
    assert normal_with_ring_target == pytest.approx(1.0)  # Poison's own neutral response to Normal

    bronzong = make_mon("bronzong")
    ground_blocked_by_levitate = enc._type_multiplier(
        PokemonType.GROUND, bronzong.types, defending_ability="levitate"
    )
    ground_with_ring_target_and_levitate = enc._type_multiplier(
        PokemonType.GROUND, bronzong.types, defending_ability="levitate", defending_item="ringtarget"
    )
    assert ground_blocked_by_levitate == 0.0
    assert ground_with_ring_target_and_levitate == 0.0  # Ring Target doesn't touch ability immunity


def test_move_slots_are_sorted_by_move_id_not_reveal_order():
    # Move-slot identity must be stable turn to turn for a partially-revealed
    # opponent - see this phase's own Edge Cases note (same "sort bench by
    # species name" reasoning already established in this module).
    revealed_first = enc._move_views(["earthquake", "closecombat"])
    revealed_second = enc._move_views(["closecombat", "earthquake"])
    assert [m.move_id for m in revealed_first if m.known] == \
        [m.move_id for m in revealed_second if m.known] == \
        sorted(["earthquake", "closecombat"])


def test_move_slots_pad_missing_slots_as_unknown():
    views = enc._move_views(["earthquake"])
    assert len(views) == enc.MAX_MOVES
    assert views[0].known is True
    assert all(v.known is False for v in views[1:])
    # An unknown slot's full feature block must be all-zero, matching this
    # module's existing "unknown/zero" padding convention (PokemonView.unknown()).
    unknown_vec = enc._move_slot_vector(views[1], (), (), None, None)
    assert (unknown_vec == 0.0).all()


def test_DW_1_4_vector_len_is_the_exact_expected_value():
    # A future accidental size change (e.g. a reordered/miscounted per-move
    # field) must be caught immediately, not just implicitly via a shape
    # check - MAX_MOVES(4) * _MOVE_VEC_LEN(46) per Pokemon-slot, 7
    # Pokemon-slots (1 my_active + 5 bench + 1 opp_active) added on top of
    # the pre-Phase-1 665.
    #
    # Phase 2 (DW-2.3): _MOVE_VEC_LEN 46 -> 50 (4 new per-move scalars:
    # bypasses_protect, recoil_fraction, drain_fraction, is_self_ko), plus
    # 3 new per-Pokemon-slot scalars (preparing, semi_invulnerable,
    # must_recharge) across all 7 Pokemon-slots - 1953 + 7*(4*4 + 3) = 2086.
    #
    # Phase 3 (DW-3.3): _MOVE_VEC_LEN 50 -> 51 (1 new per-move scalar,
    # needs_charge_turn), 7 Pokemon-slots * 4 moves * 1 = 28, plus 6 new
    # GLOBAL (not per-Pokemon-slot) scalars - my/opp_active_speed_doubled,
    # my/opp_terrain_sleep_immune, my/opp_terrain_status_immune - added once,
    # not per-slot: 2086 + 7*4*1 + 6 = 2120.
    #
    # Phase 4 (DW-4.3): _MOVE_VEC_LEN unchanged at 51 (no new per-move
    # field this phase) - +4 new per-Pokemon-slot scalars (toxic_counter,
    # has_leech_seed, has_substitute, is_confused) across all 7
    # Pokemon-slots = +28, +1 new hazard token (safeguard) across both sides
    # = +2, +6 new GLOBAL scalars (my/opp_spikes_layers,
    # my/opp_toxic_spikes_layers, my/opp_used_tera) added once, not per-slot:
    # 2120 + 7*4 + 2 + 6 = 2156.
    assert enc._MOVE_VEC_LEN == 51
    assert len(enc._HAZARD_TOKENS) == 9
    assert VECTOR_LEN == 2156


def test_move_view_reads_secondary_effect_chance_and_kind():
    scald = enc._move_view("scald")  # 30% burn
    assert scald.secondary_chance == pytest.approx(0.30)
    assert scald.secondary_kind == "status"

    ironhead = enc._move_view("ironhead")  # 30% flinch
    assert ironhead.secondary_kind == "flinch"

    moonblast = enc._move_view("moonblast")  # 30% target spa drop
    assert moonblast.secondary_kind == "boost_drop"

    tackle = enc._move_view("tackle")  # no secondary at all
    assert tackle.secondary_chance == 0.0
    assert tackle.secondary_kind is None


def test_move_view_reads_unconditional_self_boost_not_chance_based_or_top_level():
    draco_meteor = enc._move_view("dracometeor")  # self.boosts = {spa: -2}, unconditional
    assert draco_meteor.self_boost_chance == 1.0
    assert draco_meteor.self_boost_magnitude == pytest.approx(-2.0 / 6.0)

    # Swords Dance's boost is top-level `boosts` (already MoveSummary's
    # has_setup_boost), not movedex `self.boosts` - deliberately not covered
    # by self_boost_* (see MoveView's own docstring).
    swords_dance = enc._move_view("swordsdance")
    assert swords_dance.self_boost_chance == 0.0

    # Steel Wing's self-boost is chance-based, nested under
    # secondary.self.boosts - also deliberately not covered.
    steel_wing = enc._move_view("steelwing")
    assert steel_wing.self_boost_chance == 0.0


def test_move_view_reads_category_flags_from_movedex_flags():
    mach_punch = enc._move_view("machpunch")
    assert mach_punch.is_contact is True
    assert mach_punch.is_punch is True
    assert mach_punch.is_sound is False

    hyper_voice = enc._move_view("hypervoice")
    assert hyper_voice.is_sound is True
    assert hyper_voice.is_contact is False

    crunch = enc._move_view("crunch")
    assert crunch.is_bite is True

    bulletseed = enc._move_view("bulletseed")
    assert bulletseed.is_bullet is True
    assert bulletseed.multi_hit is True


def test_move_view_reads_fixed_damage_and_protect_family():
    seismic_toss = enc._move_view("seismictoss")
    assert seismic_toss.fixed_damage is True
    assert seismic_toss.base_power == 0  # damage isn't basePower - see damage.py's own precedent

    tackle = enc._move_view("tackle")
    assert tackle.fixed_damage is False

    protect = enc._move_view("protect")
    assert protect.is_protect_counter is True
    wide_guard = enc._move_view("wideguard")
    assert wide_guard.is_protect_counter is True
    mat_block = enc._move_view("matblock")  # deliberately excluded, matches poke-env's own list
    assert mat_block.is_protect_counter is False


def test_move_view_accuracy_normalizes_always_hits_to_one():
    aerial_ace = enc._move_view("aerialace")  # accuracy: True in the real dex entry
    assert aerial_ace.accuracy == 1.0

    focus_blast = enc._move_view("focusblast")  # accuracy: 70 (a real percent)
    assert focus_blast.accuracy == pytest.approx(0.70)


def test_move_view_returns_none_for_an_unrecognized_move_id():
    # Same typo-guard convention as _move_summary_features - skip silently
    # rather than crash on a bad/garbled move id.
    assert enc._move_view("thisisnotarealmove") is None


def test_move_slots_present_on_both_live_and_replay_adapters():
    live_view = battle_view_from_poke_env(
        _battle([make_mon("garchomp")], make_mon("garchomp"), [make_mon("dragapult")], make_mon("dragapult"))
    )
    assert any(m.known for m in live_view.my_active.move_slots) is False  # make_mon has no real moveset


# --- Phase 2: protect-family, charge/semi-invulnerable, recharge/recoil/
# drain/self-KO ---------------------------------------------------------
#
# Real gap this phase exists to fix (see module docstring): nothing in the
# vector said whether a specific move can even be blocked by Protect,
# whether an active Pokemon is untouchable this turn behind a charge move's
# invulnerability, or whether it's locked into recharging - all real,
# common reasons a move that looks good on paper is actually a bad choice
# right now.

# _move_slot_vector's scalar block gained 4 new fields at the end this
# phase (bypasses_protect, recoil_fraction, drain_fraction, is_self_ko) -
# same _SCALARS_OFFSET-relative-index pattern Phase 1 established for
# _STAB_IDX/_EFFECTIVENESS_IDX, so these track any future layout change
# automatically rather than hardcoding an absolute position.
_BYPASSES_PROTECT_IDX = _SCALARS_OFFSET + 20
_RECOIL_FRACTION_IDX = _SCALARS_OFFSET + 21
_DRAIN_FRACTION_IDX = _SCALARS_OFFSET + 22
_IS_SELF_KO_IDX = _SCALARS_OFFSET + 23


def test_move_view_reads_bypasses_protect():
    feint = enc._move_view("feint")  # real dex flags lack "protect" entirely
    assert feint.bypasses_protect is True

    tackle = enc._move_view("tackle")  # real dex flags include protect: 1
    assert tackle.bypasses_protect is False


def test_move_view_reads_recoil_and_drain_fraction():
    flare_blitz = enc._move_view("flareblitz")  # real dex recoil: [33, 100]
    assert flare_blitz.recoil_fraction == pytest.approx(0.33)
    assert flare_blitz.drain_fraction == 0.0

    giga_drain = enc._move_view("gigadrain")  # real dex drain: [1, 2]
    assert giga_drain.drain_fraction == pytest.approx(0.5)
    assert giga_drain.recoil_fraction == 0.0

    tackle = enc._move_view("tackle")  # neither field present
    assert tackle.recoil_fraction == 0.0
    assert tackle.drain_fraction == 0.0


def test_move_view_reads_is_self_ko():
    explosion = enc._move_view("explosion")  # real dex selfdestruct: "always"
    assert explosion.is_self_ko is True

    memento = enc._move_view("memento")  # real dex selfdestruct: "ifHit", not "always"
    assert memento.is_self_ko is True

    tackle = enc._move_view("tackle")  # no selfdestruct field at all
    assert tackle.is_self_ko is False


def test_move_slot_vector_encodes_bypasses_protect_recoil_drain_self_ko():
    dusclops = make_mon("dusclops")
    feint_vec = _move_slot("feint", (PokemonType.NORMAL,), dusclops.types)
    tackle_vec = _move_slot("tackle", (PokemonType.NORMAL,), dusclops.types)
    assert feint_vec[_BYPASSES_PROTECT_IDX] == 1.0
    assert tackle_vec[_BYPASSES_PROTECT_IDX] == 0.0

    flare_blitz_vec = _move_slot("flareblitz", (PokemonType.FIRE,), dusclops.types)
    assert flare_blitz_vec[_RECOIL_FRACTION_IDX] == pytest.approx(0.33)
    assert flare_blitz_vec[_DRAIN_FRACTION_IDX] == 0.0

    giga_drain_vec = _move_slot("gigadrain", (PokemonType.GRASS,), dusclops.types)
    assert giga_drain_vec[_DRAIN_FRACTION_IDX] == pytest.approx(0.5)
    assert giga_drain_vec[_RECOIL_FRACTION_IDX] == 0.0

    explosion_vec = _move_slot("explosion", (PokemonType.NORMAL,), dusclops.types)
    assert explosion_vec[_IS_SELF_KO_IDX] == 1.0
    assert tackle_vec[_IS_SELF_KO_IDX] == 0.0


def test_DW_2_1_semi_invulnerable_charge_is_distinguishable_from_merely_charging():
    # The real diagnosed gap: a Pokemon mid-Fly is untouchable by most moves
    # this turn, a Pokemon mid-Solar-Beam-charge is NOT - two states that
    # both have preparing=True but need to read differently to a model.
    flying = make_mon("dragonite")
    flying._preparing_move = Move("fly", gen=9)
    charging = make_mon("dragonite")
    charging._preparing_move = Move("solarbeam", gen=9)
    opp = make_mon("dragapult")

    fly_view = battle_view_from_poke_env(_battle([flying], flying, [opp], opp))
    solar_view = battle_view_from_poke_env(_battle([charging], charging, [opp], opp))

    assert fly_view.my_active.preparing is True
    assert fly_view.my_active.semi_invulnerable is True
    assert solar_view.my_active.preparing is True
    assert solar_view.my_active.semi_invulnerable is False  # the actual distinguishing bit

    # Distinguishable in the ENCODED vector, not just the PokemonView -
    # preparing/semi_invulnerable/must_recharge sit at [-4]/[-3]/[-2] of a
    # single Pokemon's block (protect_counter stays last, at [-1] - see
    # module docstring).
    fly_vec = enc._encode_pokemon(fly_view.my_active, defender=fly_view.opp_active)
    solar_vec = enc._encode_pokemon(solar_view.my_active, defender=solar_view.opp_active)
    assert fly_vec[-4] == 1.0 and fly_vec[-3] == 1.0  # preparing, semi_invulnerable
    assert solar_vec[-4] == 1.0 and solar_vec[-3] == 0.0
    assert not np.array_equal(fly_vec, solar_vec)


def test_DW_2_2_must_recharge_is_read_from_the_live_adapter():
    recharging = make_mon("dragonite")
    recharging.must_recharge = True
    opp = make_mon("dragapult")
    view = battle_view_from_poke_env(_battle([recharging], recharging, [opp], opp))

    assert view.my_active.must_recharge is True
    vec = enc._encode_pokemon(view.my_active, defender=view.opp_active)
    assert vec[-2] == 1.0  # must_recharge's position (see module docstring)

    # Sanity: a Pokemon that hasn't just used a recharge move reads False -
    # isolates that must_recharge is really being read, not defaulted True.
    rested = make_mon("dragonite")
    rested_view = battle_view_from_poke_env(_battle([rested], rested, [opp], opp))
    assert rested_view.my_active.must_recharge is False
    rested_vec = enc._encode_pokemon(rested_view.my_active, defender=rested_view.opp_active)
    assert rested_vec[-2] == 0.0


def test_charge_and_recharge_state_defaults_false_on_replay_adapter():
    # Documented live-only gap (see module docstring for the real-replay-
    # sample verification: no equivalent field exists anywhere in
    # Metamon's schema, checked directly, not assumed).
    mon = _replay_pokemon("dragonite")
    state = _replay_state(mon, _replay_pokemon("dragapult"))

    single_state_view = battle_view_from_replay_state(state)
    assert single_state_view.my_active.preparing is False
    assert single_state_view.my_active.semi_invulnerable is False
    assert single_state_view.my_active.must_recharge is False

    multi_state_views = battle_views_from_replay([state])
    assert multi_state_views[0].my_active.preparing is False
    assert multi_state_views[0].my_active.semi_invulnerable is False
    assert multi_state_views[0].my_active.must_recharge is False


def test_charge_and_recharge_state_defaults_false_for_bench_and_unknown_pokemon():
    unknown = PokemonView.unknown()
    assert unknown.preparing is False
    assert unknown.semi_invulnerable is False
    assert unknown.must_recharge is False

    mine = make_mon("garchomp")
    mine._preparing_move = Move("fly", gen=9)
    bench_mon = make_mon("blissey")
    opp = make_mon("dragapult")
    view = battle_view_from_poke_env(_battle([mine, bench_mon], mine, [opp], opp))

    assert view.my_active.preparing is True  # the active mon really is mid-charge
    assert view.my_bench[0].preparing is False  # a benched mon structurally can't be
    assert view.my_bench[0].semi_invulnerable is False
    assert view.my_bench[0].must_recharge is False


def test_poke_env_semi_invulnerable_is_false_when_not_preparing_any_move():
    resting = make_mon("dragonite")  # preparing_move is None
    assert enc._poke_env_semi_invulnerable(resting) is False

    mon = _replay_pokemon("garchomp", moves=[{"name": "earthquake"}, {"name": "dragonclaw"}])
    state = _replay_state(mon, _replay_pokemon("dragapult"))
    replay_view = battle_view_from_replay_state(state)
    known_ids = sorted(m.move_id for m in replay_view.my_active.move_slots if m.known)
    assert known_ids == sorted(["earthquake", "dragonclaw"])


# --- Phase 3 (2026-08-26): weather/terrain-conditional move behavior -------
#
# The real diagnosed gap this phase exists to fix (see module docstring):
# nothing in the vector cross-referenced individual moves/abilities against
# the battle's actual current weather/terrain - Solar Beam always looked
# like a charge move even in sun, Swift Swim's speed doubling was invisible,
# etc. Every hardcoded table below was verified against the real local
# pokemon-showdown/data/moves.ts and data/abilities.ts checkout (see module
# docstring for the exact reads), not assumed.

_NEEDS_CHARGE_TURN_IDX = _SCALARS_OFFSET + 24
_ACCURACY_IDX = _SCALARS_OFFSET + 3
_BASE_POWER_IDX = _SCALARS_OFFSET + 2


def test_DW_3_1_solar_beam_skips_charge_turn_in_sun_not_otherwise():
    dusclops = make_mon("dusclops")
    sun_vec = _move_slot("solarbeam", (PokemonType.GRASS,), dusclops.types, weather="sunnyday")
    no_weather_vec = _move_slot("solarbeam", (PokemonType.GRASS,), dusclops.types)
    rain_vec = _move_slot("solarbeam", (PokemonType.GRASS,), dusclops.types, weather="raindance")

    assert sun_vec[_NEEDS_CHARGE_TURN_IDX] == 0.0  # skips the charge turn in sun
    assert no_weather_vec[_NEEDS_CHARGE_TURN_IDX] == 1.0
    assert rain_vec[_NEEDS_CHARGE_TURN_IDX] == 1.0  # only sun exempts it, not every weather

    # Solar Blade shares the exact same real onTryMove sun-skip logic.
    solar_blade_sun = _move_slot("solarblade", (PokemonType.GRASS,), dusclops.types, weather="sunnyday")
    assert solar_blade_sun[_NEEDS_CHARGE_TURN_IDX] == 0.0


def test_needs_charge_turn_has_no_sun_exemption_for_other_charge_moves():
    # Sky Attack is a real charge move but was never verified to skip in
    # sun (see module docstring's desolateland grep) - it must always still
    # need its charge turn, distinguishing it from Solar Beam/Solar Blade.
    dusclops = make_mon("dusclops")
    vec = _move_slot("skyattack", (PokemonType.FLYING,), dusclops.types, weather="sunnyday")
    assert vec[_NEEDS_CHARGE_TURN_IDX] == 1.0


def test_needs_charge_turn_is_zero_for_a_non_charge_move():
    dusclops = make_mon("dusclops")
    vec = _move_slot("tackle", (PokemonType.NORMAL,), dusclops.types, weather="sunnyday")
    assert vec[_NEEDS_CHARGE_TURN_IDX] == 0.0


def test_weather_ball_type_and_power_change_with_real_weather():
    dusclops = make_mon("dusclops")
    normal_idx = enc._ALL_TYPES.index(PokemonType.NORMAL)
    fire_idx = enc._ALL_TYPES.index(PokemonType.FIRE)
    water_idx = enc._ALL_TYPES.index(PokemonType.WATER)
    rock_idx = enc._ALL_TYPES.index(PokemonType.ROCK)
    ice_idx = enc._ALL_TYPES.index(PokemonType.ICE)

    base_vec = _move_slot("weatherball", (PokemonType.NORMAL,), dusclops.types)
    assert base_vec[normal_idx] == 1.0  # no weather - stays its real dex type, Normal
    assert base_vec[_BASE_POWER_IDX] == pytest.approx(50 / enc._MAX_BASE_POWER_SCALE)

    sun_vec = _move_slot("weatherball", (PokemonType.NORMAL,), dusclops.types, weather="sunnyday")
    assert sun_vec[fire_idx] == 1.0 and sun_vec[normal_idx] == 0.0
    assert sun_vec[_BASE_POWER_IDX] == pytest.approx(100 / enc._MAX_BASE_POWER_SCALE)  # doubled

    rain_vec = _move_slot("weatherball", (PokemonType.NORMAL,), dusclops.types, weather="raindance")
    assert rain_vec[water_idx] == 1.0
    assert rain_vec[_BASE_POWER_IDX] == pytest.approx(100 / enc._MAX_BASE_POWER_SCALE)

    sand_vec = _move_slot("weatherball", (PokemonType.NORMAL,), dusclops.types, weather="sandstorm")
    assert sand_vec[rock_idx] == 1.0

    snow_vec = _move_slot("weatherball", (PokemonType.NORMAL,), dusclops.types, weather="snow")
    assert snow_vec[ice_idx] == 1.0


def test_weather_ball_effectiveness_uses_its_weather_dependent_type_not_its_dex_type():
    torkoal = make_mon("torkoal")  # pure Fire - resists Fire (0.5x), weak to Water (2x)
    sun_vec = _move_slot("weatherball", (PokemonType.NORMAL,), torkoal.types, weather="sunnyday")
    rain_vec = _move_slot("weatherball", (PokemonType.NORMAL,), torkoal.types, weather="raindance")
    assert sun_vec[_EFFECTIVENESS_IDX] == pytest.approx(0.5)  # Fire vs Fire
    assert rain_vec[_EFFECTIVENESS_IDX] == pytest.approx(2.0)  # Water vs Fire


def test_thunder_and_hurricane_accuracy_change_with_weather():
    dusclops = make_mon("dusclops")
    for move_id, move_type in (("thunder", PokemonType.ELECTRIC), ("hurricane", PokemonType.FLYING)):
        base_vec = _move_slot(move_id, (move_type,), dusclops.types)
        rain_vec = _move_slot(move_id, (move_type,), dusclops.types, weather="raindance")
        sun_vec = _move_slot(move_id, (move_type,), dusclops.types, weather="sunnyday")
        assert base_vec[_ACCURACY_IDX] == pytest.approx(0.70), move_id
        assert rain_vec[_ACCURACY_IDX] == pytest.approx(1.0), move_id  # always hits in rain
        assert sun_vec[_ACCURACY_IDX] == pytest.approx(0.5), move_id  # halved in sun

    # MoveView.accuracy itself stays the static dex value - the override is
    # only applied at _move_slot_vector time (see module docstring).
    assert enc._move_view("thunder").accuracy == pytest.approx(0.70)


def test_blizzard_accuracy_is_perfect_in_snow_not_other_weather():
    dusclops = make_mon("dusclops")
    base_vec = _move_slot("blizzard", (PokemonType.ICE,), dusclops.types)
    snow_vec = _move_slot("blizzard", (PokemonType.ICE,), dusclops.types, weather="snow")
    sand_vec = _move_slot("blizzard", (PokemonType.ICE,), dusclops.types, weather="sandstorm")
    assert base_vec[_ACCURACY_IDX] == pytest.approx(0.70)
    assert snow_vec[_ACCURACY_IDX] == pytest.approx(1.0)
    assert sand_vec[_ACCURACY_IDX] == pytest.approx(0.70)  # only snow, not every weather


def test_terrain_power_boost_applies_only_to_a_grounded_matching_type_attacker():
    dusclops = make_mon("dusclops")
    boosted = _move_slot(
        "thunderbolt", (PokemonType.ELECTRIC,), dusclops.types,
        terrain="electricterrain", user_grounded=True,
    )
    airborne = _move_slot(
        "thunderbolt", (PokemonType.ELECTRIC,), dusclops.types,
        terrain="electricterrain", user_grounded=False,
    )
    no_terrain = _move_slot("thunderbolt", (PokemonType.ELECTRIC,), dusclops.types)

    assert boosted[_BASE_POWER_IDX] == pytest.approx(
        90 * enc._TERRAIN_POWER_MULTIPLIER / enc._MAX_BASE_POWER_SCALE
    )
    assert airborne[_BASE_POWER_IDX] == pytest.approx(90 / enc._MAX_BASE_POWER_SCALE)
    assert no_terrain[_BASE_POWER_IDX] == pytest.approx(90 / enc._MAX_BASE_POWER_SCALE)


def test_misty_terrain_does_not_boost_fairy_type_moves():
    # Real mechanic verified against moves.ts (see module docstring): unlike
    # Electric/Grassy/Psychic Terrain, Misty Terrain's condition block has
    # no onBasePower boost for Fairy-type moves at all.
    dusclops = make_mon("dusclops")
    boosted_attempt = _move_slot(
        "moonblast", (PokemonType.FAIRY,), dusclops.types,
        terrain="mistyterrain", user_grounded=True,
    )
    no_terrain = _move_slot("moonblast", (PokemonType.FAIRY,), dusclops.types)
    assert boosted_attempt[_BASE_POWER_IDX] == pytest.approx(no_terrain[_BASE_POWER_IDX])


def test_terrain_sleep_immune_true_under_electric_or_misty_terrain_for_a_grounded_mon():
    garchomp = make_mon("garchomp")  # Dragon/Ground - ordinarily grounded
    view = battle_view_from_poke_env(
        _battle([garchomp], garchomp, [make_mon("dragapult")], make_mon("dragapult"))
    ).my_active
    assert enc._terrain_sleep_immune(view, "electricterrain") is True
    assert enc._terrain_sleep_immune(view, "mistyterrain") is True
    assert enc._terrain_sleep_immune(view, "grassyterrain") is False  # doesn't block sleep
    assert enc._terrain_sleep_immune(view, None) is False


def test_terrain_status_immune_true_only_under_misty_terrain():
    garchomp = make_mon("garchomp")
    view = battle_view_from_poke_env(
        _battle([garchomp], garchomp, [make_mon("dragapult")], make_mon("dragapult"))
    ).my_active
    assert enc._terrain_status_immune(view, "mistyterrain") is True
    # Electric Terrain's real scope is sleep-only, not full status immunity.
    assert enc._terrain_status_immune(view, "electricterrain") is False


def test_terrain_status_immunity_requires_groundedness():
    talonflame = make_mon("talonflame")  # Fire/Flying - not grounded
    view = battle_view_from_poke_env(
        _battle([talonflame], talonflame, [make_mon("dragapult")], make_mon("dragapult"))
    ).my_active
    assert enc._terrain_sleep_immune(view, "electricterrain") is False
    assert enc._terrain_status_immune(view, "mistyterrain") is False


def test_DW_3_2_swift_swim_speed_doubling_flag_true_in_rain_false_otherwise():
    kingdra = make_mon("kingdra")
    kingdra.ability = "swiftswim"
    opp = make_mon("dragapult")

    rain_view = battle_view_from_poke_env(
        _battle([kingdra], kingdra, [opp], opp, weather={Weather.RAINDANCE: 1})
    )
    no_weather_view = battle_view_from_poke_env(_battle([kingdra], kingdra, [opp], opp))

    assert enc._ability_speed_doubled(
        rain_view.my_active, rain_view.weather, rain_view.terrain
    ) is True
    assert enc._ability_speed_doubled(
        no_weather_view.my_active, no_weather_view.weather, no_weather_view.terrain
    ) is False

    # Distinguishable in the ENCODED vector too - the new Phase 3 global
    # tail block places my/opp_active_speed_doubled at [-9]/[-8] (matchup
    # score stays [-3], hazard-immunity stays [-2]/[-1] - both already-
    # anchored tests, unaffected - see module docstring).
    rain_vec = encode(rain_view)
    no_weather_vec = encode(no_weather_view)
    assert rain_vec[-9] == 1.0
    assert no_weather_vec[-9] == 0.0
    assert rain_vec[-3] == no_weather_vec[-3]  # matchup score untouched by this feature


def test_ability_speed_doubled_requires_the_exact_matching_weather_or_terrain():
    kingdra = make_mon("kingdra")
    kingdra.ability = "swiftswim"
    sun_view = battle_view_from_poke_env(
        _battle([kingdra], kingdra, [make_mon("dragapult")], make_mon("dragapult"),
                weather={Weather.SUNNYDAY: 1})
    )
    assert enc._ability_speed_doubled(sun_view.my_active, sun_view.weather, sun_view.terrain) is False

    raichu = make_mon("raichualola")  # Alolan Raichu - Surge Surfer is real for this form
    raichu.ability = "surgesurfer"
    terrain_view = battle_view_from_poke_env(
        _battle([raichu], raichu, [make_mon("dragapult")], make_mon("dragapult"),
                fields={Field.ELECTRIC_TERRAIN: 1})
    )
    assert enc._ability_speed_doubled(
        terrain_view.my_active, terrain_view.weather, terrain_view.terrain
    ) is True


def test_is_grounded_flying_levitate_and_air_balloon_all_block_groundedness():
    talonflame = make_mon("talonflame")  # Flying-type
    assert enc._is_grounded(battle_view_from_poke_env(
        _battle([talonflame], talonflame, [make_mon("garchomp")], make_mon("garchomp"))
    ).my_active) is False

    bronzong = make_mon("bronzong")
    bronzong.ability = "levitate"
    assert enc._is_grounded(battle_view_from_poke_env(
        _battle([bronzong], bronzong, [make_mon("garchomp")], make_mon("garchomp"))
    ).my_active) is False

    garchomp = make_mon("garchomp")
    garchomp.item = "airballoon"
    assert enc._is_grounded(battle_view_from_poke_env(
        _battle([garchomp], garchomp, [make_mon("dragapult")], make_mon("dragapult"))
    ).my_active) is False

    # An ordinary grounded Pokemon reads True.
    plain_garchomp = make_mon("garchomp")
    assert enc._is_grounded(battle_view_from_poke_env(
        _battle([plain_garchomp], plain_garchomp, [make_mon("dragapult")], make_mon("dragapult"))
    ).my_active) is True

    # Unknown types default to True (grounded) - see _is_grounded's own
    # docstring for why this direction matters for _is_hazard_immune below.
    assert enc._is_grounded(PokemonView.unknown()) is True


def test_air_balloon_holder_is_now_hazard_immune_via_the_shared_grounded_helper():
    # A real, incidental correctness fix from factoring _is_grounded out of
    # _is_hazard_immune (see both functions' docstrings): Air Balloon
    # genuinely blocks groundedness in real Showdown (sim/pokemon.ts's
    # isGrounded), previously unmodeled by _is_hazard_immune's old inline
    # Flying/Levitate-only check.
    garchomp = make_mon("garchomp")
    garchomp.item = "airballoon"
    view = battle_view_from_poke_env(
        _battle([garchomp], garchomp, [make_mon("dragapult")], make_mon("dragapult"))
    )
    assert enc._is_hazard_immune(view.my_active) is True


def test_is_hazard_immune_existing_behavior_is_unchanged_by_the_refactor():
    # Every pre-Phase-3 _is_hazard_immune case must still hold exactly -
    # re-asserted here directly against the refactored implementation
    # (see _is_hazard_immune's own Phase 3 docstring note for the by-hand
    # trace this test backs up).
    talonflame = make_mon("talonflame")
    assert enc._is_hazard_immune(battle_view_from_poke_env(
        _battle([talonflame], talonflame, [make_mon("garchomp")], make_mon("garchomp"))
    ).my_active) is True

    bronzong = make_mon("bronzong")
    bronzong.ability = "levitate"
    assert enc._is_hazard_immune(battle_view_from_poke_env(
        _battle([bronzong], bronzong, [make_mon("garchomp")], make_mon("garchomp"))
    ).my_active) is True

    garchomp_boots = make_mon("garchomp")
    garchomp_boots.item = "heavydutyboots"
    assert enc._is_hazard_immune(battle_view_from_poke_env(
        _battle([garchomp_boots], garchomp_boots, [make_mon("dragapult")], make_mon("dragapult"))
    ).my_active) is True

    plain_garchomp = make_mon("garchomp")
    assert enc._is_hazard_immune(battle_view_from_poke_env(
        _battle([plain_garchomp], plain_garchomp, [make_mon("dragapult")], make_mon("dragapult"))
    ).my_active) is False

    assert enc._is_hazard_immune(PokemonView.unknown()) is False


def test_weather_terrain_context_defaults_are_backward_compatible():
    # Every pre-Phase-3 caller of _move_slot_vector/_move_slots_vector/
    # _encode_pokemon (including this module's own Phase 1/2 tests, which
    # never pass weather/terrain/user_grounded) must see identical output
    # to an explicit "no weather, no terrain, grounded" call.
    dusclops = make_mon("dusclops")
    move = enc._move_view("solarbeam")
    implicit = enc._move_slot_vector(move, (PokemonType.GRASS,), dusclops.types, None, None)
    explicit = enc._move_slot_vector(
        move, (PokemonType.GRASS,), dusclops.types, None, None,
        weather=None, terrain=None, user_grounded=True,
    )
    assert np.array_equal(implicit, explicit)


# --- Phase 4 (2026-08-27): side-condition completeness, hazard stacking,
# status severity, tera-used ---------------------------------------------
#
# The real gaps this phase closes (see module docstring): Safeguard was
# absent from the hazard-token vocabulary, hazard stacking (Spikes/Toxic
# Spikes layer count) was presence-only, badly-poisoned severity had no
# magnitude, Leech Seed/Substitute/Confusion were invisible, and whether a
# side had spent its one-time Tera resource wasn't encoded at all.
#
# Per-Pokemon-block tail indices (see _encode_pokemon's docstring - these
# sit BEFORE the already-anchored preparing/semi_invulnerable/
# must_recharge/protect_counter quartet at [-4]/[-3]/[-2]/[-1]), verified
# directly against the real implementation, not just derived by hand:
_TOXIC_COUNTER_IDX = -8
_HAS_LEECH_SEED_IDX = -7
_HAS_SUBSTITUTE_IDX = -6
_IS_CONFUSED_IDX = -5

# Full-vector (encode()) global tail indices - sit BEFORE the already-
# anchored Phase 3 6-scalar block at [-9]..[-4], matchup score at [-3], and
# hazard-immunity pair at [-2]/[-1] - also verified directly:
_MY_SPIKES_LAYERS_IDX = -15
_MY_TOXIC_SPIKES_LAYERS_IDX = -14
_OPP_SPIKES_LAYERS_IDX = -13
_OPP_TOXIC_SPIKES_LAYERS_IDX = -12
_MY_USED_TERA_IDX = -11
_OPP_USED_TERA_IDX = -10


def test_safeguard_is_now_a_tracked_hazard_token():
    # Real gap #1 (see module docstring): Safeguard was entirely absent
    # from _HAZARD_TOKENS - folded in with identical shape to Reflect/Light
    # Screen (turn-tracked, not stackable).
    mine = make_mon("garchomp")
    opp = make_mon("dragapult")
    live_view = battle_view_from_poke_env(
        _battle([mine], mine, [opp], opp, my_hazards={SideCondition.SAFEGUARD: 3})
    )
    assert live_view.my_hazards == {"safeguard"}

    mon = _replay_pokemon("garchomp")
    opp_mon = _replay_pokemon("dragapult")
    replay_view = battle_view_from_replay_state(
        _replay_state(mon, opp_mon, player_conditions="safeguard")
    )
    assert replay_view.my_hazards == {"safeguard"}


def test_DW_4_2_spikes_layer_count_encodes_the_real_stack_not_just_presence():
    mine = make_mon("garchomp")
    opp = make_mon("dragapult")
    one_layer = battle_view_from_poke_env(
        _battle([mine], mine, [opp], opp, my_hazards={SideCondition.SPIKES: 1})
    )
    two_layer = battle_view_from_poke_env(
        _battle([mine], mine, [opp], opp, my_hazards={SideCondition.SPIKES: 2})
    )
    assert one_layer.my_spikes_layers == 1
    assert two_layer.my_spikes_layers == 2
    # Presence alone (my_hazards) can't distinguish these - the whole point
    # of this new field.
    assert one_layer.my_hazards == two_layer.my_hazards == {"spikes"}

    one_vec = encode(one_layer)
    two_vec = encode(two_layer)
    assert one_vec[_MY_SPIKES_LAYERS_IDX] == pytest.approx(1.0 / enc._SPIKES_MAX_LAYERS)
    assert two_vec[_MY_SPIKES_LAYERS_IDX] == pytest.approx(2.0 / enc._SPIKES_MAX_LAYERS)
    assert not np.array_equal(one_vec, two_vec)


def test_toxic_spikes_and_opponent_side_hazard_layers_also_encode_the_real_stack():
    mine = make_mon("garchomp")
    opp = make_mon("dragapult")
    my_toxic_spikes = battle_view_from_poke_env(
        _battle([mine], mine, [opp], opp, my_hazards={SideCondition.TOXIC_SPIKES: 2})
    )
    opp_spikes = battle_view_from_poke_env(
        _battle([mine], mine, [opp], opp, opp_hazards={SideCondition.SPIKES: 1})
    )
    opp_toxic_spikes = battle_view_from_poke_env(
        _battle([mine], mine, [opp], opp, opp_hazards={SideCondition.TOXIC_SPIKES: 1})
    )
    assert my_toxic_spikes.my_toxic_spikes_layers == 2
    assert opp_spikes.opp_spikes_layers == 1
    assert opp_toxic_spikes.opp_toxic_spikes_layers == 1

    assert encode(my_toxic_spikes)[_MY_TOXIC_SPIKES_LAYERS_IDX] == pytest.approx(
        2.0 / enc._TOXIC_SPIKES_MAX_LAYERS
    )
    assert encode(opp_spikes)[_OPP_SPIKES_LAYERS_IDX] == pytest.approx(1.0 / enc._SPIKES_MAX_LAYERS)
    assert encode(opp_toxic_spikes)[_OPP_TOXIC_SPIKES_LAYERS_IDX] == pytest.approx(
        1.0 / enc._TOXIC_SPIKES_MAX_LAYERS
    )


def test_hazard_stack_layers_clamp_beyond_the_real_showdown_cap():
    mine = make_mon("garchomp")
    opp = make_mon("dragapult")
    over_capped = battle_view_from_poke_env(
        _battle([mine], mine, [opp], opp, my_hazards={SideCondition.SPIKES: 99})
    )
    assert encode(over_capped)[_MY_SPIKES_LAYERS_IDX] == pytest.approx(1.0)  # clamped, not > 1.0


def test_DW_4_2_hazard_stack_layers_default_zero_on_replay_adapter_documented_gap():
    # Structural gap (see module docstring): no "spikes2"/"spikes3"-style
    # token exists anywhere in the real replay vocabulary - the field only
    # ever holds the bare condition name, so this can't be reconstructed
    # even with the whole turn sequence.
    mon = _replay_pokemon("garchomp")
    opp = _replay_pokemon("dragapult")
    state = _replay_state(mon, opp, player_conditions="spikes", opponent_conditions="toxicspikes")

    single_view = battle_view_from_replay_state(state)
    multi_view = battle_views_from_replay([state])[0]

    for view in (single_view, multi_view):
        assert view.my_spikes_layers == 0
        assert view.my_toxic_spikes_layers == 0
        assert view.opp_spikes_layers == 0
        assert view.opp_toxic_spikes_layers == 0


def test_toxic_counter_is_read_from_the_live_adapter_only_while_badly_poisoned():
    toxic = make_mon("garchomp", status=Status.TOX)
    toxic._status_counter = 3
    opp = make_mon("dragapult")
    view = battle_view_from_poke_env(_battle([toxic], toxic, [opp], opp)).my_active
    assert view.toxic_counter == 3

    # Sanity: status_counter is dual-purpose (also tracks sleep turns) - a
    # Pokemon with a DIFFERENT status must read 0 even if the underlying
    # poke-env field happens to be nonzero, isolating that the TOX gate
    # itself is doing the work, not just a pass-through.
    paralyzed = make_mon("garchomp", status=Status.PAR)
    paralyzed._status_counter = 3
    par_view = battle_view_from_poke_env(_battle([paralyzed], paralyzed, [opp], opp)).my_active
    assert par_view.toxic_counter == 0


def test_toxic_counter_encodes_as_a_normalized_scalar_and_clamps():
    view = PokemonView.unknown()
    view.known = True  # unknown() zeroes toxic_counter too; only care about that field here
    defender = PokemonView.unknown()

    vec_at_0 = enc._encode_pokemon(replace(view, toxic_counter=0), defender)
    vec_at_8 = enc._encode_pokemon(replace(view, toxic_counter=8), defender)
    vec_at_999 = enc._encode_pokemon(replace(view, toxic_counter=999), defender)

    assert vec_at_0[_TOXIC_COUNTER_IDX] == pytest.approx(0.0)
    assert vec_at_8[_TOXIC_COUNTER_IDX] == pytest.approx(8.0 / enc._TOXIC_COUNTER_SCALE)
    assert vec_at_999[_TOXIC_COUNTER_IDX] == pytest.approx(1.0)  # far beyond any real streak - must clamp


def test_toxic_counter_defaults_zero_on_replay_adapter_documented_gap():
    # No equivalent field exists anywhere in Metamon's per-state schema
    # (only the current status token, no turn count) - see module docstring.
    mon = _replay_pokemon("garchomp", status="tox")
    state = _replay_state(mon, _replay_pokemon("dragapult"))
    single_view = battle_view_from_replay_state(state)
    multi_view = battle_views_from_replay([state])[0]
    assert single_view.my_active.toxic_counter == 0
    assert multi_view.my_active.toxic_counter == 0


def test_DW_4_1_leech_seed_substitute_confusion_encode_on_live_adapter():
    seeded = make_mon("garchomp")
    seeded.effects[Effect.LEECH_SEED] = 0
    substituted = make_mon("garchomp")
    substituted.effects[Effect.SUBSTITUTE] = 0
    confused = make_mon("garchomp")
    confused.effects[Effect.CONFUSION] = 0
    plain = make_mon("garchomp")
    opp = make_mon("dragapult")

    seeded_view = battle_view_from_poke_env(_battle([seeded], seeded, [opp], opp)).my_active
    substituted_view = battle_view_from_poke_env(_battle([substituted], substituted, [opp], opp)).my_active
    confused_view = battle_view_from_poke_env(_battle([confused], confused, [opp], opp)).my_active
    plain_view = battle_view_from_poke_env(_battle([plain], plain, [opp], opp)).my_active

    assert seeded_view.has_leech_seed is True
    assert seeded_view.has_substitute is False
    assert seeded_view.is_confused is False
    assert substituted_view.has_substitute is True
    assert confused_view.is_confused is True
    assert plain_view.has_leech_seed is False
    assert plain_view.has_substitute is False
    assert plain_view.is_confused is False

    # Distinguishable in the encoded per-Pokemon block, not just the view.
    seeded_vec = enc._encode_pokemon(seeded_view, defender=plain_view)
    assert seeded_vec[_HAS_LEECH_SEED_IDX] == 1.0
    assert seeded_vec[_HAS_SUBSTITUTE_IDX] == 0.0
    assert seeded_vec[_IS_CONFUSED_IDX] == 0.0

    substituted_vec = enc._encode_pokemon(substituted_view, defender=plain_view)
    assert substituted_vec[_HAS_SUBSTITUTE_IDX] == 1.0

    confused_vec = enc._encode_pokemon(confused_view, defender=plain_view)
    assert confused_vec[_IS_CONFUSED_IDX] == 1.0


def test_DW_4_1_leech_seed_substitute_confusion_encode_on_replay_adapter():
    seeded = _replay_pokemon("garchomp", effect="leechseed")
    substituted = _replay_pokemon("dragapult", effect="substitute")
    confused = _replay_pokemon("blissey", effect="confusion")
    plain = _replay_pokemon("tinkaton", effect="noeffect")

    seeded_view = battle_view_from_replay_state(_replay_state(seeded, plain)).my_active
    substituted_view = battle_view_from_replay_state(_replay_state(substituted, plain)).my_active
    confused_view = battle_view_from_replay_state(_replay_state(confused, plain)).my_active
    plain_view = battle_view_from_replay_state(_replay_state(plain, plain)).my_active

    assert seeded_view.has_leech_seed is True
    assert seeded_view.has_substitute is False
    assert substituted_view.has_substitute is True
    assert confused_view.is_confused is True
    assert plain_view.has_leech_seed is False
    assert plain_view.has_substitute is False
    assert plain_view.is_confused is False

    seeded_vec = enc._encode_pokemon(seeded_view, defender=plain_view)
    assert seeded_vec[_HAS_LEECH_SEED_IDX] == 1.0

    # battle_views_from_replay (the multi-state adapter) must agree.
    multi_view = battle_views_from_replay([_replay_state(seeded, plain)])[0].my_active
    assert multi_view.has_leech_seed is True


def test_fainted_teammate_leech_seed_substitute_confusion_flags_are_explicitly_cleared():
    # A fainted teammate's last-seen snapshot may still show a stale
    # "effect" from when it was alive - must NOT be trusted (see
    # _replay_pokemon_view_fainted's docstring: these don't survive a faint
    # any more than boosts do).
    seeded_then_fainted = _replay_pokemon("zapdos", effect="leechseed")
    states = [
        _replay_state(
            _replay_pokemon("garchomp"), _replay_pokemon("dragapult"),
            available_switches=[seeded_then_fainted],
        ),
        # zapdos no longer appears anywhere (fainted) - reconstructed as a
        # known-but-fainted bench slot.
        _replay_state(_replay_pokemon("garchomp"), _replay_pokemon("dragapult")),
    ]
    views = battle_views_from_replay(states)
    fainted_zapdos = next(p for p in views[1].my_bench if p.known)
    assert fainted_zapdos.fainted is True
    assert fainted_zapdos.has_leech_seed is False


def test_DW_4_1_my_used_tera_encodes_correctly_on_both_adapters():
    active = make_mon("garchomp")
    opp = make_mon("dragapult")
    not_used = battle_view_from_poke_env(_battle([active], active, [opp], opp))
    assert not_used.my_used_tera is False

    tera_active = make_mon("garchomp")
    tera_active._terastallized = True
    used = battle_view_from_poke_env(_battle([tera_active], tera_active, [opp], opp))
    assert used.my_used_tera is True

    # Replay: can_tera is verified monotonic (see module docstring) - False
    # means tera has already been used this battle, exact from one state.
    mon = _replay_pokemon("garchomp")
    replay_not_used = battle_view_from_replay_state(
        _replay_state(mon, _replay_pokemon("dragapult"), can_tera=True)
    )
    replay_used = battle_view_from_replay_state(
        _replay_state(mon, _replay_pokemon("dragapult"), can_tera=False)
    )
    assert replay_not_used.my_used_tera is False
    assert replay_used.my_used_tera is True

    # battle_views_from_replay (multi-state adapter) must agree.
    multi_used = battle_views_from_replay(
        [_replay_state(mon, _replay_pokemon("dragapult"), can_tera=False)]
    )[0]
    assert multi_used.my_used_tera is True

    # Distinguishable in the encoded vector too.
    assert encode(used)[_MY_USED_TERA_IDX] == 1.0
    assert encode(not_used)[_MY_USED_TERA_IDX] == 0.0


def test_DW_4_1_used_tera_checks_the_whole_team_not_just_the_active_mon():
    active = make_mon("garchomp")  # currently active, NOT terastallized
    benched_tera = make_mon("blissey")
    benched_tera._terastallized = True  # terastallized earlier, now benched
    opp = make_mon("dragapult")

    view = battle_view_from_poke_env(_battle([active, benched_tera], active, [opp], opp))
    assert view.my_used_tera is True  # a benched mon's past tera use still counts


def test_DW_4_1_opp_used_tera_is_live_adapter_only_documented_gap():
    mine = make_mon("garchomp")
    opp_tera = make_mon("dragapult")
    opp_tera._terastallized = True
    live_view = battle_view_from_poke_env(_battle([mine], mine, [opp_tera], opp_tera))
    assert live_view.opp_used_tera is True  # live-exact
    assert encode(live_view)[_OPP_USED_TERA_IDX] == 1.0

    mon = _replay_pokemon("garchomp")
    opp_mon = _replay_pokemon("dragapult")
    replay_state = _replay_state(mon, opp_mon, can_tera=True)
    single_view = battle_view_from_replay_state(replay_state)
    multi_view = battle_views_from_replay([replay_state])[0]
    # Documented gap - always False, no opponent_can_tera field exists in
    # the real replay schema (see module docstring).
    assert single_view.opp_used_tera is False
    assert multi_view.opp_used_tera is False
