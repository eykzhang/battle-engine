from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.battle.field import Field
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
    item="unknownitem", moves=None,
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
    player_prev_move=None, opponent_prev_move=None,
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

    vec_at_0 = enc._encode_pokemon(replace(view_at_0, protect_counter=0))
    vec_at_2 = enc._encode_pokemon(replace(view_at_0, protect_counter=2))
    vec_at_999 = enc._encode_pokemon(replace(view_at_0, protect_counter=999))

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
