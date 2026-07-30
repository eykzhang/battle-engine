from types import SimpleNamespace

import numpy as np
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
    battle_view_from_poke_env,
    battle_view_from_replay_state,
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
        "moves": [],
        **{f"{stat}_boost": boosts.get(stat, 0) for stat in
           ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
        **{f"base_{stat}": base_stats[stat] for stat in
           ("hp", "atk", "def", "spa", "spd", "spe")},
    }


def _replay_state(
    player_active, opponent_active, available_switches=None,
    opponents_remaining=6, player_conditions="noconditions",
    opponent_conditions="noconditions", weather="noweather", battle_field="nofield",
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

    # matchup score is the last dimension of the vector.
    assert favorable[-1] > 0.0
    assert unfavorable[-1] < 0.0


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
    assert levitate_vec[-1] < unrevealed_vec[-1]


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


def test_replay_adapter_maps_unknown_ability_token_to_none():
    mon = _replay_pokemon("dragapult", ability="unknownability")
    revealed = _replay_pokemon("garchomp", ability="roughskin")
    state = _replay_state(revealed, mon)

    view = battle_view_from_replay_state(state)

    assert view.opp_active.ability is None
    assert view.my_active.ability == "roughskin"


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
