"""Phase 6 / M4: the Smogon chaos loader.

The tests that matter here are the ones pinning arithmetic that is easy to get
wrong and impossible to notice: the denominator behind every probability, and
the teammate deviation formula. Both were established empirically against the
real file (see the module docstring), so both get pinned against a synthetic
file whose right answers can be computed by hand.
"""

from __future__ import annotations

import math
import random

import pytest

from battle_engine.usage_stats import (
    NEUTRAL_SPREAD,
    Distribution,
    Spread,
    parse_chaos,
)


# One team-slot's worth of weight per set, so the four one-draw-per-set
# distributions each sum to `set_count` and Moves sums to 4x it - exactly the
# shape the real file has.
def _species(usage, set_count, *, moves, items, abilities, spreads, tera, teammates, raw=None):
    return {
        "usage": usage,
        "Raw count": raw if raw is not None else set_count * 10,
        "Abilities": abilities,
        "Items": items,
        "Moves": moves,
        "Spreads": spreads,
        "Tera Types": tera,
        "Teammates": teammates,
    }


@pytest.fixture
def payload():
    return {
        "info": {"metagame": "gen9ou", "cutoff": 1500, "number of battles": 100},
        "data": {
            "Great Tusk": _species(
                0.5,
                100,
                # Sums to 400 = 4 * set_count: every set has four moves.
                moves={"headlongrush": 100.0, "rapidspin": 80.0, "icespinner": 120.0, "knockoff": 100.0},
                items={"heavydutyboots": 70.0, "leftovers": 30.0},
                abilities={"protosynthesis": 100.0},
                spreads={"Jolly:0/252/0/0/4/252": 60.0, "Impish:252/0/252/0/4/0": 40.0},
                tera={"steel": 60.0, "water": 40.0},
                # P(Gholdengo | Great Tusk) = 0.25 + 25/100 = 0.50
                teammates={"Gholdengo": 25.0},
            ),
            "Gholdengo": _species(
                0.25,
                50,
                moves={"makeitrain": 50.0, "shadowball": 50.0, "recover": 50.0, "nastyplot": 50.0},
                items={"airballoon": 50.0},
                abilities={"goodasgold": 50.0},
                spreads={"Timid:0/0/0/252/4/252": 50.0},
                tera={"fairy": 50.0},
                teammates={"Great Tusk": 25.0},
            ),
            # Not a real species: must be dropped rather than sampled, because
            # poke-engine turns an unknown id into a statless `NONE`.
            "Fakemon": _species(
                0.1,
                10,
                moves={"tackle": 40.0},
                items={"leftovers": 10.0},
                abilities={"levitate": 10.0},
                spreads={"Serious:0/0/0/0/0/0": 10.0},
                tera={"normal": 10.0},
                teammates={"Great Tusk": 1.0},
            ),
        },
    }


class TestSpread:
    def test_parses_a_real_chaos_key(self):
        spread = Spread.parse("Jolly:0/252/4/0/0/252")
        assert spread == Spread(nature="jolly", evs=(0, 252, 4, 0, 0, 252))
        assert spread.ev("spe") == 252
        assert spread.ev_total == 508

    @pytest.mark.parametrize(
        "key",
        [
            "Kanto:0/252/4/0/0/252",  # nature that does not exist
            "Jolly:0/252/4/0/252",  # five stats
            "Jolly:0/252/4/0/0/300",  # EV above the 252 cap
            "Jolly:0/252/4/0/0/x",  # not a number
            "nonsense",
        ],
    )
    def test_returns_none_rather_than_raising(self, key):
        # The chaos file is scraped from real ladder teams, so it carries
        # whatever those teams packed. A caller wants those skipped, not a
        # crash 200 species into a load.
        assert Spread.parse(key) is None

    def test_stat_formula_matches_showdown(self):
        # Great Tusk: base 131 Atk / 87 Spe, Jolly 252 Atk / 252 Spe.
        # Hand-computed from the gen-3+ formula with 31 IVs.
        spread = Spread(nature="jolly", evs=(0, 252, 4, 0, 0, 252))
        assert spread.stat(131, "atk") == 361
        assert spread.stat(87, "spe") == 300  # 273 * 1.1, floored
        # A hindering nature floors the same way: 361 * 0.9 = 324.9 -> 324.
        assert Spread(nature="jolly", evs=(0, 0, 0, 252, 0, 0)).stat(131, "spa") == 324

    def test_hp_has_its_own_arm_and_ignores_nature(self):
        # Great Tusk, base 115 HP, 252 HP EVs: (230 + 31 + 63) + 100 + 10.
        spread = Spread(nature="jolly", evs=(252, 0, 0, 0, 0, 0))
        assert spread.stat(115, "hp") == 434
        assert spread.nature_multiplier("hp") == 1.0

    def test_neutral_spread_reproduces_the_translator_convention(self):
        # The 0-EV / 31-IV / neutral estimate `poke_engine_state._dex_profile`
        # applies to anything unknown. "No prediction" and "predicted the
        # neutral spread" have to be the same number or the M4 comparison is
        # measuring an offset rather than an improvement.
        assert NEUTRAL_SPREAD.stat(131, "atk") == int((2 * 131 + 31) * 100 / 100) + 5
        assert NEUTRAL_SPREAD.stat(115, "hp") == int((2 * 115 + 31) * 100 / 100) + 110


class TestDistribution:
    def test_drops_non_positive_weights(self):
        d = Distribution({"a": 3.0, "b": 0.0, "c": -1.0})
        assert len(d) == 1 and "b" not in d and "c" not in d

    def test_probability_and_mode(self):
        d = Distribution({"a": 3.0, "b": 1.0})
        assert d.probability("a") == 0.75
        assert d.most_likely() == "a"
        assert d.probability("missing") == 0.0

    def test_empty_distribution_is_falsy_and_has_no_mode(self):
        d = Distribution({})
        assert not d
        assert d.most_likely() is None
        assert d.most_likely("fallback") == "fallback"
        assert d.probability("a") == 0.0

    def test_sample_without_replacement_never_repeats_and_respects_the_pool(self):
        d = Distribution({"a": 5.0, "b": 4.0, "c": 3.0})
        drawn = d.sample_without_replacement(random.Random(0), 5)
        assert len(drawn) == 3 == len(set(drawn))

    def test_sampling_follows_the_weights(self):
        d = Distribution({"a": 9.0, "b": 1.0})
        rng = random.Random(0)
        draws = [d.sample(rng) for _ in range(2000)]
        assert 0.85 < draws.count("a") / len(draws) < 0.95

    def test_without_and_reweighted_do_not_mutate(self):
        d = Distribution({"a": 1.0, "b": 1.0})
        assert d.without(["a"]).most_likely() == "b"
        assert len(d) == 2
        doubled = d.reweighted(lambda k: 2.0 if k == "a" else 1.0)
        assert doubled.most_likely() == "a"
        assert d.probability("a") == 0.5


class TestParseChaos:
    def test_keys_are_ids_not_display_names(self, payload):
        stats = parse_chaos(payload)
        assert "greattusk" in stats and "Great Tusk" not in stats

    def test_drops_species_poke_engine_cannot_represent(self, payload):
        stats = parse_chaos(payload)
        # Silent failure otherwise: poke-engine accepts an unknown species id
        # and serializes it as a statless, typeless `NONE`.
        assert "fakemon" not in stats
        assert stats.report.species_dropped == ("fakemon",)
        assert stats.report.species_kept == 2

    def test_set_count_is_the_denominator_not_raw_count(self, payload):
        stats = parse_chaos(payload)
        tusk = stats.entry("greattusk")
        assert tusk.set_count == 100.0
        assert tusk.raw_count == 1000.0
        # Ice Spinner is on 120 of 100 sets' worth of move slots -> capped at 1.
        assert tusk.move_probability("icespinner") == 1.0
        assert tusk.move_probability("rapidspin") == pytest.approx(0.8)
        # Dividing by Raw count instead would give 0.08, off by the weighting
        # factor - the bug this test exists to prevent.
        assert tusk.move_probability("rapidspin") != pytest.approx(80.0 / tusk.raw_count)

    def test_move_probability_is_zero_for_an_unseen_move(self, payload):
        assert parse_chaos(payload).entry("greattusk").move_probability("splash") == 0.0

    def test_teammate_deviation_becomes_a_conditional_probability(self, payload):
        stats = parse_chaos(payload)
        # usage(Gholdengo) + delta / set_count(Great Tusk) = 0.25 + 25/100.
        assert stats.conditional_probability("greattusk", "gholdengo") == pytest.approx(0.5)
        # No recorded deviation falls back to the base rate rather than to zero.
        assert stats.conditional_probability("gholdengo", "gholdengo") == pytest.approx(0.25)

    def test_teammate_deltas_pointing_at_dropped_species_are_removed(self, payload):
        stats = parse_chaos(payload)
        assert "fakemon" not in stats.entry("greattusk").teammate_deltas

    def test_conditional_usage_excludes_the_known_team(self, payload):
        stats = parse_chaos(payload)
        pool = stats.conditional_usage(["greattusk"])
        # Species Clause: a slot may not repeat what is already on the team.
        assert "greattusk" not in pool
        assert pool.most_likely() == "gholdengo"

    def test_conditional_usage_with_no_known_team_is_plain_usage(self, payload):
        stats = parse_chaos(payload)
        assert stats.conditional_usage([]).probability("greattusk") == pytest.approx(
            stats.usage.probability("greattusk")
        )

    def test_teammate_lift_sharpens_toward_a_core(self, payload):
        stats = parse_chaos(payload)
        # P(Gholdengo | Great Tusk) is double its base usage, so conditioning
        # on Great Tusk must raise Gholdengo's share of what is left.
        assert stats.conditional_usage(["greattusk"]).probability("gholdengo") == 1.0
        assert stats.usage.probability("gholdengo") < 1.0

    def test_empty_move_slots_are_counted_not_predicted(self):
        payload = {
            "info": {"metagame": "gen9ou", "cutoff": 1500},
            "data": {
                "Great Tusk": _species(
                    0.5,
                    100,
                    moves={"headlongrush": 100.0, "": 42.0},
                    items={"leftovers": 100.0},
                    abilities={"protosynthesis": 100.0},
                    spreads={"Jolly:0/252/0/0/4/252": 100.0},
                    tera={"steel": 100.0},
                    teammates={},
                )
            },
        }
        stats = parse_chaos(payload)
        assert stats.report.empty_move_slots == 42.0
        assert "" not in stats.entry("greattusk").moves

    def test_unmodelled_items_are_kept_on_purpose(self, payload):
        # Air Balloon has no poke-engine mechanics. Dropping it would
        # renormalize Gholdengo's item distribution onto whatever is left and
        # overstate how often it holds something the engine does model; the
        # translator downgrades it to `unknownitem` and records the downgrade.
        stats = parse_chaos(payload)
        assert stats.entry("gholdengo").items.most_likely() == "airballoon"

    def test_likeliest_moves_respects_what_is_already_known(self, payload):
        stats = parse_chaos(payload)
        assert stats.likeliest_moves("greattusk", known=("icespinner",), n=4) == (
            "headlongrush",
            "knockoff",
            "rapidspin",
        )

    def test_sample_moves_fills_only_the_empty_slots(self, payload):
        stats = parse_chaos(payload)
        drawn = stats.sample_moves("greattusk", random.Random(3), known=("icespinner", "knockoff"))
        assert len(drawn) == 2
        assert "icespinner" not in drawn and "knockoff" not in drawn

    def test_unknown_species_answers_emptily_rather_than_raising(self, payload):
        stats = parse_chaos(payload)
        assert stats.get("missingno") is None
        assert stats.sample_moves("missingno", random.Random(0)) == ()
        with pytest.raises(KeyError):
            stats.entry("missingno")

    def test_usage_sums_to_one_slot_per_team_member(self):
        # Not synthetic: the real file's usage values sum to ~6, one per team
        # slot. This is the check that `usage` means "fraction of teams
        # containing" rather than a share of some other total.
        pytest.importorskip("poke_env")
        from battle_engine.usage_stats import find_cached, load_usage_stats

        path = find_cached(cutoff=1500)
        if path is None:
            pytest.skip("no cached usage stats; run scripts/fetch_usage_stats.py")
        stats = load_usage_stats(path)
        total = sum(stats.entry(s).usage for s in stats)
        assert math.isclose(total, 6.0, rel_tol=0.02)
