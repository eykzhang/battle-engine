"""Tests for the Showdown replay-log parser.

Two layers. Most cases are hand-written protocol fixtures - a real ladder replay
is 500 lines and cannot isolate a mechanic, while a six-line fixture pins exactly
one. The last test runs the parser over the real fetched corpus and skips cleanly
when it is absent, because fixtures only prove the parser handles the message
shapes someone thought to write down.

Every fixture line here was copied from the shape the simulator actually emits
(`pokemon-showdown/sim/`, plus real logs), not invented from the spec - three of
the behaviors below contradict `sim/SIM-PROTOCOL.md` and would be untestable
against a fixture written from the documentation.
"""

import json
from pathlib import Path

import pytest

from battle_engine.replay_log import (
    UNKNOWN,
    ActionKind,
    ReplayParseError,
    is_known,
    parse_replay_json,
    parse_replay_log,
)

_CORPUS_DIR = Path("data/replays_showdown")

_P1_TEAM = ["Gliscor, M", "Clefable, F", "Zamazenta-*"]
_P2_TEAM = ["Raging Bolt", "Kingambit, M", "Great Tusk"]


def _log(*turn_blocks: str, p1_lead: str = "Gliscor, M",
         p2_lead: str = "Raging Bolt", gametype: str = "singles") -> str:
    """Assemble a full protocol log: real header, team preview, leads, then turns.

    The header is not decoration. `|tier|[Gen 9] OU` in particular is a regression
    guard: its `[Gen 9]` prefix looks exactly like a `[from]`-style bracket tag,
    and an earlier version of the kwarg splitter ate it and left `|tier|` with no
    fields at all.
    """
    lines = [
        "|j|☆alice",
        f"|gametype|{gametype}",
        "|player|p1|alice|hugh|1521",
        "|player|p2|bob|170|1490",
        "|gen|9",
        "|tier|[Gen 9] OU",
        "|rated|",
        "|rule|HP Percentage Mod: HP is shown in percentages",
        "|clearpoke",
    ]
    lines += [f"|poke|p1|{details}|" for details in _P1_TEAM]
    lines += [f"|poke|p2|{details}|" for details in _P2_TEAM]
    lines += [
        "|teampreview",
        f"|teamsize|p1|{len(_P1_TEAM)}",
        f"|teamsize|p2|{len(_P2_TEAM)}",
        "|start",
        f"|switch|p1a: {p1_lead.split(',')[0]}|{p1_lead}|100/100",
        f"|switch|p2a: {p2_lead.split(',')[0]}|{p2_lead}|100/100",
    ]
    for block in turn_blocks:
        lines.append(block.strip("\n"))
    return "\n".join(lines) + "\n"


def _one_turn(*body: str, number: int = 1) -> str:
    """One complete turn: the turn marker, the body, and `|upkeep|`.

    No trailing `|turn|` - the parser emits a transition when the *next* turn
    starts or when the log runs out, so blocks chain by simply following each
    other and the last block needs no terminator.
    """
    return "\n".join([f"|turn|{number}", *body, "|upkeep"])


# ---------------------------------------------------------------------------
# DW-M2.2 / DW-M2.3 - the transition shape and the parse entry points
# ---------------------------------------------------------------------------

def test_DW_M2_2_transition_carries_before_both_actions_and_after():
    # The whole point of M2: M3 must be able to ask "given this state and what
    # BOTH players did, what does a forward model predict". A transition missing
    # either side's action cannot drive that question, which is exactly why the
    # existing Metamon corpus is unusable here.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
        "|-damage|p2a: Raging Bolt|62/100",
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
        "|-damage|p1a: Gliscor|71/100",
    )))
    transition = replay.transitions[0]
    assert transition.turn == 1
    assert transition.state_before.p1.active.hp_fraction == 1.0
    assert transition.p1_action.move == "earthquake"
    assert transition.p2_action.move == "thunderclap"
    assert transition.state_after.p1.active.hp_fraction == pytest.approx(0.71)
    assert transition.state_after.p2.active.hp_fraction == pytest.approx(0.62)


def test_DW_M2_3_parse_replay_json_returns_one_transition_per_turn():
    # parse_replay_json is the entry point the corpus uses, so it has to accept a
    # raw `<id>.json` payload and carry its metadata through, not just the log.
    payload = {
        "id": "gen9ou-123", "formatid": "gen9ou", "rating": 1500,
        "players": ["alice", "bob"],
        "log": _log(
            _one_turn("|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt", number=1),
            _one_turn("|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt", number=2),
            _one_turn("|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt", number=3),
            "|win|alice",
        ),
    }
    replay = parse_replay_json(payload)
    assert [t.turn for t in replay.transitions] == [1, 2, 3]
    assert replay.battle_id == "gen9ou-123"
    assert replay.rating == 1500
    assert replay.winner == "alice"
    assert replay.leads == {"p1": "p1: Gliscor", "p2": "p2: Raging Bolt"}


def test_parse_rejects_a_log_it_cannot_honestly_represent():
    # Returning an empty list would make a doubles battle or a truncated log look
    # identical to a zero-turn one, and a fidelity harness would score the silence
    # as agreement. Loud failure, per docs/code-standards.md.
    with pytest.raises(ReplayParseError, match="no .turn. messages"):
        parse_replay_log(_log())
    with pytest.raises(ReplayParseError, match="singles"):
        parse_replay_log(
            _log(_one_turn("|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt"),
                 gametype="doubles"))


# ---------------------------------------------------------------------------
# DW-M2.4 - the four action shapes are distinguishable
# ---------------------------------------------------------------------------

def test_DW_M2_4_move_switch_tera_and_forced_replacement_are_distinct_kinds():
    # All four have to be told apart, because they mean different things to a
    # forward model: a move and a switch are alternative turn choices, a Tera move
    # additionally burns the side's once-per-battle Tera, and a replacement is a
    # decision made after the turn has already resolved.
    replay = parse_replay_log(_log(
        # Turn 1: p1 switches, p2 Teras and attacks, killing the switch-in.
        "|turn|1",
        "|switch|p1a: Clefable|Clefable, F|100/100",
        "|-terastallize|p2a: Raging Bolt|Fairy",
        "|move|p2a: Raging Bolt|Draining Kiss|p1a: Clefable",
        "|-damage|p1a: Clefable|0 fnt",
        "|faint|p1a: Clefable",
        "|upkeep",
        "|switch|p1a: Gliscor|Gliscor, M|100/100",
        # Turn 2: a plain move, so all four kinds appear in one replay.
        "|turn|2",
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
        "|upkeep",
        "|turn|3",
    ))
    first, second = replay.transitions[0], replay.transitions[1]

    assert first.p1_action.kind is ActionKind.SWITCH
    assert first.p1_action.switch_in == "p1: Clefable"
    assert first.p1_action.terastallized is False

    assert first.p2_action.kind is ActionKind.MOVE
    assert first.p2_action.move == "drainingkiss"
    assert first.p2_action.terastallized is True
    assert first.p2_action.tera_type == "Fairy"

    assert first.p1_replacement.kind is ActionKind.REPLACEMENT
    assert first.p1_replacement.switch_in == "p1: Gliscor"
    assert first.p2_replacement is None

    assert second.p1_action.kind is ActionKind.MOVE
    assert second.p1_action.terastallized is False


def test_tera_is_recorded_even_when_the_move_it_enabled_never_went_off():
    # -terastallize is emitted before the move resolves, so a Pokemon that Teras
    # and then flinches really did use its Terastallization. Dropping the flag
    # would tell a forward model the side still has Tera available.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Fake Out|p2a: Raging Bolt",
        "|-terastallize|p2a: Raging Bolt|Fairy",
        "|cant|p2a: Raging Bolt|flinch",
    )))
    action = replay.transitions[0].p2_action
    assert action.kind is ActionKind.BLOCKED
    assert action.blocked_by == "flinch"
    assert action.terastallized is True
    assert replay.transitions[0].state_after.p2.tera_used is True


# ---------------------------------------------------------------------------
# DW-M2.6 - unknown is explicit, and refuses to pass as a real value
# ---------------------------------------------------------------------------

def test_DW_M2_6_unknown_is_not_a_default_and_refuses_truthiness():
    # This is the single most important property of the module. `None` already
    # means "known to hold nothing", so a falsy UNKNOWN would let `if mon.item:`
    # silently report an unrevealed Choice Band as an empty hand - and M3 would
    # score that fabrication as if it were an observation.
    with pytest.raises(TypeError, match="no truth value"):
        bool(UNKNOWN)
    assert is_known(None) is True
    assert is_known(0) is True
    assert is_known(UNKNOWN) is False
    assert repr(UNKNOWN) == "UNKNOWN"


def test_DW_M2_6_unrevealed_item_is_unknown_and_a_removed_one_is_none():
    # Three states that must not collapse into two: never revealed (UNKNOWN),
    # revealed and held (an id), revealed and gone (None).
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Knock Off|p2a: Raging Bolt",
        "|-enditem|p2a: Raging Bolt|Leftovers|[from] move: Knock Off",
        "|-damage|p2a: Raging Bolt|60/100",
    )))
    before = replay.transitions[0].state_before
    after = replay.transitions[0].state_after
    # Before anything was revealed, item, ability and Tera type are all unknown.
    assert before.p2.active.item is UNKNOWN
    assert before.p2.active.ability is UNKNOWN
    assert before.p2.active.tera_type is UNKNOWN
    # Knocked off: known to hold nothing now, which is not the same as unknown.
    assert after.p2.active.item is None


def test_an_item_is_revealed_by_the_from_tag_on_someone_elses_damage():
    # `|-damage|VICTIM|HP|[from] item: Rocky Helmet|[of] HOLDER` is the only
    # announcement Rocky Helmet ever gets, and the damaged Pokemon is not the one
    # holding it - the `[of]` tag names the holder. Getting that backwards would
    # assign the item to the wrong side.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
        "|-damage|p2a: Raging Bolt|70/100",
        "|-damage|p1a: Gliscor|88/100|[from] item: Rocky Helmet|[of] p2a: Raging Bolt",
    )))
    after = replay.transitions[0].state_after
    assert after.p2.active.item == "rockyhelmet"
    assert after.p1.active.item is UNKNOWN
    assert after.p1.active.hp_fraction == pytest.approx(0.88)


def test_DW_M2_6_a_cant_message_hides_which_move_was_selected():
    # `|cant|p1a: X|par` says a move was chosen and failed, not which one. That is
    # a genuinely unknown action, not a missing one, and must not read as "no move".
    replay = parse_replay_log(_log(_one_turn(
        "|cant|p1a: Gliscor|par",
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
    )))
    action = replay.transitions[0].p1_action
    assert action.kind is ActionKind.BLOCKED
    assert action.move is UNKNOWN
    assert action.blocked_by == "par"


def test_a_cant_message_that_names_the_move_records_it():
    # Disable and Taunt name the blocked move, so the choice IS observable there.
    replay = parse_replay_log(_log(_one_turn(
        "|cant|p1a: Gliscor|move: Taunt|Swords Dance",
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
    )))
    action = replay.transitions[0].p1_action
    assert action.move == "swordsdance"
    assert "swordsdance" in replay.transitions[0].state_after.p1.active.revealed_moves


def test_a_side_that_fainted_before_acting_is_unobserved_not_absent():
    # 14.4% of turns in the real corpus end this way. "We saw no evidence" and
    # "they did nothing" are different claims, and only the first one is true here.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
        "|-damage|p1a: Gliscor|0 fnt",
        "|faint|p1a: Gliscor",
    )))
    assert replay.transitions[0].p1_action.kind is ActionKind.UNOBSERVED
    assert replay.transitions[0].p1_action.move is None


# ---------------------------------------------------------------------------
# DW-M2.5 - per-side state reconstruction
# ---------------------------------------------------------------------------

def test_DW_M2_5_team_preview_gives_both_full_teams_before_anything_is_revealed():
    # Team preview is the only place both complete teams appear. It is what makes
    # "how many opponent Pokemon are still unrevealed" answerable for M4.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt")))
    before = replay.transitions[0].state_before
    assert [entry.species for entry in before.p2.preview] == [
        "Raging Bolt", "Kingambit", "Great Tusk"]
    # Only the lead has actually appeared, so only it has observed state.
    assert list(before.p2.team) == ["p2: Raging Bolt"]
    assert before.p2.unrevealed_count == 2
    # A `-*` preview entry hides the forme; the asterisk is preserved rather than
    # guessed into a concrete forme.
    zamazenta = [e for e in before.p1.preview if e.base_species == "Zamazenta"][0]
    assert zamazenta.forme_unrevealed is True
    assert zamazenta.linked_ident is None


def test_DW_M2_5_hp_status_and_boosts_are_tracked_per_pokemon():
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Swords Dance|p1a: Gliscor",
        "|-boost|p1a: Gliscor|atk|2",
        "|move|p2a: Raging Bolt|Thunder Wave|p1a: Gliscor",
        "|-status|p1a: Gliscor|par",
        "|-damage|p1a: Gliscor|84/100 par|[from] item: Life Orb",
    )))
    mon = replay.transitions[0].state_after.p1.active
    assert mon.boosts["atk"] == 2
    assert mon.status == "par"
    assert mon.hp_fraction == pytest.approx(0.84)
    # The denominator is kept because a percentage log is only accurate to ~0.5%,
    # and M3 needs to size its comparison tolerance rather than assume exactness.
    assert mon.hp_denominator == 100


def test_DW_M2_5_hazard_layers_are_counted_and_the_move_prefix_is_normalized():
    # Real logs are inconsistent here: `Spikes` arrives bare while `Stealth Rock`
    # and `Toxic Spikes` arrive as `move: ...`. Both must land on the same id, or
    # a consumer would see two different hazards.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Spikes|p1a: Gliscor",
        "|-sidestart|p1: alice|Spikes",
        "|-sidestart|p1: alice|Spikes",
        "|-sidestart|p1: alice|move: Stealth Rock",
        "|-sidestart|p1: alice|move: Toxic Spikes",
        "|move|p1a: Gliscor|Rapid Spin|p2a: Raging Bolt",
        "|-sideend|p1: alice|move: Stealth Rock|[from] move: Rapid Spin",
    )))
    conditions = replay.transitions[0].state_after.p1.side_conditions
    assert conditions == {"spikes": 2, "toxicspikes": 1}
    assert "stealthrock" not in conditions


def test_DW_M2_5_weather_terrain_and_other_field_effects_are_separated():
    # Terrain gets its own field because a forward model treats it differently
    # from Trick Room; folding both into one bag would lose that.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Sunny Day|p1a: Gliscor",
        "|-weather|SunnyDay",
        "|move|p2a: Raging Bolt|Grassy Terrain|p2a: Raging Bolt",
        "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge|[of] p2a: Raging Bolt",
        "|-fieldstart|move: Trick Room|[of] p2a: Raging Bolt",
        "|-weather|SunnyDay|[upkeep]",
    )))
    after = replay.transitions[0].state_after
    assert after.weather == "sunnyday"
    assert after.terrain == "grassyterrain"
    assert set(after.fields) == {"trickroom"}
    # The [from] ability: tag names the ability and the [of] tag names its owner -
    # that is a real reveal, not an inference.
    assert after.p2.active.ability == "grassysurge"


def test_DW_M2_5_a_clear_field_is_a_known_absence_not_an_unknown():
    # A battle starts with no weather and every change is announced, so "no
    # weather" is an observation. Marking it UNKNOWN would be as wrong as
    # defaulting an unrevealed item to None.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt")))
    before = replay.transitions[0].state_before
    assert before.weather is None and is_known(before.weather)
    assert before.terrain is None and is_known(before.terrain)


def test_DW_M2_5_revealed_moves_accumulate_and_abilities_are_picked_up():
    replay = parse_replay_log(_log(
        _one_turn("|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
                  "|-ability|p2a: Raging Bolt|Protosynthesis", number=1),
        _one_turn("|move|p1a: Gliscor|Knock Off|p2a: Raging Bolt", number=2),
    ))
    mon = replay.transitions[1].state_after.p1.active
    assert mon.revealed_moves == ["earthquake", "knockoff"]
    assert replay.transitions[1].state_after.p2.active.ability == "protosynthesis"


def test_boosts_and_volatiles_reset_when_a_pokemon_leaves_the_field():
    # Boosts are only fully knowable because they reset on switch and every change
    # is announced. Carrying a +2 through a switch-out would make every later
    # damage comparison wrong for that Pokemon.
    replay = parse_replay_log(_log(
        _one_turn("|move|p1a: Gliscor|Swords Dance|p1a: Gliscor",
                  "|-boost|p1a: Gliscor|atk|2",
                  "|-start|p1a: Gliscor|confusion", number=1),
        _one_turn("|switch|p1a: Clefable|Clefable, F|100/100", number=2),
    ))
    gliscor = replay.transitions[1].state_after.p1.team["p1: Gliscor"]
    assert gliscor.boosts["atk"] == 0
    assert gliscor.volatiles == {}
    # Knowledge about the Pokemon survives leaving the field; battle state does not.
    assert gliscor.revealed_moves == ["swordsdance"]


def test_boosts_clamp_at_the_game_limit():
    # The log reports the requested amount, not the clamped result: three Swords
    # Dances announce +2 each. Storing +6 rather than +8 is what keeps the state
    # comparable to a forward model's.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Swords Dance|p1a: Gliscor",
        "|-boost|p1a: Gliscor|atk|2",
        "|-boost|p1a: Gliscor|atk|2",
        "|-boost|p1a: Gliscor|atk|2",
        "|-boost|p1a: Gliscor|atk|2",
    )))
    assert replay.transitions[0].state_after.p1.active.boosts["atk"] == 6


def test_copyboost_direction_follows_the_emitter_not_the_spec():
    # sim/SIM-PROTOCOL.md says boosts go from SOURCE to TARGET. Psych Up
    # (data/moves.ts) emits ('-copyboost', source, target) and its onHit does
    # `source.boosts[i] = target.boosts[i]` - the opposite. Parsed against the
    # emitter, and this test is the record of that decision.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Calm Mind|p2a: Raging Bolt",
        "|-boost|p2a: Raging Bolt|spa|1",
        "|move|p1a: Gliscor|Psych Up|p2a: Raging Bolt",
        "|-copyboost|p1a: Gliscor|p2a: Raging Bolt|[from] move: Psych Up",
    )))
    after = replay.transitions[0].state_after
    assert after.p1.active.boosts["spa"] == 1  # the Psych Up user gained the boost
    assert after.p2.active.boosts["spa"] == 1  # the target keeps its own


# ---------------------------------------------------------------------------
# DW-M2.7 - the mechanics the fidelity harness will trip over
# ---------------------------------------------------------------------------

def test_DW_M2_7_both_players_moving_in_one_turn_is_attributed_by_speed_order():
    # Attribution must follow the Pokemon ident, not log position. A parser that
    # assumed p1 always moves first would silently swap both actions on every
    # turn the opponent outsped, which is roughly half of them.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
        "|-damage|p1a: Gliscor|40/100",
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
        "|-damage|p2a: Raging Bolt|55/100",
    )))
    transition = replay.transitions[0]
    assert transition.p1_action.move == "earthquake"
    assert transition.p2_action.move == "thunderclap"


def test_DW_M2_7_priority_move_going_first_does_not_change_attribution():
    # Same guard from the other direction: Thunderclap has +1 priority, so the
    # slower Pokemon's line appears first. The action still belongs to its user.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
    ), p1_lead="Gliscor, M"))
    assert replay.transitions[0].p1_action.move == "earthquake"
    assert replay.transitions[0].p2_action.kind is ActionKind.MOVE


def test_DW_M2_7_a_voluntary_switch_and_a_pivot_switch_are_different_actions():
    # Switch actions resolve before every move, so position in the turn decides:
    # a switch before the first move is the turn's choice, one after is a
    # consequence of a move already made. Conflating them would tell M3 that a
    # U-turn turn was a switch turn.
    replay = parse_replay_log(_log(
        _one_turn("|switch|p1a: Clefable|Clefable, F|100/100",
                  "|move|p2a: Raging Bolt|Thunderclap|p1a: Clefable", number=1),
        _one_turn("|move|p1a: Clefable|U-turn|p2a: Raging Bolt",
                  "|-damage|p2a: Raging Bolt|80/100",
                  "|switch|p1a: Gliscor|Gliscor, M|100/100|[from] U-turn",
                  "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor", number=2),
    ))
    voluntary, pivot_turn = replay.transitions[0], replay.transitions[1]
    assert voluntary.p1_action.kind is ActionKind.SWITCH
    assert voluntary.mid_turn_switches == ()

    assert pivot_turn.p1_action.kind is ActionKind.MOVE
    assert pivot_turn.p1_action.move == "uturn"
    # The forward model has to be told which Pokemon replaced the pivot user.
    assert pivot_turn.p1_action.pivot_switch_in == "p1: Gliscor"
    assert [a.kind for a in pivot_turn.mid_turn_switches] == [ActionKind.PIVOT]


def test_DW_M2_7_an_eject_button_switch_is_not_a_chosen_turn_action():
    # The awkward case for any position-based rule: the switching side never moved
    # this turn, so a naive "did they move first?" test would call this voluntary.
    # The positional rule (any switch after the turn's first move line is a
    # consequence) gets it right, and the item is recorded as the cause.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
        "|-damage|p1a: Gliscor|40/100",
        "|-enditem|p1a: Gliscor|Eject Button|[from] move: Thunderclap",
        "|switch|p1a: Clefable|Clefable, F|100/100|[from] item: Eject Button",
    )))
    transition = replay.transitions[0]
    assert transition.p1_action.kind is ActionKind.UNOBSERVED
    assert [a.kind for a in transition.mid_turn_switches] == [ActionKind.PIVOT]
    assert transition.mid_turn_switches[0].from_effect == "item: ejectbutton"


def test_DW_M2_7_a_dragged_pokemon_was_not_chosen_at_all():
    # Whirlwind picks the replacement at random. Recording it as a switch action
    # would teach a set-prediction model that the opponent chose to bring it in.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Whirlwind|p1a: Gliscor",
        "|drag|p1a: Clefable|Clefable, F|100/100",
    )))
    transition = replay.transitions[0]
    assert transition.p1_action.kind is ActionKind.UNOBSERVED
    assert [a.kind for a in transition.mid_turn_switches] == [ActionKind.DRAGGED]
    assert transition.state_after.p1.active_ident == "p1: Clefable"


def test_DW_M2_7_a_faint_replacement_lands_after_the_state_after_snapshot():
    # Verified in real logs: a mid-turn faint's replacement |switch| always comes
    # after |upkeep|. state_after is cut there on purpose - a forward model is
    # handed a state and both moves and returns the post-residual state; it does
    # not choose replacements. Folding the replacement in would make every
    # post-faint turn look like a model divergence.
    replay = parse_replay_log(_log(
        "|turn|1",
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
        "|-damage|p1a: Gliscor|0 fnt",
        "|faint|p1a: Gliscor",
        "|upkeep",
        "|switch|p1a: Clefable|Clefable, F|100/100",
        "|-damage|p1a: Clefable|88/100|[from] Stealth Rock",
        "|turn|2",
        "|upkeep",
        "|turn|3",
    ))
    first, second = replay.transitions[0], replay.transitions[1]
    assert first.faints == ("p1: Gliscor",)
    # At the snapshot the fainted Pokemon is still the active one - nothing has
    # replaced it yet.
    assert first.state_after.p1.active_ident == "p1: Gliscor"
    assert first.state_after.p1.active.fainted is True
    assert first.p1_replacement.switch_in == "p1: Clefable"
    # The hazard chip on the way in belongs to the replacement phase, and is
    # therefore visible in the NEXT turn's state_before, not in this state_after.
    assert [c.phase for c in first.hp_changes] == ["turn", "replacement"]
    assert second.state_before.p1.active.hp_fraction == pytest.approx(0.88)


def test_DW_M2_7_residual_damage_is_recorded_with_the_effect_that_caused_it():
    # M3's done-when asks for divergence "broken down by cause", so burn, poison,
    # sand and Leftovers have to be distinguishable from move damage rather than
    # summed into one net delta.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
        "|-damage|p2a: Raging Bolt|70/100",
        "|-weather|Sandstorm|[upkeep]",
        "|-damage|p2a: Raging Bolt|64/100|[from] Sandstorm",
        "|-heal|p1a: Gliscor|100/100|[from] item: Leftovers",
        "|-damage|p2a: Raging Bolt|58/100|[from] brn",
        "|-damage|p2a: Raging Bolt|52/100|[from] psn",
    )))
    changes = replay.transitions[0].hp_changes
    assert [(c.source, c.source_kind) for c in changes] == [
        (None, None),           # direct move damage carries no [from] tag
        ("sandstorm", None),
        ("leftovers", "item"),
        ("brn", None),
        ("psn", None),
    ]
    assert changes[0].before == 1.0 and changes[0].after == pytest.approx(0.70)


def test_DW_M2_7_a_multi_hit_move_records_its_hit_count():
    # Population Bomb hitting 7 times and hitting 3 times are very different
    # observations. None (not 1) when the log did not say, because a default 1
    # would be a fabricated observation.
    replay = parse_replay_log(_log(
        _one_turn("|move|p1a: Gliscor|Population Bomb|p2a: Raging Bolt",
                  "|-damage|p2a: Raging Bolt|90/100",
                  "|-damage|p2a: Raging Bolt|80/100",
                  "|-damage|p2a: Raging Bolt|70/100",
                  "|-hitcount|p2a: Raging Bolt|3", number=1),
        _one_turn("|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt", number=2),
    ))
    assert replay.transitions[0].p1_action.hit_count == 3
    assert replay.transitions[1].p1_action.hit_count is None


def test_DW_M2_7_a_missed_move_is_flagged_in_both_of_its_protocol_forms():
    # Showdown announces a miss either as a `[miss]` tag on the move line or as a
    # separate `|-miss|` message, depending on the move. Both have to set the flag,
    # or a fidelity harness would score a missed attack as a damage prediction
    # failure.
    tagged = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Bleakwind Storm|p2a: Raging Bolt|[miss]",
        "|-miss|p1a: Gliscor|p2a: Raging Bolt",
    )))
    assert tagged.transitions[0].p1_action.missed is True

    separate = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Thunder|p1a: Gliscor",
        "|-miss|p2a: Raging Bolt|p1a: Gliscor",
    )))
    assert separate.transitions[0].p2_action.missed is True
    assert separate.transitions[0].p1_action.missed is False


def test_a_critical_hit_is_attached_to_the_move_that_landed_it():
    # Damage rolls are the noisiest part of any fidelity comparison, and a crit
    # roughly doubles one. Knowing which turns had one lets M3 separate "the model
    # is wrong" from "the roll was extreme".
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
        "|-crit|p1a: Gliscor",
        "|-damage|p1a: Gliscor|20/100",
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
        "|-damage|p2a: Raging Bolt|70/100",
    )))
    assert replay.transitions[0].p2_action.critical_hit is True
    assert replay.transitions[0].p1_action.critical_hit is False


def test_a_called_move_does_not_replace_the_action_or_pollute_the_moveset():
    # Sleep Talk, Dancer and Metronome all emit the chosen move first and the
    # called move second with a [from] tag, so taking the first line per side
    # needs no whitelist of calling moves. The called move is also left out of
    # revealed_moves: a Metronome pick is not in the user's moveset at all, and
    # recording neither is the choice that can never be wrong.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Metronome|p1a: Gliscor",
        "|move|p1a: Gliscor|Fire Blast|p2a: Raging Bolt|[from]Metronome",
        "|-damage|p2a: Raging Bolt|50/100",
    )))
    transition = replay.transitions[0]
    assert transition.p1_action.move == "metronome"
    assert transition.state_after.p1.active.revealed_moves == ["metronome"]


def test_a_locked_move_continuation_is_an_action_with_no_choice_behind_it():
    # Outrage's second turn is emitted as a move with [from]lockedmove and no
    # preceding line. It is the turn's action, but the player did not select it -
    # from_effect is what lets M3 tell those apart.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Outrage|p2a: Raging Bolt|[from]lockedmove",
        "|-damage|p2a: Raging Bolt|40/100",
    )))
    action = replay.transitions[0].p1_action
    assert action.kind is ActionKind.MOVE
    assert action.move == "outrage"
    assert action.from_effect == "lockedmove"


def test_a_fainted_pokemon_keeps_no_phantom_status_from_the_hp_field():
    # The spec is explicit: when HP is 0 the trailing token is `fnt` and must be
    # ignored. Storing it as a status condition would invent one.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p2a: Raging Bolt|Thunderclap|p1a: Gliscor",
        "|-damage|p1a: Gliscor|0 fnt",
        "|faint|p1a: Gliscor",
    )))
    mon = replay.transitions[0].state_after.p1.active
    assert mon.hp_fraction == 0.0
    assert mon.fainted is True
    # It switched in healthy and was never statused, so its status stays the
    # observed None - the `fnt` token never becomes a status condition.
    assert mon.status is None


def test_an_unmodelled_protocol_line_becomes_a_note_rather_than_silence():
    # The escape hatch that keeps the corpus honest: a message this parser does
    # not model must be visible to M3 so it can exclude the turn, instead of being
    # scored as a divergence the parser caused.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
        "|-somethingnobodyhasimplemented|p1a: Gliscor|wat",
    )))
    assert any("somethingnobodyhas" in note
               for note in replay.transitions[0].notes)


def test_illusion_is_flagged_because_earlier_observations_were_about_another_pokemon():
    # |replace| means everything recorded for that slot described the Zoroark.
    # Unwinding it correctly is guesswork, so the transition is flagged and the
    # consumer decides - silently keeping the wrong history would be worse.
    replay = parse_replay_log(_log(_one_turn(
        "|move|p1a: Gliscor|Earthquake|p2a: Raging Bolt",
        "|replace|p2a: Raging Bolt|Zoroark-Hisui, M|100/100",
    )))
    assert any("illusion" in note.lower() for note in replay.transitions[0].notes)


# ---------------------------------------------------------------------------
# DW-M2.8 - the real corpus
# ---------------------------------------------------------------------------

def _corpus_files() -> list[Path]:
    return sorted(_CORPUS_DIR.glob("*.json"))


def test_DW_M2_8_real_corpus_parses_and_holds_its_invariants():
    # Fixtures only prove the parser handles the message shapes someone thought to
    # write down. Real ladder games are where the surprises are, so the corpus is
    # checked for three properties fixtures cannot establish: everything parses,
    # nothing unmodelled slips through, and state_after really does agree with the
    # HP events attributed to the turn.
    files = _corpus_files()
    if not files:
        pytest.skip("no fetched replay corpus at data/replays_showdown "
                    "(run scripts/fetch_showdown_replays.py)")

    unmodelled: list[str] = []
    hp_mismatches: list[tuple] = []
    turns = 0
    for path in files:
        replay = parse_replay_json(json.loads(path.read_text()))
        assert replay.transitions, f"{path.name} parsed to zero turns"
        turns += len(replay.transitions)
        for transition in replay.transitions:
            # A note that starts with '|' is a raw unmodelled protocol line. The
            # prose notes (Illusion, Transform) are known, accepted limitations.
            unmodelled += [n for n in transition.notes if n.startswith("|")]

            last_turn_phase = {c.target: c.after for c in transition.hp_changes
                               if c.phase == "turn"}
            for ident, expected in last_turn_phase.items():
                mon = transition.state_after.side(ident[:2]).team.get(ident)
                if mon is None or not is_known(mon.hp_fraction):
                    continue
                # Carve-out: a Pokemon that faints without a damage line (Healing
                # Wish, Explosion, Final Gambit) is announced only by |faint|, and
                # no synthetic HP event is invented for it.
                if mon.fainted and expected > 0:
                    continue
                if abs(mon.hp_fraction - expected) > 1e-9:
                    hp_mismatches.append((path.name, transition.turn, ident))

    assert not unmodelled, f"unmodelled protocol lines: {unmodelled[:5]}"
    assert not hp_mismatches, f"state_after disagrees with hp_changes: {hp_mismatches[:5]}"
    assert turns > 0


def test_DW_M2_8_real_corpus_observes_both_sides_on_most_turns():
    # The corpus exists to answer "what did BOTH players do". If the parser could
    # only attribute one side, that would show up here as a collapsed rate rather
    # than as a crash - the failure mode the Metamon corpus already has.
    files = _corpus_files()
    if not files:
        pytest.skip("no fetched replay corpus at data/replays_showdown "
                    "(run scripts/fetch_showdown_replays.py)")

    both = total = 0
    for path in files:
        for transition in parse_replay_json(
                json.loads(path.read_text())).transitions:
            total += 1
            both += (transition.p1_action.kind is not ActionKind.UNOBSERVED
                     and transition.p2_action.kind is not ActionKind.UNOBSERVED)
    # Measured at 85% over the first 50-replay corpus; the shortfall is entirely
    # Pokemon that fainted before acting, which is genuinely unobservable. The
    # threshold is set well below the measurement so it catches a regression
    # rather than normal variation between corpora.
    assert both / total > 0.75, f"only {both}/{total} turns have both actions"
