"""Phase 6 / M3: does the forward model actually predict what happened?

Phase 4's diagnosis - "the search is fine, the model cannot represent the
game" - has been an assertion since 2026-08-30. This module turns it into a
measurement, and does the same for poke-engine before Phase 6 stakes
everything on it. A forward model that is wrong in a way nobody measured is
exactly how Phase 4 spent four milestones building a correct search over an
unusable model.

The method: take a real gen9ou game off Showdown's replay API, stop at the
start of each turn, ask the model what both players' chosen actions will do,
and compare its answer to what the log says actually happened.

Two decisions shape everything below.

**poke-env is the state carrier, on both ends.** A replay log drives a real
`poke_env.battle.Battle` (verified: the only protocol messages poke-env
refuses are `|t:|` and `|win|`, both inert here), so `state_before` and
`state_after` are both produced by `poke_engine_state.state_from_poke_env` -
the same tested translator the player will use in M5, not a second
replay-specific one written for the harness. Predicted and observed states
are then the same type and comparable field by field. M2's parser is used
for exactly one thing that a `Battle` cannot supply: **what each player
chose** on a given turn. That is what M2 exists for.

**Two numbers, not one, because two different things can be wrong.**

- *Representability* - could the model even be asked this question from what
  the battle had revealed? At turn 3 a Lokix that has never used Knock Off
  has no Knock Off in its moveset, and poke-engine resolves an action by
  name against the active Pokemon's *current* moves, so `generate_
  instructions(..., "knockoff", ...)` raises. Nothing is wrong with the
  forward model there. What is missing is set prediction, and the rate at
  which this happens is M4's baseline, stated as a measurement instead of
  Foul Play's assertion that set prediction is where the strength lives.
- *Fidelity* - given the action, is the resulting state right? Measured
  under an oracle that injects precisely the knowledge the turn requires
  (the move about to be used, the species about to be switched to, the Tera
  type about to be revealed) and nothing else. This isolates model error
  from knowledge error.

The oracle goes in through M3's `UnknownFiller` seam rather than around it.
That is deliberate: it is the same seam M4's usage-statistics filler uses,
so the harness exercises the interface M4 depends on, and the filler
structurally cannot overwrite an observation - `OracleFiller` cannot quietly
paper over a real divergence.

**Stochastic outcomes get two scores, and the gap between them is the
finding.** `generate_instructions` returns every outcome of a move pair with
its probability - damage rolls, crits, misses, secondary effects. A single
observed turn took one of those branches. So each turn is scored twice:
`modal` (the likeliest branch, i.e. what a model asked for one prediction
would say) and `best` (the closest branch to what happened, i.e. whether the
model could represent the observed outcome at all). A turn where `modal`
diverges and `best` matches is a damage roll, not a bug. A turn where `best`
diverges too is a mechanic the model does not have.

**Only observable quantities are compared.** A replay never shows EVs,
natures, exact HP under HP Percentage Mod, screen durations, or an
unrevealed Pokemon - so those are excluded rather than scored against
values the harness itself invented. HP is compared as a fraction with a
tolerance sized to the percent quantization, and reported with its
magnitude, because "off by 1%" and "off by 60%" are not the same finding.

**One backend is implemented.** `ForwardModelBackend` is a real seam and the
comparison code is backend-agnostic, but only `PokeEngineBackend` exists.
Our own `cpp/src/forward_model.cpp` is not wired up: the integration shape
this milestone was meant to decide was already decided on throughput grounds
(notes/decision-poke-engine-search-replaces-the-cpp-search.md), so scoring
the C++ model now would document a component that is no longer on the
critical path, at the cost of a `resolve_turn` binding that does not exist
yet. The seam is here so that decision stays reversible.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

import poke_engine
from poke_env.battle.battle import Battle
from poke_env.data import to_id_str

from battle_engine.poke_engine_state import (
    UNKNOWN_ITEM,
    RevealedOnlyFiller,
    SideObservation,
    SlotFill,
    UnknownToPokeEngine,
    is_known_move,
    is_known_species,
    state_from_poke_env,
)
from battle_engine.replay_log import (
    Action,
    ActionKind,
    ParsedReplay,
    TurnTransition,
    is_known,
    parse_replay_json,
)

# poke-engine's placeholder species id, and simultaneously the string
# `MoveChoice::from_string` reserves for "no action" - see NO_ACTION below.
PLACEHOLDER_ID = "none"
NO_ACTION = "none"
NO_WEATHER = "none"  # also poke-engine's "no terrain"

# A replay under HP Percentage Mod reports HP to the nearest whole percent,
# on both sides. So an exactly-correct prediction can still differ from the
# observation by up to half a percent of max HP in each direction, and the
# two roundings compound. 2% is that, doubled again for headroom - anything
# inside it is not evidence of a modelling error. The magnitude is always
# reported alongside the verdict so a reader can see what this threshold is
# hiding.
HP_TOLERANCE = 0.02

# A turn whose only error is an HP figure inside this band predicted the right
# events - who moved first, what landed, what fainted, what was inflicted -
# and got the damage number wrong. For a search that is a far cheaper error
# than a missed mechanic, so the report tiers it separately instead of
# lumping it in with "wrong".
NEAR_MISS_HP = 0.10

# The two scoring conditions, keyed by the `hindsight` flag that selects them.
CONDITIONS = {False: "action-oracle", True: "hindsight-oracle"}

# Categories that are reported but do not count toward "the model got this
# turn right", because a difference in them is at least as likely to be the
# harness as the model:
#
# - `item_revealed`: the model's value is the explicit `unknownitem`
#   sentinel. That is not a wrong prediction, it is a declined one - the item
#   became visible *during* the turn being scored (Leftovers healing at
#   upkeep is the common case). Its real cost already lands in `hp`. The
#   count is kept because it is a direct measure of how often a turn hands
#   set prediction a new fact.
# - `volatile`: the observed set comes from a translator that documents
#   dropping 139 of poke-env's 224 `Effect` members, so an absent volatile
#   says nothing about whether the battle had one. Measured examples that are
#   translator artifacts rather than model error: Heal Block (dropped by us,
#   correctly set by poke-engine) and Roost's typechange (cleared by
#   poke-engine at end of turn, still live in poke-env at `|upkeep|`).
INFORMATIONAL = frozenset({"item_revealed", "volatile"})


# ---------------------------------------------------------------------------
# Driving a replay log through poke-env
# ---------------------------------------------------------------------------


class ReplayDriver:
    """Feeds a raw `|`-protocol log to a real `poke_env.battle.Battle`.

    Yields the live battle at the two instants a turn is bracketed by:
    `|turn|N` (before either player acts) and the `|upkeep|` that closes turn
    N (after residuals, before any faint replacement). That cut is the same
    one `replay_log.TurnTransition` uses, so the two agree about what "after"
    means without either having to know about the other.

    The battle is **live and mutated after each yield** - translate it inside
    the loop, do not keep the reference.

    Protocol messages poke-env has no handler for raise `NotImplementedError`
    out of `parse_message`. They are counted in `unhandled` rather than
    silently dropped: over the gen9ou corpus the only two are `|t:|`
    (timestamps) and `|win|`, neither of which carries battle state, but a
    third one appearing is a signal, not noise.

    **The username matters and is not cosmetic.** poke-env derives
    `player_role` by comparing each `|player|` line's username to the
    `Battle`'s own, and *infers the opposite role when they do not match*
    (`abstract_battle.py`: `else: self._player_role = "p1" if player == "p2"
    else "p2"`). Passing a placeholder username happens to land on the right
    role after both opening `|player|` lines have been seen - and then a
    mid-battle re-announcement (a player disconnecting and reconnecting emits
    another `|player|p1|...`) flips it, after which p1's Pokemon are filed
    into `opponent_team` and the battle dies with "p2's team already has 6
    pokemons". Two of 300 corpus replays hit exactly that. So the real p1
    username is read out of the log and used.
    """

    def __init__(self, log_text: str, battle_tag: str, *, player_username: Optional[str] = None) -> None:
        self.log_text = log_text
        self.battle_tag = battle_tag
        self.player_username = player_username or self._p1_username(log_text)
        self.unhandled: Counter = Counter()

    @staticmethod
    def _p1_username(log_text: str) -> str:
        """p1's username, from the log's own first `|player|p1|` line.

        Read from the log rather than from the replay JSON's `players` array,
        so the driver works on a bare protocol log and does not depend on that
        array being ordered p1-first.
        """
        for line in log_text.split("\n"):
            fields = line.split("|")
            if len(fields) > 3 and fields[1] == "player" and fields[2] == "p1" and fields[3]:
                return fields[3]
        return "p1"

    def __iter__(self) -> Iterator[Tuple[str, int, Battle]]:
        # poke-env logs a warning per unrecognized effect; a 300-replay run
        # would emit tens of thousands of them and drown the report.
        battle = Battle(self.battle_tag, self.player_username, logging.getLogger("battle_engine.fidelity"), gen=9)
        for line in self.log_text.split("\n"):
            if not line.startswith("|"):
                continue
            fields = line.split("|")[1:]
            if not fields:
                continue
            try:
                # poke-env expects the leading empty field a raw split on the
                # full line would produce.
                battle.parse_message([""] + fields)
            except NotImplementedError:
                self.unhandled[fields[0]] += 1
                continue
            if fields[0] == "turn":
                yield "turn", int(fields[1]), battle
            elif fields[0] == "upkeep":
                yield "upkeep", battle.turn, battle


# ---------------------------------------------------------------------------
# What each player chose, as something poke-engine will accept
# ---------------------------------------------------------------------------


class UnscorableTurn(Exception):
    """This turn cannot be scored, and the reason is not a model failure.

    `reason` is the aggregation key. It is deliberately a small closed
    vocabulary rather than free text - the report ranks by it.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class EngineAction:
    """One side's chosen action, in the string form `generate_instructions`
    takes, plus what the state must already know for that string to resolve.

    poke-engine addresses an action by *name*, against the side's current
    contents: a move by its id among the active Pokemon's moves, a switch by
    the **species id** of the Pokemon being switched to (`MoveChoice::
    from_string`, genx/state.rs). Two consequences that shape this harness:

    - You cannot ask it to simulate an action the state does not already
      believe is available. Hence `needs_move` / `needs_species`.
    - The string `"none"` is reserved for `MoveChoice::None`, and is also the
      id of the placeholder Pokemon `RevealedOnlyFiller` puts in unrevealed
      slots. A switch into an unrevealed slot is therefore not merely
      unlikely to be right, it is unaddressable. That is a constraint on M5's
      action space, not just on this file.
    """

    text: str
    kind: str  # "move" | "switch"
    needs_move: Optional[str] = None
    needs_species: Optional[str] = None
    needs_tera_type: Optional[str] = None


def engine_action(action: Action, transition: TurnTransition) -> EngineAction:
    """Translate one observed action, or raise `UnscorableTurn` saying why not.

    The kinds that raise are not failures of anything - they are turns where
    the log does not record a decision that could be replayed:

    - `UNOBSERVED`: no evidence either way (usually a Pokemon that fainted
      before it acted). 7.7% of actions over the corpus.
    - `BLOCKED`: a `|cant|` turn. The selection was made but did not execute,
      and is often not even named.
    - `DRAGGED`: Whirlwind or Roar moved this Pokemon; nobody chose it.
    - `PIVOT`: an item-forced mid-turn switch (Eject Button). The destination
      was chosen, the timing was not, so it is not this turn's action.
    - `REPLACEMENT`: a post-faint send-in, which happens after the turn has
      resolved and lives on the transition, not in the action slot.
    """
    if action.kind is ActionKind.UNOBSERVED:
        raise UnscorableTurn("unobserved_action")
    if action.kind is ActionKind.BLOCKED:
        raise UnscorableTurn("blocked_action", action.blocked_by or "")
    if action.kind is ActionKind.DRAGGED:
        raise UnscorableTurn("dragged")
    if action.kind is ActionKind.PIVOT:
        raise UnscorableTurn("forced_pivot")
    if action.kind is ActionKind.REPLACEMENT:
        raise UnscorableTurn("replacement_in_action_slot")

    if action.kind is ActionKind.SWITCH:
        species = _switch_target_species(action, transition)
        return EngineAction(text=species, kind="switch", needs_species=species)

    if not is_known(action.move) or action.move is None:
        raise UnscorableTurn("move_unknown")
    move_id = to_id_str(action.move)
    if not is_known_move(move_id):
        # Z-moves and G-Max moves are the only gen9-legal gap measured in
        # M3, and neither is legal in OU - so this is a naming bug if it
        # ever fires, and is worth seeing rather than absorbing.
        raise UnscorableTurn("move_not_in_engine", move_id)

    tera_type = None
    text = move_id
    if action.terastallized:
        text = f"{move_id}-tera"
        tera_type = action.tera_type
    return EngineAction(text=text, kind="move", needs_move=move_id, needs_tera_type=tera_type)


def _switch_target_species(action: Action, transition: TurnTransition) -> str:
    """The species id of the Pokemon a SWITCH action brought in.

    `Action.switch_in` is an ident (`p1: Nickname`), and poke-engine wants a
    species, so the transition's own post-state is where the mapping lives.
    `species` is used, never `base_species`: they differ for exactly the
    formes poke-engine stats separately (`urshifurapidstrike` collapses to
    `urshifu`), which produces a real id with the wrong stats and no error -
    the M3 finding in notes/gotcha-poke-env-poke-engine-name-mismatches.md.
    """
    ident = action.switch_in
    if ident is None:
        raise UnscorableTurn("switch_without_target")
    mon = transition.state_after.side(action.player).team.get(ident)
    if mon is None:
        mon = transition.state_before.side(action.player).team.get(ident)
    if mon is None:
        raise UnscorableTurn("switch_target_not_in_snapshot", ident)
    species = to_id_str(mon.species)
    if not is_known_species(species):
        raise UnscorableTurn("species_not_in_engine", species)
    if species == PLACEHOLDER_ID:
        raise UnscorableTurn("switch_target_is_placeholder", ident)
    return species


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hindsight:
    """Everything one battle ever reveals about one Pokemon, keyed by species.

    Used to build the second scoring condition. Only *monotone* attributes go
    in here - ones that, once shown, were true from turn 1:

    - ability: shown by a proc (`|-ability|`, Intimidate, Sand Stream, ...).
    - tera type: shown by `|-terastallize|`.
    - moves: cumulative; a move used on turn 15 was in the moveset on turn 1.

    **Items are deliberately absent.** An item is not monotone - Boots stay
    on, a Berry is eaten, Knock Off removes one - so a battle-final item is
    not a valid turn-N item and back-filling it would fabricate observations.
    That leaves items as the largest known gap in this condition, and half of
    them have no poke-engine mechanics anyway.

    EVs, IVs and natures are never revealed by any replay, so no oracle built
    from a log can supply them. That is the floor on this whole measurement,
    and the report says so.
    """

    ability: Optional[str] = None
    tera_type: Optional[str] = None
    moves: Tuple[str, ...] = ()


def hindsight_knowledge(parsed: ParsedReplay) -> Dict[str, Dict[str, Hindsight]]:
    """{"p1"/"p2": {species id: Hindsight}} from a whole parsed battle.

    Read from the last transition's post-state, where the parser's running
    per-Pokemon knowledge has accumulated everything the log ever showed.

    Measured over the corpus, this is far from omniscient: a gen9 OU battle
    eventually reveals only 40.5% of the abilities and 26.8% of the items of
    the Pokemon that actually appear. That number is the reason Foul Play
    uses Smogon usage statistics rather than in-battle inference alone, and
    it means this condition is a *lower* bound on set prediction's headroom,
    not an upper one.
    """
    out: Dict[str, Dict[str, Hindsight]] = {"p1": {}, "p2": {}}
    if not parsed.transitions:
        return out
    final = parsed.transitions[-1].state_after
    for player in ("p1", "p2"):
        for mon in final.side(player).team.values():
            species = to_id_str(mon.species)
            out[player][species] = Hindsight(
                ability=to_id_str(mon.ability) if is_known(mon.ability) and mon.ability else None,
                tera_type=mon.tera_type if is_known(mon.tera_type) and mon.tera_type else None,
                moves=tuple(to_id_str(m) for m in mon.revealed_moves),
            )
    return out


@dataclass
class _SideOracle:
    """What one side's state must be told for this turn's action to resolve,
    plus (in the hindsight condition) what the battle will eventually show."""

    active_species: Optional[str] = None
    move: Optional[str] = None
    switch_species: Optional[str] = None
    tera_type: Optional[str] = None
    knowledge: Dict[str, Hindsight] = field(default_factory=dict)


class OracleFiller:
    """Supplies exactly the unrevealed facts this turn's actions require.

    This is a measurement instrument, not a set-prediction strategy: it is
    told the answer. It exists so that a divergence can be attributed to the
    forward model rather than to the battle not having revealed enough yet.
    Everything it is *not* explicitly given stays exactly as
    `RevealedOnlyFiller` would leave it, so unrevealed slots are still
    placeholders and unrevealed items are still `unknownitem` - the oracle
    lifts one specific constraint, not the fog of war.

    It goes through `UnknownFiller` rather than mutating the built state
    because the seam already guarantees the thing this harness most needs:
    a fill can never overwrite an observation, so the oracle cannot silently
    correct a real divergence into a match.
    """

    name = "oracle"

    def __init__(self, ours: _SideOracle, theirs: _SideOracle) -> None:
        self._by_side = {True: ours, False: theirs}

    def fill_side(self, observation: SideObservation) -> Sequence[SlotFill]:
        oracle = self._by_side[observation.is_ours]
        revealed = {slot.species for slot in observation.slots if slot.species}
        # Only fill the switch target into an empty slot if it is genuinely
        # unrevealed; a revealed one is already addressable by species.
        pending_species = (
            oracle.switch_species
            if oracle.switch_species and oracle.switch_species not in revealed
            else None
        )
        fills: List[SlotFill] = []
        for slot in observation.slots:
            if slot.species is not None and slot.species == oracle.active_species:
                known = oracle.knowledge.get(slot.species, Hindsight())
                # The action's move goes first so it survives the four-move
                # cap; the rest is hindsight and matters less for one turn.
                moves = ((oracle.move,) if oracle.move else ()) + known.moves
                fills.append(
                    SlotFill(
                        moves=moves,
                        # The turn's own Tera type wins: it is about to be
                        # observed, whereas hindsight may name a later one.
                        tera_type=oracle.tera_type or known.tera_type,
                        ability=known.ability,
                    )
                )
            elif slot.species is None and pending_species is not None:
                known = oracle.knowledge.get(pending_species, Hindsight())
                fills.append(
                    SlotFill(
                        species=pending_species,
                        moves=known.moves,
                        tera_type=known.tera_type,
                        ability=known.ability,
                    )
                )
                pending_species = None
            elif slot.species is not None and slot.species in oracle.knowledge:
                known = oracle.knowledge[slot.species]
                fills.append(
                    SlotFill(moves=known.moves, tera_type=known.tera_type, ability=known.ability)
                )
            else:
                fills.append(SlotFill())
        return fills


def _side_oracle(
    battle: Battle, action: EngineAction, *, ours: bool, knowledge: Optional[Dict[str, Hindsight]] = None
) -> _SideOracle:
    active = battle.active_pokemon if ours else battle.opponent_active_pokemon
    return _SideOracle(
        active_species=active.species if active is not None else None,
        move=action.needs_move,
        switch_species=action.needs_species,
        tera_type=action.needs_tera_type,
        knowledge=knowledge or {},
    )


def _addressable(state: Any, side_name: str, action: EngineAction) -> bool:
    """Would `action` resolve against `state` as it stands, with no oracle?

    This is the representability question, answered without calling
    `generate_instructions` (which would need the *pair* to be addressable
    and could not tell the two sides apart in its error).
    """
    side = getattr(state, side_name)
    if action.kind == "switch":
        return any(p.id.lower() == action.text and str(i) != side.active_index for i, p in enumerate(side.pokemon))
    active = side.pokemon[int(side.active_index)]
    return any(m.id.lower() == action.needs_move for m in active.moves)


# ---------------------------------------------------------------------------
# Comparing two states on what a replay can actually see
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    """One observable the model got wrong.

    `magnitude` is meaningful only for `hp`; it is the absolute difference in
    HP fraction. Everything else is categorical and leaves it at 0.
    """

    category: str
    side: str
    subject: str
    observed: Any
    predicted: Any
    magnitude: float = 0.0


# Screens and Tailwind are excluded on purpose: poke-engine stores turns
# remaining, poke-env stores the turn a screen started, and Light Clay is
# not observable until the item is revealed - so the translator's own value
# is an assumption (it records it as one) and scoring against it would be
# scoring the harness's guess. Presence is compared instead, below.
_HAZARD_FIELDS = ("spikes", "toxic_spikes", "stealth_rock", "sticky_web")
_SCREEN_FIELDS = ("reflect", "light_screen", "aurora_veil", "tailwind")
_BOOST_FIELDS = (
    "attack_boost",
    "defense_boost",
    "special_attack_boost",
    "special_defense_boost",
    "speed_boost",
    "accuracy_boost",
    "evasion_boost",
)


def _slots_by_species(side: Any) -> Dict[str, Any]:
    """Real Pokemon on a side, keyed by species id.

    Keyed by species rather than index because the two states being compared
    do not agree on indices: a switch during the turn reveals a Pokemon, so
    the observed state has a slot the predicted state filled with a
    placeholder. Placeholders are dropped - there is nothing to compare, and
    several of them share the id `none`.

    Species Clause guarantees the key is unique within a side in gen9 OU.
    """
    out: Dict[str, Any] = {}
    for mon in side.pokemon:
        species = mon.id.lower()
        if species == PLACEHOLDER_ID:
            continue
        out[species] = mon
    return out


def _volatiles(side: Any) -> frozenset:
    """A side's volatile statuses, normalized.

    `none` is discarded: poke-engine's own serializer round-trip inserts a
    spurious `NONE` into the set (M3 module docstring, point 7), so its
    presence says nothing about the battle.
    """
    return frozenset(v.lower() for v in side.volatile_statuses) - {PLACEHOLDER_ID}


def _field_divergence(
    category: str, before: str, observed: str, predicted: str
) -> Optional[Divergence]:
    """Weather / terrain, with expiry split out from everything else.

    How long a weather has left is not observable: poke-env records the turn
    it started, Smooth/Heat/Damp/Icy Rock extends the base 5 turns to 8, and
    poke-engine does not implement any of those four rocks (they are among
    the 59.5% of items it has no mechanics for). So the translator supplies
    an assumed counter and the model will sometimes clear a weather a real
    battle still has, or keep one it has lost.

    That is a real cost to a search, so it is reported - but it is a
    different finding from "the model failed to notice Sand Stream", so it
    gets its own category rather than being counted as a missed event.
    """
    if observed == predicted:
        return None
    if predicted == NO_WEATHER and observed == before:
        return Divergence(f"{category}_expired_early", "field", "", observed, predicted)
    if observed == NO_WEATHER and predicted == before:
        return Divergence(f"{category}_expired_late", "field", "", observed, predicted)
    return Divergence(category, "field", "", observed, predicted)


def compare_states(observed: Any, predicted: Any, before: Any) -> Tuple[Divergence, ...]:
    """Every observable difference between what happened and what was predicted.

    Deliberately excluded, because a replay does not show them and the values
    on both sides come from the harness's own assumptions rather than from
    the battle: EVs, natures, exact stats, screen and weather *durations*,
    unrevealed Pokemon, PP, and abilities/items that were never revealed.
    Comparing those would measure this harness against itself.
    """
    out: List[Divergence] = []
    for side_name in ("side_one", "side_two"):
        obs_side = getattr(observed, side_name)
        pred_side = getattr(predicted, side_name)

        obs_active = obs_side.pokemon[int(obs_side.active_index)].id.lower()
        pred_active = pred_side.pokemon[int(pred_side.active_index)].id.lower()
        if obs_active != pred_active:
            out.append(Divergence("active", side_name, "", obs_active, pred_active))

        obs_mons = _slots_by_species(obs_side)
        pred_mons = _slots_by_species(pred_side)
        for species in sorted(obs_mons.keys() & pred_mons.keys()):
            o, p = obs_mons[species], pred_mons[species]
            o_frac = o.hp / o.maxhp if o.maxhp else 0.0
            p_frac = p.hp / p.maxhp if p.maxhp else 0.0
            delta = abs(o_frac - p_frac)
            if delta > HP_TOLERANCE:
                out.append(Divergence("hp", side_name, species, round(o_frac, 4), round(p_frac, 4), delta))
            if (o.hp == 0) != (p.hp == 0):
                out.append(Divergence("fainted", side_name, species, o.hp == 0, p.hp == 0))
            if o.status.lower() != p.status.lower():
                out.append(Divergence("status", side_name, species, o.status.lower(), p.status.lower()))
            o_item, p_item = o.item.lower(), p.item.lower()
            if o_item != p_item:
                category = "item_revealed" if p_item == UNKNOWN_ITEM else "item"
                out.append(Divergence(category, side_name, species, o_item, p_item))
            if o.terastallized != p.terastallized:
                out.append(Divergence("terastallized", side_name, species, o.terastallized, p.terastallized))

        for boost in _BOOST_FIELDS:
            o_val, p_val = getattr(obs_side, boost), getattr(pred_side, boost)
            if o_val != p_val:
                out.append(Divergence("boost", side_name, boost, o_val, p_val))

        for hazard in _HAZARD_FIELDS:
            o_val = getattr(obs_side.side_conditions, hazard)
            p_val = getattr(pred_side.side_conditions, hazard)
            if o_val != p_val:
                out.append(Divergence("hazard", side_name, hazard, o_val, p_val))
        for screen in _SCREEN_FIELDS:
            o_val = getattr(obs_side.side_conditions, screen) > 0
            p_val = getattr(pred_side.side_conditions, screen) > 0
            if o_val != p_val:
                out.append(Divergence("screen", side_name, screen, o_val, p_val))

        obs_vol, pred_vol = _volatiles(obs_side), _volatiles(pred_side)
        if obs_vol != pred_vol:
            out.append(
                Divergence("volatile", side_name, "", tuple(sorted(obs_vol)), tuple(sorted(pred_vol)))
            )

    for category, attribute in (("weather", "weather"), ("terrain", "terrain")):
        divergence = _field_divergence(
            category,
            getattr(before, attribute).lower(),
            getattr(observed, attribute).lower(),
            getattr(predicted, attribute).lower(),
        )
        if divergence is not None:
            out.append(divergence)
    if observed.trick_room != predicted.trick_room:
        out.append(Divergence("trick_room", "field", "", observed.trick_room, predicted.trick_room))
    return tuple(out)


# ---------------------------------------------------------------------------
# The backend seam
# ---------------------------------------------------------------------------


class ForwardModelBackend(Protocol):
    """A forward model, as this harness needs to see one.

    See the module docstring for why `PokeEngineBackend` is currently the
    only implementation.
    """

    name: str

    def branches(self, state: Any, side_one_action: str, side_two_action: str) -> Sequence[Tuple[float, Any]]:
        """Every outcome of the move pair, as (probability, resulting state).

        Probabilities are fractions summing to ~1. The states must be
        comparable to a state produced by `state_from_poke_env`.
        """
        ...


class PokeEngineBackend:
    """poke-engine v0.0.48, gen9 build, driven from Python.

    `generate_instructions` enumerates outcomes as *instruction lists* rather
    than states; `State.apply_instructions` turns each into a state. Applying
    every branch is more work than a search would do (a search samples one),
    which is the point: the harness needs `best`, not just `modal`.
    """

    name = "poke-engine"

    def branches(self, state: Any, side_one_action: str, side_two_action: str) -> Sequence[Tuple[float, Any]]:
        outcomes = poke_engine.generate_instructions(state, side_one_action, side_two_action)
        return [(o.percentage / 100.0, state.apply_instructions(o)) for o in outcomes]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def scoring_divergences(divergences: Sequence[Divergence]) -> Tuple[Divergence, ...]:
    """The subset that counts toward the verdict. See `INFORMATIONAL`."""
    return tuple(d for d in divergences if d.category not in INFORMATIONAL)


@dataclass(frozen=True)
class TurnScore:
    """One scored turn."""

    battle_id: str
    turn: int
    p1_kind: str
    p2_kind: str
    representable: bool
    """True if both actions resolved against the revealed-only state - i.e.
    the model could have been asked this question during a real game with no
    set prediction at all."""
    missing: Tuple[str, ...]
    """What the oracle had to supply, as `p1:move` / `p2:species` tags. The
    M4 baseline is built from these."""
    n_branches: int
    modal_probability: float
    modal_divergences: Tuple[Divergence, ...]
    best_probability: float
    best_divergences: Tuple[Divergence, ...]
    seconds: float

    @property
    def modal_exact(self) -> bool:
        return not scoring_divergences(self.modal_divergences)

    @property
    def best_exact(self) -> bool:
        return not scoring_divergences(self.best_divergences)


@dataclass
class FidelityReport:
    """Aggregated over a corpus. Every field is a count of turns unless named
    otherwise."""

    backend: str
    condition: str = "action-oracle"
    replays: int = 0
    turns_seen: int = 0
    scores: List[TurnScore] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    unhandled_protocol: Counter = field(default_factory=Counter)
    panics: List[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def scored(self) -> int:
        return len(self.scores)

    def _rate(self, n: int, d: int) -> str:
        return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else f"{n}/0 = n/a"

    def render(self) -> str:
        lines: List[str] = []
        add = lines.append
        add(f"Forward-model fidelity - backend: {self.backend}   condition: {self.condition}")
        add(f"  replays: {self.replays}   turns in corpus: {self.turns_seen}   scored: {self.scored}")

        add("")
        add("Turns not scored (no decision in the log to replay, or a translation gap):")
        skipped_total = sum(self.skipped.values())
        for reason, n in self.skipped.most_common():
            share = f"  ({100.0 * n / self.turns_seen:.1f}% of corpus)" if self.turns_seen else ""
            add(f"  {reason:<34} {n:>6}{share}")
        add(f"  {'TOTAL':<34} {skipped_total:>6}")

        if not self.scores:
            return "\n".join(lines)

        add("")
        add("Representability from revealed information only (the M4 baseline):")
        representable = sum(1 for s in self.scores if s.representable)
        add(f"  {'both actions addressable':<34}{self._rate(representable, self.scored)}")
        missing = Counter(tag for s in self.scores for tag in s.missing)
        for tag, n in missing.most_common():
            add(f"  oracle had to supply {tag:<15} {n:>6}  ({100.0 * n / self.scored:.1f}% of scored turns)")

        add("")
        add("Fidelity, given the action (oracle-filled):")
        modal_exact = sum(1 for s in self.scores if s.modal_exact)
        best_exact = sum(1 for s in self.scores if s.best_exact)
        add(f"  {'modal branch exactly right':<34}{self._rate(modal_exact, self.scored)}")
        add(f"  {'some branch exactly right':<34}{self._rate(best_exact, self.scored)}")
        add(f"  {'-> stochastic-only divergence':<34}{self._rate(best_exact - modal_exact, self.scored)}")
        add(f"  {'-> unrepresentable outcome':<34}{self._rate(self.scored - best_exact, self.scored)}")
        near = sum(
            1
            for s in self.scores
            if (divergences := scoring_divergences(s.best_divergences))
            and all(d.category == "hp" and d.magnitude < NEAR_MISS_HP for d in divergences)
        )
        add(f"  {f'+ only error is HP within {NEAR_MISS_HP:.0%}':<34}{self._rate(near, self.scored)}")
        add(f"  {'= right or near-right':<34}{self._rate(best_exact + near, self.scored)}")

        signed = [
            float(d.predicted) - float(d.observed)
            for s in self.scores
            for d in s.best_divergences
            if d.category == "hp"
        ]
        if signed:
            add("")
            add("HP error, where the model and the battle disagree (best branch):")
            ordered = sorted(signed)
            median = ordered[len(ordered) // 2]
            under = sum(1 for x in signed if x > 0)
            add(f"  {'mean signed (predicted - real)':<32}{sum(signed) / len(signed):+.1%}")
            add(f"  {'median signed':<32}{median:+.1%}")
            add(f"  {'model under-damaged':<32}{self._rate(under, len(signed))}")
            buckets: Counter = Counter()
            for magnitude in (d.magnitude for s in self.scores for d in s.best_divergences if d.category == "hp"):
                for edge, label in ((0.05, "<5%"), (0.10, "5-10%"), (0.20, "10-20%"), (0.40, "20-40%")):
                    if magnitude < edge:
                        buckets[label] += 1
                        break
                else:
                    buckets[">40%"] += 1
            for label in ("<5%", "5-10%", "10-20%", "20-40%", ">40%"):
                add(f"  |dHP| {label:<28}{self._rate(buckets[label], len(signed))}")

        for label, key in (("modal", "modal_divergences"), ("best", "best_divergences")):
            add("")
            add(f"Divergence by cause ({label} branch), ranked by turns affected:")
            per_cause: Counter = Counter()
            magnitude: Dict[str, List[float]] = {}
            for score in self.scores:
                seen = set()
                for div in getattr(score, key):
                    seen.add(div.category)
                    if div.category == "hp":
                        magnitude.setdefault("hp", []).append(div.magnitude)
                per_cause.update(seen)
            for cause, n in per_cause.most_common():
                extra = "   [informational, not scored]" if cause in INFORMATIONAL else ""
                if cause == "hp" and magnitude.get("hp"):
                    vals = magnitude["hp"]
                    extra = f"   mean |dHP| {100 * sum(vals) / len(vals):.1f}%  max {100 * max(vals):.1f}%"
                add(f"  {cause:<20} {n:>6}  ({100.0 * n / self.scored:.1f}% of scored turns){extra}")

        add("")
        branches = [s.n_branches for s in self.scores]
        elapsed = sum(s.seconds for s in self.scores)
        add("Throughput (single-threaded, Python-driven):")
        add("  Timed around the backend call only - enumerating every outcome of the move")
        add("  pair and applying each. Translation and comparison are harness cost, not")
        add("  model cost, and are in the wall-clock line instead.")
        add(f"  {'mean branches per move pair':<34}{sum(branches) / len(branches):.2f}  max {max(branches)}")
        add(f"  {'move pairs per second':<34}{len(self.scores) / elapsed:.0f}")
        add(f"  {'branch states per second':<34}{sum(branches) / elapsed:.0f}")
        add(f"  {'wall clock over the corpus':<34}{self.seconds:.1f}s")

        if self.panics:
            add("")
            add(f"Backend panics ({len(self.panics)}) - a Rust panic through pyo3, first few:")
            for line in self.panics[:5]:
                add(f"  {line}")

        if self.unhandled_protocol:
            add("")
            add("Protocol messages poke-env has no handler for (informational):")
            for msg, n in self.unhandled_protocol.most_common():
                add(f"  |{msg}|{'':<28} {n:>6}")
        return "\n".join(lines)


def score_replay(
    path: Path | str,
    backend: Optional[ForwardModelBackend] = None,
    report: Optional[FidelityReport] = None,
    *,
    hindsight: bool = False,
) -> FidelityReport:
    """Score every scorable turn of one replay, accumulating into `report`.

    `hindsight` selects the second condition: the oracle additionally supplies
    every ability, Tera type and move the battle will *eventually* reveal.
    See `Hindsight` for what that does and does not cover.
    """
    backend = backend or PokeEngineBackend()
    report = report or FidelityReport(backend=backend.name, condition=CONDITIONS[hindsight])
    path = Path(path)
    payload = json.loads(path.read_text())
    # Parsed from the payload already in hand rather than by path: one read,
    # and it keeps this function usable on a payload that never touched disk.
    parsed: ParsedReplay = parse_replay_json(payload)
    by_turn = {t.turn: t for t in parsed.transitions}
    knowledge = (
        hindsight_knowledge(parsed) if hindsight else {"p1": {}, "p2": {}}
    )

    battle_id = payload.get("id") or path.stem
    driver = ReplayDriver(payload["log"], battle_id)
    pending: Optional[Tuple[TurnTransition, Any, Tuple[str, ...], bool, EngineAction, EngineAction]] = None

    for marker, turn, battle in driver:
        if marker == "turn":
            pending = None
            transition = by_turn.get(turn)
            if transition is None:
                # A turn the parser did not produce a transition for - the
                # last turn of a battle that ends before |upkeep|, normally.
                continue
            report.turns_seen += 1
            try:
                pending = _prepare(transition, battle, knowledge)
            except UnscorableTurn as exc:
                report.skipped[exc.reason] += 1
        elif marker == "upkeep" and pending is not None:
            transition, state_before, missing, representable, a1, a2 = pending
            pending = None
            try:
                observed = state_from_poke_env(battle).state
            except (ValueError, UnknownToPokeEngine):
                report.skipped["observed_state_untranslatable"] += 1
                continue
            try:
                report.scores.append(
                    _score(
                        backend,
                        battle_id,
                        transition.turn,
                        state_before,
                        observed,
                        a1,
                        a2,
                        missing,
                        representable,
                    )
                )
            except ValueError:
                # generate_instructions rejects an action string it cannot
                # resolve. With the oracle in place this should not happen;
                # when it does the shape is worth seeing, not swallowing.
                report.skipped["action_rejected_by_engine"] += 1
            except BaseException as exc:  # noqa: BLE001
                # A Rust panic crosses pyo3 as `PanicException`, which
                # derives from BaseException and so slips past every
                # `except Exception` in this file and in `score_corpus`.
                # One of them (Encore with no last-used move) cost a whole
                # corpus run before it was caught; it is fixed in the
                # translator, but a panic must never again be able to end a
                # run at replay 250 of 300.
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                report.skipped[f"backend_panic:{type(exc).__name__}"] += 1
                report.panics.append(f"{battle_id} turn {transition.turn}: {exc}")

    report.replays += 1
    report.unhandled_protocol.update(driver.unhandled)
    return report


def _prepare(
    transition: TurnTransition, battle: Battle, knowledge: Dict[str, Dict[str, Hindsight]]
) -> Tuple[TurnTransition, Any, Tuple[str, ...], bool, EngineAction, EngineAction]:
    """Everything that has to happen while the battle is at `|turn|N`.

    Raises `UnscorableTurn` for a turn there is no point simulating, which is
    why this is separated from `_score`: the reason has to be attributed
    before any model is called, or a set-prediction gap would be counted as a
    modelling error.
    """
    if transition.notes:
        # The parser flags a transition it may have modelled incompletely
        # (Illusion, Transform, an unmodelled protocol line). Scoring one
        # would attribute the parser's gap to the forward model.
        raise UnscorableTurn("parser_flagged_transition")

    if transition.p1_action.pivot_switch_in or transition.p2_action.pivot_switch_in:
        # U-turn and friends. poke-engine does not resolve the switch inside
        # the move: it records `slow_uturn_move` / `switch_out_move_second_
        # saved_move` and expects the switch as the *next* choice. The replay
        # has already made it by `|upkeep|`. So the two disagree about who is
        # active and about who took the hazard chip on the way in - a turn
        # boundary convention difference, not a wrong prediction, and scoring
        # it would charge the model for both. 8.3% of turns over the corpus.
        raise UnscorableTurn("pivot_turn_boundary")
    if any(a.kind is ActionKind.DRAGGED for a in transition.mid_turn_switches):
        # Whirlwind / Roar / Red Card. Same boundary problem, plus the
        # replacement is drawn at random from the unrevealed remainder, which
        # the model's placeholder slots cannot represent at all. 0.8%.
        raise UnscorableTurn("forced_switch_turn_boundary")

    a1 = engine_action(transition.p1_action, transition)
    a2 = engine_action(transition.p2_action, transition)

    try:
        baseline = state_from_poke_env(battle, filler=RevealedOnlyFiller()).state
    except (ValueError, UnknownToPokeEngine):
        raise UnscorableTurn("state_before_untranslatable")

    missing: List[str] = []
    representable = True
    for label, side_name, action in (("p1", "side_one", a1), ("p2", "side_two", a2)):
        if not _addressable(baseline, side_name, action):
            representable = False
            missing.append(f"{label}:{action.kind}")
    # A Tera type is never observable before the Tera, so it is always
    # supplied - counted separately so it does not swamp the two above.
    for label, action in (("p1", a1), ("p2", a2)):
        if action.needs_tera_type:
            missing.append(f"{label}:tera_type")

    oracle = OracleFiller(
        _side_oracle(battle, a1, ours=True, knowledge=knowledge["p1"]),
        _side_oracle(battle, a2, ours=False, knowledge=knowledge["p2"]),
    )
    try:
        state_before = state_from_poke_env(battle, filler=oracle).state
    except (ValueError, UnknownToPokeEngine) as exc:
        raise UnscorableTurn("oracle_state_untranslatable", str(exc))

    return transition, state_before, tuple(missing), representable, a1, a2


def _score(
    backend: ForwardModelBackend,
    battle_id: str,
    turn: int,
    state_before: Any,
    observed: Any,
    a1: EngineAction,
    a2: EngineAction,
    missing: Tuple[str, ...],
    representable: bool,
) -> TurnScore:
    start = time.perf_counter()
    branches = backend.branches(state_before, a1.text, a2.text)
    elapsed = time.perf_counter() - start
    if not branches:
        raise ValueError("backend returned no branches")

    scored = [(prob, compare_states(observed, state, state_before)) for prob, state in branches]
    modal_prob, modal_div = max(scored, key=lambda pair: pair[0])
    # Fewest divergences wins; ties broken toward the likelier branch, so
    # "some branch matched" never depends on an arbitrary ordering.
    best_prob, best_div = min(scored, key=lambda pair: (len(scoring_divergences(pair[1])), -pair[0]))

    return TurnScore(
        battle_id=battle_id,
        turn=turn,
        p1_kind=a1.kind,
        p2_kind=a2.kind,
        representable=representable,
        missing=missing,
        n_branches=len(branches),
        modal_probability=modal_prob,
        modal_divergences=modal_div,
        best_probability=best_prob,
        best_divergences=best_div,
        seconds=elapsed,
    )


def score_corpus(
    paths: Sequence[Path | str],
    backend: Optional[ForwardModelBackend] = None,
    on_replay: Optional[Callable[[int, Path | str, FidelityReport], None]] = None,
    *,
    hindsight: bool = False,
) -> FidelityReport:
    """Score a whole corpus. `on_replay(index, path, report)` is called after
    each file, for progress reporting - a 300-replay run is minutes long and
    a silent one is the failure mode
    notes/gotcha-benchmark-runs-need-empirical-timing-and-progress-visibility.md
    already recorded once.
    """
    backend = backend or PokeEngineBackend()
    report = FidelityReport(backend=backend.name, condition=CONDITIONS[hindsight])
    start = time.perf_counter()
    for index, path in enumerate(paths):
        try:
            score_replay(path, backend=backend, report=report, hindsight=hindsight)
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the run
            report.skipped[f"replay_failed:{type(exc).__name__}"] += 1
        if on_replay is not None:
            on_replay(index, path, report)
    report.seconds = time.perf_counter() - start
    return report
