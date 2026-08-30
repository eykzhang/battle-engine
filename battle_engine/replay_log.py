"""Turn a Showdown replay's raw `|`-protocol log into ordered turn transitions.

Phase 6 M3 needs to ask, per turn: given the state before turn N and what *both*
players actually did, what does a forward model predict, and how far is that from
what really happened? That question needs `(state_before, p1_action, p2_action,
state_after)` tuples, which is what `parse_replay_log` produces.

Written against `pokemon-showdown/sim/SIM-PROTOCOL.md` plus the simulator source
where the spec is wrong or silent - see "Where the spec is wrong" below.

Usage::

    from battle_engine.replay_log import parse_replay_file
    replay = parse_replay_file("data/replays_showdown/gen9ou-2672946156.json")
    for transition in replay.transitions:
        ...  # transition.state_before, .p1_action, .p2_action, .state_after

The single most important property of this module
------------------------------------------------
A replay log is a stream of *observations*, not a state dump. Plenty about a
battle is never observable from it. Where a value cannot be known, this module
stores the `UNKNOWN` sentinel rather than a plausible default, because M3
compares predictions against observations: a fabricated `0` that looks like a
real observation would silently corrupt the divergence metric in the direction
that makes the model look better than it is.

`UNKNOWN` is deliberately not falsy - `bool(UNKNOWN)` raises `TypeError`. `None`
already means "known to be absent" (no item, no status, no weather), so a falsy
unknown would let `if mon.item:` quietly treat an unrevealed Choice Band as an
empty hand. Crashing at the mistake is worth the ergonomic cost. Use
`is_known(value)` to branch.

What a log can and cannot tell you
----------------------------------
Observable, and tracked exactly: the active Pokemon per side, HP as a percentage,
status, stat boosts (they reset on switch and every change is announced), side
conditions and hazard layers, weather, terrain and other field effects, which
moves each Pokemon has used, items and abilities once revealed, Tera type and
whether each side has used its Terastallization, faints, and both sides' chosen
actions.

Not observable, and therefore `UNKNOWN` (or simply absent) rather than guessed:

- **Absolute HP.** Replay logs are spectator logs - they carry no `|request|`, and
  the `HP Percentage Mod` rule means every HP is `n/100` for both sides. Damage in
  real HP points cannot be recovered, and every HP fraction carries up to half a
  percentage point of rounding error. `PokemonState.hp_denominator` records the
  precision so a consumer can size its tolerance instead of assuming exactness.
- **Items and abilities before they are revealed** - `UNKNOWN`, distinct from a
  Pokemon whose item is known to have been consumed or knocked off (`None`).
- **Unrevealed moves.** `revealed_moves` is what has been *used*, so it is a lower
  bound on the moveset, never the moveset.
- **A Tera type before Terastallization.** `-terastallize` is the only reveal.
- **EVs, IVs, natures, PP, sleep and Toxic counters.** Never announced.
- **Which move a `|cant|` hid.** `|cant|p1a: X|par` says a move was chosen and
  failed, not which one, so the action is `BLOCKED` with `move=UNKNOWN`.
- **Unrevealed team members.** Team preview gives both full teams at the species
  level, but a Pokemon that never switches in has no observed state at all;
  `SideState.preview` holds it, `SideState.team` does not.

Where the spec is wrong (verified against the simulator source, 2026-08-30)
--------------------------------------------------------------------------
- **`|-terastallize|POKEMON|TYPE` is undocumented.** It is emitted at
  `sim/battle-actions.ts:1943` and appears in neither `sim/SIM-PROTOCOL.md` nor
  `PROTOCOL.md`, despite Tera being the defining gen9 mechanic. Parsed against the
  emitter.
- **`|-copyboost|SOURCE|TARGET` copies the other way round from the spec.** The
  spec says boosts go from `SOURCE` to `TARGET`; Psych Up (`data/moves.ts:14240`)
  emits `('-copyboost', source, target)` and its `onHit` does
  `source.boosts[i] = target.boosts[i]`. Implemented per the emitter.
- **`|-activate|` takes a Pokemon first.** The spec documents `|-activate|EFFECT`;
  every real line is `|-activate|POKEMON|EFFECT`.
- **`|-sidestart|SIDE|CONDITION`'s SIDE is `p1: Username`,** not `p1`.
- **Side-condition names are inconsistently prefixed.** `Reflect` and `Spikes` are
  bare while `move: Light Screen`, `move: Stealth Rock` and `move: Toxic Spikes`
  are prefixed, per-move rather than systematically. All names are normalized to
  ids here.
- **`|-swapboost|`'s STATS field is optional.** Heart Swap emits none.

Deliberate simplifications, and why
-----------------------------------
- **Singles only.** `|swap|` and multi-slot targeting are recorded as notes rather
  than modelled; gen9 OU is singles and doubles support would double the
  positional bookkeeping for no Phase 6 benefit.
- **The turn's chosen move is a side's *first* `|move|` line.** Sleep Talk, Dancer
  and Metronome emit the chosen move first and the called move second with a
  `[from]` tag, so this needs no whitelist of calling moves. Only that first move
  is added to `revealed_moves` (a Metronome-called Earthquake is not in the
  moveset); a `[from]lockedmove` continuation is recorded as the action with its
  `from_effect` set, since no choice existed that turn.
- **A switch before the turn's first `|move|`/`|cant|` is voluntary; one after is
  not.** Switch actions resolve before all move actions in singles, and Pursuit -
  the one exception - does not exist in gen9. This classifies U-turn pivots and
  Eject Button switches correctly without a whitelist, including the Eject Button
  case where a side switches without having moved.
- **`state_after` is captured at `|upkeep|`, before the replacement phase.**
  Verified in real logs: a mid-turn faint's replacement `|switch|` always comes
  after `|upkeep|`. A forward model is handed a state and both moves and returns
  the post-residual state; it does not choose replacements. So `state_after(N)`
  and `state_before(N+1)` are deliberately *not* the same snapshot, and differ
  exactly by `p1_replacement`/`p2_replacement` and any hazard damage they took.
- **An unrecognized message degrades to a note, never to a silent default.** Every
  message is handled, verified cosmetic, or appended to `TurnTransition.notes`
  with its raw text, so M3 can exclude or account for those turns instead of
  scoring a divergence this parser caused.
- **Illusion (`|replace|`) and Transform are flagged, not unwound.** Both mean
  earlier observations for that slot described a different Pokemon. Rewriting the
  history is guesswork, so the transition gets a note and the consumer decides.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class UnknownType:
    """The "this is not observable from the log" sentinel.

    Not falsy on purpose: `None` already means "known to be absent", so a falsy
    unknown would make `if mon.item:` silently wrong. Raising here turns that
    mistake into a stack trace at the line that made it.
    """

    __slots__ = ()
    _instance: UnknownType | None = None

    def __new__(cls) -> UnknownType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        raise TypeError(
            "UNKNOWN has no truth value - it means 'not observable from the log', "
            "which is not the same as None ('known absent') or 0. Branch on "
            "is_known(value) instead."
        )

    def __copy__(self) -> UnknownType:
        return self

    def __deepcopy__(self, memo: dict) -> UnknownType:
        return self


UNKNOWN = UnknownType()

# `Maybe[T]` reads at the call site as exactly the three-state value it is:
# a T, or None where None is meaningful, or UNKNOWN.
type Maybe[T] = T | UnknownType


def is_known(value: object) -> bool:
    """True when `value` is a real observation rather than the UNKNOWN sentinel."""
    return value is not UNKNOWN


BOOST_STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

# The four terrains get their own snapshot field because a forward model treats
# terrain and the other pseudo-weathers (Trick Room, Gravity, ...) differently.
TERRAINS = frozenset({"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"})

# Multi-layer hazards. Everything else is present-or-absent, so its "layers"
# stays 1 and a repeat -sidestart would be a protocol surprise worth noting.
_STACKING_SIDE_CONDITIONS = {"spikes": 3, "toxicspikes": 2}

_DETAILS_TERA_RE = re.compile(r"^tera:(?P<type>.+)$")


class ActionKind(Enum):
    """What a player did on a turn, as far as the log shows it."""

    MOVE = "move"
    """A move was selected. Check `Action.terastallized` for a Tera + move turn."""

    SWITCH = "switch"
    """A voluntary switch, selected at the start of the turn."""

    REPLACEMENT = "replacement"
    """Sending in a Pokemon after a faint. A real choice, but not a turn action -
    it happens after the turn has fully resolved, so it lives on
    `TurnTransition.p1_replacement`/`p2_replacement` rather than in the turn's
    action slot."""

    PIVOT = "pivot"
    """A mid-turn switch caused by the player's own move (U-turn, Baton Pass) or by
    an item (Eject Button, Eject Pack). The destination was chosen; the timing was
    not."""

    DRAGGED = "dragged"
    """Whirlwind, Roar, Red Card: the Pokemon that came in was not chosen at all."""

    BLOCKED = "blocked"
    """A `|cant|` turn: a move was selected but did not execute. `move` is UNKNOWN
    unless the log named it."""

    UNOBSERVED = "unobserved"
    """Nothing in the log attributes an action to this side this turn. Distinct
    from "did nothing" - it means the parser saw no evidence either way.

    The common cause is a Pokemon that fainted to the opponent's move before it
    could act: 14.4% of turns over the first 50-replay corpus have at least one
    side UNOBSERVED, and every sampled case was that. One
    inference is available to a consumer and is deliberately left to it rather
    than baked in here: switch actions always resolve, so a side that is
    UNOBSERVED and did not switch must have selected a move. That is an inference,
    not an observation, and this module only records observations."""


@dataclass
class Action:
    """One player's action for one turn, as observed."""

    player: str
    kind: ActionKind
    move: Maybe[str] | None = None
    """Move id (`toID` form), UNKNOWN when a `|cant|` hid it, None for non-moves."""
    move_name: str | None = None
    target: str | None = None
    switch_in: str | None = None
    """Ident (`p1: Nickname`) of the Pokemon brought in, for switch-shaped kinds."""
    pivot_switch_in: str | None = None
    """Set on a MOVE action when the move pivoted the user out (U-turn and friends).
    The forward model needs to be told which Pokemon replaced the user."""
    terastallized: bool = False
    tera_type: str | None = None
    missed: bool = False
    critical_hit: bool = False
    hit_count: int | None = None
    """`-hitcount` for multi-hit moves. None when the move is not multi-hit or the
    log did not say - not 1, which would be a fabricated observation."""
    from_effect: str | None = None
    """The `[from]` tag on the line that produced this action, e.g. `lockedmove`
    (Outrage continuation, no choice made) or `item: Eject Button`."""
    blocked_by: str | None = None
    """The `|cant|` reason: `par`, `slp`, `flinch`, `recharge`, a move name, ..."""


@dataclass
class HpChange:
    """One `-damage` / `-heal` / `-sethp` event, with what caused it.

    M3's done-when asks for divergence "broken down by cause", so the cause is
    kept rather than only the net HP delta. `source` is None for direct move
    damage (no `[from]` tag) and a normalized effect id otherwise (`psn`,
    `item: leftovers` -> `leftovers`, `stealthrock`, `recoil`, ...).
    """

    target: str
    before: Maybe[float]
    after: float
    source: str | None = None
    source_kind: str | None = None
    """`item`, `ability`, `move`, or None when the `[from]` tag carried no prefix."""
    of: str | None = None
    phase: str = "turn"
    """`turn` for anything up to `|upkeep|`, `replacement` for the hazard chip a
    post-faint replacement takes on the way in. The distinction is load-bearing:
    only `turn` events are reflected in `TurnTransition.state_after`, because a
    replacement is a separate decision made after the turn has resolved."""


@dataclass
class PreviewEntry:
    """One `|poke|` team-preview entry: a species, before anything else is known."""

    species: str
    base_species: str
    forme_unrevealed: bool
    """True for a `Zamazenta-*` style entry, where team preview hides the forme."""
    level: int
    gender: str | None
    linked_ident: str | None = None
    """Filled in when a revealed Pokemon matches this slot unambiguously. Left None
    rather than guessed when two preview entries could both match - a wrong link
    would fabricate a set-prediction observation for M4."""


@dataclass
class PokemonState:
    """Everything the log has revealed about one Pokemon."""

    ident: str
    """`p1: Nickname` - stable within a side across switches, unlike `p1a: Nickname`."""
    player: str
    nickname: str
    species: str
    level: int = 100
    gender: str | None = None
    hp_fraction: Maybe[float] = UNKNOWN
    hp_denominator: Maybe[int] = UNKNOWN
    """Denominator the log reported HP against (100 under HP Percentage Mod). The
    precision of `hp_fraction`, so a consumer can size its comparison tolerance."""
    fainted: bool = False
    status: Maybe[str | None] = UNKNOWN
    boosts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BOOST_STATS, 0))
    volatiles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    single_turn: set[str] = field(default_factory=set)
    single_move: set[str] = field(default_factory=set)
    revealed_moves: list[str] = field(default_factory=list)
    item: Maybe[str | None] = UNKNOWN
    ability: Maybe[str] = UNKNOWN
    terastallized: bool = False
    tera_type: Maybe[str] = UNKNOWN
    transformed: bool = False

    def reset_on_switch_out(self) -> None:
        """Clear everything that does not survive leaving the field.

        Boosts and volatiles reset; revealed moves, item, ability and Tera state
        persist because they are knowledge about the Pokemon, not battle state.
        """
        self.boosts = dict.fromkeys(BOOST_STATS, 0)
        self.volatiles.clear()
        self.single_turn.clear()
        self.single_move.clear()


@dataclass
class SideState:
    player: str
    username: str = ""
    team_size: Maybe[int] = UNKNOWN
    preview: list[PreviewEntry] = field(default_factory=list)
    team: dict[str, PokemonState] = field(default_factory=dict)
    """Only Pokemon that have actually appeared. A team-preview entry that never
    switches in has no state to observe, so it stays in `preview` alone."""
    active_ident: str | None = None
    side_conditions: dict[str, int] = field(default_factory=dict)
    tera_used: bool = False

    @property
    def active(self) -> PokemonState | None:
        return self.team.get(self.active_ident) if self.active_ident else None

    @property
    def unrevealed_count(self) -> Maybe[int]:
        if not is_known(self.team_size):
            return UNKNOWN
        return max(0, self.team_size - len(self.team))


@dataclass
class BattleSnapshot:
    """The observed battle state at one instant."""

    turn: int
    p1: SideState
    p2: SideState
    weather: Maybe[str | None] = None
    """None means "known clear" - a battle starts with no weather and every change
    is announced, so this is an observation rather than a default."""
    terrain: Maybe[str | None] = None
    fields: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Non-terrain field effects: Trick Room, Gravity, Magic Room, ..."""

    def side(self, player: str) -> SideState:
        return self.p1 if player == "p1" else self.p2


@dataclass
class TurnTransition:
    """One turn: what was true before, what both players did, what was true after.

    `state_after` is captured at `|upkeep|` - after residuals, before any faint
    replacement. `state_before` of the *next* transition therefore differs from
    this one's `state_after` by exactly the replacement phase, which is recorded
    in `p1_replacement`/`p2_replacement`. See the module docstring for why the cut
    is there.
    """

    turn: int
    state_before: BattleSnapshot
    p1_action: Action
    p2_action: Action
    state_after: BattleSnapshot
    p1_replacement: Action | None = None
    p2_replacement: Action | None = None
    mid_turn_switches: tuple[Action, ...] = ()
    """Pivots and drags, which are switches that happened but were not the turn's
    selected action."""
    hp_changes: tuple[HpChange, ...] = ()
    """Every HP event of the turn, each tagged with the phase it happened in.

    The last `phase == "turn"` event for a Pokemon agrees with its HP in
    `state_after` (verified over the M2 corpus), with one carve-out: a Pokemon
    that faints without a damage line - Healing Wish, Explosion, Final Gambit -
    is announced only by `|faint|`, and no synthetic HP event is invented for it,
    since a made-up event with no cause would look exactly like observed move
    damage. `phase == "replacement"` events happen after the `state_after`
    snapshot and are not reflected in it."""
    faints: tuple[str, ...] = ()
    """Idents that fainted during the turn's resolution."""
    replacement_faints: tuple[str, ...] = ()
    """Idents that fainted in the replacement phase - a Pokemon sent in after a
    faint and killed by entry hazards on the way. Rare, and separated because it
    is not something the turn's forward-model call could have predicted."""
    notes: tuple[str, ...] = ()
    """Raw text of any protocol line this parser did not model, plus flags for
    Illusion and Transform. A non-empty `notes` means the transition may be
    incomplete and a fidelity harness should say so rather than score it."""

    def action(self, player: str) -> Action:
        return self.p1_action if player == "p1" else self.p2_action


@dataclass
class ParsedReplay:
    battle_id: str | None
    format_id: str | None
    players: tuple[str, ...]
    rating: int | None
    gen: int | None
    gametype: str | None
    rules: tuple[str, ...]
    leads: dict[str, str]
    winner: str | None
    transitions: list[TurnTransition]
    notes: tuple[str, ...]
    """Lines that were unmodelled outside any turn (setup and post-battle)."""


class ReplayParseError(Exception):
    """The log is not a parseable singles battle (wrong gametype, truncated, ...)."""


# --------------------------------------------------------------------------
# Protocol scanning helpers
# --------------------------------------------------------------------------

def to_id(text: str) -> str:
    """Showdown's `toID`: lowercase, strip everything but letters and digits."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _effect_id(text: str) -> tuple[str, str | None]:
    """Split an effect string into (id, kind).

    Effect strings arrive as `move: Stealth Rock`, `item: Leftovers`,
    `ability: Drought`, or bare (`Reflect`, `psn`). The prefix is inconsistent per
    move rather than systematic, so it is stripped and reported separately instead
    of being folded into the id.
    """
    if ":" in text:
        prefix, _, rest = text.partition(":")
        kind = prefix.strip().lower()
        if kind in ("move", "item", "ability", "pokemon"):
            return to_id(rest), kind
    return to_id(text), None


# Every bracket tag Showdown emits is a single run of letters ([from], [of],
# [miss], [fromitem], [Type], ...), collected by grepping the simulator source.
# The pattern matters: `|tier|[Gen 9] OU` also starts with a bracket, and a looser
# rule swallowed the format name as a tag and left `|tier|` with no fields at all.
_TAG_NAME_RE = re.compile(r"[A-Za-z]+$")


def _split_kwargs(fields: list[str]) -> tuple[list[str], dict[str, str]]:
    """Peel `[from] X` / `[of] Y` / `[miss]` style tags off a message's fields.

    Showdown emits these with and without a space after the bracket
    (`[from]lockedmove` and `[from] ability: Drizzle` both occur), so the split is
    on the closing bracket rather than on whitespace.
    """
    positional: list[str] = []
    kwargs: dict[str, str] = {}
    for item in fields:
        stripped = item.strip()
        name, closer, value = stripped[1:].partition("]")
        if stripped.startswith("[") and closer and _TAG_NAME_RE.fullmatch(name):
            kwargs[name.lower()] = value.strip()
        else:
            positional.append(item)
    return positional, kwargs


def _parse_ident(token: str) -> tuple[str, str, str]:
    """`p1a: Draymoney Green` -> (`p1`, `p1: Draymoney Green`, `Draymoney Green`).

    The position letter is dropped from the key so that a Pokemon keeps one
    identity across switching out and back in.
    """
    position, _, nickname = token.partition(": ")
    position = position.strip()
    player = position[:2]
    if player not in ("p1", "p2"):
        raise ReplayParseError(f"unsupported player position {token!r} (singles only)")
    return player, f"{player}: {nickname}", nickname


def _parse_details(details: str) -> dict:
    """`Sawsbuck, L50, F, shiny, tera:Fairy` -> species/level/gender/tera.

    Team-preview details omit level and shininess and give an unrevealed forme as
    `Arceus-*`; that asterisk is preserved in `species` rather than guessed away.
    """
    parts = [part.strip() for part in details.split(",")]
    parsed = {"species": parts[0], "level": 100, "gender": None, "tera_type": None}
    for part in parts[1:]:
        if re.fullmatch(r"L\d+", part):
            parsed["level"] = int(part[1:])
        elif part in ("M", "F"):
            parsed["gender"] = part
        elif (match := _DETAILS_TERA_RE.match(part)) is not None:
            parsed["tera_type"] = match["type"]
    return parsed


def _parse_hp(token: str) -> tuple[float | None, int | None, str | None]:
    """`40/100 par` -> (0.40, 100, 'par'); `0 fnt` -> (0.0, None, None).

    Per the spec, when HP is 0 the trailing status is `fnt` and must be ignored -
    it is a display artifact, not the Pokemon's status condition.
    """
    token = token.strip()
    if not token:
        return None, None, None
    hp_part, _, status_part = token.partition(" ")
    status = status_part.strip() or None
    if "/" in hp_part:
        numerator, _, denominator = hp_part.partition("/")
        try:
            num, den = float(numerator), int(denominator)
        except ValueError:
            return None, None, None
        if den == 0:
            return None, None, None
        if num == 0:
            return 0.0, den, None
        return num / den, den, status
    try:
        value = float(hp_part)
    except ValueError:
        return None, None, None
    return (0.0, None, None) if value == 0 else (value, None, status)


# Messages verified to carry nothing this module tracks, either because they are
# chat and room noise or because they only restate something already recorded
# (`-supereffective` after the `-damage` it explains). Listed explicitly so that a
# genuinely new message type shows up in `notes` instead of being dropped: an
# unmodelled line has to be visible to M3, which would otherwise score a
# divergence this parser caused.
_COSMETIC = frozenset({
    "", "t:", "j", "J", "l", "L", "n", "N", "c", "c:", "chat", "join", "leave",
    "name", "raw", "html", "uhtml", "uhtmlchange", "inactive", "inactiveoff",
    "debug", "bigerror", "error", "message", "popup", "notify", "seed", "unlink",
    "expire", "askreg", "-message", "-hint", "-anim", "-supereffective",
    "-resisted", "-fail", "-notarget", "-block", "-center", "-combine",
    "-waiting", "-nothing", "-zpower", "-zbroken", "-ohko", "-candynamax",
    "-fieldactivate", "-mega", "-primal", "-burst", "rated", "start",
    "clearpoke", "teampreview", "showteam", "updatepoke",
})


class _BattleTracker:
    """Applies protocol lines to a running battle state and emits transitions.

    Kept private: `parse_replay_log` is the whole public surface, so the mutable
    state machine never escapes and a caller cannot half-apply a log.
    """

    def __init__(self) -> None:
        self.p1 = SideState("p1")
        self.p2 = SideState("p2")
        self.weather: Maybe[str | None] = None
        self.terrain: Maybe[str | None] = None
        self.fields: dict[str, tuple[str, ...]] = {}

        self.gen: int | None = None
        self.gametype: str | None = None
        self.format_id: str | None = None
        self.rules: list[str] = []
        self.winner: str | None = None
        self.leads: dict[str, str] = {}

        self.transitions: list[TurnTransition] = []
        self.notes: list[str] = []

        self._turn: int | None = None
        self._state_before: BattleSnapshot | None = None
        self._state_after: BattleSnapshot | None = None
        self._actions: dict[str, Action] = {}
        self._replacements: dict[str, Action] = {}
        self._mid_turn: list[Action] = []
        self._hp_changes: list[HpChange] = []
        self._faints: list[str] = []
        self._replacement_faints: list[str] = []
        self._turn_notes: list[str] = []
        self._first_action_seen = False
        self._in_replacement_phase = False
        self._last_move_action: Action | None = None
        self._pending_switch_cause: dict[str, str] = {}

    # -- side/pokemon lookup ------------------------------------------------

    def side(self, player: str) -> SideState:
        return self.p1 if player == "p1" else self.p2

    def _mon(self, token: str) -> PokemonState:
        """Look up (or create) the Pokemon a message refers to.

        Creating on demand matters: `|-damage|` can name a Pokemon this parser has
        not seen switch in yet if a log is joined mid-battle, and inventing a
        blank-but-present record is more honest than dropping the observation.
        """
        player, ident, nickname = _parse_ident(token)
        side = self.side(player)
        mon = side.team.get(ident)
        if mon is None:
            mon = PokemonState(ident=ident, player=player, nickname=nickname,
                               species=nickname)
            side.team[ident] = mon
        return mon

    # -- snapshots ----------------------------------------------------------

    def snapshot(self) -> BattleSnapshot:
        """A deep copy of the live state.

        Deep, not shallow: a snapshot handed to M3 must not change under it as the
        rest of the log is applied, and every interesting field is a mutable dict
        or list.
        """
        return BattleSnapshot(
            turn=self._turn or 0,
            p1=copy.deepcopy(self.p1),
            p2=copy.deepcopy(self.p2),
            weather=self.weather,
            terrain=self.terrain,
            fields=copy.deepcopy(self.fields),
        )

    # -- turn boundaries ----------------------------------------------------

    def _begin_turn(self, number: int) -> None:
        self._turn = number
        self._state_before = self.snapshot()
        self._state_after = None
        self._actions = {}
        self._replacements = {}
        self._mid_turn = []
        self._hp_changes = []
        self._faints = []
        self._replacement_faints = []
        self._turn_notes = []
        self._first_action_seen = False
        self._in_replacement_phase = False
        self._last_move_action = None
        self._pending_switch_cause = {}
        for side in (self.p1, self.p2):
            for mon in side.team.values():
                mon.single_turn.clear()

    def _finish_turn(self) -> None:
        """Emit the transition for the turn that just ended, if there was one."""
        if self._turn is None or self._state_before is None:
            return
        # A battle that ends mid-turn (the winning hit) never reaches |upkeep|, so
        # the end-of-resolution snapshot is taken here instead.
        state_after = self._state_after if self._state_after is not None else self.snapshot()
        self.transitions.append(TurnTransition(
            turn=self._turn,
            state_before=self._state_before,
            p1_action=self._actions.get("p1", Action("p1", ActionKind.UNOBSERVED)),
            p2_action=self._actions.get("p2", Action("p2", ActionKind.UNOBSERVED)),
            state_after=state_after,
            p1_replacement=self._replacements.get("p1"),
            p2_replacement=self._replacements.get("p2"),
            mid_turn_switches=tuple(self._mid_turn),
            hp_changes=tuple(self._hp_changes),
            faints=tuple(self._faints),
            replacement_faints=tuple(self._replacement_faints),
            notes=tuple(self._turn_notes),
        ))
        self._turn = None
        self._state_before = None

    def _note(self, text: str) -> None:
        (self._turn_notes if self._turn is not None else self.notes).append(text)

    # -- dispatch -----------------------------------------------------------

    def apply(self, line: str) -> None:
        if not line.startswith("|"):
            return
        raw = line[1:].split("|")
        message = raw[0]
        fields, kwargs = _split_kwargs(raw[1:])
        handler = getattr(self, f"_on_{message.replace('-', 'minor_')}", None)
        if handler is not None:
            handler(fields, kwargs, line)
            return
        if message in _COSMETIC:
            return
        self._note(line)

    # -- battle setup -------------------------------------------------------

    def _on_player(self, fields, kwargs, line) -> None:
        if len(fields) >= 2 and fields[0] in ("p1", "p2"):
            self.side(fields[0]).username = fields[1]

    def _on_teamsize(self, fields, kwargs, line) -> None:
        if fields[0] in ("p1", "p2"):
            self.side(fields[0]).team_size = int(fields[1])

    def _on_gametype(self, fields, kwargs, line) -> None:
        self.gametype = fields[0]

    def _on_gen(self, fields, kwargs, line) -> None:
        self.gen = int(fields[0])

    def _on_tier(self, fields, kwargs, line) -> None:
        self.format_id = to_id(fields[0])

    def _on_rule(self, fields, kwargs, line) -> None:
        self.rules.append(fields[0])

    def _on_poke(self, fields, kwargs, line) -> None:
        # The ITEM field is documented as `item` when the Pokemon is holding one,
        # but every observed gen9ou replay leaves it blank even for Pokemon that
        # later reveal an item - so it is not read here.
        player = fields[0]
        if player not in ("p1", "p2"):
            return
        details = _parse_details(fields[1])
        species = details["species"]
        self.side(player).preview.append(PreviewEntry(
            species=species,
            base_species=species.removesuffix("-*"),
            forme_unrevealed=species.endswith("-*"),
            level=details["level"],
            gender=details["gender"],
        ))

    def _on_turn(self, fields, kwargs, line) -> None:
        self._finish_turn()
        self._begin_turn(int(fields[0]))

    def _on_upkeep(self, fields, kwargs, line) -> None:
        # The boundary between the turn resolving and the faint-replacement phase.
        self._state_after = self.snapshot()
        self._in_replacement_phase = True

    def _on_win(self, fields, kwargs, line) -> None:
        self.winner = fields[0] if fields else None
        self._finish_turn()

    def _on_tie(self, fields, kwargs, line) -> None:
        self._finish_turn()

    # -- major actions ------------------------------------------------------

    def _on_move(self, fields, kwargs, line) -> None:
        player, _, _ = _parse_ident(fields[0])
        move_name = fields[1]
        move_id = to_id(move_name)
        from_effect = kwargs.get("from")
        self._first_action_seen = True

        existing = self._actions.get(player)
        if existing is not None and existing.move_name is not None:
            # A second |move| for the same side is a called move (Sleep Talk,
            # Dancer, Metronome). The chosen action is the first line, already
            # recorded, and a called move is not evidence of the moveset. The
            # move_name test is what distinguishes a real earlier move line from
            # the empty placeholder _on_minor_terastallize leaves behind.
            self._last_move_action = existing
            return

        action = Action(
            player=player,
            kind=ActionKind.MOVE,
            move=move_id,
            move_name=move_name,
            target=fields[2] if len(fields) > 2 and fields[2] else None,
            missed="miss" in kwargs,
            from_effect=from_effect,
        )
        mon = self.side(player).active
        # _on_minor_terastallize may already have opened a placeholder action for
        # this side: Tera resolves before every move on the turn, so the flag
        # arrives first and the move completes it rather than replacing it.
        if existing is not None and existing.terastallized:
            action.terastallized = True
            action.tera_type = existing.tera_type
        self._actions[player] = action
        self._last_move_action = action

        if from_effect is None or to_id(from_effect) == "lockedmove":
            if mon is not None and move_id not in mon.revealed_moves:
                mon.revealed_moves.append(move_id)

    def _on_cant(self, fields, kwargs, line) -> None:
        player, _, _ = _parse_ident(fields[0])
        reason = fields[1] if len(fields) > 1 else None
        self._first_action_seen = True
        existing = self._actions.get(player)
        if existing is not None and existing.move_name is not None:
            return
        named_move = fields[2] if len(fields) > 2 else None
        action = Action(
            player=player,
            kind=ActionKind.BLOCKED,
            # A `|cant|` without a named move genuinely hides which move was
            # selected. UNKNOWN, not None: a move *was* chosen.
            move=to_id(named_move) if named_move else UNKNOWN,
            move_name=named_move,
            blocked_by=reason,
        )
        if existing is not None and existing.terastallized:
            # Terastallized, then flinched: the Tera happened and is observed even
            # though the move never went off.
            action.terastallized = True
            action.tera_type = existing.tera_type
        self._actions[player] = action
        if named_move:
            mon = self.side(player).active
            if mon is not None and to_id(named_move) not in mon.revealed_moves:
                mon.revealed_moves.append(to_id(named_move))

    def _switch_in(self, player: str, ident: str, nickname: str,
                   details: str, hp_token: str) -> PokemonState:
        side = self.side(player)
        outgoing = side.active
        if outgoing is not None and outgoing.ident != ident:
            outgoing.reset_on_switch_out()

        parsed = _parse_details(details)
        mon = side.team.get(ident)
        if mon is None:
            mon = PokemonState(ident=ident, player=player, nickname=nickname,
                               species=parsed["species"])
            side.team[ident] = mon
        mon.species = parsed["species"]
        mon.level = parsed["level"]
        mon.gender = parsed["gender"]
        if parsed["tera_type"] is not None:
            mon.terastallized = True
            mon.tera_type = parsed["tera_type"]
        mon.reset_on_switch_out()
        mon.fainted = False

        fraction, denominator, status = _parse_hp(hp_token)
        if fraction is not None:
            mon.hp_fraction = fraction
            mon.hp_denominator = denominator if denominator is not None else UNKNOWN
            mon.status = status
        side.active_ident = ident
        self._link_preview(side, mon)
        return mon

    def _link_preview(self, side: SideState, mon: PokemonState) -> None:
        """Attach a revealed Pokemon to its team-preview slot, when unambiguous.

        Team preview hides formes behind `-*`, so a `Zamazenta-*` entry has to be
        matched by base species. Where two entries could match, the link is left
        None rather than picked - M4 reads these to score set prediction, and a
        wrong link would be a fabricated observation.
        """
        if any(entry.linked_ident == mon.ident for entry in side.preview):
            return
        free = [entry for entry in side.preview if entry.linked_ident is None]
        exact = [entry for entry in free if entry.species == mon.species]
        if len(exact) == 1:
            exact[0].linked_ident = mon.ident
            return
        base = mon.species.split("-")[0]
        loose = [entry for entry in free
                 if entry.forme_unrevealed and entry.base_species == base]
        if len(loose) == 1:
            loose[0].linked_ident = mon.ident

    def _on_switch(self, fields, kwargs, line) -> None:
        player, ident, nickname = _parse_ident(fields[0])
        self._switch_in(player, ident, nickname, fields[1],
                        fields[2] if len(fields) > 2 else "")

        if self._turn is None:
            self.leads[player] = ident
            return
        if self._in_replacement_phase:
            self._replacements[player] = Action(
                player=player, kind=ActionKind.REPLACEMENT, switch_in=ident)
            return
        if not self._first_action_seen:
            # Switch actions resolve before every move action in singles, so a
            # switch seen before the turn's first move is the turn's action.
            self._actions[player] = Action(
                player=player, kind=ActionKind.SWITCH, switch_in=ident)
            return

        cause = self._pending_switch_cause.pop(player, None)
        pivot = Action(player=player, kind=ActionKind.PIVOT, switch_in=ident,
                       from_effect=cause)
        self._mid_turn.append(pivot)
        own_action = self._actions.get(player)
        if own_action is not None and own_action.kind is ActionKind.MOVE:
            own_action.pivot_switch_in = ident

    def _on_drag(self, fields, kwargs, line) -> None:
        player, ident, nickname = _parse_ident(fields[0])
        self._switch_in(player, ident, nickname, fields[1],
                        fields[2] if len(fields) > 2 else "")
        if self._turn is not None:
            self._mid_turn.append(Action(player=player, kind=ActionKind.DRAGGED,
                                         switch_in=ident,
                                         from_effect=kwargs.get("from")))

    def _on_replace(self, fields, kwargs, line) -> None:
        # Illusion dropped: everything previously recorded for this slot described
        # the Illusion user, not this Pokemon. Flagged rather than retroactively
        # rewritten, because unwinding it correctly is guesswork.
        player, ident, nickname = _parse_ident(fields[0])
        self._switch_in(player, ident, nickname, fields[1],
                        fields[2] if len(fields) > 2 else "")
        self._note(f"illusion revealed for {ident}; earlier observations for this "
                   f"slot describe a different Pokemon")

    def _on_detailschange(self, fields, kwargs, line) -> None:
        mon = self._mon(fields[0])
        parsed = _parse_details(fields[1])
        mon.species = parsed["species"]
        if parsed["tera_type"] is not None:
            mon.terastallized = True
            mon.tera_type = parsed["tera_type"]

    _on_minor_formechange = _on_detailschange

    def _on_faint(self, fields, kwargs, line) -> None:
        mon = self._mon(fields[0])
        mon.hp_fraction = 0.0
        mon.fainted = True
        target = (self._replacement_faints if self._in_replacement_phase
                  else self._faints)
        target.append(mon.ident)

    def _on_swap(self, fields, kwargs, line) -> None:
        self._note(f"|swap| is doubles-only and not modelled: {line}")

    # -- HP, status, boosts -------------------------------------------------

    def _record_hp(self, token: str, hp_token: str, kwargs: dict) -> None:
        mon = self._mon(token)
        fraction, denominator, status = _parse_hp(hp_token)
        if fraction is None:
            return
        before = mon.hp_fraction
        mon.hp_fraction = fraction
        if denominator is not None:
            mon.hp_denominator = denominator
        if fraction == 0.0:
            mon.fainted = True
        else:
            mon.status = status
        source_id, source_kind = (_effect_id(kwargs["from"]) if "from" in kwargs
                                  else (None, None))
        self._hp_changes.append(HpChange(
            target=mon.ident, before=before, after=fraction,
            source=source_id, source_kind=source_kind, of=kwargs.get("of"),
            phase="replacement" if self._in_replacement_phase else "turn",
        ))

    def _on_minor_damage(self, fields, kwargs, line) -> None:
        self._record_hp(fields[0], fields[1], kwargs)
        self._reveal_from_kwargs(kwargs)

    _on_minor_heal = _on_minor_damage

    def _on_minor_sethp(self, fields, kwargs, line) -> None:
        self._record_hp(fields[0], fields[1], kwargs)

    def _on_minor_status(self, fields, kwargs, line) -> None:
        self._mon(fields[0]).status = fields[1]
        self._reveal_from_kwargs(kwargs)

    def _on_minor_curestatus(self, fields, kwargs, line) -> None:
        self._mon(fields[0]).status = None
        self._reveal_from_kwargs(kwargs)

    def _on_minor_cureteam(self, fields, kwargs, line) -> None:
        player, _, _ = _parse_ident(fields[0])
        for mon in self.side(player).team.values():
            if not mon.fainted:
                mon.status = None

    def _on_minor_boost(self, fields, kwargs, line, sign: int = 1) -> None:
        mon = self._mon(fields[0])
        stat = fields[1]
        # Boosts cap at +/-6; the log reports the requested amount, not the
        # clamped result, so clamping here is what keeps the state truthful.
        mon.boosts[stat] = max(-6, min(6, mon.boosts.get(stat, 0) + sign * int(fields[2])))
        self._reveal_from_kwargs(kwargs)

    def _on_minor_unboost(self, fields, kwargs, line) -> None:
        self._on_minor_boost(fields, kwargs, line, sign=-1)

    def _on_minor_setboost(self, fields, kwargs, line) -> None:
        self._mon(fields[0]).boosts[fields[1]] = int(fields[2])
        self._reveal_from_kwargs(kwargs)

    def _on_minor_clearboost(self, fields, kwargs, line) -> None:
        self._mon(fields[0]).boosts = dict.fromkeys(BOOST_STATS, 0)

    def _on_minor_clearallboost(self, fields, kwargs, line) -> None:
        for side in (self.p1, self.p2):
            for mon in side.team.values():
                mon.boosts = dict.fromkeys(BOOST_STATS, 0)

    def _on_minor_clearnegativeboost(self, fields, kwargs, line) -> None:
        boosts = self._mon(fields[0]).boosts
        for stat, value in boosts.items():
            if value < 0:
                boosts[stat] = 0

    def _on_minor_clearpositiveboost(self, fields, kwargs, line) -> None:
        boosts = self._mon(fields[0]).boosts
        for stat, value in boosts.items():
            if value > 0:
                boosts[stat] = 0

    def _on_minor_invertboost(self, fields, kwargs, line) -> None:
        boosts = self._mon(fields[0]).boosts
        for stat in boosts:
            boosts[stat] = -boosts[stat]

    def _on_minor_copyboost(self, fields, kwargs, line) -> None:
        # Direction per the emitter, not the spec: Psych Up emits
        # ('-copyboost', source, target) and copies target's boosts onto source.
        source = self._mon(fields[0])
        target = self._mon(fields[1])
        source.boosts = dict(target.boosts)

    def _on_minor_swapboost(self, fields, kwargs, line) -> None:
        source = self._mon(fields[0])
        target = self._mon(fields[1])
        # STATS is optional - Heart Swap swaps everything and emits no stat list.
        stats = ([stat.strip() for stat in fields[2].split(",")]
                 if len(fields) > 2 and fields[2] else list(BOOST_STATS))
        for stat in stats:
            source.boosts[stat], target.boosts[stat] = (
                target.boosts.get(stat, 0), source.boosts.get(stat, 0))

    # -- items, abilities, volatiles ---------------------------------------

    def _reveal_from_kwargs(self, kwargs: dict) -> None:
        """Reveal the item or ability named in a `[from]` tag on its `[of]` owner.

        `|-damage|p2a: Kingambit|73/100|[from] item: Rocky Helmet|[of] p1a: X` is
        the only announcement Rocky Helmet ever gets, so skipping it would leave a
        genuinely observed item marked UNKNOWN.
        """
        source = kwargs.get("from")
        if not source:
            return
        effect_id, kind = _effect_id(source)
        owner = kwargs.get("of")
        if kind == "item" and owner:
            self._mon(owner).item = effect_id
        elif kind == "ability" and owner:
            self._mon(owner).ability = effect_id

    def _on_minor_item(self, fields, kwargs, line) -> None:
        mon = self._mon(fields[0])
        mon.item = to_id(fields[1])
        self._reveal_from_kwargs(kwargs)

    def _on_minor_enditem(self, fields, kwargs, line) -> None:
        mon = self._mon(fields[0])
        mon.item = None  # known to hold nothing now, as opposed to UNKNOWN
        item_id = to_id(fields[1])
        if item_id in ("ejectbutton", "ejectpack"):
            # The switch this causes lands later in the turn and is a forced
            # relocation, not a chosen pivot - remember why for _on_switch.
            self._pending_switch_cause[mon.player] = f"item: {item_id}"

    def _on_minor_ability(self, fields, kwargs, line) -> None:
        self._mon(fields[0]).ability = to_id(fields[1])
        self._reveal_from_kwargs(kwargs)

    def _on_minor_endability(self, fields, kwargs, line) -> None:
        # The ability is suppressed, not forgotten: keep the reveal, flag the state.
        self._mon(fields[0]).volatiles["abilitysuppressed"] = ()

    def _on_minor_activate(self, fields, kwargs, line) -> None:
        # Real lines are |-activate|POKEMON|EFFECT, not the spec's |-activate|EFFECT.
        if not fields:
            return
        mon = self._mon(fields[0])
        if len(fields) > 1:
            effect_id, kind = _effect_id(fields[1])
            if kind == "ability":
                mon.ability = effect_id
            elif kind == "item":
                mon.item = effect_id
            elif effect_id == "ejectpack":
                self._pending_switch_cause[mon.player] = "item: ejectpack"
        self._reveal_from_kwargs(kwargs)

    def _on_minor_start(self, fields, kwargs, line) -> None:
        mon = self._mon(fields[0])
        effect_id, kind = _effect_id(fields[1]) if len(fields) > 1 else ("", None)
        if kind == "ability":
            mon.ability = effect_id
        mon.volatiles[effect_id] = tuple(fields[2:])
        self._reveal_from_kwargs(kwargs)

    def _on_minor_end(self, fields, kwargs, line) -> None:
        mon = self._mon(fields[0])
        effect_id, _ = _effect_id(fields[1]) if len(fields) > 1 else ("", None)
        mon.volatiles.pop(effect_id, None)

    def _on_minor_singleturn(self, fields, kwargs, line) -> None:
        self._mon(fields[0]).single_turn.add(_effect_id(fields[1])[0])

    def _on_minor_singlemove(self, fields, kwargs, line) -> None:
        self._mon(fields[0]).single_move.add(_effect_id(fields[1])[0])

    def _on_minor_mustrecharge(self, fields, kwargs, line) -> None:
        self._mon(fields[0]).volatiles["mustrecharge"] = ()

    def _on_minor_prepare(self, fields, kwargs, line) -> None:
        # A charging two-turn move (Solar Beam, Fly) is real state the forward
        # model must carry into the next turn, so the move id is kept with it.
        self._mon(fields[0]).volatiles["twoturnmove"] = (to_id(fields[1]),)

    def _on_minor_transform(self, fields, kwargs, line) -> None:
        mon = self._mon(fields[0])
        mon.transformed = True
        self._note(f"{mon.ident} transformed into {fields[1]}; its boosts, moves and "
                   f"stats are now copies and are not tracked through the change")

    def _on_minor_terastallize(self, fields, kwargs, line) -> None:
        # Undocumented in both protocol files; parsed against sim/battle-actions.ts.
        mon = self._mon(fields[0])
        mon.terastallized = True
        mon.tera_type = fields[1]
        self.side(mon.player).tera_used = True
        # Tera resolves before every move on the turn, so the move line for this
        # side has not been seen yet; stash the flag on a placeholder that _on_move
        # will complete.
        action = self._actions.get(mon.player)
        if action is None:
            action = Action(player=mon.player, kind=ActionKind.MOVE, move=UNKNOWN)
            self._actions[mon.player] = action
        action.terastallized = True
        action.tera_type = fields[1]

    # -- field and side conditions -----------------------------------------

    def _on_minor_weather(self, fields, kwargs, line) -> None:
        value = fields[0] if fields else "none"
        if "upkeep" in kwargs:
            return  # weather unchanged, just ticking down
        self.weather = None if to_id(value) in ("none", "") else to_id(value)
        self._reveal_from_kwargs(kwargs)

    def _on_minor_fieldstart(self, fields, kwargs, line) -> None:
        effect_id, _ = _effect_id(fields[0])
        if effect_id in TERRAINS:
            self.terrain = effect_id
        else:
            self.fields[effect_id] = tuple(fields[1:])
        self._reveal_from_kwargs(kwargs)

    def _on_minor_fieldend(self, fields, kwargs, line) -> None:
        effect_id, _ = _effect_id(fields[0])
        if effect_id in TERRAINS:
            if self.terrain == effect_id:
                self.terrain = None
        else:
            self.fields.pop(effect_id, None)

    def _on_minor_sidestart(self, fields, kwargs, line) -> None:
        player = fields[0].split(":")[0].strip()
        if player not in ("p1", "p2"):
            return
        effect_id, _ = _effect_id(fields[1])
        conditions = self.side(player).side_conditions
        cap = _STACKING_SIDE_CONDITIONS.get(effect_id, 1)
        conditions[effect_id] = min(cap, conditions.get(effect_id, 0) + 1)

    def _on_minor_sideend(self, fields, kwargs, line) -> None:
        player = fields[0].split(":")[0].strip()
        if player not in ("p1", "p2"):
            return
        self.side(player).side_conditions.pop(_effect_id(fields[1])[0], None)

    def _on_minor_swapsideconditions(self, fields, kwargs, line) -> None:
        self.p1.side_conditions, self.p2.side_conditions = (
            self.p2.side_conditions, self.p1.side_conditions)

    # -- action annotations -------------------------------------------------

    def _on_minor_crit(self, fields, kwargs, line) -> None:
        if self._last_move_action is not None:
            self._last_move_action.critical_hit = True

    def _on_minor_miss(self, fields, kwargs, line) -> None:
        if self._last_move_action is not None:
            self._last_move_action.missed = True

    def _on_minor_hitcount(self, fields, kwargs, line) -> None:
        if self._last_move_action is not None:
            self._last_move_action.hit_count = int(fields[1])

    def _on_minor_immune(self, fields, kwargs, line) -> None:
        self._reveal_from_kwargs(kwargs)


def parse_replay_log(log: str, *, battle_id: str | None = None,
                     players: tuple[str, ...] = (), rating: int | None = None,
                     format_id: str | None = None) -> ParsedReplay:
    """Parse a raw `|`-protocol log into ordered turn transitions.

    Raises `ReplayParseError` for a log this module cannot honestly represent -
    a non-singles gametype, or one with no turns at all. Callers batching over a
    corpus are expected to catch it per replay; the alternative (returning an
    empty result) would make an unparseable battle indistinguishable from a
    zero-turn one.
    """
    tracker = _BattleTracker()
    for line in log.split("\n"):
        tracker.apply(line.rstrip("\r"))
    tracker._finish_turn()

    if tracker.gametype not in (None, "singles"):
        raise ReplayParseError(
            f"gametype {tracker.gametype!r} is not singles; this parser tracks one "
            f"active Pokemon per side and would silently mis-attribute actions"
        )
    if not tracker.transitions:
        raise ReplayParseError("log contains no |turn| messages")

    return ParsedReplay(
        battle_id=battle_id,
        format_id=format_id or tracker.format_id,
        players=players or (tracker.p1.username, tracker.p2.username),
        rating=rating,
        gen=tracker.gen,
        gametype=tracker.gametype,
        rules=tuple(tracker.rules),
        leads=dict(tracker.leads),
        winner=tracker.winner,
        transitions=tracker.transitions,
        notes=tuple(tracker.notes),
    )


def parse_replay_json(payload: dict) -> ParsedReplay:
    """Parse a `<id>.json` payload as returned by Showdown's replay API."""
    log = payload.get("log")
    if not isinstance(log, str):
        raise ReplayParseError("replay payload has no 'log' string")
    players = payload.get("players") or ()
    return parse_replay_log(
        log,
        battle_id=payload.get("id"),
        players=tuple(players),
        rating=payload.get("rating"),
        format_id=payload.get("formatid"),
    )


def parse_replay_file(path: str | Path) -> ParsedReplay:
    """Parse a replay cached by `scripts/fetch_showdown_replays.py`."""
    return parse_replay_json(json.loads(Path(path).read_text()))
