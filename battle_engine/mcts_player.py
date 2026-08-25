"""M5 (translator) / M7 (MctsPlayer, this milestone): converts a live
poke-env AbstractBattle into a be::BattleState for the C++ forward model /
search to operate on. Lives here, not in C++/bindings, per the plan's own
explicit decision ("Python-side is unambiguously easier") - fixing a
contradiction the original plan draft had between this file's stated
location (in the M4 section) and an M7 code sketch that implied a C++-side
`battle_state_from_poke_env`.

MctsPlayer (a poke-env Player wired to native.search()) mirrors
battle_engine.ppo_eval.load_ppo_player / self_play.FrozenPolicyPlayer's own
Player-wrapping shape: __init__(*args, **kwargs) -> super().__init__(...),
choose_move(battle) -> BattleOrder built from one translated action id.
Root-only ActionId -> BattleOrder translation (this file's own
_action_id_to_order) is the ONLY place native ActionIds ever cross into
poke-env's action vocabulary - M6's internal search tree never touches it.

Team-preview-order assumption (verified against poke-env's real source, not
guessed): AbstractBattle._team and ._opponent_team are populated by
ordinary dict insertion, and the only place either is ever reassigned
wholesale (get_pokemon()'s nickname-matching branch, abstract_battle.py)
replaces an existing entry's KEY at its EXISTING list index rather than
moving it - so dict insertion order is stable positionally for the whole
battle, for both dicts. For `battle.team` (always fully known from team
preview onward, since it's your own team) this gives a genuine, fixed
team-preview-order match to ActionId's switch targets (action.hpp) -
_action_id_to_order's switch branch relies on this directly, a plain
list-index lookup, NOT the species-sorted mapping action_space.py's
different (Metamon 13-way) action scheme needs; conflating the two would
silently mismap switches. For `battle.opponent_team`, insertion order is
REVEAL order, not necessarily the opponent's real team-preview order
(poke-env has no way to know that before a slot is revealed) - consistent
with Tier 1's already-accepted "opponent modeling is revealed-only"
limitation (see action.hpp's own doc comment on legal_actions()), not a new
gap this translator introduces.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status as PokeEnvStatus
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from battle_engine import _native
# M4b: reuses encoding.py's own already-verified adapter helpers directly
# (move-summary derivation, item's unknown-token sentinel, weather/terrain
# single-most-recent-value reduction) rather than re-deriving equivalent
# logic here - see this module's own M4b design note above
# _move_summary_to_native for why (movedex_table.hpp can't drive this: it
# lacks the heal/sideCondition/selfSwitch/boosts/target flags these need,
# and extending it is out of this milestone's file scope). Leading
# underscores crossed on purpose: this translator is effectively a third
# adapter alongside encoding.py's own poke-env/replay pair, sharing their
# one verified derivation instead of duplicating it - matches this
# project's own two-adapter-plus-shared-core convention.
from battle_engine.encoding import (
    _move_summary_features,
    _poke_env_item,
    _poke_env_terrain,
    _poke_env_weather,
)

# Only the 18 real gen-9 types have a be::Type counterpart (see types.hpp's
# own comment on why Stellar is excluded - Terastallization-only, and
# poke-env's own damage_multiplier special-cases it to a no-op 1.0
# unconditionally). THREE_QUESTION_MARKS/STELLAR fall through
# _type_to_native's "unsupported" branch below, same as a real,
# named simplification elsewhere in this project - not a crash.
_TYPE_MAP = {t.name: getattr(_native.Type, t.name) for t in PokemonType if hasattr(_native.Type, t.name)}
_STATUS_MAP = {s.name: getattr(_native.Status, s.name) for s in PokeEnvStatus}

_TEAM_SIZE = 6
_MOVESET_SIZE = 4

# M4b: matches cpp/include/be/types.hpp's Type enum declaration order
# exactly - the index a MoveSummary.move_types bool sits at (be::Type's own
# order), NOT poke-env's PokemonType alphabetical order (_ALL_TYPES,
# encoding.py's concept - the permutation between the two lives entirely in
# battle_state.cpp's encode_native(), not here).
_BE_TYPE_NAMES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING",
    "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST",
    "DRAGON", "DARK", "STEEL", "FAIRY",
]

# M4b: single-token weather/terrain -> be::Weather/be::Terrain. Built from
# encoding.py's own _WEATHER_FROM_POKE_ENV/_TERRAIN_FROM_POKE_ENV string
# vocabulary (via _poke_env_weather/_poke_env_terrain below), not a
# separate hand-derived mapping.
_WEATHER_TO_NATIVE = {
    "sandstorm": _native.Weather.SANDSTORM,
    "raindance": _native.Weather.RAINDANCE,
    "sunnyday": _native.Weather.SUNNYDAY,
    "snow": _native.Weather.SNOW,
}
_TERRAIN_TO_NATIVE = {
    "electricterrain": _native.Terrain.ELECTRIC,
    "grassyterrain": _native.Terrain.GRASSY,
    "mistyterrain": _native.Terrain.MISTY,
    "psychicterrain": _native.Terrain.PSYCHIC,
}

# M4b: the 6 turn-tracked hazard/screen SideConditions - see
# battle_state.hpp's own comment on the turn-tracked vs. stack-tracked
# split (spikes/toxic_spikes, handled separately below, are the only
# stack-tracked pair).
_HAZARD_TURN_FIELDS = {
    "stealth_rock_turn": SideCondition.STEALTH_ROCK,
    "sticky_web_turn": SideCondition.STICKY_WEB,
    "reflect_turn": SideCondition.REFLECT,
    "light_screen_turn": SideCondition.LIGHT_SCREEN,
    "aurora_veil_turn": SideCondition.AURORA_VEIL,
    "tailwind_turn": SideCondition.TAILWIND,
}


def _type_to_native(poke_env_type: Optional[PokemonType]) -> Any:
    if poke_env_type is None:
        return _native.Type.NONE
    return _TYPE_MAP.get(poke_env_type.name, _native.Type.NONE)


def _status_to_native(poke_env_status: Optional[PokeEnvStatus]) -> Any:
    if poke_env_status is None:
        return _native.Status.NONE
    return _STATUS_MAP[poke_env_status.name]


def _stat_block(base_stats: dict) -> Any:
    block = _native.StatBlock()
    block.hp = base_stats["hp"]
    block.atk = base_stats["atk"]
    block.def_ = base_stats["def"]
    block.spa = base_stats["spa"]
    block.spd = base_stats["spd"]
    block.spe = base_stats["spe"]
    return block


def _move_summary_to_native(move_ids) -> Any:
    """M4b: builds a be::MoveSummary by calling encoding.py's own
    _move_summary_features directly - the exact function encode()'s live
    adapter already uses - rather than re-deriving the same dex-flag logic
    against movedex_table.hpp (which doesn't carry the heal/sideCondition/
    selfSwitch/boosts/target flags this needs, and is out of this
    milestone's file scope to extend - see this module's own top-of-file
    note). This is what makes encode_native()'s output correct by
    construction rather than by porting-then-hoping-it-agrees.
    """
    summary = _move_summary_features(move_ids)
    native = _native.MoveSummary()
    native.has_recovery = summary.has_recovery
    native.has_hazard_setup = summary.has_hazard_setup
    native.has_hazard_removal = summary.has_hazard_removal
    native.has_setup_boost = summary.has_setup_boost
    native.has_pivot = summary.has_pivot
    native.has_priority = summary.has_priority
    native.max_base_power = summary.max_base_power
    move_type_names = {t.name for t in summary.move_types}
    native.move_types = [name in move_type_names for name in _BE_TYPE_NAMES]
    return native


def _pokemon_slot(mon: Pokemon) -> Any:
    slot = _native.PokemonSlot()
    slot.revealed = True
    slot.fainted = mon.fainted
    slot.level = mon.level
    slot.hp_fraction = mon.current_hp_fraction
    slot.status = _status_to_native(mon.status)
    slot.type1 = _type_to_native(mon.type_1)
    slot.type2 = _type_to_native(mon.type_2)
    slot.base_stats = _stat_block(mon.base_stats)
    slot.boost_spe = mon.boosts.get("spe", 0)
    known_spe = mon.stats.get("spe") if mon.stats else None
    slot.spe_stat = known_spe if known_spe is not None else -1

    move_ids = list(mon.moves.keys())[:_MOVESET_SIZE]
    slot.moves = move_ids + [""] * (_MOVESET_SIZE - len(move_ids))

    # M4b: everything encode_native() needs beyond default_eval's fields.
    # species is base_species (NOT species/name) per the plan's own
    # constraint - a real bug precedent in encoding.py's history (form
    # changes like Terapagos rename `name` but not `base_species`).
    slot.species = mon.base_species
    item = _poke_env_item(mon)
    slot.item = item if item is not None else ""
    slot.ability = mon.ability or ""
    # poke-env already resets protect_counter to 0 on switch-out, so
    # reading it off a bench slot needs no special-casing (matches
    # encoding.py's own PokemonView.protect_counter docstring).
    slot.protect_counter = mon.protect_counter
    slot.boost_atk = mon.boosts.get("atk", 0)
    slot.boost_def = mon.boosts.get("def", 0)
    slot.boost_spa = mon.boosts.get("spa", 0)
    slot.boost_spd = mon.boosts.get("spd", 0)
    slot.boost_accuracy = mon.boosts.get("accuracy", 0)
    slot.boost_evasion = mon.boosts.get("evasion", 0)
    slot.move_summary = _move_summary_to_native(list(mon.moves.keys()))
    return slot


def _team_slots(team: dict) -> list:
    mons = list(team.values())
    if len(mons) > _TEAM_SIZE:
        # Should not happen for any real gen9 singles team - a real bug
        # (e.g. a doubles battle, or a malformed request) if it ever does.
        raise ValueError(f"team has {len(mons)} Pokemon, expected at most {_TEAM_SIZE}")
    slots = [_pokemon_slot(mon) for mon in mons]
    slots += [_native.PokemonSlot() for _ in range(_TEAM_SIZE - len(slots))]
    return slots


def _active_slot_index(team: dict, active: Optional[Pokemon]) -> int:
    """-1 means "no well-defined active slot right now" - both when poke-env
    has no active_pokemon at all (team preview) AND when the active
    Pokemon has just fainted. This second case is a real, previously-
    unhandled bug, not an edge case: poke-env's own Battle.active_pokemon
    property (battle.py) returns whichever team member has `.active ==
    True`, with NO fainted check - a Pokemon stays "active" until the
    replacement switch actually lands, so `active is None` alone never
    catches "I just fainted, a forced switch is required." Without this,
    my_active_slot pointed at a fainted mon's real index instead of -1,
    and since that mon's known moveset is still populated,
    legal_actions() (action.hpp) incorrectly offered MOVE actions for a
    Pokemon that cannot move - the server always rejects them
    ("[Invalid choice] Can't move: You need a switch response"), and
    poke-env's own retry-until-legal loop only has a 1/1000 chance per
    retry of giving up and using a safe default order instead (see
    Player.DEFAULT_CHOICE_CHANCE) - an expected ~1000 wasted searches
    (minutes to tens of minutes) per single faint. Found 2026-08-25 by
    directly instrumenting and watching a live diagnostic run, not
    inferred - the real invariant this restores is the one
    BattleState/PokemonSlot already documented as required
    ("my_active_slot is either -1, or a valid index into a slot that is
    both revealed and not fainted") but this translator never actually
    enforced.
    """
    if active is None or active.fainted:
        return -1
    mons = list(team.values())
    return mons.index(active)


def _side_conditions(conditions: dict) -> Any:
    result = _native.SideConditions()
    result.spikes_layers = conditions.get(SideCondition.SPIKES, 0)
    result.toxic_spikes_layers = conditions.get(SideCondition.TOXIC_SPIKES, 0)
    result.stealth_rock = SideCondition.STEALTH_ROCK in conditions
    result.sticky_web = SideCondition.STICKY_WEB in conditions
    # M4b: real turn numbers for encode_native()'s single-most-recent-
    # hazard derivation (see _HAZARD_TURN_FIELDS above) - -1 = not active.
    for field_name, condition in _HAZARD_TURN_FIELDS.items():
        setattr(result, field_name, conditions.get(condition, -1))
    return result


def _weather_to_native(weather: dict) -> Any:
    token = _poke_env_weather(weather)
    return _WEATHER_TO_NATIVE.get(token, _native.Weather.NONE)


def _terrain_to_native(fields: dict) -> Any:
    token = _poke_env_terrain(fields)
    return _TERRAIN_TO_NATIVE.get(token, _native.Terrain.NONE)


def battle_state_from_poke_env(battle: Any) -> Any:
    """Builds a be::BattleState snapshot of `battle`'s current turn. Raises
    ValueError if either team somehow has more than 6 Pokemon (a real bug
    upstream, not a case this translator papers over - mirrors
    encoding.py's own team-preview ValueError precedent for an analogous
    "this shouldn't be possible" state).

    M4b: `battle.weather`/`battle.fields` are read via getattr with an
    empty-dict default, not direct attribute access - a real AbstractBattle
    always has both (never hits the fallback), but this module's own
    pre-existing test fixtures (tests/test_native_legality.py,
    tests/test_mcts_player.py - both out of this milestone's file scope)
    build a SimpleNamespace that doesn't set either, and DW-4.2 requires
    those to keep passing unchanged. The fallback ({} = "no weather/terrain
    info") is the exact real state those minimal fixtures represent anyway.
    """
    state = _native.BattleState()
    state.my_team = _team_slots(battle.team)
    state.opp_team = _team_slots(battle.opponent_team)
    state.my_active_slot = _active_slot_index(battle.team, battle.active_pokemon)
    state.opp_active_slot = _active_slot_index(battle.opponent_team, battle.opponent_active_pokemon)
    state.my_hazards = _side_conditions(battle.side_conditions)
    state.opp_hazards = _side_conditions(battle.opponent_side_conditions)
    state.weather = _weather_to_native(getattr(battle, "weather", {}))
    state.terrain = _terrain_to_native(getattr(battle, "fields", {}))
    return state


def _action_id_to_order(action_id: int, battle: Any) -> BattleOrder:
    """Translates one root ActionId (action.hpp's fixed 0-9 scheme) into a
    submittable poke-env BattleOrder for `battle`'s current turn. Pure
    function of (action_id, battle) - no Player instance needed (Player.
    create_order is itself a staticmethod), which is what makes this
    directly unit-testable without a live connection.

    Switch (0-5): list(battle.team.values())[action_id] - a direct index,
    relying on this module's own docstring finding that battle.team's dict-
    insertion order IS team-preview order, the same order ActionId 0-5
    assumes. Caller-owned precondition (not re-checked here, matching
    action.hpp/mcts.hpp's "callers own their inputs" convention throughout
    this codebase): action_id must be one search() actually returned as
    legal for this exact state, so the indexed slot is guaranteed to exist,
    be non-fainted, and not already the active Pokemon.

    Move (6-9): list(battle.active_pokemon.moves.keys())[action_id - 6] -
    the same ordering _pokemon_slot() used to build the moves list
    legal_actions() legality-checked against, so translating back through
    the identical ordering keeps the two sides of that check in agreement.
    Same caller-owned precondition: battle.active_pokemon must exist (the
    C++ side never returns a move action when my_active_slot == -1, since
    legal_actions() has no move slots to offer in that state) and the move
    slot must be a real, non-empty entry.
    """
    if action_id < _native.NUM_SWITCH_ACTIONS:
        target = list(battle.team.values())[action_id]
        return Player.create_order(target)
    move_slot = action_id - _native.MOVE_ACTION_OFFSET
    move_id = list(battle.active_pokemon.moves.keys())[move_slot]
    return Player.create_order(battle.active_pokemon.moves[move_id])


class MctsPlayer(Player):
    """M7: a poke-env Player driven by M6's native.search() (open-loop
    MCTS/DUCT, plain UCB1, the C++-fixed default_eval leaf evaluator - no
    PPO enhancement track here, that's M6b/a later phase's
    search_puct/PolicyWeights, a different native binding entirely).

    Real-gameplay finding from M7 bring-up, not previously named anywhere
    in cpp/include/be/ (action.hpp/forward_model.hpp/battle_state.hpp
    track no PP, choice-lock, or trapping state at all): a real Showdown
    turn can restrict `battle.available_moves` to fewer than the active
    mon's 4 known moves (Choice item lock, Disable, Encore, 0 PP, ...),
    but `legal_actions()` always treats every known move slot as legal
    regardless. If search() picks a move slot that's real-game-illegal
    this turn, poke-env's own request/response protocol just re-prompts
    for the same turn (a client-side retry, not a crash or a hang -
    `Player._handle_battle_request` sends a fresh `choose_move` on every
    re-sent `|request|`) until either a legal choice lands or poke-env's
    own probabilistic DEFAULT_CHOICE_CHANCE fallback breaks the loop.
    Measured impact: real wall-clock stayed under ~0.6s/battle even with
    this overhead (10-battle samples vs RandomPlayer/MaxBasePowerPlayer at
    n_simulations=200, gen9randombattle - see this phase's Execution Log
    entry), so it doesn't block a real 500-battle benchmark run, but it's
    a real, load-bearing gap for a future phase to close (PP/choice-lock/
    trapping modeling), not something this phase's file scope can fix -
    named here rather than left to be silently rediscovered.

    MctsPlayer IS-A Player: same shape as every other Player subclass in
    this codebase (FrozenPolicyPlayer, TwoPlySearchPlayer) - choose_move is
    the only overridden method, no empty overrides, LSP holds.

    n_simulations has no built-in default: M7's own scope explicitly
    includes MEASURING a real ms/turn number before picking one (see this
    module's own history / CLAUDE.md's Phase 4 status) rather than
    guessing - forcing a caller to state it keeps that measurement honest
    instead of silently encoding a guess as this class's default.

    seed seeds an internal random.Random that draws one fresh 64-bit
    search-seed per choose_move() call (search()'s own seed parameter),
    NOT the same literal seed reused every turn - mirrors
    FrozenPolicyPlayer's own seed-param convention (a fixed construction
    seed makes a whole battle's sequence of choices reproducible, while
    still giving every individual search() call its own seed rather than
    correlating every turn's simulation trace against an identical value).
    """

    def __init__(self, *args, n_simulations: int, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_simulations = n_simulations
        self._rng = random.Random(seed)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        state = battle_state_from_poke_env(battle)
        result = _native.search(state, self._n_simulations, self._rng.getrandbits(64))
        if result.best_action == _native.NO_ACTION:
            # No legal root action at all (e.g. every non-active team member
            # fainted/unrevealed and the active mon's every known move slot
            # is somehow empty - action.hpp's own documented edge case for
            # legal_actions()). Fall back to Showdown's own "first legal
            # order" default rather than propagating an invalid order to
            # poke-env - the plan's own explicit requirement for this case.
            return self.choose_default_move()
        return _action_id_to_order(result.best_action, battle)


class MctsPuctPlayer(Player):
    """M6b: a poke-env Player driven by native.search_puct() - PUCT search
    with the trained PPO actor as a per-node move prior and its critic as
    the leaf value (see cpp/include/be/mcts.hpp's search_puct() doc comment
    for the full design), in place of MctsPlayer's fixed UCB1 + default_eval.

    A SEPARATE class from MctsPlayer, not a mode flag on it - same "no
    logical-cohesion flag branch" reasoning this milestone applies to
    select_puct_action vs. select_ucb1_action C++-side (see mcts.hpp/
    docs/code-standards.md's cohesion standard), applied here to the two
    Player classes for the identical reason: their choose_move bodies differ
    in which native function they call and what they load at construction,
    not in a shared body branching on a flag.

    Loads PolicyWeights ONCE at construction (ppo_bin_path, see
    _native.PolicyWeights.load's own doc comment for the exact binary
    format read - scripts/export_weights.py's output) and reuses the loaded
    handle for every choose_move() call - re-reading the weight file per
    turn would blow the ms/turn budget this milestone measured (see
    cpp/tests/test_mcts.cpp's own DW-5.3 microbenchmark).

    Otherwise mirrors MctsPlayer exactly: same root-only ActionId ->
    BattleOrder translation (_action_id_to_order), same NO_ACTION fallback
    to choose_default_move(), same n_simulations-has-no-default rationale
    (a real measured ms/turn number should drive the choice, not a guess),
    same per-call fresh 64-bit seed convention. MctsPuctPlayer IS-A Player -
    same LSP precedent as MctsPlayer/FrozenPolicyPlayer/TwoPlySearchPlayer.
    """

    def __init__(self, *args, ppo_bin_path: str, n_simulations: int, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights = _native.PolicyWeights.load(ppo_bin_path)
        self._n_simulations = n_simulations
        self._rng = random.Random(seed)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        state = battle_state_from_poke_env(battle)
        result = _native.search_puct(state, self._weights, self._n_simulations, self._rng.getrandbits(64))
        if result.best_action == _native.NO_ACTION:
            return self.choose_default_move()
        return _action_id_to_order(result.best_action, battle)
