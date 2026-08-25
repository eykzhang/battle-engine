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
    if active is None:
        return -1
    mons = list(team.values())
    return mons.index(active)


def _side_conditions(conditions: dict) -> Any:
    result = _native.SideConditions()
    result.spikes_layers = conditions.get(SideCondition.SPIKES, 0)
    result.toxic_spikes_layers = conditions.get(SideCondition.TOXIC_SPIKES, 0)
    result.stealth_rock = SideCondition.STEALTH_ROCK in conditions
    result.sticky_web = SideCondition.STICKY_WEB in conditions
    return result


def battle_state_from_poke_env(battle: Any) -> Any:
    """Builds a be::BattleState snapshot of `battle`'s current turn. Raises
    ValueError if either team somehow has more than 6 Pokemon (a real bug
    upstream, not a case this translator papers over - mirrors
    encoding.py's own team-preview ValueError precedent for an analogous
    "this shouldn't be possible" state).
    """
    state = _native.BattleState()
    state.my_team = _team_slots(battle.team)
    state.opp_team = _team_slots(battle.opponent_team)
    state.my_active_slot = _active_slot_index(battle.team, battle.active_pokemon)
    state.opp_active_slot = _active_slot_index(battle.opponent_team, battle.opponent_active_pokemon)
    state.my_hazards = _side_conditions(battle.side_conditions)
    state.opp_hazards = _side_conditions(battle.opponent_side_conditions)
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
