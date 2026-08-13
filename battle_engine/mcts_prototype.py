"""Phase 4, milestone M0: a pure-Python open-loop MCTS/DUCT prototype - NOT
production code. Its only job is to answer one question before committing to
the multi-week C++ implementation (see /Users/edward/.claude/plans/
precious-crafting-bachman.md): does a real, sampled, branching search beat
search.py's 1-ply "my action, then opponent's single best-known-move reply"
expected-value projection? This reuses battle_engine.damage's formula pieces
and battle_engine.search's projection helpers directly rather than
reimplementing them - only the damage sampling and the tree/search shape are
new.

Simplifications accepted here (beyond damage.py/evaluation.py's own, already-
documented ones) that are explicitly NOT carried into the real C++ Tier-1
implementation, which does model these:
- No switch-in hazard chip damage (the real gap search.py's docstring names -
  the C++ forward model closes it; this prototype doesn't need to, since it's
  only testing whether deeper *branching* search helps at all).
- No non-damage move effects (status, boosts, hazard setup/removal) - matches
  evaluate()'s own scope.
- Opponent's action space at internal (non-root) nodes is limited to their
  currently-revealed moves and currently-revealed, non-active, non-fainted
  team members - the same Tier-1 opponent-modeling restriction the C++ plan
  uses (see the plan's "Opponent modeling" section for why).
- Internal-node legality (both sides) ignores PP/trapped/choice-lock - only
  the real root's candidate list uses poke-env's actual
  battle.available_moves/available_switches.
- Triple Kick/Triple Axel's real per-hit rising base power isn't modeled,
  just repeated 90%-accuracy single-power hits - a rare-move corner case not
  worth prototype time.

Value backup exploits a real, structurally-verified property of evaluate():
it is exactly antisymmetric under swapping (team, opponent_team,
active_pokemon, opponent_active_pokemon, side_conditions,
opponent_side_conditions) - every term is a my-minus-opponent difference or a
strictly order-reversing comparison. So the opponent's value at a node is
exactly -my_value, not an approximation - no need to construct an actual
POV-flipped state to score it from their side.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from battle_engine.damage import _attack_defense_stats, _crit_chance, _fixed_damage, estimate_stat
from battle_engine.evaluation import evaluate
from battle_engine.search import _apply_damage, _replace_mon

Action = Union[Move, Pokemon]  # a move to use, or a Pokemon to switch into

DEFAULT_N_SIMULATIONS = 200
DEFAULT_MAX_DEPTH = 4
# The textbook UCB1 default (~1.4) assumes rewards in [0,1]; evaluate()'s
# actual scale is roughly [-4,4] (per search.py's own documented scale), so a
# 1.4 constant makes the exploration term negligible relative to typical
# value differences - scaled up accordingly, not re-derived from first
# principles (a real, plausible confound checked before trusting the first
# M0 benchmark result, not tuned to make search win).
UCB1_EXPLORATION_CONSTANT = 4.0

# Real Showdown multi-hit distribution for the standard 2-5-hit move family
# (matches the (2+3)/3 + (4+5)/6 expectation damage.py's Move.expected_hits
# already assumes, sampled instead of averaged here).
_MULTI_HIT_COUNTS = (2, 3, 4, 5)
_MULTI_HIT_WEIGHTS = (0.35, 0.35, 0.15, 0.15)


def _sample_hit_count(move: Move, rng: random.Random) -> int:
    if move._id in ("triplekick", "tripleaxel"):
        hits = 0
        for _ in range(3):
            if rng.random() >= 0.9:
                break
            hits += 1
        return hits
    min_hits, max_hits = move.n_hit
    if min_hits == max_hits:
        return min_hits
    return rng.choices(_MULTI_HIT_COUNTS, weights=_MULTI_HIT_WEIGHTS)[0]


def sample_damage_fraction(attacker: Pokemon, defender: Pokemon, move: Move, rng: random.Random) -> float:
    """A *sampled* counterpart to damage.expected_damage: same formula
    pieces, reused directly (not reimplemented), but crit/roll/accuracy/
    multi-hit are each rolled once rather than folded into an expectation -
    the actual mechanism change this milestone exists to test.
    """
    if move.category == MoveCategory.STATUS:
        return 0.0

    fixed = _fixed_damage(attacker, defender, move)
    if fixed is not None:
        if defender.damage_multiplier(move.type) == 0:
            return 0.0
        if rng.random() >= move.accuracy:
            return 0.0
        return fixed / estimate_stat(defender, "hp")

    if not move.base_power:
        return 0.0
    if rng.random() >= move.accuracy:
        return 0.0

    type_effectiveness = defender.damage_multiplier(move.type)
    if type_effectiveness == 0:
        return 0.0

    atk_stat, def_stat = _attack_defense_stats(move)
    atk = estimate_stat(attacker, atk_stat)
    defense = estimate_stat(defender, def_stat)
    base = int(int(int(2 * attacker.level / 5 + 2) * move.base_power * atk / defense) / 50) + 2
    stab = attacker.stab_multiplier if move.type in attacker.types else 1.0

    total = 0.0
    for _ in range(_sample_hit_count(move, rng)):
        roll = rng.randint(85, 100) / 100.0
        crit_mult = 1.5 if rng.random() < _crit_chance(move) else 1.0
        total += base * stab * type_effectiveness * roll * crit_mult

    return total / estimate_stat(defender, "hp")


def _boosted_speed(mon: Pokemon) -> float:
    stage = mon.boosts["spe"]
    multiplier = (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
    speed = estimate_stat(mon, "spe") * multiplier
    if mon.status == Status.PAR:
        speed *= 0.5
    return speed


def _my_actions(state) -> List[Action]:
    active = state.active_pokemon
    actions: List[Action] = []
    if active is not None and not active.fainted:
        actions.extend(active.moves.values())
    actions.extend(mon for mon in state.team.values() if mon is not active and not mon.fainted)
    return actions


def _opp_actions(state) -> List[Action]:
    active = state.opponent_active_pokemon
    actions: List[Action] = []
    if active is not None and not active.fainted:
        actions.extend(active.moves.values())
    actions.extend(mon for mon in state.opponent_team.values() if mon is not active and not mon.fainted)
    return actions


def resolve_turn(state, my_action: Action, opp_action: Optional[Action], rng: random.Random):
    """One sampled turn: both actions resolve (switch = free, no damage;
    move = ordered by priority then speed, tie broken randomly, then
    sampled damage), returning a new state. opp_action=None models "no
    revealed information" the same way search.py's _opponent_best_reply
    does - they're assumed to deal no damage that turn.
    """
    my_move = my_action if isinstance(my_action, Move) else None
    my_active = state.active_pokemon if my_move is not None else my_action
    # opp_action is None when nothing's been revealed about the opponent yet
    # (see module docstring) - opp_active must still default to their real
    # current active mon in that case, not None (a switch's target when
    # opp_action IS a Pokemon; their current active either when opp_action
    # is a revealed Move or when it's None).
    opp_move = opp_action if isinstance(opp_action, Move) else None
    opp_active = opp_action if isinstance(opp_action, Pokemon) else state.opponent_active_pokemon

    order = []
    if my_move is not None:
        order.append(("me", my_move))
    if opp_move is not None:
        order.append(("opp", opp_move))
    order.sort(
        key=lambda e: (
            -e[1].priority,
            -_boosted_speed(my_active if e[0] == "me" else opp_active),
            rng.random(),
        )
    )

    my_hp_mon, opp_hp_mon = my_active, opp_active
    for side, move in order:
        attacker = my_hp_mon if side == "me" else opp_hp_mon
        defender = opp_hp_mon if side == "me" else my_hp_mon
        if defender.fainted:
            continue
        dmg = sample_damage_fraction(attacker, defender, move, rng)
        if side == "me":
            opp_hp_mon = _apply_damage(opp_hp_mon, dmg)
        else:
            my_hp_mon = _apply_damage(my_hp_mon, dmg)

    return SimpleNamespace(
        team=_replace_mon(state.team, my_active, my_hp_mon),
        opponent_team=_replace_mon(state.opponent_team, opp_active, opp_hp_mon),
        active_pokemon=my_hp_mon,
        opponent_active_pokemon=opp_hp_mon,
        side_conditions=state.side_conditions,
        opponent_side_conditions=state.opponent_side_conditions,
    )


@dataclass
class VisitStats:
    visits: int = 0
    value_sum: float = 0.0

    def ucb1(self, total_visits: int, c: float = UCB1_EXPLORATION_CONSTANT) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value_sum / self.visits + c * math.sqrt(math.log(total_visits) / self.visits)


@dataclass
class DecisionNode:
    """Open-loop: this node represents a path of (my_action, opp_action)
    choices from the root, not a stored canonical state - every visit
    resamples resolve_turn() fresh. my_actions/opp_actions were enumerated
    from the ONE sample that first created this node, so they may not
    exactly match every later resample's true legal set (e.g. a mon that
    fainted in one sample but not another) - accepted, matching the same
    tradeoff named in the approved C++ plan's M6 design.
    """

    my_actions: List[Action]
    opp_actions: List[Action]
    my_stats: List[VisitStats] = field(init=False)
    opp_stats: List[VisitStats] = field(init=False)
    children: Dict[tuple, "DecisionNode"] = field(default_factory=dict)

    def __post_init__(self):
        self.my_stats = [VisitStats() for _ in self.my_actions]
        self.opp_stats = [VisitStats() for _ in self.opp_actions]


def _select_ucb1(stats: List[VisitStats]) -> int:
    total = sum(s.visits for s in stats) + 1
    return max(range(len(stats)), key=lambda i: stats[i].ucb1(total))


def _search(state, node: DecisionNode, rng: random.Random, depth: int, max_depth: int) -> float:
    """Returns state's value from state.active_pokemon's perspective."""
    if (
        depth >= max_depth
        or state.active_pokemon.fainted
        or state.opponent_active_pokemon.fainted
        or not node.my_actions
    ):
        return evaluate(state)

    i = _select_ucb1(node.my_stats)
    j = _select_ucb1(node.opp_stats) if node.opp_actions else None
    opp_action = node.opp_actions[j] if j is not None else None

    next_state = resolve_turn(state, node.my_actions[i], opp_action, rng)
    key = (i, j)

    if key not in node.children:
        node.children[key] = DecisionNode(_my_actions(next_state), _opp_actions(next_state))
        value = evaluate(next_state)
    else:
        value = _search(next_state, node.children[key], rng, depth + 1, max_depth)

    node.my_stats[i].visits += 1
    node.my_stats[i].value_sum += value
    if j is not None:
        node.opp_stats[j].visits += 1
        node.opp_stats[j].value_sum += -value  # evaluate()'s exact antisymmetry - see module docstring

    return value


def search(battle, n_simulations: int, rng: random.Random, max_depth: int) -> Optional[Action]:
    """Root candidates use battle's real, PP/trapped-aware legal-action
    lists (available_moves/available_switches); internal nodes use the
    simpler _my_actions/_opp_actions - see module docstring.
    """
    my_actions: List[Action] = list(battle.available_moves) + list(battle.available_switches)
    if not my_actions:
        return None
    opp_actions = _opp_actions(battle)

    root = DecisionNode(my_actions, opp_actions)
    for _ in range(n_simulations):
        _search(battle, root, rng, 0, max_depth)

    best_i = max(range(len(root.my_actions)), key=lambda i: root.my_stats[i].visits)
    return root.my_actions[best_i]


class MctsPrototypePlayer(Player):
    """M0's benchmark-facing player - see module docstring. Not the real
    Phase 4 deliverable (that's M7's MctsPlayer, pybind11-backed); this
    exists purely to measure whether real branching search beats
    TwoPlySearchPlayer before investing in the C++ implementation.
    """

    def __init__(
        self,
        *args,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._n_simulations = n_simulations
        self._max_depth = max_depth
        self._rng = random.Random(seed)

    def choose_move(self, battle) -> BattleOrder:
        if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            return self.choose_random_move(battle)
        best_action = search(battle, self._n_simulations, self._rng, self._max_depth)
        if best_action is None:
            return self.choose_random_move(battle)
        return self.create_order(best_action)
