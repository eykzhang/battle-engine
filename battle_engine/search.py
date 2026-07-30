"""Phase-1 bot: 2-ply lookahead over battle_engine.evaluation.evaluate, using
battle_engine.damage.expected_damage to project outcomes.

For every legal action (attack or switch), this assumes the opponent replies
with their best *known* move against the resulting state, evaluates that
projected state, and picks the action with the highest score. That's the "my
action, then their best response" simplification of the real simultaneous
matrix game discussed in the roadmap (a full game-theoretic solve is stretch
Phase 4's DUCT/MCTS search).

Named simplifications (consistent with damage.py/evaluation.py):
- The opponent is assumed to best-respond to whichever of my actions is under
  consideration, not to move simultaneously with imperfect information about
  my choice.
- Their reply is chosen only among moves they've already revealed; if they
  haven't shown any moves yet, they're assumed to deal no damage that turn.
- Only the two active Pokemon's HP changes in the projection — switching in
  doesn't take hazard chip damage, and non-damaging move effects (status,
  boosts, hazards) aren't projected forward, matching evaluate()'s own scope.

Phase 2 (milestone E): TwoPlySearchPlayer's scoring function is injectable
(`eval_fn`), not hardcoded to evaluate() — the search *shape* (2-ply,
best-known-reply, switch-urgency patch) stays the same, only the function
that scores a projected state changes. `win_prob.make_eval_fn` adapts a
trained WinProbModel into this same shape for a head-to-head benchmark
against the Phase-1 hand-crafted bot.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Union

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from battle_engine.damage import expected_damage
from battle_engine.evaluation import evaluate, type_matchup_score

EvalFn = Callable[[object], float]

# How much a shallow 2-ply search undervalues escaping a bad matchup: an
# attack's HP-swing value is directly visible within the search horizon, but
# switching's payoff (avoiding several future turns of damage) mostly isn't,
# so switching structurally loses to attacking even when it's clearly the
# better plan (found via replay inspection: the bot never switched
# voluntarily, e.g. tanking three straight burns instead of pivoting out).
# This adds back a bonus to switch candidates, scaled by how bad the current
# matchup actually is, rather than a hard switch/stay rule.
SWITCH_URGENCY_WEIGHT = 1.0


def _apply_damage(mon: Pokemon, damage_fraction: float) -> Pokemon:
    """A shallow copy of mon with HP reduced by damage_fraction (of max HP)."""
    projected = copy.copy(mon)
    new_hp = max(0, mon.current_hp - round(damage_fraction * mon.max_hp))
    projected._current_hp = new_hp
    if new_hp == 0:
        projected._status = Status.FNT
    return projected


def _replace_mon(
    team: Dict[str, Pokemon], target: Pokemon, replacement: Pokemon
) -> Dict[str, Pokemon]:
    return {
        ident: (replacement if mon is target else mon) for ident, mon in team.items()
    }


def _opponent_best_reply(opp_active: Pokemon, my_active: Pokemon) -> Optional[Move]:
    """Opponent's most damaging revealed move against my_active, or None if
    they haven't shown any moves yet."""
    known_moves = list(opp_active.moves.values())
    if not known_moves:
        return None
    return max(known_moves, key=lambda m: expected_damage(opp_active, my_active, m))


def _project_after_action(
    battle: AbstractBattle,
    my_active_after_action: Pokemon,
    my_move: Optional[Move],
) -> SimpleNamespace:
    """A lightweight snapshot exposing evaluate()'s expected interface,
    reflecting the state after my_move (or a switch, if my_move is None)
    resolves and the opponent plays their assumed best reply.
    """
    opp_active = battle.opponent_active_pokemon

    projected_opp_active = opp_active
    if my_move is not None:
        dmg = expected_damage(my_active_after_action, opp_active, my_move)
        projected_opp_active = _apply_damage(opp_active, dmg)

    # A fainted opponent doesn't get to move — without this, a guaranteed KO
    # always looked no better than a non-KO hit, since the projection still
    # charged the full retaliation either way.
    opp_reply = (
        None
        if projected_opp_active.fainted
        else _opponent_best_reply(opp_active, my_active_after_action)
    )
    projected_my_active = my_active_after_action
    if opp_reply is not None:
        dmg = expected_damage(opp_active, my_active_after_action, opp_reply)
        projected_my_active = _apply_damage(my_active_after_action, dmg)

    return SimpleNamespace(
        team=_replace_mon(battle.team, my_active_after_action, projected_my_active),
        opponent_team=_replace_mon(
            battle.opponent_team, opp_active, projected_opp_active
        ),
        active_pokemon=projected_my_active,
        opponent_active_pokemon=projected_opp_active,
        side_conditions=battle.side_conditions,
        opponent_side_conditions=battle.opponent_side_conditions,
        # Passed through unchanged (a 1-turn damage projection doesn't
        # change field conditions) - evaluate() doesn't read these, but
        # encoding.battle_view_from_poke_env does, and would AttributeError
        # without them. Needed for milestone E's learned eval_fn to work.
        weather=battle.weather,
        fields=battle.fields,
    )


def _switch_urgency_bonus(battle: AbstractBattle, weight: float) -> float:
    """How much extra credit to give switch candidates this turn, scaled by
    how unfavorable the current matchup already is (0 if it's fine)."""
    current_matchup = type_matchup_score(
        battle.active_pokemon, battle.opponent_active_pokemon
    )
    return max(0.0, -current_matchup) * weight


class TwoPlySearchPlayer(Player):
    def __init__(
        self,
        *args,
        eval_fn: EvalFn = evaluate,
        switch_urgency_weight: float = SWITCH_URGENCY_WEIGHT,
        **kwargs,
    ):
        """eval_fn scores a projected battle-like state from its own
        perspective — see evaluation.evaluate()'s docstring for the exact
        duck-typed interface it (and any replacement) must accept. Defaults
        to the Phase-1 hand-crafted eval, unchanged from before this
        parameter existed.

        switch_urgency_weight is separate from eval_fn on purpose: it was
        tuned empirically for evaluate()'s scale (roughly [-4, 4], summed
        hand-picked weights) and doesn't necessarily transfer to a
        differently-scaled eval_fn — e.g. win_prob.make_eval_fn's [0, 1]
        probability output, where even one unfavorable type matchup's
        default bonus (up to ~4.0) would swamp any real difference between
        candidates (typically well under 0.1). Pass 0.0 explicitly for such
        an eval_fn rather than silently reusing a weight tuned for a
        different scale.
        """
        super().__init__(*args, **kwargs)
        self._eval_fn = eval_fn
        self._switch_urgency_weight = switch_urgency_weight

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            return self.choose_random_move(battle)

        # Each candidate: (order to submit, my move if attacking else None,
        # my active Pokemon after the action resolves).
        candidates: List[tuple[Union[Move, Pokemon], Optional[Move], Pokemon]] = [
            (move, move, battle.active_pokemon) for move in battle.available_moves
        ] + [(switch, None, switch) for switch in battle.available_switches]

        if not candidates:
            return self.choose_random_move(battle)

        switch_bonus = _switch_urgency_bonus(battle, self._switch_urgency_weight)

        def score(candidate: tuple[Union[Move, Pokemon], Optional[Move], Pokemon]) -> float:
            _, my_move, my_active_after = candidate
            value = self._eval_fn(_project_after_action(battle, my_active_after, my_move))
            if my_move is None:  # this candidate is a switch
                value += switch_bonus
            return value

        best_order, _, _ = max(candidates, key=score)
        return self.create_order(best_order)
