// M6: MCTS with DUCT (Decoupled UCT) - the actual search, and per the plan's
// own review notes (plans/precious-crafting-bachman.md's M6 section) the
// milestone needing the MOST structure, not the least, since node-model
// ambiguity is exactly where an implementer gets stuck for days. Every
// design decision below that isn't forced by the plan is called out
// explicitly as a deliberate choice, not left implicit.
//
// EXPLICITLY OPEN-LOOP, NOT CLOSED-LOOP: nodes represent action-index paths
// from the root, not a stored canonical BattleState. Each traversal
// resamples resolve_turn() fresh from `root` using the accumulated path of
// (my_action, opp_action) pairs chosen so far. This composes cleanly with
// action.hpp's fixed, state-independent action scheme (that's WHY the
// scheme was designed that way) and sidesteps "which state does this node's
// statistics correspond to" entirely - at the cost of legal_actions()
// potentially differing slightly between two resamples of the SAME path
// (e.g. a mon fainted via hazard chip damage in one sample, not another,
// since sample_damage_fraction/apply_switch_in_hazards are stochastic).
// This is a real, accepted approximation, not a bug - see the plan's own
// discussion.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "be/action.hpp"
#include "be/battle_state.hpp"

namespace be {

// Per-action running statistics for UCB1/DUCT - one of these per legal
// action in a node's action list (index-aligned with that list, NOT keyed
// by ActionId directly, since ActionId values aren't contiguous from 0 for
// a given node's actual legal set).
struct VisitStats {
  int visits = 0;
  float value_sum = 0.0f;
};

// Sentinel used in a packed child key (see pack_action_pair below) for "this
// side did not independently choose an action this node" - i.e. the
// non-acting side's slot on a kForcedSwitch node. Not a legal ActionId
// (those are always >= 0 - see action.hpp), so it can never collide with a
// real action value.
inline constexpr ActionId kNoAction = -1;

// Packs a (my_action, opp_action) pair into a single unordered_map key.
// ActionId is int8_t (range 0-9 for a real DecisionNode, or kNoAction for
// the unused slot on a kForcedSwitch node - see SearchNode::kind below) -
// cast through uint8_t first so kNoAction's negative bit pattern packs
// predictably instead of relying on implementation-defined sign-extension
// behavior during the shift.
uint32_t pack_action_pair(ActionId my_action, ActionId opp_action);

// Which shape a SearchNode has. A tagged single struct (below) was chosen
// over a std::variant<DecisionNode, ForcedSwitchNode>-style split-type
// design specifically because a recursive std::variant whose alternatives
// each hold unique_ptr<SearchNode> children hits real C++ trouble: forming
// variant<A, B> requires A and B to be COMPLETE types at that point, but A
// and B each need SearchNode (the variant's own alias) to exist first to
// declare their unique_ptr<SearchNode> members - a genuine chicken-and-egg
// incomplete-type problem, not a style preference. The tag-enum design
// below sidesteps it entirely (a struct containing unordered_map<uint32_t,
// unique_ptr<SearchNode>> as a member of itself is completely ordinary C++,
// no forward-declaration tricks needed) while still satisfying the plan's
// "its own type/tagged variant, not an implicit empty-opponent-action-list"
// requirement: kForcedSwitch is a real, explicit tag, not inferred from
// opp_actions happening to be empty.
enum class NodeKind : uint8_t {
  kDecision,      // both sides choose simultaneously (the normal case)
  kForcedSwitch,  // exactly one side must choose a replacement switch after
                  // its active fainted mid-turn (TurnResolution::kMyFainted
                  // / kOppFainted) - see forced_switch_side below for which
};

// One node in the open-loop search tree.
//
// kDecision: my_actions/opp_actions/my_stats/opp_stats are ALL populated
// (independent UCB1 tables per side - the "decoupled" in DUCT). Children
// are keyed by pack_action_pair(my_action, opp_action) for whichever pair
// was actually selected and simulated through resolve_turn().
//
// kForcedSwitch: only ONE of (my_actions, my_stats) / (opp_actions,
// opp_stats) is populated - WHICHEVER PAIR MATCHES forced_switch_side, not
// always my_actions/my_stats. This is deliberate, not an arbitrary choice:
// my_actions/my_stats must always mean "the real searching agent's own
// data" and opp_actions/opp_stats must always mean "the real opponent's
// own data," at EVERY node in the tree regardless of kind - see EvalFn's
// own doc comment below for why backup depends on that invariant holding
// uniformly. If forced_switch_side == Side::Opp, populate
// opp_actions/opp_stats (the opponent's own legal switch options) and
// leave my_actions/my_stats empty - not the reverse. Either way, the
// populated list is restricted to that side's legal SWITCH actions only
// (query legal_actions() with Side restricted to switch-only results,
// since only a subset of the side's actions are legal in this situation -
// a caller responsibility, same as legal_actions()'s own "callers are
// responsible for restricting to what's actually choosable" convention
// elsewhere in this codebase). Children are keyed by
// pack_action_pair(chosen_switch, kNoAction) if forced_switch_side == Me,
// or pack_action_pair(kNoAction, chosen_switch) if == Opp - so the packed
// key's two slots keep meaning "my choice" / "opp choice" consistently
// too, not "whichever side happened to be forced."
//
// TurnResolution::kBothFainted is NOT a third NodeKind. Since neither
// side's forced replacement switch damages the other side's already-
// decided active (apply_switch_in_hazards only ever touches the incoming
// mon's own hazards), the two replacements are independent and can be
// modeled as two CHAINED kForcedSwitch nodes (side A's forced-switch node's
// single child is side B's forced-switch node) in either order - order
// doesn't affect the resulting state. This is a suggested resolution, not
// mandated by the plan; the tree-walk logic that actually constructs this
// chain lives in search()'s implementation, not this header.
struct SearchNode {
  NodeKind kind = NodeKind::kDecision;
  Side forced_switch_side = Side::Me;  // only meaningful when kind == kForcedSwitch

  std::vector<ActionId> my_actions;
  std::vector<ActionId> opp_actions;  // empty when kind == kForcedSwitch
  std::vector<VisitStats> my_stats;   // index-aligned with my_actions
  std::vector<VisitStats> opp_stats;  // index-aligned with opp_actions; empty when kind == kForcedSwitch

  std::unordered_map<uint32_t, std::unique_ptr<SearchNode>> children;
};

// Leaf value function: given a state, returns a score from the REAL
// searching agent's own perspective - state.my_team/my_active_slot always
// means the actual "me" driving this search() call, at EVERY node in the
// tree, at every depth. This is NOT an alternating-turn game like chess,
// where a symmetric evaluator conventionally scores "from whoever's about
// to move" and a caller has to flip sign every ply to account for the
// alternation. resolve_turn() resolves BOTH sides' actions simultaneously
// every turn, and BattleState's my_team/opp_team identities never swap as
// the open-loop tree gets deeper (my_action always applies to
// state.my_team, opp_action always applies to state.opp_team, at every
// resolve_turn() call along the whole path from root). So leaf_eval(state)
// means the same, fixed thing - "how good is this for the real me" -
// everywhere in the tree; there is no per-depth sign flip to apply during
// backup at all.
//
// The ONLY place a sign flip belongs is between the tree's two independent
// per-side DUCT tables at a given node (my_stats vs opp_stats - or, at a
// kForcedSwitch node, whichever ONE of those pairs is actually populated,
// per SearchNode's own doc comment above): my_stats always accumulates
// the raw leaf value v (real me benefits when v is high), opp_stats always
// accumulates -v (so UCB1 maximizing over opp_stats is equivalent to the
// opponent minimizing my real value - the standard negamax trick, applied
// once per relevant node, not depth-dependent). Get this "which TABLE,
// not which DEPTH" distinction right - conflating the two (as this
// header's own earlier draft briefly did) silently corrupts every backed-
// up value.
//
// CRITICAL, plan-correcting note on the value-backup convention: the plan's
// M6 section says "opponent's value is 1 - v (valid given zero-sum,
// POV-relative scoring)" - that convention is correct ONLY for a bounded
// [0,1] win-probability-style evaluator (WinProbModel, the M6b enhancement
// track's leaf evaluator). It is NOT correct for default_eval below.
// Verified directly against eval.cpp's actual formula (not assumed):
// evaluate() is exactly ANTISYMMETRIC under swapping which side is "my"
// vs "opp" - every term (HP-fraction diff, alive-count diff, status diff,
// hazard diff, type_matchup_score, speed_control_score) flips sign under
// that swap. So for default_eval specifically, the correct backup is
// opponent_value = -v, NOT 1 - v. Using 1-v here would silently corrupt
// backpropagated values (e.g. v=2.0 -> "opponent's value" of -1.0 under
// 1-v, vs the correct +(-2.0) under -v - these disagree by more than sign).
// If/when a WinProbModel-based EvalFn is wired in later (M6b), switch back
// to 1-v for THAT evaluator specifically - don't apply one convention
// universally without checking which shape the active leaf_eval actually
// has.
using EvalFn = std::function<float(const BattleState&)>;

// Default leaf evaluator - the C++ port of evaluation.py's hand-crafted
// eval (eval.hpp's evaluate()), per the plan's Scope decision on why the
// hand-crafted eval, not a learned model, is M6/M7's default. A thin
// forward - the real content is evaluate() itself.
float default_eval(const BattleState& state);

// Standard UCB1 selection over one node's per-side action/stats table:
// argmax_i ( visits_i == 0 ? +infinity
//                          : value_sum_i / visits_i
//                            + exploration_constant * sqrt(ln(parent_visits) / visits_i) )
// An unvisited action (visits == 0) is always selected before any visited
// one, regardless of exploration_constant - standard UCB1 cold-start
// behavior, not something to special-case away. `actions` and `stats` must
// be the same length and index-aligned (stats[i] tracks actions[i]) - a
// caller bug otherwise, not checked here, same "callers own their inputs"
// convention as legal_actions()/resolve_turn(). Returns kNoAction if
// `actions` is empty (nothing to select - the caller's problem, e.g. a
// forced-switch side with zero legal replacements). This DOES happen for a
// real is_valid() state, routinely, not just in theory: M7 integration
// found it via ASan against realistic early-battle states (opponent's
// bench mostly unrevealed - the Tier-1 limitation legal_actions() already
// documents), not a synthetic case. search()'s own tree-walk handles a
// kForcedSwitch node hitting this by treating it as a terminal leaf for
// that simulation (see the kNoAction check in search()'s implementation) -
// this function itself still just returns kNoAction and leaves the
// decision to the caller, per the "caller owns their inputs" convention
// above.
//
// Exposed as its own function (not inlined into search()'s tree walk) so
// it's independently unit-testable against a synthetic bandit with a
// known-correct answer - see cpp/tests/test_mcts.cpp.
ActionId select_ucb1_action(const std::vector<ActionId>& actions, const std::vector<VisitStats>& stats,
                             int parent_visits, float exploration_constant);

// M0's Python prototype (see git history / CLAUDE.md's Phase 4 M0 section)
// swept c in {0.7, 1.4, 4.0} against TwoPlySearchPlayer and found no
// statistically distinguishable difference between them at that milestone's
// scale - 1.4 (~sqrt(2), the textbook UCB1 constant) is used here as a
// reasonable, but NOT deeply-tuned, starting point. Real tuning is a
// post-M7 step once search() is wired into the benchmark harness and a
// real head-to-head number exists to tune against - same "measure, don't
// guess" standard as every other phase's hyperparameter (see e.g.
// switch_urgency_weight's own sweep history).
inline constexpr float kDefaultExplorationConstant = 1.4f;

// The result of one search() call from `root`.
struct SearchResult {
  ActionId best_action;
  // The ROOT node's "my" side visit distribution specifically (my_actions
  // paired with final visits from my_stats) - NOT the opponent's
  // independent table, which is an internal DUCT-only construct never
  // exposed to a caller (a real caller only ever needs to know what MY
  // side should play, plus diagnostics for that choice). The plan's own
  // struct sketch names this field without specifying which side's table
  // it reports - resolved here explicitly, since leaving it ambiguous is
  // exactly the kind of node-model gap this milestone's own plan section
  // warns about.
  std::vector<std::pair<ActionId, int>> root_visit_distribution;
};

// Runs n_simulations of MCTS/DUCT from `root`, using `leaf_eval` to score
// leaf states (see EvalFn's own doc comment above for the value-backup
// convention a given leaf_eval must satisfy), and returns the root's best
// action for "my" side plus its visit distribution. Deterministic given a
// fixed seed (an internal Rng(seed) drives every stochastic choice: which
// action UCB1/random-untried-action expansion picks, and every resolve_turn
// call's own damage/crit/accuracy/multi-hit/turn-order sampling) - same
// seeded-Rng-for-reproducibility convention as forward_model.hpp's own
// functions.
//
// `root` itself is never mutated - every simulation resolves a FRESH copy
// via resolve_turn() starting from `root`'s own values (open-loop, per this
// header's own top-of-file note), not `root` directly.
//
// EMPTY-ACTION-LIST NOTE, found during M7 integration against realistic
// battle states (not a synthetic worry - reproduces from TURN 1 of most
// real battles): a plain kDecision node's my_actions/opp_actions can be
// genuinely empty for a side with no known moves and no legal switch
// target (most commonly the OPPONENT before they've revealed any move,
// with only their lead known). select_ucb1_action() already documents
// returning kNoAction for this ("the caller's problem") - search()'s own
// tree-walk is that caller, and treats it as a terminal leaf for the
// simulation (evaluate the current state as-is, stop descending) rather
// than passing kNoAction into resolve_turn(), which requires a real action
// from both sides. The same handling applies at a kForcedSwitch node with
// no legal replacement (SearchNode's own doc comment).
//
// LATENCY NOTE for M7 (not this milestone's own concern, but real and
// worth stating where the function that will need it is declared):
// Player.choose_move runs on poke-env's asyncio loop, so M7's pybind11
// binding of this function MUST release the GIL
// (py::gil_scoped_release) around the call - a long blocking C++ call
// here would otherwise stall the websocket for every OTHER concurrently-
// running battle, not just this one. Also: M7 needs a real measured
// ms/turn number at whatever n_simulations gets chosen, so a 500-battle
// benchmark run finishes in a reasonable window - measure this once
// search() actually works, don't guess at a simulation count.
SearchResult search(const BattleState& root, const EvalFn& leaf_eval, int n_simulations, uint64_t seed);

}  // namespace be
