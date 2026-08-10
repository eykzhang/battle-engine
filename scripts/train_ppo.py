"""Phase 3: PPO training via stable-baselines3 (+ sb3-contrib for action
masking), wired to MetamonActionSinglesEnv - the "working plumbing before
hand-rolling anything" step from the roadmap.

--opponent random trains against a single fixed baseline (RandomPlayer) -
the original, simplest version, still useful for quick smoke tests.
--opponent self-play (the default, and the roadmap's actual Phase 3 target
- "self-play + play vs frozen past versions") trains against
battle_engine/self_play.py's FrozenPolicyPlayer instead: a pool of frozen
snapshots of the trainee's own past weights, refreshed periodically during
training via SelfPlaySnapshotCallback, one snapshot sampled per battle (not
per turn). Seeded with the trainee's own initial (possibly warm-started)
weights so there's a real opponent from battle 1, not an empty pool.

--search-bot-fraction (default 0.2, self-play only) mixes a real
TwoPlySearchPlayer instance into that opponent pool via
self_play.MixedOpponentPlayer, at the given per-battle probability. Added
after a real training run (1,000,000 timesteps, see CLAUDE.md's Phase 3
sanity-run status) plateaued at a ~28-32% win rate against
TwoPlySearchPlayer: pure self-play only ever tests the trainee against
itself, so nothing in training required "beats recent snapshots" to
transfer to "beats 2-ply lookahead" - only --eval's periodic real games
against TwoPlySearchPlayer ever measured that, and measuring isn't
training. 0.0 disables mixing and reproduces the old pure-self-play
behavior exactly. DEFAULT_MAX_SNAPSHOTS also widened 5 -> 20 in the same
pass (a second, independent plateau contributor: 5 snapshots at the default
4096-step interval is a rolling ~2% window of a 1,000,000-timestep run) -
see the constant's own comment for the exact reasoning.

Warm-starts from the Phase-2 models by default (--no-warm-start to train
from scratch instead): battle_engine/ppo_warm_start.py copies
imitation.py's trained trunk+output into the policy (actor) network and
win_prob.py's into the value (critic) network - both roadmap-anticipated
warm-start candidates, now actually usable together because masking (see
below) means the network no longer needs to see action_mask as an input
feature, which is what let ppo_warm_start.py drop it and make PPO's trunk
shape-identical to both Phase-2 models' trunks (verified end-to-end: the
warm-started policy reproduces bit-identical logits/values to the source
models on the same input - see tests/test_ppo_warm_start.py). An earlier
version of this docstring said this was blocked by CombinedExtractor's
[action_mask(ACTION_SPACE_SIZE), observation(VECTOR_LEN)] concatenation
(verified real, Gymnasium sorts Dict keys alphabetically) - that's still
true of SB3's *default* architecture, just no longer the one this script
actually uses.

Uses sb3-contrib's MaskablePPO + ActionMasker, not plain stable-baselines3
PPO: an independent review (2026-07-31) measured that without real masking,
letting illegal picks fall back to a randomly-substituted legal move
(env.strict=False) is a real training-quality problem, not just a
crash-avoidance shortcut - under a fresh (near-uniform) policy, 56.2% of
sampled actions are illegal on real gen9ou states, and each one gets a
*different, randomly chosen* move actually played, so plain PPO credits the
action it sampled with the outcome of a different action poke-env silently
substituted. MaskableActorCriticPolicy instead sets illegal actions' logits
to a large negative value before sampling (see
sb3_contrib.common.maskable.distributions.MaskableCategorical.apply_masking),
so the policy can never actually pick one - env.strict=False stays on only
as defense in depth (e.g. a hypothetical mask/legality mismatch), not as
the primary mechanism anymore.

Two more pieces (--eval/--checkpoint, both on by default), added once a real,
multi-hour training run became the actual next step rather than a feasibility
check: PPO's own logged metrics (loss, entropy, explained_variance) don't
directly answer "is this actually getting better at winning" - --eval
periodically freezes the live policy and plays it for --eval-battles real
games against TwoPlySearchPlayer (the Phase-1 bot - the actual thing Phase 3's
gate needs to beat) via the same real benchmark harness every phase gate in
this project uses (battle_engine.benchmark.run_benchmark), not a proxy metric.
--checkpoint periodically saves the model (data/models/checkpoints/ by
default) so an overnight run survives a crash, sleep, or early stop without
losing all progress - distinct from --save, which only saves once at the end.

--resume-from PATH loads one of --checkpoint's saved .zip files and continues
training from there (SB3's reset_num_timesteps=False - --timesteps is then
"how many MORE steps to run", same convention SB3 itself uses, not a new
absolute target) instead of building a fresh model - first used to continue a
sanity run that was deliberately stopped partway through rather than let run
to completion, so its already-checkpointed progress didn't have to be
discarded. --warm-start is forced off when resuming regardless of the flag's
own value (with a loud message, not silently): the resumed checkpoint's
weights ARE the actual trained state to continue from, and re-applying
load_warm_start_weights on top of them would overwrite that progress with the
original Phase-2 checkpoints again. self-play's snapshot pool and the eval
opponent are NOT part of what's saved/resumed (CheckpointCallback only saves
the MaskablePPO model) - re-seeded fresh from the resumed model's current
(already-trained) weights, same code path as a cold start's initial seeding.

Usage:
    .venv/bin/python scripts/train_ppo.py --timesteps 2000
    .venv/bin/python scripts/train_ppo.py --resume-from data/models/checkpoints/ppo_500000_steps.zip --timesteps 500000

Needs the local Showdown server running (see README) and the `ml` extra
installed (`.venv/bin/pip install -e ".[ml,dev]"` - now includes both
stable-baselines3 and sb3-contrib).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player import Player, RandomPlayer
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from battle_engine.imitation import ImitationModel
from battle_engine.ppo_eval import EvalVsOpponentCallback
from battle_engine.ppo_warm_start import load_warm_start_weights, warm_start_policy_kwargs
from battle_engine.rl_env import MetamonActionSinglesEnv, sb3_contrib_action_mask_fn
from battle_engine.search import TwoPlySearchPlayer
from battle_engine.self_play import FrozenPolicyPlayer, MixedOpponentPlayer, SelfPlaySnapshotCallback
from battle_engine.teams import RandomTeamFromPool
from battle_engine.win_prob import WinProbModel

DEFAULT_MODEL_PATH = Path("data/models/ppo.zip")
DEFAULT_IMITATION_MODEL_PATH = Path("data/models/imitation.pt")
DEFAULT_WIN_PROB_MODEL_PATH = Path("data/models/win_prob.pt")
DEFAULT_SNAPSHOT_INTERVAL = 4096
# Was 5 (a rolling ~2% window of a 1,000,000-timestep run - snapshot_interval
# * max_snapshots = 20,480 steps of history) - flagged by the sanity-run
# plateau diagnosis (see CLAUDE.md's Phase 3 status) as one likely
# contributor: a pool this narrow mostly tests the trainee against very
# recent versions of itself, giving little protection against forgetting
# older strategies or cycling. 20 widens the covered window to ~8.2% of that
# same run (81,920 steps) at negligible cost - each snapshot is just one
# ~186k-param MLP state_dict (~750KB), cheap to keep even a few dozen of.
DEFAULT_MAX_SNAPSHOTS = 20
# Fraction of self-play training battles played against a real
# TwoPlySearchPlayer instance instead of a frozen self-play snapshot - the
# sanity-run diagnosis's other likely contributor: the trainee never once
# played the actual Phase-3 gate opponent during training, only itself, so
# "beats recent self" wasn't required to transfer to "beats 2-ply lookahead".
# 0.2 keeps self-play as the dominant signal (cheaper, faster-iterating
# opponent pool) while guaranteeing real, repeated exposure to the actual
# target throughout training, not just at periodic --eval checkpoints (which
# measure against it but never train against it). Only used when --opponent
# self-play; 0.0 disables mixing and reproduces the old pure-self-play
# behavior exactly.
DEFAULT_SEARCH_BOT_FRACTION = 0.2
DEFAULT_EVAL_INTERVAL = 4096
DEFAULT_EVAL_BATTLES = 20
DEFAULT_CHECKPOINT_INTERVAL = 8192
DEFAULT_CHECKPOINT_DIR = Path("data/models/checkpoints")


def build_env(opponent: Player) -> ActionMasker:
    """A fresh env + the given opponent, wrapped for SB3's single-agent Gym
    API and then for action masking. strict=False remains as defense in
    depth, not the primary illegal-action mechanism now that masking
    prevents the policy from sampling an illegal action in the first place
    (see module docstring).

    The opponent should be built with start_listening=False and no team= -
    SingleAgentWrapper never actually connects `opponent` to the server or
    has it play a real match (verified in single_agent_wrapper.py): it only
    calls opponent.choose_move(battle2)/opponent.teampreview(battle2), pure
    decision functions over the battle object poke-env's own env.agent2
    (the actually-connected side) is playing. A version that passed team=
    and left start_listening at its True default was reviewed and found to
    leak a live, logged-in, unclosed websocket connection per call for no
    purpose.
    """
    env = MetamonActionSinglesEnv(
        battle_format="gen9ou", team=RandomTeamFromPool(), strict=False
    )
    wrapped = SingleAgentWrapper(env, opponent)
    return ActionMasker(wrapped, sb3_contrib_action_mask_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO rollout length before each policy update. Was 256 (an 8x-smaller-"
        "than-SB3's-own-default dev-speed shortcut, chosen only so an early laptop "
        "feasibility check didn't have to wait for a full rollout to get a first "
        "timing measurement) - left at that value for every real training run since, "
        "including the sanity runs that plateaued (see CLAUDE.md's Phase 3 status). "
        "Restored to SB3's own default (2048) 2026-08-09: a smaller rollout means "
        "noisier per-update gradient estimates (fewer battle-turns of real data "
        "before each policy update), a plausible, never-actually-tested contributor "
        "to that plateau. batch_size stays at SB3's default (64), so this is still a "
        "clean 32 minibatches/epoch with n_envs=1.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"save the trained model to {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"initialize from {DEFAULT_IMITATION_MODEL_PATH} (actor) and "
        f"{DEFAULT_WIN_PROB_MODEL_PATH} (critic) instead of random weights "
        "(default: on; --no-warm-start to train from scratch)",
    )
    parser.add_argument(
        "--opponent",
        choices=["self-play", "random"],
        default="self-play",
        help="self-play (default): frozen snapshots of the trainee's own past weights. "
        "random: a fixed RandomPlayer, for quick smoke tests.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=DEFAULT_SNAPSHOT_INTERVAL,
        help="timesteps between pushing a new frozen snapshot (--opponent self-play only)",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=DEFAULT_MAX_SNAPSHOTS,
        help="frozen-snapshot pool size (--opponent self-play only)",
    )
    parser.add_argument(
        "--search-bot-fraction",
        type=float,
        default=DEFAULT_SEARCH_BOT_FRACTION,
        help="fraction of training battles played against a real TwoPlySearchPlayer "
        "instead of a frozen self-play snapshot (--opponent self-play only) - trains "
        "directly against the actual Phase-3 gate opponent, not just self-play "
        "snapshots. 0.0 reproduces the old pure-self-play behavior.",
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="periodically benchmark the live policy against the Phase-1 search bot "
        "(TwoPlySearchPlayer) via real games, for an interpretable progress signal "
        "beyond PPO's own loss/entropy metrics (default: on; --no-eval to skip)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=DEFAULT_EVAL_INTERVAL,
        help="timesteps between eval-vs-search-bot checks",
    )
    parser.add_argument(
        "--eval-battles",
        type=int,
        default=DEFAULT_EVAL_BATTLES,
        help="battles played per eval check (small - this runs often, not a full "
        "phase-gate benchmark)",
    )
    parser.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="periodically save the model so a long run survives a crash/sleep/"
        "interruption without losing all progress (default: on; --no-checkpoint to skip)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="timesteps between checkpoint saves",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="directory for periodic checkpoints (distinct from --save's single "
        "final-model path)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="continue training from a saved checkpoint (--checkpoint's output) instead "
        "of building a fresh model - --timesteps means MORE steps on top of the "
        "checkpoint's own count, not a new absolute target. Forces --warm-start off "
        "(the checkpoint's weights are what's actually being resumed).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.search_bot_fraction <= 1.0:
        # Real footgun caught by independent review (2026-08-09): weights
        # this far out of range don't raise (random.choices only rejects a
        # non-positive *total*, and self-play's weight is always >= 0 here),
        # they silently resolve to "always pick the search bot" - a typo'd
        # fraction would read as a successful run of a different experiment,
        # not an error.
        parser.error("--search-bot-fraction must be between 0.0 and 1.0")

    if args.resume_from is not None and args.warm_start:
        print("--resume-from given: ignoring --warm-start (resumed weights take precedence)")
        args.warm_start = False

    self_play_opponent: Optional[FrozenPolicyPlayer] = None
    if args.opponent == "self-play":
        self_play_opponent = FrozenPolicyPlayer(
            battle_format="gen9ou", start_listening=False, max_snapshots=args.max_snapshots
        )
        if args.search_bot_fraction > 0.0:
            search_bot_opponent = TwoPlySearchPlayer(
                battle_format="gen9ou", start_listening=False
            )
            opponent: Player = MixedOpponentPlayer(
                battle_format="gen9ou",
                start_listening=False,
                players=[
                    (1.0 - args.search_bot_fraction, self_play_opponent),
                    (args.search_bot_fraction, search_bot_opponent),
                ],
            )
        else:
            opponent = self_play_opponent
    else:
        opponent = RandomPlayer(battle_format="gen9ou", start_listening=False)

    env = build_env(opponent)
    if args.resume_from is not None:
        # env= reconnects the loaded model to this run's fresh
        # SingleAgentWrapper/ActionMasker chain (a new opponent instance,
        # new underlying connection) rather than trying to restore whatever
        # env object existed at save time, which MaskablePPO.load doesn't
        # attempt to serialize in the first place.
        try:
            model = MaskablePPO.load(args.resume_from, env=env)
        except Exception:
            # Real leak caught by independent review (2026-08-09): build_env()
            # above already opened real websocket connections (trainee +
            # opponent) - a shape mismatch (e.g. resuming a checkpoint from
            # before an encoding change) or any other load failure left both
            # open for the rest of the process's life with no path to
            # env.close() afterward.
            env.close()
            raise
        print(
            f"resumed from {args.resume_from} at {model.num_timesteps} timesteps "
            f"(n_steps={model.n_steps}, restored from the checkpoint - --n-steps is "
            f"ignored when resuming)"
        )
    else:
        # Always the warm-start architecture (net_arch=[128,64], ReLU,
        # ObservationOnlyExtractor), even with --no-warm-start - an
        # independent review found that gating policy_kwargs itself on
        # --warm-start meant --no-warm-start silently trained a DIFFERENT,
        # SB3-default network (net_arch [64,64] per head, Tanh, sees
        # action_mask as a feature, ~95k params vs ~186k) - confounding
        # architecture with initialization, exactly the two things a "does
        # warm-start actually help" comparison needs to isolate.
        # --no-warm-start now only skips load_warm_start_weights below,
        # changing nothing else.
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            n_steps=args.n_steps,
            policy_kwargs=warm_start_policy_kwargs(),
            verbose=1,
        )
    if args.warm_start:
        if not DEFAULT_IMITATION_MODEL_PATH.exists() or not DEFAULT_WIN_PROB_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"--warm-start needs both {DEFAULT_IMITATION_MODEL_PATH} and "
                f"{DEFAULT_WIN_PROB_MODEL_PATH} (see CLAUDE.md's Commands for how to "
                "build them) - or pass --no-warm-start to train from scratch"
            )
        load_warm_start_weights(
            model.policy,
            ImitationModel.load(DEFAULT_IMITATION_MODEL_PATH),
            WinProbModel.load(DEFAULT_WIN_PROB_MODEL_PATH),
        )
        print(
            f"warm-started actor from {DEFAULT_IMITATION_MODEL_PATH}, "
            f"critic from {DEFAULT_WIN_PROB_MODEL_PATH}"
        )

    callbacks = []
    if self_play_opponent is not None:
        # Seed the pool with the trainee's own starting weights - possibly
        # warm-started, so battle 1's opponent is a reasonable player, not
        # an empty pool FrozenPolicyPlayer would otherwise raise on. Always
        # the inner FrozenPolicyPlayer, even when it's wrapped in a
        # MixedOpponentPlayer for search-bot mixing - that's the only one of
        # the two sub-players with a snapshot pool to seed/refresh.
        self_play_opponent.add_snapshot(model.policy.state_dict())
        callbacks.append(
            SelfPlaySnapshotCallback(self_play_opponent, snapshot_interval=args.snapshot_interval)
        )

    eval_opponent = None
    if args.eval:
        # TwoPlySearchPlayer (Phase 1's bot, defaults: hand-crafted eval,
        # no learned model involved) - the actual thing Phase 3's gate
        # needs to beat, so tracking progress against it during training is
        # more meaningful than tracking it against RandomPlayer or a
        # same-strength self-play snapshot. One long-lived instance reused
        # across every eval call (not rebuilt per call, unlike the
        # temporary evaluator EvalVsOpponentCallback builds internally).
        eval_opponent = TwoPlySearchPlayer(
            battle_format="gen9ou", team=RandomTeamFromPool()
        )
        callbacks.append(
            EvalVsOpponentCallback(
                eval_opponent,
                eval_interval=args.eval_interval,
                n_battles=args.eval_battles,
                battle_format="gen9ou",
                verbose=1,
            )
        )

    if args.checkpoint:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_interval,
                save_path=str(args.checkpoint_dir),
                name_prefix="ppo",
                verbose=1,
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    # reset_num_timesteps=False on resume: model.num_timesteps keeps
    # counting up from the checkpoint's own value (SB3's own convention -
    # see _setup_learn) rather than restarting at 0. This project's own two
    # callbacks (EvalVsOpponentCallback, SelfPlaySnapshotCallback) are keyed
    # on self.num_timesteps, so eval/snapshot intervals continue on the same
    # absolute schedule across a resume - verified directly on a real
    # resumed run (eval fired at exactly the next multiple of the interval
    # counting from 0, matching the resumed model's own absolute
    # num_timesteps). SB3's own CheckpointCallback is NOT keyed the same way
    # (an independent review, 2026-08-01, caught an earlier version of this
    # comment claiming it was) - it uses self.n_calls, which resets to 0
    # every fresh main() invocation regardless of resume, so its save
    # cadence is calls-since-THIS-run-started, not the absolute schedule
    # (the saved filename's own timestep number is still absolute and
    # correct, via self.num_timesteps - only the interval timing isn't).
    # Harmless for a resume that lands on a round multiple of both intervals
    # (as this project's own sanity-run resume did), but worth knowing if
    # resuming from an arbitrary offset.
    timesteps_before = model.num_timesteps
    start = time.perf_counter()
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        reset_num_timesteps=args.resume_from is None,
    )
    elapsed = time.perf_counter() - start

    new_timesteps = model.num_timesteps - timesteps_before
    print(
        f"\n{new_timesteps} new timesteps in {elapsed:.1f}s "
        f"({new_timesteps / elapsed:.1f} steps/s) - {model.num_timesteps} total"
    )

    if args.save:
        DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(DEFAULT_MODEL_PATH)
        print(f"saved to {DEFAULT_MODEL_PATH}")

    env.close()
    if eval_opponent is not None:
        # Unlike the self-play/random opponent (which SingleAgentWrapper
        # only ever calls choose_move on directly, never a real connected
        # match - see build_env's docstring), eval_opponent genuinely plays
        # real, connected battles: EvalVsOpponentCallback's benchmark calls
        # run_benchmark(evaluator, eval_opponent, ...), which is real
        # Player.battle_against - so it has a real websocket to close here.
        asyncio.run(eval_opponent.ps_client.stop_listening())


if __name__ == "__main__":
    main()
