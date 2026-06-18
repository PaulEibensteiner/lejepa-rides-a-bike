from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from bike.environment import BikeEnv
from train import (
    ManualControllerPolicy,
    StickyGaussianTorquePolicy,
    collect_episode,
    random_policy,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for policy rollout and video export."""
    parser = argparse.ArgumentParser(
        description="Sample one trajectory from a policy and save it as MP4."
    )
    parser.add_argument(
        "--policy",
        choices=["random", "sticky", "manual"],
        default="sticky",
        help="Policy to rollout.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/videos/policy_sample.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--wind-std", type=float, default=0.01)
    parser.add_argument(
        "--control-interval-steps",
        type=int,
        default=1,
        help="Repeat each policy action for this many env steps.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frame-rate used during MP4 encoding.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help=(
            "Render 1 frame every N simulation steps. "
            "Defaults to auto (1 / (fps * dt)), e.g. 50 for fps=20 and dt=0.001."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Simulation time-step; used for auto frame-skip calculation.",
    )

    # Sticky policy params
    parser.add_argument("--p-new", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=20.0)

    # Manual policy params
    parser.add_argument("--desired-heading", type=float, default=0.0)
    parser.add_argument("--c1", type=float, default=-1.0)
    parser.add_argument("--c2", type=float, default=100.0)
    parser.add_argument("--c3", type=float, default=100.0)

    return parser.parse_args()


def _make_policy(args: argparse.Namespace):
    """Construct the selected policy object or callable.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A callable policy compatible with collect_episode.
    """
    if args.policy == "random":
        return random_policy
    if args.policy == "sticky":
        return StickyGaussianTorquePolicy(p_new=args.p_new, sigma=args.sigma)
    return ManualControllerPolicy(
        desired_heading=args.desired_heading,
        c1=args.c1,
        c2=args.c2,
        c3=args.c3,
    )


def main() -> None:
    """Roll out one episode from a chosen policy and export an MP4.

    Args:
        None.
    """
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    np.random.seed(args.seed)

    env = BikeEnv(
        max_steps=args.max_steps, wind_std=args.wind_std, video_path=args.output
    )
    try:
        policy = _make_policy(args)
        states, actions, _next_states, fell = collect_episode(
            env,
            policy,
            episode_seed=args.seed,
            control_interval_steps=max(1, int(args.control_interval_steps)),
        )
    finally:
        env.close()

    logger.info(
        "Saved trajectory video to %s | steps=%d | fell=%s",
        args.output,
        len(actions),
        fell,
    )


if __name__ == "__main__":
    main()
