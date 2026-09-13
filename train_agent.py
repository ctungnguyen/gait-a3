#!/usr/bin/env python3
"""Train one style's PPO model with checkpoints, logs, and final evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from part2.arena import (
    ArenaConfig,
    ControlStyle,
    action_names,
    make_direct_env,
    make_progression_reward,
    make_rotation_env,
)
from part2.training.artifacts import PROJECT_ROOT
from part2.training.config import load_training_settings
from part2.training.pipeline import train_style


def build_parser(default_style: ControlStyle | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GAIT Part II PPO agent")
    if default_style is None:
        parser.add_argument("--style", choices=("rotation", "direct"), required=True)
    parser.add_argument("--preset", choices=("baseline", "exploratory", "stable"), default="baseline")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--final-eval-episodes", type=int)
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate 500 steps without SB3 training")
    parser.add_argument("--arena-config", type=Path, default=PROJECT_ROOT / "config" / "arena.json")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "config" / "training.json")
    return parser


def _dry_run(style: ControlStyle, arena_config: ArenaConfig, settings) -> None:
    factory = make_rotation_env if style is ControlStyle.ROTATION else make_direct_env
    env = factory(
        config=arena_config,
        reward_fn=make_progression_reward(settings.reward.to_dict()),
        seed=settings.seed,
    )
    observation, info = env.reset(seed=settings.seed)
    total_reward = 0.0
    resets = 0
    try:
        for _ in range(500):
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if not env.observation_space.contains(observation):
                raise AssertionError("Observation escaped the declared space")
            env.core.assert_invariants()
            total_reward += reward
            if terminated or truncated:
                resets += 1
                observation, info = env.reset()
    finally:
        env.close()
    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "control_style": style.value,
                "action_meanings": list(action_names(style)),
                "observation_shape": list(observation.shape),
                "steps": 500,
                "episode_resets": resets,
                "sample_total_reward": total_reward,
                "training_settings": settings.to_dict(),
            },
            indent=2,
        )
    )


def main(
    default_style: ControlStyle | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    parser = build_parser(default_style)
    args = parser.parse_args(argv)
    style = default_style or ControlStyle(args.style)
    arena_config = ArenaConfig.from_json(args.arena_config)
    settings = load_training_settings(
        args.training_config,
        preset=args.preset,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        final_eval_episodes=args.final_eval_episodes,
    )
    if args.dry_run:
        _dry_run(style, arena_config, settings)
        return

    summary = train_style(
        style=style,
        arena_config=arena_config,
        settings=settings,
        run_name=args.run_name,
        promote=True,
        resume_model=args.resume,
        device=args.device,
        progress_bar=not args.no_progress_bar,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
