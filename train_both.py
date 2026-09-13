#!/usr/bin/env python3
"""Train both mandatory PPO agents sequentially with the same settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from part2.arena import ArenaConfig, ControlStyle
from train_agent import _dry_run
from part2.training.artifacts import PROJECT_ROOT
from part2.training.config import load_training_settings
from part2.training.pipeline import train_style
from part2.training.plots import plot_evaluation_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Train both required Part II control styles")
    presets = ("baseline", "exploratory", "stable", "rotation_focus", "direct_focus")
    parser.add_argument(
        "--preset",
        choices=presets,
        help="Optional shared preset override for both styles",
    )
    parser.add_argument("--rotation-preset", choices=presets, default="rotation_focus")
    parser.add_argument("--direct-preset", choices=presets, default="direct_focus")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--final-eval-episodes", type=int)
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--resume-rotation", type=Path)
    parser.add_argument("--resume-direct", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arena-config", type=Path, default=PROJECT_ROOT / "config" / "arena.json")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "config" / "training.json")
    args = parser.parse_args()

    arena_config = ArenaConfig.from_json(args.arena_config)
    settings_by_style = {
        style: load_training_settings(
            args.training_config,
            preset=(
                args.preset
                or (args.rotation_preset if style is ControlStyle.ROTATION else args.direct_preset)
            ),
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            final_eval_episodes=args.final_eval_episodes,
        )
        for style in (ControlStyle.ROTATION, ControlStyle.DIRECT)
    }
    if args.dry_run:
        _dry_run(ControlStyle.ROTATION, arena_config, settings_by_style[ControlStyle.ROTATION])
        _dry_run(ControlStyle.DIRECT, arena_config, settings_by_style[ControlStyle.DIRECT])
        return

    summaries = {}
    for style, resume in (
        (ControlStyle.ROTATION, args.resume_rotation),
        (ControlStyle.DIRECT, args.resume_direct),
    ):
        summaries[style.value] = train_style(
            style=style,
            arena_config=arena_config,
            settings=settings_by_style[style],
            run_name=args.run_name,
            promote=True,
            resume_model=resume,
            device=args.device,
            progress_bar=not args.no_progress_bar,
        )

    curve_path = plot_evaluation_curves(run_name=args.run_name)
    print(json.dumps(summaries, indent=2))
    print(f"Saved combined training curves: {curve_path}")


if __name__ == "__main__":
    main()
