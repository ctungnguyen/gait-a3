#!/usr/bin/env python3
"""Compatibility entry point for the canonical PPO training pipeline.

The assessment permits either PPO or DQN. This project intentionally selects
PPO for both mandatory control styles so their comparison changes only the
action representation. Use train_rotation.py, train_direct.py, or train_both.py
for the full set of options.
"""

from __future__ import annotations

import argparse

from part2.arena import ArenaConfig, ControlStyle
from part2.training.artifacts import PROJECT_ROOT
from part2.training.config import load_training_settings
from part2.training.pipeline import train_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=("rotation", "direct", "both"), default="both")
    parser.add_argument("--preset", choices=("baseline", "exploratory", "stable"), default="baseline")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args()

    arena_config = ArenaConfig.from_json(PROJECT_ROOT / "config" / "arena.json")
    settings = load_training_settings(
        PROJECT_ROOT / "config" / "training.json",
        preset=args.preset,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )
    styles = (
        (ControlStyle.ROTATION, ControlStyle.DIRECT)
        if args.style == "both"
        else (ControlStyle(args.style),)
    )
    for style in styles:
        train_style(
            style=style,
            arena_config=arena_config,
            settings=settings,
            run_name=args.run_name,
            promote=True,
            device=args.device,
            progress_bar=not args.no_progress_bar,
        )


if __name__ == "__main__":
    main()
