#!/usr/bin/env python3
"""Run measured PPO preset trials instead of claiming unsupported tuning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from part2.arena import ArenaConfig, ControlStyle
from part2.training.artifacts import PROJECT_ROOT, write_json
from part2.training.config import load_training_settings
from part2.training.pipeline import train_style


PRESETS = ("baseline", "exploratory", "stable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three meaningful PPO configurations")
    parser.add_argument("--style", choices=("both", "rotation", "direct"), default="both")
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--trial-timesteps", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--arena-config", type=Path, default=PROJECT_ROOT / "config" / "arena.json")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "config" / "training.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "tuning")
    args = parser.parse_args()
    if args.trial_timesteps <= 0 or args.eval_episodes <= 0:
        parser.error("trial timesteps and evaluation episodes must be positive")

    styles = (
        (ControlStyle.ROTATION, ControlStyle.DIRECT)
        if args.style == "both"
        else (ControlStyle(args.style),)
    )
    arena_config = ArenaConfig.from_json(args.arena_config)
    rows = []
    full_summaries = {}

    for style in styles:
        style_summaries = {}
        for preset in args.presets:
            settings = load_training_settings(
                args.training_config,
                preset=preset,
                total_timesteps=args.trial_timesteps,
                n_envs=args.n_envs,
                seed=args.seed,
                final_eval_episodes=args.eval_episodes,
            )
            run_name = f"tune_{preset}"
            summary = train_style(
                style=style,
                arena_config=arena_config,
                settings=settings,
                run_name=run_name,
                promote=False,
                device=args.device,
                progress_bar=not args.no_progress_bar,
            )
            result = summary["final_evaluation"]
            style_summaries[preset] = summary
            rows.append(
                {
                    "control_style": style.value,
                    "preset": preset,
                    "timesteps": args.trial_timesteps,
                    "seed": settings.seed,
                    "mean_return": result["mean_return"],
                    "std_return": result["std_return"],
                    "success_rate": result["success_rate"],
                    "mean_max_phase": result["mean_max_phase"],
                    "mean_episode_steps": result["mean_episode_steps"],
                    "model": summary["model"],
                }
            )

        ranked = sorted(
            args.presets,
            key=lambda name: (
                style_summaries[name]["final_evaluation"]["mean_return"],
                style_summaries[name]["final_evaluation"]["success_rate"],
                style_summaries[name]["final_evaluation"]["mean_max_phase"],
            ),
            reverse=True,
        )
        full_summaries[style.value] = {
            "ranking": ranked,
            "recommended_preset": ranked[0],
            "trials": style_summaries,
        }

    shared_ranking = None
    if len(styles) == 2:
        shared_scores = {
            preset: sum(
                full_summaries[style.value]["trials"][preset]["final_evaluation"][
                    "mean_return"
                ]
                for style in styles
            )
            / len(styles)
            for preset in args.presets
        }
        shared_ranking = sorted(args.presets, key=shared_scores.get, reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "hyperparameter_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "hyperparameter_trials.json"
    write_json(
        json_path,
        {
            "selection_rule": "Highest mean return, then success rate, then mean max phase",
            "shared_fair_comparison_ranking": shared_ranking,
            "shared_recommended_preset": None if shared_ranking is None else shared_ranking[0],
            "results": full_summaries,
        },
    )
    print(json.dumps({style: data["ranking"] for style, data in full_summaries.items()}, indent=2))
    if shared_ranking is not None:
        print(f"Shared preset ranking for a controlled comparison: {shared_ranking}")
    print(f"Saved tuning table: {csv_path}")
    print(f"Saved complete tuning evidence: {json_path}")
    print("Retrain each canonical final model with its recommended preset and full timesteps.")


if __name__ == "__main__":
    main()
