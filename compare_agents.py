#!/usr/bin/env python3
"""Evaluate both trained control styles on identical seeds and export evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from arena import ArenaConfig, ControlStyle
from training.artifacts import PROJECT_ROOT, write_json
from training.config import load_training_settings
from training.evaluation import evaluate_model, load_ppo_model, validate_model_contract
from training.plots import plot_evaluation_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair headless comparison of both PPO agents")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--arena-config", type=Path, default=PROJECT_ROOT / "config" / "arena.json")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "config" / "training.json")
    parser.add_argument("--rotation-model", type=Path, default=PROJECT_ROOT / "models" / "rotation_agent.zip")
    parser.add_argument("--direct-model", type=Path, default=PROJECT_ROOT / "models" / "direct_agent.zip")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    arena_config = ArenaConfig.from_json(args.arena_config)
    settings = load_training_settings(args.training_config)
    model_paths = {
        ControlStyle.ROTATION: args.rotation_model,
        ControlStyle.DIRECT: args.direct_model,
    }
    summaries = {}
    all_rows = []
    for style in (ControlStyle.ROTATION, ControlStyle.DIRECT):
        model_path = model_paths[style]
        if not model_path.exists():
            raise SystemExit(f"Missing {style.value} model: {model_path}")
        model = load_ppo_model(model_path, device=args.device)
        validate_model_contract(model, style, model_path)
        rows, summary = evaluate_model(
            model,
            style=style,
            arena_config=arena_config,
            reward_config=settings.reward,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
        )
        summaries[style.value] = summary
        all_rows.extend({"control_style": style.value, **row} for row in rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "control_style_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = args.output_dir / "control_style_comparison.json"
    write_json(
        summary_path,
        {
            "algorithm": "PPO",
            "episodes_per_style": args.episodes,
            "shared_episode_seed_start": args.seed,
            "fair_comparison": "Same Arena, reward, deterministic policy, and episode seeds",
            "results": summaries,
        },
    )

    try:
        curve_path = plot_evaluation_curves(run_name=args.run_name)
    except FileNotFoundError:
        curve_path = None

    print(json.dumps(summaries, indent=2))
    print(f"Saved episode comparison: {csv_path}")
    print(f"Saved summary: {summary_path}")
    if curve_path is not None:
        print(f"Saved curves: {curve_path}")


if __name__ == "__main__":
    main()
