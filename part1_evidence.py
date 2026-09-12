#!/usr/bin/env python3
"""Generate reproducible Part I learning curves and policy evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

import part1


ROOT = Path(__file__).resolve().parent


def evaluate(agent, level_id: int, max_steps: int) -> dict:
    env = part1.GridWorld(part1.MAPS[level_id], part1.CONFIG["monsterMoveChance"])
    state = env.reset()
    total = 0.0
    for step in range(1, max_steps + 1):
        result = env.step(agent.choose_action(state, 0.0, evaluate=True))
        total += result.reward
        state = result.next_state
        if result.done:
            return {
                "level": level_id,
                "return": total,
                "steps": step,
                "event": result.info.get("event", "unknown"),
            }
    return {"level": level_id, "return": total, "steps": max_steps, "event": "timeout"}


def train(level_id: int, algorithm: str, intrinsic: bool = False):
    settings = dict(part1.CONFIG)
    settings.update(part1.LEVEL_OVERRIDES.get(level_id, {}))
    agent, rows = part1.train_tabular(
        level_id,
        algorithm,
        settings,
        use_intrinsic_reward=intrinsic,
    )
    return agent, rows


def write_rows(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["episode", "return", "steps", "epsilon", "died"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Part I report evidence")
    parser.add_argument("--episodes", type=int, default=part1.CONFIG["episodes"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "part1")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    part1.CONFIG["episodes"] = args.episodes
    args.output.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for level_id, algorithms in ((4, ("q_learning", "sarsa")), (5, ("q_learning", "sarsa"))):
        for algorithm in algorithms:
            agent, rows = train(level_id, algorithm)
            key = f"level{level_id}_{algorithm}"
            write_rows(args.output / f"{key}.csv", rows)
            experiments[key] = evaluate(agent, level_id, part1.CONFIG["maxStepsPerEpisode"])

    for intrinsic in (False, True):
        agent, rows = train(6, "q_learning", intrinsic)
        key = "level6_q_learning_intrinsic" if intrinsic else "level6_q_learning_no_intrinsic"
        write_rows(args.output / f"{key}.csv", rows)
        experiments[key] = evaluate(agent, 6, part1.CONFIG["maxStepsPerEpisode"])

    level1_results = {}
    for algorithm in ("q_learning", "sarsa"):
        agent, rows = train(1, algorithm)
        key = f"level1_{algorithm}"
        write_rows(args.output / f"{key}.csv", rows)
        level1_results[algorithm] = evaluate(agent, 1, part1.CONFIG["maxStepsPerEpisode"])
    experiments["level1_comparison"] = level1_results

    with (args.output / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(experiments, output, indent=2)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for csv_path in sorted(args.output.glob("*.csv")):
        if "level1_" not in csv_path.name and "level4_" not in csv_path.name and "level5_" not in csv_path.name and "level6_" not in csv_path.name:
            continue
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        episodes = [int(row["episode"]) for row in rows]
        returns = [float(row["return"]) for row in rows]
        window = max(1, min(50, len(returns)))
        moving = [
            sum(returns[max(0, index - window + 1): index + 1]) / len(returns[max(0, index - window + 1): index + 1])
            for index in range(len(returns))
        ]
        label = csv_path.stem.replace("_", " ")
        axes[0].plot(episodes, moving, label=label)
        if "level6" in csv_path.name:
            axes[1].plot(episodes, returns, label=label)
    axes[0].set_title("Part I learning curves (50-episode moving average)")
    axes[0].set_ylabel("Return")
    axes[1].set_title("Level 6 intrinsic reward comparison")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Environment return")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output / "part1_learning_curves.png", dpi=180)
    plt.close(figure)
    print(f"Saved Part I evidence to {args.output}")


if __name__ == "__main__":
    main()
