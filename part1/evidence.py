"""Generate all Part I logs, policies, comparisons, and rubric figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, PART1_ROOT
from .training import (
    TrainingResult,
    evaluate,
    evaluation_summary,
    greedy_policy_trace,
    moving_average,
    save_evaluation,
    save_training_result,
    shortest_collectible_steps,
    shortest_external_reward_steps,
    train,
)

RUNS = (
    (0, "q_learning", False),
    (1, "q_learning", False),
    (1, "sarsa", False),
    (2, "q_learning", False),
    (2, "sarsa", False),
    (3, "q_learning", False),
    (3, "sarsa", False),
    (4, "q_learning", False),
    (4, "sarsa", False),
    (5, "q_learning", False),
    (5, "sarsa", False),
    (6, "q_learning", False),
    (6, "q_learning", True),
)


def _plot_group(
    runs: list[TrainingResult],
    filename: Path,
    title: str,
    window: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency setup
        raise RuntimeError("Install requirements-rl.txt to generate figures") from exc

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for result in runs:
        label = f"L{result.level.number} {result.algorithm.replace('_', ' ').title()}"
        if result.intrinsic_enabled:
            label += " + intrinsic"
        x_values = [record.episode for record in result.episodes]
        returns = moving_average(
            (record.environment_return for record in result.episodes), window
        )
        successes = moving_average((record.success for record in result.episodes), window)
        axes[0].plot(x_values, returns, label=label, linewidth=1.8)
        axes[1].plot(x_values, successes, label=label, linewidth=1.8)
    axes[0].set_title(title)
    axes[0].set_ylabel("Mean environment return")
    axes[1].set_ylabel("Success rate")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=170)
    plt.close(figure)


def _plot_intrinsic(runs: list[TrainingResult], filename: Path, window: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements-rl.txt to generate figures") from exc

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for result in sorted(runs, key=lambda item: item.intrinsic_enabled):
        label = "With intrinsic reward" if result.intrinsic_enabled else "Without intrinsic reward"
        x_values = [record.episode for record in result.episodes]
        successes = moving_average((record.success for record in result.episodes), window)
        env_returns = moving_average(
            (record.environment_return for record in result.episodes), window
        )
        axes[0].plot(x_values, successes, label=label, linewidth=2)
        axes[1].plot(x_values, env_returns, label=label, linewidth=2)
    axes[0].set_title("Task 5 - Level 6 Count-Based Exploration")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Environment return only")
    axes[1].set_xlabel("Training episode")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=170)
    plt.close(figure)


def _plot_level1_routes(runs: list[TrainingResult], filename: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements-rl.txt to generate figures") from exc

    level = runs[0].level
    traces = [greedy_policy_trace(run, seed=30_001) for run in runs]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for axis, result, trace in zip(axes, runs, traces):
        for y, row in enumerate(level.layout):
            for x, symbol in enumerate(row):
                color = "#e95f6f" if symbol == "F" else "#1c2a39"
                axis.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#40566d"))
                if symbol == "S":
                    axis.text(x + 0.5, y + 0.5, "S", ha="center", va="center", color="white", weight="bold")
                elif symbol == "A":
                    axis.text(x + 0.5, y + 0.5, "A", ha="center", va="center", color="#a8f0bd", weight="bold")
        positions = trace["positions"]
        axis.plot(
            [position[0] + 0.5 for position in positions],
            [position[1] + 0.5 for position in positions],
            color="#53d7ef" if result.algorithm == "q_learning" else "#ffd06b",
            linewidth=3,
            marker="o",
            markersize=3,
        )
        axis.set_title(
            f"{result.algorithm.replace('_', ' ').title()}\n"
            f"{trace['steps']} steps, fire clearance {trace['minimum_manhattan_fire_clearance']}"
        )
        axis.set_xlim(0, len(level.layout[0]))
        axis.set_ylim(len(level.layout), 0)
        axis.set_aspect("equal")
        axis.set_xticks(range(len(level.layout[0]) + 1))
        axis.set_yticks(range(len(level.layout) + 1))
        axis.grid(alpha=0.15)
    figure.suptitle("Task 2 - Learned Epsilon-Zero Policy Routes")
    figure.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=170)
    plt.close(figure)


def generate_all(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
) -> dict:
    reports_dir = PART1_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[tuple[int, str, bool], TrainingResult] = {}
    summaries: list[dict] = []

    for index, (level, algorithm, intrinsic) in enumerate(RUNS, start=1):
        run_label = f"L{level} {algorithm}{' intrinsic' if intrinsic else ''}"

        def progress(current: int, total: int, _record, label=run_label) -> None:
            interval = max(1, total // 5)
            if current == 1 or current == total or current % interval == 0:
                print(f"[{index:02d}/{len(RUNS)}] {label}: {current}/{total}")

        result = train(
            level,
            algorithm,
            config_path=config_path,
            episodes=episodes,
            intrinsic=intrinsic,
            progress=progress,
        )
        save_training_result(result)
        eval_records = evaluate(result, episodes=evaluation_episodes)
        save_evaluation(result, eval_records)
        summary = {
            "run_name": result.run_name,
            "level": level,
            "algorithm": algorithm,
            "intrinsic_enabled": intrinsic,
            **evaluation_summary(eval_records),
            "greedy_policy_trace": greedy_policy_trace(result),
        }
        if level == 0:
            optimum = shortest_collectible_steps(result.level)
            summary["shortest_possible_steps"] = optimum
            summary["mean_excess_steps_on_success"] = (
                summary["mean_success_steps"] - optimum if optimum is not None else None
            )
        if level == 6:
            summary["minimum_steps_to_first_external_reward"] = (
                shortest_external_reward_steps(result.level)
            )
            summary["interpretation"] = (
                "The no-intrinsic run is a deliberately equal-budget sparse-reward "
                "baseline. A zero success rate is a valid experimental result, not "
                "an unreachable map; the intrinsic run changes learning reward only."
            )
        summaries.append(summary)
        completed[(level, algorithm, intrinsic)] = result

    window = max(25, (episodes or 5_000) // 40)
    _plot_group(
        [completed[(0, "q_learning", False)]],
        reports_dir / "task1_level0_q_learning.png",
        "Task 1 - Level 0 Q-Learning",
        window,
    )
    _plot_level1_routes(
        [completed[(1, "q_learning", False)], completed[(1, "sarsa", False)]],
        reports_dir / "task2_level1_policy_routes.png",
    )
    _plot_group(
        [completed[(1, "q_learning", False)], completed[(1, "sarsa", False)]],
        reports_dir / "task2_level1_q_vs_sarsa.png",
        "Task 2 - Level 1 Q-Learning vs On-Policy SARSA",
        window,
    )
    _plot_group(
        [
            completed[(2, "q_learning", False)],
            completed[(2, "sarsa", False)],
            completed[(3, "q_learning", False)],
            completed[(3, "sarsa", False)],
        ],
        reports_dir / "task3_levels2_3_extension.png",
        "Task 3 - Both Algorithms on Multi-Objective Levels",
        window,
    )
    _plot_group(
        [
            completed[(4, "q_learning", False)],
            completed[(4, "sarsa", False)],
            completed[(5, "q_learning", False)],
            completed[(5, "sarsa", False)],
        ],
        reports_dir / "task4_monster_levels4_5.png",
        "Task 4 - Learning Under 40% Stochastic Monster Movement",
        window,
    )
    _plot_intrinsic(
        [completed[(6, "q_learning", False)], completed[(6, "q_learning", True)]],
        reports_dir / "task5_level6_intrinsic_comparison.png",
        window,
    )

    fieldnames = sorted({key for summary in summaries for key in summary})
    with (reports_dir / "part1_evaluation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "environment_reward_contract": {"apple": 1, "key": 0, "chest": 2, "death": 0},
        "monster_move_probability": 0.4,
        "intrinsic_formula": "intrinsicRewardStrength / sqrt(n(s) + 1)",
        "note": (
            "Intrinsic reward affects learning updates only; charts use unchanged "
            "environment return. Level 6 is intentionally solvable but sparse, so "
            "zero baseline success within the fixed budget is valid evidence."
        ),
        "runs": summaries,
    }
    (reports_dir / "part1_evaluation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all Part I rubric evidence")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--episodes",
        type=int,
        help="Use one episode count for every run (quick pipeline testing only)",
    )
    parser.add_argument("--evaluation-episodes", type=int)
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    generate_all(
        config_path=args.config,
        episodes=args.episodes,
        evaluation_episodes=args.evaluation_episodes,
    )


if __name__ == "__main__":
    main()
