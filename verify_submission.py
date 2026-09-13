#!/usr/bin/env python3
"""One-command verification of generated Part I and Part II code artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def report(label: str, passed: bool, detail: str) -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {label}: {detail}")
    return passed


def verify_part1() -> bool:
    reports = ROOT / "part1" / "reports"
    summary_path = reports / "part1_evaluation_summary.json"
    passed = report("Part I summary", summary_path.exists(), str(summary_path))
    if not summary_path.exists():
        return False
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rewards_ok = payload.get("environment_reward_contract") == {
        "apple": 1,
        "key": 0,
        "chest": 2,
        "death": 0,
    }
    passed &= report("Fixed Gridworld rewards", rewards_ok, str(payload.get("environment_reward_contract")))
    runs = {run["run_name"]: run for run in payload.get("runs", [])}

    expected_runs = {
        "level0_q_learning",
        "level1_q_learning", "level1_sarsa",
        "level2_q_learning", "level2_sarsa",
        "level3_q_learning", "level3_sarsa",
        "level4_q_learning", "level4_sarsa",
        "level5_q_learning", "level5_sarsa",
        "level6_q_learning", "level6_q_learning_intrinsic",
    }
    passed &= report(
        "Required Part I runs",
        expected_runs <= set(runs),
        f"{len(expected_runs & set(runs))}/{len(expected_runs)} present",
    )
    if not expected_runs <= set(runs):
        return False

    level0 = runs["level0_q_learning"]
    passed &= report(
        "Level 0 shortest policy",
        level0["success_rate"] == 1 and level0["mean_excess_steps_on_success"] == 0,
        f"success={level0['success_rate']:.0%}, steps={level0['mean_success_steps']}, optimum={level0['shortest_possible_steps']}",
    )
    q_trace = runs["level1_q_learning"]["greedy_policy_trace"]
    sarsa_trace = runs["level1_sarsa"]["greedy_policy_trace"]
    passed &= report(
        "Level 1 SARSA policy difference",
        sarsa_trace["minimum_manhattan_fire_clearance"] > q_trace["minimum_manhattan_fire_clearance"],
        f"Q clearance={q_trace['minimum_manhattan_fire_clearance']}, SARSA clearance={sarsa_trace['minimum_manhattan_fire_clearance']}",
    )
    monster_names = [
        f"level{level}_{algorithm}"
        for level in (4, 5)
        for algorithm in ("q_learning", "sarsa")
    ]
    monster_ok = all(
        runs[name]["success_rate"] >= 0.5 and runs[name]["mean_monster_moves"] > 0
        for name in monster_names
    )
    passed &= report("Monster-level learning/evidence", monster_ok, "Levels 4-5, both algorithms")
    baseline = runs["level6_q_learning"]["success_rate"]
    intrinsic = runs["level6_q_learning_intrinsic"]["success_rate"]
    passed &= report(
        "Level 6 intrinsic comparison",
        intrinsic > baseline,
        f"without={baseline:.0%}, with={intrinsic:.0%}",
    )

    figures = (
        "task1_level0_q_learning.png",
        "task2_level1_q_vs_sarsa.png",
        "task2_level1_policy_routes.png",
        "task3_levels2_3_extension.png",
        "task4_monster_levels4_5.png",
        "task5_level6_intrinsic_comparison.png",
    )
    missing = [name for name in figures if not (reports / name).exists()]
    passed &= report("Part I report figures", not missing, "all present" if not missing else str(missing))
    return bool(passed)


def verify_part2() -> bool:
    result = subprocess.run(
        [sys.executable, str(ROOT / "verify_block6_submission.py")],
        cwd=ROOT,
        check=False,
    )
    return report("Part II strict verifier", result.returncode == 0, f"exit code {result.returncode}")


def main() -> None:
    passed = verify_part1()
    passed &= verify_part2()
    passed &= report("Unified launcher", (ROOT / "main.py").exists(), "main.py")
    print("\nCode/artifact verification:", "PASS" if passed else "FAIL")
    print("External obligations not machine-verifiable: final PDF, contribution statement, and group video URL.")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
