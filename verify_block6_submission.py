#!/usr/bin/env python3
"""Strict pre-submission verification for the Part II Block 6 deliverables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from part2.arena import ControlStyle, make_direct_env, make_rotation_env
from part2.training.artifacts import PROJECT_ROOT, load_metadata_for_model
from part2.training.config import load_training_settings

REPOSITORY_ROOT = Path(__file__).resolve().parent


def report(name: str, passed: bool, detail: str) -> bool:
    marker = "PASS" if passed else "FAIL"
    print(f"[{marker}] {name}: {detail}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all Block 6 code and evidence")
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Check source contracts without requiring trained artifacts",
    )
    args = parser.parse_args()

    passed = True
    settings = load_training_settings(PROJECT_ROOT / "config" / "training.json")
    passed &= report("Algorithm", settings.algorithm == "PPO", settings.algorithm)
    passed &= report(
        "Neural network",
        len(settings.net_arch) >= 1,
        f"MLP hidden layers {list(settings.net_arch)}",
    )

    for style, factory, expected_actions in (
        (ControlStyle.ROTATION, make_rotation_env, 5),
        (ControlStyle.DIRECT, make_direct_env, 6),
    ):
        env = factory(seed=settings.seed)
        try:
            passed &= report(
                f"{style.value} action space",
                env.action_space.n == expected_actions,
                f"Discrete({env.action_space.n})",
            )
            observation, _info = env.reset(seed=settings.seed)
            passed &= report(
                f"{style.value} observation",
                env.observation_space.contains(observation),
                f"shape={observation.shape}, dtype={observation.dtype}",
            )
        finally:
            env.close()

    required_scripts = (
        "train_rotation.py",
        "train_direct.py",
        "train_both.py",
        "evaluate_rotation.py",
        "evaluate_direct.py",
        "tune_hyperparameters.py",
        "compare_agents.py",
        "plot_training_curves.py",
    )
    missing_scripts = [name for name in required_scripts if not (REPOSITORY_ROOT / name).exists()]
    passed &= report(
        "Training/evaluation scripts",
        not missing_scripts,
        "all present" if not missing_scripts else f"missing {missing_scripts}",
    )

    if args.code_only:
        print("Code-only verification complete; trained evidence was intentionally not checked.")
        raise SystemExit(0 if passed else 1)

    try:
        from part2.training.evaluation import load_ppo_model, validate_model_contract
    except ImportError as exc:
        report("SB3 import", False, str(exc))
        raise SystemExit(1) from exc

    for style in (ControlStyle.ROTATION, ControlStyle.DIRECT):
        model_path = PROJECT_ROOT / "models" / f"{style.value}_agent.zip"
        metadata = load_metadata_for_model(model_path) if model_path.exists() else None
        passed &= report(
            f"{style.value} trained model",
            model_path.exists(),
            str(model_path),
        )
        passed &= report(
            f"{style.value} metadata",
            metadata is not None,
            str(model_path.with_suffix('.metadata.json')),
        )
        if model_path.exists():
            try:
                model = load_ppo_model(model_path)
                validate_model_contract(model, style, model_path)
            except Exception as exc:
                passed &= report(f"{style.value} model contract", False, str(exc))
            else:
                passed &= report(f"{style.value} model contract", True, "PPO/action metadata match")

        tensorboard_files = list(
            (PROJECT_ROOT / "logs" / "tensorboard" / style.value).rglob("events.out.tfevents.*")
        )
        passed &= report(
            f"{style.value} TensorBoard log",
            bool(tensorboard_files),
            f"{len(tensorboard_files)} event file(s)",
        )
        evaluation_file = (
            PROJECT_ROOT
            / "logs"
            / "evaluations"
            / style.value
            / "final"
            / "evaluations.npz"
        )
        passed &= report(
            f"{style.value} periodic evaluations",
            evaluation_file.exists(),
            str(evaluation_file),
        )
        summary_file = PROJECT_ROOT / "reports" / f"{style.value}_final_summary.json"
        passed &= report(
            f"{style.value} final metrics",
            summary_file.exists(),
            str(summary_file),
        )

    for name in (
        "tuning/hyperparameter_trials.csv",
        "tuning/hyperparameter_trials.json",
        "control_style_comparison.csv",
        "control_style_comparison.json",
        "training_curves_final.png",
        "model_selection/model_selection.csv",
        "model_selection/model_selection.json",
    ):
        path = PROJECT_ROOT / "reports" / name
        passed &= report(name, path.exists(), str(path))

    print("Block 6 submission verification:", "PASS" if passed else "FAIL")
    if not passed:
        print("Run final training and compare_agents.py, then verify again.")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
