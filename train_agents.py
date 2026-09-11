#!/usr/bin/env python3
"""Train Arena agents with Stable Baselines3 and TensorBoard logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arena import ControlStyle, make_direct_env, make_rotation_env


POLICY_KWARGS = {"net_arch": [128, 128]}


PRESETS: dict[str, dict[str, Any]] = {
    "short": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "gamma": 0.99,
        "ent_coef": 0.01,
    },
    "long": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.995,
        "ent_coef": 0.005,
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=("direct", "rotation", "both"), default="both")
    parser.add_argument("--algorithm", choices=("ppo", "dqn"), default="ppo")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="short")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    return parser


def _train_one(
    style: ControlStyle,
    algorithm: str,
    preset: str,
    timesteps: int,
    seed: int,
    output_dir: Path,
    log_dir: Path,
) -> None:
    try:
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "Stable Baselines3 is required. Install with "
            "'python -m pip install -r requirements-rl.txt'."
        ) from exc

    factory = make_direct_env if style is ControlStyle.DIRECT else make_rotation_env
    env = Monitor(factory(seed=seed))
    params = dict(PRESETS[preset])
    style_log_dir = log_dir / f"{algorithm}_{style.value}_{preset}"
    style_log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    params.update(
        policy="MlpPolicy",
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        seed=seed,
        tensorboard_log=str(style_log_dir),
    )
    if algorithm == "dqn":
        # DQN uses replay-buffer parameters rather than PPO rollout parameters.
        params.pop("n_steps")
        params.pop("batch_size")
        params.pop("ent_coef")
        params.update(buffer_size=50_000, learning_starts=2_000, train_freq=4)
        model = DQN(env=env, **params)
    else:
        model = PPO(env=env, **params)
    effective_hyperparameters = {
        key: value
        for key, value in params.items()
        if key not in {"policy", "policy_kwargs", "verbose", "seed", "tensorboard_log"}
    }

    try:
        model.learn(total_timesteps=timesteps, progress_bar=False)
        model.save(str(output_dir / f"{style.value}_agent"))
    finally:
        env.close()
    metadata = {
        "style": style.value,
        "algorithm": algorithm,
        "preset": preset,
        "timesteps": timesteps,
        "seed": seed,
        "hyperparameters": effective_hyperparameters,
        "policy_kwargs": POLICY_KWARGS,
        "tensorboard_log": str(style_log_dir),
    }
    (output_dir / f"{style.value}_agent.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _parser().parse_args()
    if args.timesteps <= 0:
        raise SystemExit("--timesteps must be positive")
    styles = (
        (ControlStyle.DIRECT, ControlStyle.ROTATION)
        if args.style == "both"
        else (ControlStyle(args.style),)
    )
    for style in styles:
        _train_one(
            style,
            args.algorithm,
            args.preset,
            args.timesteps,
            args.seed,
            args.output_dir,
            args.log_dir,
        )


if __name__ == "__main__":
    main()
