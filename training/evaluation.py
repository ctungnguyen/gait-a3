"""Headless metrics and model-contract validation shared by Block 6 tools."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np

from arena import ArenaConfig, ControlStyle, action_names, make_direct_env, make_rotation_env
from arena.adapters import OBSERVATION_SIZE
from arena.rewards import RewardConfig, ProgressionReward
from .artifacts import load_metadata_for_model, sha256_file


EPISODE_INFO_KEYS = (
    "episode_max_phase",
    "episode_phases_advanced",
    "episode_enemies_destroyed",
    "episode_spawners_destroyed",
    "episode_shots_fired",
    "episode_damage_dealt",
    "episode_damage_taken",
    "is_success",
)


def load_ppo_model(model_path: str | Path, device: str = "auto"):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "Stable Baselines3 is required. Install requirements-rl.txt first."
        ) from exc
    return PPO.load(str(model_path), device=device)


def validate_model_contract(
    model,
    style: ControlStyle,
    model_path: str | Path,
    *,
    warn_missing_metadata: bool = True,
) -> None:
    expected_count = len(action_names(style))
    actual_count = getattr(model.action_space, "n", None)
    if actual_count != expected_count:
        raise ValueError(
            f"Wrong model for {style.value}: model uses Discrete({actual_count}), "
            f"but this style requires Discrete({expected_count})."
        )

    metadata = load_metadata_for_model(model_path)
    if metadata is None:
        if warn_missing_metadata:
            warnings.warn(
                f"No metadata sidecar found for {model_path}; validating action count only.",
                stacklevel=2,
            )
        return

    expected_actions = list(action_names(style))
    mismatches: list[str] = []
    if metadata.get("algorithm") != "PPO":
        mismatches.append(f"algorithm={metadata.get('algorithm')!r}")
    if metadata.get("control_style") != style.value:
        mismatches.append(f"control_style={metadata.get('control_style')!r}")
    if metadata.get("action_count") != expected_count:
        mismatches.append(f"action_count={metadata.get('action_count')!r}")
    if metadata.get("action_meanings") != expected_actions:
        mismatches.append("action_meanings differ")
    if metadata.get("observation_size") != OBSERVATION_SIZE:
        mismatches.append(f"observation_size={metadata.get('observation_size')!r}")
    recorded_hash = metadata.get("model_sha256")
    model_file = Path(model_path)
    if recorded_hash and model_file.exists() and recorded_hash != sha256_file(model_file):
        mismatches.append("model SHA-256 does not match metadata")
    if mismatches:
        raise ValueError("Model metadata is incompatible: " + ", ".join(mismatches))


def evaluate_model(
    model,
    *,
    style: ControlStyle,
    arena_config: ArenaConfig,
    reward_config: RewardConfig,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one model on reproducible episode seeds without rendering."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    factory = make_rotation_env if style is ControlStyle.ROTATION else make_direct_env
    env = factory(
        config=arena_config,
        reward_fn=ProgressionReward(reward_config),
        render_mode=None,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            observation, _info = env.reset(seed=episode_seed)
            episode_return = 0.0
            terminated = truncated = False
            info: dict[str, Any] = {}

            while not (terminated or truncated):
                prediction, _state = model.predict(observation, deterministic=deterministic)
                action = int(np.asarray(prediction).item())
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)

            rows.append(
                {
                    "episode": episode_index + 1,
                    "seed": episode_seed,
                    "return": episode_return,
                    "steps": int(info["step"]),
                    "termination": "death" if terminated else "time_limit",
                    **{key: info[key] for key in EPISODE_INFO_KEYS},
                }
            )
    finally:
        env.close()

    return rows, summarize_episodes(rows)


def summarize_episodes(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("At least one episode is required")

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    returns = values("return")
    steps = values("steps")
    phases = values("episode_max_phase")
    successes = values("is_success")
    return {
        "episodes": len(rows),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_episode_steps": float(np.mean(steps)),
        "mean_max_phase": float(np.mean(phases)),
        "highest_phase": int(np.max(phases)),
        "success_rate": float(np.mean(successes)),
        "mean_enemies_destroyed": float(np.mean(values("episode_enemies_destroyed"))),
        "mean_spawners_destroyed": float(np.mean(values("episode_spawners_destroyed"))),
        "mean_damage_dealt": float(np.mean(values("episode_damage_dealt"))),
        "mean_damage_taken": float(np.mean(values("episode_damage_taken"))),
        "mean_shots_fired": float(np.mean(values("episode_shots_fired"))),
    }


def write_episode_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty episode CSV")
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
