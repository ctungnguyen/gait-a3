"""Validated JSON configuration for PPO training and tuning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from arena.rewards import RewardConfig


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    algorithm: str
    preset: str
    total_timesteps: int
    n_envs: int
    seed: int
    eval_freq: int
    n_eval_episodes: int
    checkpoint_freq: int
    final_eval_episodes: int
    net_arch: tuple[int, ...]
    activation: str
    ppo: dict[str, Any]
    reward: RewardConfig

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["net_arch"] = list(self.net_arch)
        result["reward"] = self.reward.to_dict()
        return result


def _merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_nested(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_training_settings(
    path: str | Path,
    preset: str = "baseline",
    *,
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    seed: int | None = None,
    final_eval_episodes: int | None = None,
) -> TrainingSettings:
    """Load one named preset and reject configurations likely to fail silently."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    presets = raw.get("presets", {})
    if preset not in presets:
        choices = ", ".join(sorted(presets)) or "none"
        raise ValueError(f"Unknown training preset {preset!r}; available: {choices}")

    merged = _merge_nested({key: value for key, value in raw.items() if key != "presets"}, presets[preset])
    if total_timesteps is not None:
        merged["total_timesteps"] = total_timesteps
    if n_envs is not None:
        merged["n_envs"] = n_envs
    if seed is not None:
        merged["seed"] = seed
    if final_eval_episodes is not None:
        merged["final_eval_episodes"] = final_eval_episodes

    algorithm = str(merged.get("algorithm", "")).upper()
    if algorithm != "PPO":
        raise ValueError("This fair-comparison pipeline intentionally uses PPO for both styles")

    policy = dict(merged.get("policy", {}))
    architecture = tuple(int(width) for width in policy.get("net_arch", ()))
    if not architecture or any(width <= 0 for width in architecture):
        raise ValueError("policy.net_arch must contain at least one positive hidden-layer width")
    activation = str(policy.get("activation", "tanh")).lower()
    if activation not in {"tanh", "relu"}:
        raise ValueError("policy.activation must be 'tanh' or 'relu'")

    ppo = dict(merged.get("ppo", {}))
    required_ppo = {
        "learning_rate",
        "linear_learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    }
    missing = sorted(required_ppo - set(ppo))
    if missing:
        raise ValueError(f"Missing PPO setting(s): {', '.join(missing)}")

    positive_integer_fields = (
        "total_timesteps",
        "n_envs",
        "eval_freq",
        "n_eval_episodes",
        "checkpoint_freq",
        "final_eval_episodes",
    )
    for name in positive_integer_fields:
        value = int(merged[name])
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    ppo["n_steps"] = int(ppo["n_steps"])
    ppo["batch_size"] = int(ppo["batch_size"])
    ppo["n_epochs"] = int(ppo["n_epochs"])
    if min(ppo["n_steps"], ppo["batch_size"], ppo["n_epochs"]) <= 0:
        raise ValueError("PPO n_steps, batch_size, and n_epochs must be positive")

    rollout_size = ppo["n_steps"] * int(merged["n_envs"])
    if ppo["batch_size"] > rollout_size or rollout_size % ppo["batch_size"] != 0:
        raise ValueError(
            "PPO batch_size must divide n_steps * n_envs exactly "
            f"({ppo['batch_size']} does not divide {rollout_size})"
        )
    if not 0.0 < float(ppo["gamma"]) <= 1.0:
        raise ValueError("PPO gamma must be in (0, 1]")
    if not 0.0 <= float(ppo["gae_lambda"]) <= 1.0:
        raise ValueError("PPO gae_lambda must be in [0, 1]")

    return TrainingSettings(
        algorithm=algorithm,
        preset=preset,
        total_timesteps=int(merged["total_timesteps"]),
        n_envs=int(merged["n_envs"]),
        seed=int(merged["seed"]),
        eval_freq=int(merged["eval_freq"]),
        n_eval_episodes=int(merged["n_eval_episodes"]),
        checkpoint_freq=int(merged["checkpoint_freq"]),
        final_eval_episodes=int(merged["final_eval_episodes"]),
        net_arch=architecture,
        activation=activation,
        ppo=ppo,
        reward=RewardConfig.from_dict(merged.get("reward")),
    )
