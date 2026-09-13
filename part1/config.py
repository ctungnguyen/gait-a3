"""Validated, file-backed training settings for Part I."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PART1_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PART1_ROOT / "config" / "gridworld.json"


@dataclass(frozen=True)
class GridworldSettings:
    episodes: int = 5_000
    alpha: float = 0.25
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 3_500
    max_steps_per_episode: int = 350
    evaluation_episodes: int = 30
    fps_visual: int = 12
    tile_size: int = 56
    seed: int = 42
    intrinsic_reward_strength: float = 0.01
    monster_move_chance: float = 0.40

    def validate(self) -> "GridworldSettings":
        if self.episodes < 1 or self.max_steps_per_episode < 1:
            raise ValueError("episodes and maxStepsPerEpisode must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilonEnd must be <= epsilonStart and both in [0, 1]")
        if self.epsilon_decay_episodes < 0:
            raise ValueError("epsilonDecayEpisodes cannot be negative")
        if not 0.0 <= self.monster_move_chance <= 1.0:
            raise ValueError("monsterMoveChance must be in [0, 1]")
        if self.intrinsic_reward_strength < 0.0:
            raise ValueError("intrinsicRewardStrength cannot be negative")
        if self.evaluation_episodes < 1 or self.fps_visual < 1 or self.tile_size < 24:
            raise ValueError("evaluationEpisodes/fpsVisual/tileSize are invalid")
        return self

    def with_overrides(self, **values: Any) -> "GridworldSettings":
        known = {field.name for field in self.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown Gridworld setting(s): {sorted(unknown)}")
        return replace(self, **values).validate()


_JSON_TO_FIELD = {
    "episodes": "episodes",
    "alpha": "alpha",
    "gamma": "gamma",
    "epsilonStart": "epsilon_start",
    "epsilonEnd": "epsilon_end",
    "epsilonDecayEpisodes": "epsilon_decay_episodes",
    "maxStepsPerEpisode": "max_steps_per_episode",
    "evaluationEpisodes": "evaluation_episodes",
    "fpsVisual": "fps_visual",
    "tileSize": "tile_size",
    "seed": "seed",
    "intrinsicRewardStrength": "intrinsic_reward_strength",
    "monsterMoveChance": "monster_move_chance",
}


def _translate(raw: dict[str, Any]) -> dict[str, Any]:
    unknown = set(raw) - set(_JSON_TO_FIELD)
    if unknown:
        raise ValueError(f"Unknown gridworld config key(s): {sorted(unknown)}")
    return {_JSON_TO_FIELD[key]: value for key, value in raw.items()}


def load_gridworld_settings(
    path: str | Path = DEFAULT_CONFIG_PATH,
    level: int | None = None,
) -> GridworldSettings:
    """Load base settings and, when requested, merge a level override."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Gridworld config must contain a JSON object")

    base_raw = raw.get("training", raw)
    if not isinstance(base_raw, dict):
        raise ValueError("training must be a JSON object")
    settings = GridworldSettings(**_translate(base_raw)).validate()

    if level is not None:
        level_overrides = raw.get("levelOverrides", {})
        override = level_overrides.get(str(level), {})
        if not isinstance(override, dict):
            raise ValueError(f"levelOverrides.{level} must be a JSON object")
        settings = settings.with_overrides(**_translate(override))
    return settings
