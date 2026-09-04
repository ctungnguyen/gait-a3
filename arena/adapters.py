"""Replaceable observation and reward adapters for teammate handoff.

Block 2 needs a working ``step`` return value, so this file provides a compact
baseline observation and a neutral reward. The owners of Blocks 3 and 5 can
replace either callable without changing :class:`arena.core.ArenaCore`.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .core import ArenaCore
from .entities import StepOutcome
from .math2d import length


OBSERVATION_LABELS = (
    "player_x",
    "player_y",
    "player_velocity_x",
    "player_velocity_y",
    "player_facing_sin",
    "player_facing_cos",
    "nearest_enemy_relative_x",
    "nearest_enemy_relative_y",
    "nearest_enemy_distance",
    "nearest_enemy_health",
    "nearest_enemy_present",
    "nearest_spawner_relative_x",
    "nearest_spawner_relative_y",
    "nearest_spawner_distance",
    "nearest_spawner_health",
    "nearest_spawner_present",
    "player_health",
    "phase",
    "shoot_cooldown",
    "enemy_count",
    "spawner_count",
)
OBSERVATION_SIZE = len(OBSERVATION_LABELS)


class ObservationFunction(Protocol):
    def __call__(self, core: ArenaCore) -> np.ndarray: ...


class RewardFunction(Protocol):
    def __call__(self, core: ArenaCore, outcome: StepOutcome) -> float: ...


def build_baseline_observation(core: ArenaCore) -> np.ndarray:
    """Return a normalized, fixed-size vector that already covers the brief."""

    cfg = core.config
    player = core.player
    diagonal = math.hypot(cfg.width, cfg.height)
    enemy = core.nearest_enemy()
    spawner = core.nearest_spawner()

    values: list[float] = [
        player.pos[0] / cfg.width * 2.0 - 1.0,
        player.pos[1] / cfg.height * 2.0 - 1.0,
        float(np.clip(player.vel[0] / cfg.player_max_speed, -1.0, 1.0)),
        float(np.clip(player.vel[1] / cfg.player_max_speed, -1.0, 1.0)),
        math.sin(player.angle),
        math.cos(player.angle),
    ]

    if enemy is None:
        values.extend((0.0, 0.0, 1.0, 0.0, 0.0))
    else:
        relative = enemy.pos - player.pos
        values.extend(
            (
                float(np.clip(relative[0] / cfg.width, -1.0, 1.0)),
                float(np.clip(relative[1] / cfg.height, -1.0, 1.0)),
                float(np.clip(length(relative) / diagonal, 0.0, 1.0)),
                float(np.clip(enemy.health / enemy.max_health, 0.0, 1.0)),
                1.0,
            )
        )

    if spawner is None:
        values.extend((0.0, 0.0, 1.0, 0.0, 0.0))
    else:
        relative = spawner.pos - player.pos
        values.extend(
            (
                float(np.clip(relative[0] / cfg.width, -1.0, 1.0)),
                float(np.clip(relative[1] / cfg.height, -1.0, 1.0)),
                float(np.clip(length(relative) / diagonal, 0.0, 1.0)),
                float(np.clip(spawner.health / spawner.max_health, 0.0, 1.0)),
                1.0,
            )
        )

    values.extend(
        (
            float(np.clip(player.health / player.max_health, 0.0, 1.0)),
            float(np.clip(core.phase / cfg.max_phase_for_observation, 0.0, 1.0)),
            float(np.clip(player.shoot_cooldown / cfg.player_shoot_cooldown, 0.0, 1.0)),
            float(np.clip(len(core.enemies) / cfg.max_enemies, 0.0, 1.0)),
            float(np.clip(len(core.spawners) / cfg.max_spawner_count, 0.0, 1.0)),
        )
    )
    observation = np.asarray(values, dtype=np.float32)
    if observation.shape != (OBSERVATION_SIZE,):
        raise AssertionError(f"Observation has unexpected shape {observation.shape}")
    return observation


def neutral_reward(_core: ArenaCore, _outcome: StepOutcome) -> float:
    """Block 5 hook: valid API value without silently choosing reward weights."""

    return 0.0
