"""Authoritative observation and reward adapters for GAIT Part II.

Task 3: Observation Design (Block 3).
Provides normalized, fixed-size numerical feature vectors capturing full
Markovian dynamics and relative target bearings for both control styles.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .core import ArenaCore
from .entities import StepOutcome
from .math2d import EPSILON, length


OBSERVATION_LABELS = (
    # 1. Player Kinematics (6)
    "player_x",
    "player_y",
    "player_velocity_x",
    "player_velocity_y",
    "player_facing_sin",
    "player_facing_cos",
    # 2. Nearest Enemy Features (8)
    "nearest_enemy_unit_dir_x",
    "nearest_enemy_unit_dir_y",
    "nearest_enemy_distance",
    "nearest_enemy_aim_alignment",   # cos(angle_to_enemy - ship_heading)
    "nearest_enemy_aim_ortho",       # sin(angle_to_enemy - ship_heading)
    "nearest_enemy_health",
    "nearest_enemy_present",
    "nearest_enemy_closing_speed",
    # 3. Nearest Spawner Features (6)
    "nearest_spawner_unit_dir_x",
    "nearest_spawner_unit_dir_y",
    "nearest_spawner_distance",
    "nearest_spawner_aim_alignment", # cos(angle_to_spawner - ship_heading)
    "nearest_spawner_health",
    "nearest_spawner_present",
    # 4. Global State & Cooldowns (5)
    "player_health",
    "phase",
    "shoot_cooldown",
    "enemy_density",
    "spawner_density",
)
OBSERVATION_SIZE = len(OBSERVATION_LABELS)  # 25 features


class ObservationFunction(Protocol):
    def __call__(self, core: ArenaCore) -> np.ndarray: ...


class RewardFunction(Protocol):
    def __call__(self, core: ArenaCore, outcome: StepOutcome) -> float: ...


def build_baseline_observation(core: ArenaCore) -> np.ndarray:
    """Compute an omni-directional, normalized 25-D observation vector."""
    cfg = core.config
    player = core.player
    diagonal = math.hypot(cfg.width, cfg.height)
    enemy = core.nearest_enemy()
    spawner = core.nearest_spawner()

    heading_cos = math.cos(player.angle)
    heading_sin = math.sin(player.angle)

    values: list[float] = [
        float(np.clip(player.pos[0] / cfg.width * 2.0 - 1.0, -1.0, 1.0)),
        float(np.clip(player.pos[1] / cfg.height * 2.0 - 1.0, -1.0, 1.0)),
        float(np.clip(player.vel[0] / cfg.player_max_speed, -1.0, 1.0)),
        float(np.clip(player.vel[1] / cfg.player_max_speed, -1.0, 1.0)),
        heading_sin,
        heading_cos,
    ]

    if enemy is None:
        values.extend((0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0))
    else:
        rel = enemy.pos - player.pos
        dist = length(rel)
        if dist > EPSILON:
            unit_x = float(rel[0] / dist)
            unit_y = float(rel[1] / dist)
            aim_alignment = unit_x * heading_cos + unit_y * heading_sin
            aim_ortho = unit_x * heading_sin - unit_y * heading_cos
            rel_vel = player.vel - enemy.vel
            closing_speed = float(np.dot(rel_vel, rel / dist) / (cfg.player_max_speed + cfg.enemy_max_speed))
        else:
            unit_x, unit_y = 0.0, 0.0
            aim_alignment, aim_ortho = 1.0, 0.0
            closing_speed = 0.0

        values.extend((
            float(np.clip(unit_x, -1.0, 1.0)),
            float(np.clip(unit_y, -1.0, 1.0)),
            float(np.clip(dist / diagonal, 0.0, 1.0)),
            float(np.clip(aim_alignment, -1.0, 1.0)),
            float(np.clip(aim_ortho, -1.0, 1.0)),
            float(np.clip(enemy.health / enemy.max_health, 0.0, 1.0)),
            1.0,
            float(np.clip(closing_speed, -1.0, 1.0)),
        ))

    if spawner is None:
        values.extend((0.0, 0.0, 1.0, -1.0, 0.0, 0.0))
    else:
        rel_s = spawner.pos - player.pos
        dist_s = length(rel_s)
        if dist_s > EPSILON:
            unit_sx = float(rel_s[0] / dist_s)
            unit_sy = float(rel_s[1] / dist_s)
            aim_spawner = unit_sx * heading_cos + unit_sy * heading_sin
        else:
            unit_sx, unit_sy = 0.0, 0.0
            aim_spawner = 1.0

        values.extend((
            float(np.clip(unit_sx, -1.0, 1.0)),
            float(np.clip(unit_sy, -1.0, 1.0)),
            float(np.clip(dist_s / diagonal, 0.0, 1.0)),
            float(np.clip(aim_spawner, -1.0, 1.0)),
            float(np.clip(spawner.health / spawner.max_health, 0.0, 1.0)),
            1.0,
        ))

    values.extend((
        float(np.clip(player.health / player.max_health, 0.0, 1.0)),
        float(np.clip((core.phase - 1) / max(1, cfg.max_phase_for_observation - 1), 0.0, 1.0)),
        float(np.clip(player.shoot_cooldown / cfg.player_shoot_cooldown, 0.0, 1.0)),
        float(np.clip(len(core.enemies) / cfg.max_enemies, 0.0, 1.0)),
        float(np.clip(len(core.spawners) / cfg.max_spawner_count, 0.0, 1.0)),
    ))

    obs = np.asarray(values, dtype=np.float32)
    if obs.shape != (OBSERVATION_SIZE,):
        raise AssertionError(f"Observation size mismatch: expected {OBSERVATION_SIZE}, got {obs.shape}")
    return obs


def neutral_reward(_core: ArenaCore, _outcome: StepOutcome) -> float:
    """Block 5 hook: valid API value without silently choosing reward weights."""
    return 0.0