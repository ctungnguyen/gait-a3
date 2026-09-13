"""Event-based training reward for the Part II deep-RL agents.

The environment mechanics remain in :mod:`part2.arena.core`. This module only
translates facts already emitted by the Arena into a learning signal, keeping
reward tuning isolated and auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import numpy as np

from .core import ArenaCore
from .controls import ControlStyle
from .entities import StepOutcome
from .math2d import EPSILON, forward, length, normalized


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for intentional progression without changing game mechanics."""

    step_penalty: float = -0.003
    enemy_damage_per_hp: float = 0.005
    enemy_destroyed: float = 0.5
    spawner_damage_per_hp: float = 0.080
    spawner_destroyed: float = 12.0
    phase_advanced: float = 30.0
    player_damage_per_hp: float = -0.150
    player_died: float = -25.0
    time_limit_reached: float = -2.0
    approach_progress: float = 0.020
    aim_progress: float = 0.040
    firing_lane_progress: float = 0.030
    aligned_shot: float = 0.080
    wasted_shot: float = -0.020
    aligned_shot_threshold: float = 0.92

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "RewardConfig":
        values = {} if values is None else dict(values)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown reward setting(s): {', '.join(unknown)}")
        result = cls(**values)
        if not -1.0 <= result.aligned_shot_threshold <= 1.0:
            raise ValueError("aligned_shot_threshold must be between -1 and 1")
        return result

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class ProgressionReward:
    """Stateful callable used independently by each training environment.

    Event rewards teach the main objective. A small distance-difference term
    makes early exploration less sparse by rewarding movement toward the
    nearest active spawner and symmetrically penalising movement away from it.
    Because the term is a difference, repeatedly moving out and back cannot
    create free reward; the per-step cost additionally discourages camping.
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        control_style: ControlStyle | str | None = None,
    ):
        self.config = config or RewardConfig()
        self.control_style = None if control_style is None else ControlStyle(control_style)
        self._target_id: int | None = None
        self._target_distance: float | None = None
        self._target_alignment: float | None = None
        self._target_lane_error: float | None = None
        self.last_components: dict[str, float] = {}

    def reset(self, core: ArenaCore) -> None:
        self._remember_target(core)
        self.last_components = {}

    def __call__(self, core: ArenaCore, outcome: StepOutcome) -> float:
        cfg = self.config
        components: dict[str, float] = {"step": cfg.step_penalty}

        for event in outcome.events:
            if event.name == "enemy_damaged":
                self._add(components, "enemy_damage", cfg.enemy_damage_per_hp * event.amount)
            elif event.name == "enemy_destroyed":
                self._add(components, "enemy_destroyed", cfg.enemy_destroyed)
            elif event.name == "spawner_damaged":
                self._add(components, "spawner_damage", cfg.spawner_damage_per_hp * event.amount)
            elif event.name == "spawner_destroyed":
                self._add(components, "spawner_destroyed", cfg.spawner_destroyed)
            elif event.name == "phase_advanced":
                self._add(components, "phase_advanced", cfg.phase_advanced)
            elif event.name == "player_damaged":
                self._add(components, "player_damage", cfg.player_damage_per_hp * event.amount)
            elif event.name == "player_died":
                self._add(components, "player_died", cfg.player_died)
            elif event.name == "time_limit_reached":
                self._add(components, "time_limit", cfg.time_limit_reached)
            elif event.name == "projectile_fired":
                self._add(components, "shot_quality", self._shot_quality(core))

        approach, aim, lane = self._navigation_rewards(core)
        if approach != 0.0:
            components["approach"] = approach
        if aim != 0.0:
            components["aim_progress"] = aim
        if lane != 0.0:
            components["firing_lane"] = lane

        self.last_components = components
        return float(sum(components.values()))

    @staticmethod
    def _add(components: dict[str, float], name: str, amount: float) -> None:
        components[name] = components.get(name, 0.0) + float(amount)

    def _remember_target(self, core: ArenaCore) -> None:
        target = core.nearest_spawner()
        self._target_id = None if target is None else target.entity_id
        self._target_distance = None if target is None else length(target.pos - core.player.pos)
        self._target_alignment = None if target is None else self._alignment(core, target.pos)
        self._target_lane_error = (
            None
            if target is None or self.control_style is not ControlStyle.DIRECT
            else self._firing_lane_error(core, target.pos)
        )

    def _navigation_rewards(self, core: ArenaCore) -> tuple[float, float, float]:
        """Reward distance and signed aim improvement toward the live target.

        The target ID is part of the memory.  Destroying one spawner therefore
        starts a fresh baseline for the next spawner instead of producing a
        false shaping spike.  The aim term supplies the Rotation agent with an
        immediate learning signal for turning toward that new target.
        """

        target = core.nearest_spawner()
        if target is None:
            self._target_id = None
            self._target_distance = None
            self._target_alignment = None
            self._target_lane_error = None
            return 0.0, 0.0, 0.0

        current_distance = length(target.pos - core.player.pos)
        current_alignment = self._alignment(core, target.pos)
        approach_reward = 0.0
        aim_reward = 0.0
        lane_reward = 0.0
        if self._target_id == target.entity_id and self._target_distance is not None:
            maximum_step = max(core.config.player_max_speed * core.config.fixed_dt, EPSILON)
            progress = float(
                np.clip((self._target_distance - current_distance) / maximum_step, -1.0, 1.0)
            )
            approach_reward = self.config.approach_progress * progress
            if self._target_alignment is not None:
                maximum_turn = max(
                    core.config.player_rotation_speed * core.config.fixed_dt,
                    EPSILON,
                )
                alignment_progress = float(
                    np.clip(
                        (current_alignment - self._target_alignment) / maximum_turn,
                        -1.0,
                        1.0,
                    )
                )
                aim_reward = self.config.aim_progress * alignment_progress
            if (
                self.control_style is ControlStyle.DIRECT
                and self._target_lane_error is not None
            ):
                current_lane_error = self._firing_lane_error(core, target.pos)
                lane_progress = float(
                    np.clip(
                        (self._target_lane_error - current_lane_error) / maximum_step,
                        -1.0,
                        1.0,
                    )
                )
                lane_reward = self.config.firing_lane_progress * lane_progress

        self._target_id = target.entity_id
        self._target_distance = current_distance
        self._target_alignment = current_alignment
        self._target_lane_error = (
            self._firing_lane_error(core, target.pos)
            if self.control_style is ControlStyle.DIRECT
            else None
        )
        return approach_reward, aim_reward, lane_reward

    @staticmethod
    def _alignment(core: ArenaCore, target_position: np.ndarray) -> float:
        relative = target_position - core.player.pos
        if length(relative) <= EPSILON:
            return 1.0
        return float(np.dot(forward(core.player.angle), normalized(relative)))

    @staticmethod
    def _firing_lane_error(core: ArenaCore, target_position: np.ndarray) -> float:
        """Distance to the nearest horizontal/vertical firing lane."""

        relative = target_position - core.player.pos
        return float(min(abs(float(relative[0])), abs(float(relative[1]))))

    def _shot_quality(self, core: ArenaCore) -> float:
        candidates = [*core.spawners, *core.enemies]
        if not candidates:
            return self.config.wasted_shot

        heading = forward(core.player.angle)
        qualities: list[float] = []
        for entity in candidates:
            relative = entity.pos - core.player.pos
            if length(relative) <= EPSILON:
                qualities.append(1.0)
                continue

            forward_distance = float(np.dot(relative, heading))
            lateral_distance = abs(
                float(relative[0] * heading[1] - relative[1] * heading[0])
            )
            hit_radius = float(entity.radius + core.config.projectile_radius + 2.0)
            alignment = float(np.dot(heading, normalized(relative)))
            if (
                forward_distance <= 0.0
                or lateral_distance > hit_radius
                or alignment < self.config.aligned_shot_threshold
            ):
                continue
            lane_quality = 1.0 - lateral_distance / max(hit_radius, EPSILON)
            qualities.append(float(np.clip(0.5 + 0.5 * lane_quality, 0.0, 1.0)))

        if not qualities:
            return self.config.wasted_shot
        return self.config.aligned_shot * max(qualities)


def make_progression_reward(
    values: dict[str, Any] | None = None,
    control_style: ControlStyle | str | None = None,
) -> ProgressionReward:
    """Build a fresh reward object; never share state between vector envs."""

    return ProgressionReward(RewardConfig.from_dict(values), control_style=control_style)
