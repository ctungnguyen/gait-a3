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
from .entities import StepOutcome
from .math2d import EPSILON, forward, length, normalized


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for intentional progression without changing game mechanics."""

    step_penalty: float = -0.002
    enemy_damage_per_hp: float = 0.020
    enemy_destroyed: float = 1.5
    spawner_damage_per_hp: float = 0.050
    spawner_destroyed: float = 8.0
    phase_advanced: float = 20.0
    player_damage_per_hp: float = -0.150
    player_died: float = -25.0
    time_limit_reached: float = -2.0
    approach_progress: float = 0.015
    aligned_shot: float = 0.080
    wasted_shot: float = -0.015
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

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self._target_id: int | None = None
        self._target_distance: float | None = None
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

        approach = self._approach_reward(core)
        if approach != 0.0:
            components["approach"] = approach

        self.last_components = components
        return float(sum(components.values()))

    @staticmethod
    def _add(components: dict[str, float], name: str, amount: float) -> None:
        components[name] = components.get(name, 0.0) + float(amount)

    def _remember_target(self, core: ArenaCore) -> None:
        target = core.nearest_spawner()
        self._target_id = None if target is None else target.entity_id
        self._target_distance = None if target is None else length(target.pos - core.player.pos)

    def _approach_reward(self, core: ArenaCore) -> float:
        target = core.nearest_spawner()
        if target is None:
            self._target_id = None
            self._target_distance = None
            return 0.0

        current_distance = length(target.pos - core.player.pos)
        reward = 0.0
        if self._target_id == target.entity_id and self._target_distance is not None:
            maximum_step = max(core.config.player_max_speed * core.config.fixed_dt, EPSILON)
            progress = float(
                np.clip((self._target_distance - current_distance) / maximum_step, -1.0, 1.0)
            )
            reward = self.config.approach_progress * progress

        self._target_id = target.entity_id
        self._target_distance = current_distance
        return reward

    def _shot_quality(self, core: ArenaCore) -> float:
        candidates = [*core.spawners, *core.enemies]
        if not candidates:
            return self.config.wasted_shot

        heading = forward(core.player.angle)
        alignments: list[float] = []
        for entity in candidates:
            relative = entity.pos - core.player.pos
            if length(relative) <= EPSILON:
                alignments.append(1.0)
            else:
                alignments.append(float(np.dot(heading, normalized(relative))))

        best_alignment = max(alignments)
        threshold = self.config.aligned_shot_threshold
        if best_alignment < threshold:
            return self.config.wasted_shot

        quality = (best_alignment - threshold) / max(1.0 - threshold, EPSILON)
        return self.config.aligned_shot * float(np.clip(quality, 0.0, 1.0))


def make_progression_reward(values: dict[str, Any] | None = None) -> ProgressionReward:
    """Build a fresh reward object; never share state between vector envs."""

    return ProgressionReward(RewardConfig.from_dict(values))
