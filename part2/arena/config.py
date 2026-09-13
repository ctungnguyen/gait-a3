"""Configuration values for the Arena simulation.

The simulation uses seconds and pixels. Training advances by ``fixed_dt``
instead of wall-clock time, so headless runs and rendered evaluation behave
consistently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArenaConfig:
    width: int = 960
    height: int = 680
    fixed_dt: float = 1.0 / 30.0
    max_steps: int = 5_400

    player_radius: float = 16.0
    player_health: float = 100.0
    player_max_speed: float = 260.0
    player_direct_acceleration: float = 1_000.0
    player_thrust_acceleration: float = 520.0
    player_rotation_speed: float = 3.2
    player_drag_per_30hz_step: float = 0.91
    player_shoot_cooldown: float = 0.22

    projectile_radius: float = 4.0
    projectile_speed: float = 520.0
    projectile_damage: float = 10.0
    projectile_lifetime: float = 2.4

    enemy_radius: float = 14.0
    enemy_health: float = 20.0
    enemy_max_speed: float = 105.0
    enemy_max_acceleration: float = 280.0
    enemy_prediction_time: float = 0.65
    enemy_contact_damage: float = 10.0
    enemy_contact_cooldown: float = 0.75

    spawner_radius: float = 27.0
    spawner_health: float = 50.0
    initial_spawner_count: int = 2
    max_spawner_count: int = 4
    spawner_base_interval: float = 2.4
    spawner_interval_phase_reduction: float = 0.14
    spawner_min_interval: float = 0.85
    max_enemies: int = 24

    phase_enemy_speed_growth: float = 0.08
    phase_enemy_health_growth: float = 0.10
    phase_spawner_health_growth: float = 0.12
    max_phase_for_observation: int = 10

    @property
    def max_episode_seconds(self) -> float:
        return self.max_steps * self.fixed_dt

    @classmethod
    def from_json(cls, path: str | Path) -> "ArenaConfig":
        """Load known fields from JSON and reject accidental misspellings."""

        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown ArenaConfig field(s): {', '.join(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
