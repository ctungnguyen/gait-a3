"""Data-only entities used by :mod:`arena.core` and the Pygame renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .math2d import Vec, vec


@dataclass(slots=True)
class Player:
    pos: Vec
    vel: Vec = field(default_factory=vec)
    angle: float = -math.pi / 2.0
    radius: float = 16.0
    max_health: float = 100.0
    health: float = 100.0
    shoot_cooldown: float = 0.0


@dataclass(slots=True)
class Enemy:
    entity_id: int
    pos: Vec
    vel: Vec = field(default_factory=vec)
    radius: float = 14.0
    max_health: float = 20.0
    health: float = 20.0
    max_speed: float = 105.0
    contact_cooldown: float = 0.0


@dataclass(slots=True)
class Spawner:
    entity_id: int
    pos: Vec
    radius: float = 27.0
    max_health: float = 50.0
    health: float = 50.0
    spawn_interval: float = 2.4
    spawn_timer: float = 1.0


@dataclass(slots=True)
class Projectile:
    entity_id: int
    pos: Vec
    vel: Vec
    radius: float = 4.0
    damage: float = 10.0
    lifetime: float = 2.4


@dataclass(slots=True, frozen=True)
class ArenaEvent:
    """One simulation fact that later reward logic can consume."""

    name: str
    amount: float = 0.0
    entity_id: int | None = None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {"name": self.name, "amount": self.amount, "entity_id": self.entity_id}


@dataclass(slots=True)
class StepOutcome:
    events: list[ArenaEvent]
    terminated: bool
    truncated: bool
