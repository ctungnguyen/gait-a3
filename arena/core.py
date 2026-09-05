"""Headless, deterministic Arena simulation for Part II Block 1.

The module deliberately does not import Pygame or Gymnasium. That separation
allows Stable Baselines3 to train quickly without opening a window while the
same state can still be rendered by :mod:`arena.renderer` during evaluation.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np

from .config import ArenaConfig
from .controls import ControlCommand, ControlStyle
from .entities import ArenaEvent, Enemy, Player, Projectile, Spawner, StepOutcome
from .math2d import (
    EPSILON,
    circle_overlap,
    clamp_to_arena,
    forward,
    length,
    limited,
    normalized,
    vec,
)


class ArenaCore:
    """Own all gameplay state and advance it by one fixed simulation step."""

    _SPAWNER_ANCHORS = (
        # Keep the first anchor clear of the diagnostic HUD while preserving
        # a symmetric, open-arena layout for every later phase.
        (0.16, 0.36),
        (0.84, 0.82),
        (0.84, 0.18),
        (0.16, 0.82),
    )

    def __init__(self, config: ArenaConfig | None = None, seed: int | None = None):
        self.config = config or ArenaConfig()
        self._initial_seed = seed
        self._rng = random.Random(seed)
        self._next_id = 1
        self.player: Player
        self.enemies: list[Enemy]
        self.spawners: list[Spawner]
        self.projectiles: list[Projectile]
        self.phase = 1
        self.step_count = 0
        self.elapsed_seconds = 0.0
        self.last_events: list[ArenaEvent] = []
        self.reset(seed=seed)

    def _id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def reset(self, seed: int | None = None) -> None:
        """Restore a complete episode and optionally reseed all randomness."""

        if seed is not None:
            self._initial_seed = seed
            self._rng.seed(seed)

        cfg = self.config
        self._next_id = 1
        self.phase = 1
        self.step_count = 0
        self.elapsed_seconds = 0.0
        self.last_events = []
        self.player = Player(
            pos=vec(cfg.width / 2.0, cfg.height / 2.0),
            radius=cfg.player_radius,
            max_health=cfg.player_health,
            health=cfg.player_health,
        )
        self.enemies = []
        self.spawners = []
        self.projectiles = []
        self._spawn_phase_spawners()

    def step(
        self,
        command: ControlCommand,
        style: ControlStyle | str,
        dt: float | None = None,
    ) -> StepOutcome:
        """Advance the entire world once and return events plus end flags."""

        style = ControlStyle(style)
        dt = self.config.fixed_dt if dt is None else float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive number")

        self.step_count += 1
        self.elapsed_seconds += dt
        events: list[ArenaEvent] = []

        self.player.shoot_cooldown = max(0.0, self.player.shoot_cooldown - dt)
        for enemy in self.enemies:
            enemy.contact_cooldown = max(0.0, enemy.contact_cooldown - dt)

        self._move_player(command, style, dt)
        if command.shoot:
            self._try_fire(events)

        self._update_spawners(dt, events)
        self._update_enemies(dt)
        self._update_projectiles(dt)
        self._resolve_projectile_hits(events)
        self._resolve_enemy_contacts(events)

        terminated = self.player.health <= 0.0
        if terminated and not any(event.name == "player_died" for event in events):
            events.append(ArenaEvent("player_died"))

        if not terminated and not self.spawners:
            self.phase += 1
            events.append(ArenaEvent("phase_advanced", amount=float(self.phase)))
            self._spawn_phase_spawners()

        truncated = self.step_count >= self.config.max_steps
        if truncated:
            events.append(ArenaEvent("time_limit_reached", amount=self.elapsed_seconds))

        self.last_events = events
        return StepOutcome(events=events, terminated=terminated, truncated=truncated)

    # ------------------------------------------------------------------
    # Player control
    # ------------------------------------------------------------------

    def _move_player(self, command: ControlCommand, style: ControlStyle, dt: float) -> None:
        cfg = self.config
        player = self.player

        # SHOOT is a separate rubric action, but it must not secretly behave as
        # a brake. A shoot-only command keeps the current velocity and lets the
        # ship coast while the projectile is created.
        if command.preserve_momentum:
            pass
        elif style is ControlStyle.ROTATION:
            player.angle += float(command.turn) * cfg.player_rotation_speed * dt
            if command.thrust > 0.0:
                player.vel += (
                    forward(player.angle)
                    * cfg.player_thrust_acceleration
                    * min(1.0, float(command.thrust))
                    * dt
                )
            drag = cfg.player_drag_per_30hz_step ** (dt / max(cfg.fixed_dt, EPSILON))
            player.vel *= drag
        else:
            direction = normalized(command.move)
            desired_velocity = direction * cfg.player_max_speed
            maximum_change = cfg.player_direct_acceleration * dt
            player.vel += limited(desired_velocity - player.vel, maximum_change)
            if length(direction) > EPSILON:
                player.angle = math.atan2(float(direction[1]), float(direction[0]))

        player.vel = limited(player.vel, cfg.player_max_speed)
        player.pos += player.vel * dt
        self._clamp_moving_entity(player.pos, player.vel, player.radius)

    def _try_fire(self, events: list[ArenaEvent]) -> None:
        cfg = self.config
        if self.player.shoot_cooldown > 0.0:
            return
        heading = forward(self.player.angle)
        projectile = Projectile(
            entity_id=self._id(),
            pos=self.player.pos.copy() + heading * (self.player.radius + cfg.projectile_radius + 2.0),
            vel=heading * cfg.projectile_speed + self.player.vel * 0.20,
            radius=cfg.projectile_radius,
            damage=cfg.projectile_damage,
            lifetime=cfg.projectile_lifetime,
        )
        self.projectiles.append(projectile)
        self.player.shoot_cooldown = cfg.player_shoot_cooldown
        events.append(ArenaEvent("projectile_fired", entity_id=projectile.entity_id))

    # ------------------------------------------------------------------
    # Spawners, phases, and enemy steering
    # ------------------------------------------------------------------

    def _phase_spawner_count(self) -> int:
        cfg = self.config
        growth = (self.phase - 1) // 2
        return min(cfg.max_spawner_count, cfg.initial_spawner_count + growth)

    def _spawn_phase_spawners(self) -> None:
        cfg = self.config
        count = self._phase_spawner_count()
        health = cfg.spawner_health * (1.0 + (self.phase - 1) * cfg.phase_spawner_health_growth)
        interval = max(
            cfg.spawner_min_interval,
            cfg.spawner_base_interval - (self.phase - 1) * cfg.spawner_interval_phase_reduction,
        )

        offset = (self.phase - 1) % len(self._SPAWNER_ANCHORS)
        ordered = self._SPAWNER_ANCHORS[offset:] + self._SPAWNER_ANCHORS[:offset]
        for index, (x_fraction, y_fraction) in enumerate(ordered[:count]):
            self.spawners.append(
                Spawner(
                    entity_id=self._id(),
                    pos=vec(cfg.width * x_fraction, cfg.height * y_fraction),
                    radius=cfg.spawner_radius,
                    max_health=health,
                    health=health,
                    spawn_interval=interval,
                    spawn_timer=0.65 + index * 0.28,
                )
            )

    def _update_spawners(self, dt: float, events: list[ArenaEvent]) -> None:
        if len(self.enemies) >= self.config.max_enemies:
            return
        for spawner in self.spawners:
            spawner.spawn_timer -= dt
            if spawner.spawn_timer > 0.0 or len(self.enemies) >= self.config.max_enemies:
                continue
            angle = self._rng.random() * math.tau
            direction = forward(angle)
            spawn_pos = spawner.pos + direction * (spawner.radius + self.config.enemy_radius + 6.0)
            clamp_to_arena(
                spawn_pos,
                self.config.enemy_radius,
                self.config.width,
                self.config.height,
            )
            health = self.config.enemy_health * (
                1.0 + (self.phase - 1) * self.config.phase_enemy_health_growth
            )
            speed = self.config.enemy_max_speed * (
                1.0 + (self.phase - 1) * self.config.phase_enemy_speed_growth
            )
            enemy = Enemy(
                entity_id=self._id(),
                pos=spawn_pos,
                vel=direction * speed * 0.25,
                radius=self.config.enemy_radius,
                max_health=health,
                health=health,
                max_speed=speed,
            )
            self.enemies.append(enemy)
            events.append(ArenaEvent("enemy_spawned", entity_id=enemy.entity_id))
            spawner.spawn_timer += spawner.spawn_interval

    def _update_enemies(self, dt: float) -> None:
        cfg = self.config
        for enemy in self.enemies:
            predicted_player, _prediction = self.enemy_pursuit_target(enemy)
            desired = normalized(predicted_player - enemy.pos) * enemy.max_speed
            steering = limited(desired - enemy.vel, cfg.enemy_max_acceleration * dt)
            enemy.vel = limited(enemy.vel + steering, enemy.max_speed)
            enemy.pos += enemy.vel * dt
            self._clamp_moving_entity(enemy.pos, enemy.vel, enemy.radius)

    # ------------------------------------------------------------------
    # Projectile and contact collision rules
    # ------------------------------------------------------------------

    def _update_projectiles(self, dt: float) -> None:
        cfg = self.config
        survivors: list[Projectile] = []
        for projectile in self.projectiles:
            projectile.pos += projectile.vel * dt
            projectile.lifetime -= dt
            inside = (
                -projectile.radius <= projectile.pos[0] <= cfg.width + projectile.radius
                and -projectile.radius <= projectile.pos[1] <= cfg.height + projectile.radius
            )
            if inside and projectile.lifetime > 0.0:
                survivors.append(projectile)
        self.projectiles = survivors

    def _resolve_projectile_hits(self, events: list[ArenaEvent]) -> None:
        consumed: set[int] = set()
        dead_enemies: set[int] = set()
        dead_spawners: set[int] = set()

        for projectile in self.projectiles:
            for enemy in self.enemies:
                if enemy.entity_id in dead_enemies:
                    continue
                if circle_overlap(projectile.pos, projectile.radius, enemy.pos, enemy.radius):
                    enemy.health -= projectile.damage
                    consumed.add(projectile.entity_id)
                    events.append(
                        ArenaEvent("enemy_damaged", amount=projectile.damage, entity_id=enemy.entity_id)
                    )
                    if enemy.health <= 0.0:
                        dead_enemies.add(enemy.entity_id)
                        events.append(ArenaEvent("enemy_destroyed", entity_id=enemy.entity_id))
                    break

            if projectile.entity_id in consumed:
                continue

            for spawner in self.spawners:
                if spawner.entity_id in dead_spawners:
                    continue
                if circle_overlap(projectile.pos, projectile.radius, spawner.pos, spawner.radius):
                    spawner.health -= projectile.damage
                    consumed.add(projectile.entity_id)
                    events.append(
                        ArenaEvent("spawner_damaged", amount=projectile.damage, entity_id=spawner.entity_id)
                    )
                    if spawner.health <= 0.0:
                        dead_spawners.add(spawner.entity_id)
                        events.append(ArenaEvent("spawner_destroyed", entity_id=spawner.entity_id))
                    break

        if consumed:
            self.projectiles = [p for p in self.projectiles if p.entity_id not in consumed]
        if dead_enemies:
            self.enemies = [e for e in self.enemies if e.entity_id not in dead_enemies]
        if dead_spawners:
            self.spawners = [s for s in self.spawners if s.entity_id not in dead_spawners]

    def _resolve_enemy_contacts(self, events: list[ArenaEvent]) -> None:
        cfg = self.config
        for enemy in self.enemies:
            if not circle_overlap(enemy.pos, enemy.radius, self.player.pos, self.player.radius):
                continue

            away = enemy.pos - self.player.pos
            if length(away) <= EPSILON:
                away = vec(1.0, 0.0)
            away = normalized(away)
            overlap = enemy.radius + self.player.radius - length(enemy.pos - self.player.pos)
            if overlap > 0.0:
                enemy.pos += away * (overlap + 1.0)
                enemy.vel += away * 40.0
                clamp_to_arena(enemy.pos, enemy.radius, cfg.width, cfg.height)

            if enemy.contact_cooldown <= 0.0:
                damage = min(cfg.enemy_contact_damage, self.player.health)
                self.player.health = max(0.0, self.player.health - damage)
                enemy.contact_cooldown = cfg.enemy_contact_cooldown
                events.append(ArenaEvent("player_damaged", amount=damage, entity_id=enemy.entity_id))

    # ------------------------------------------------------------------
    # State access used by observations, rewards, diagnostics, and tests
    # ------------------------------------------------------------------

    def nearest_enemy(self) -> Enemy | None:
        return self._nearest(self.enemies)

    def nearest_spawner(self) -> Spawner | None:
        return self._nearest(self.spawners)

    def enemy_pursuit_target(self, enemy: Enemy) -> tuple[np.ndarray, float]:
        """Return the exact predicted target and look-ahead used by pursuit."""

        distance = length(self.player.pos - enemy.pos)
        prediction = min(
            self.config.enemy_prediction_time,
            distance / max(enemy.max_speed + length(self.player.vel), 1.0),
        )
        return self.player.pos + self.player.vel * prediction, prediction

    def _nearest(self, entities: Iterable[Enemy | Spawner]):
        return min(entities, key=lambda entity: length(entity.pos - self.player.pos), default=None)

    def info(self, outcome: StepOutcome | None = None) -> dict:
        events = self.last_events if outcome is None else outcome.events
        return {
            "phase": self.phase,
            "step": self.step_count,
            "elapsed_seconds": self.elapsed_seconds,
            "player_health": self.player.health,
            "enemy_count": len(self.enemies),
            "spawner_count": len(self.spawners),
            "events": [event.as_dict() for event in events],
        }

    def assert_invariants(self) -> None:
        """Raise immediately if integration code corrupts core game state."""

        cfg = self.config
        all_entities = [*self.enemies, *self.spawners, *self.projectiles]
        ids = [entity.entity_id for entity in all_entities]
        if len(ids) != len(set(ids)):
            raise AssertionError("Entity IDs must be unique")
        if not 0.0 <= self.player.health <= self.player.max_health:
            raise AssertionError("Player health is outside its valid range")
        if not (
            self.player.radius <= self.player.pos[0] <= cfg.width - self.player.radius
            and self.player.radius <= self.player.pos[1] <= cfg.height - self.player.radius
        ):
            raise AssertionError("Player escaped the Arena bounds")

    def _clamp_moving_entity(self, position: np.ndarray, velocity: np.ndarray, radius: float) -> None:
        before = position.copy()
        clamp_to_arena(position, radius, self.config.width, self.config.height)
        if position[0] != before[0]:
            velocity[0] = 0.0
        if position[1] != before[1]:
            velocity[1] = 0.0
