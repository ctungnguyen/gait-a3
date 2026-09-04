"""Pygame visualization for the open, continuous Neon Arena."""

from __future__ import annotations

import math
import random

import numpy as np

from .config import ArenaConfig
from .controls import ControlStyle
from .core import ArenaCore
from .math2d import EPSILON, length


class ArenaRenderer:
    BG = (12, 18, 30)
    BG_ALT = (18, 27, 43)
    GRID = (29, 48, 67)
    EDGE = (67, 106, 136)
    TEXT = (239, 246, 251)
    MUTED = (153, 174, 192)
    CYAN = (83, 215, 239)
    BLUE = (102, 145, 255)
    LIME = (91, 231, 151)
    CORAL = (242, 99, 113)
    GOLD = (255, 198, 75)
    VIOLET = (166, 117, 255)

    def __init__(self, config: ArenaConfig, render_mode: str):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on local setup
            raise RuntimeError(
                "Pygame is required for render(). Install dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from exc

        self.pg = pygame
        self.config = config
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        if render_mode == "human":
            self.surface = pygame.display.set_mode((config.width, config.height))
            pygame.display.set_caption("GAIT Part II - Neon Pursuit Arena")
        else:
            self.surface = pygame.Surface((config.width, config.height))
        self.font = pygame.font.SysFont("consolas", 17)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.large_font = pygame.font.SysFont("consolas", 34, bold=True)
        seeded = random.Random(7_311)
        self.stars = [
            (seeded.randrange(config.width), seeded.randrange(config.height), seeded.choice((1, 1, 1, 2)))
            for _ in range(115)
        ]

    def draw(
        self,
        core: ArenaCore,
        control_style: ControlStyle,
        status: str | None = None,
        debug: bool = False,
        action_label: str = "NO ACTION YET",
    ):
        pg = self.pg
        self._draw_background(core)

        for spawner in core.spawners:
            self._draw_spawner(spawner, core.elapsed_seconds)
        for projectile in core.projectiles:
            pg.draw.circle(
                self.surface,
                self.GOLD,
                self._point(projectile.pos),
                max(2, round(projectile.radius)),
            )
            pg.draw.circle(
                self.surface,
                (255, 242, 178),
                self._point(projectile.pos),
                max(1, round(projectile.radius / 2)),
            )
        for enemy in core.enemies:
            self._draw_enemy(enemy)
        self._draw_player(core)
        if debug:
            self._draw_debug_overlays(core)
        self._draw_hud(core, control_style)
        if debug:
            self._draw_debug_panel(core, action_label)

        if status:
            self._draw_status(status)

        if self.render_mode == "human":
            pg.display.flip()
            pg.event.pump()
            return None

        pixels = pg.surfarray.array3d(self.surface)
        return np.transpose(pixels, (1, 0, 2)).copy()

    def _draw_background(self, core: ArenaCore) -> None:
        pg = self.pg
        self.surface.fill(self.BG)

        # Broad horizontal bands and radar rings create an arena, not a tile map.
        for y in range(0, self.config.height, 80):
            color = self.BG_ALT if (y // 80) % 2 == 0 else self.BG
            pg.draw.rect(self.surface, color, (0, y, self.config.width, 80))
        for x, y, radius in self.stars:
            pg.draw.circle(self.surface, (45, 68, 91), (x, y), radius)

        center = (self.config.width // 2, self.config.height // 2)
        pulse = 3.0 * math.sin(core.elapsed_seconds * 1.8)
        for base_radius in (105, 215, 325):
            radius = max(1, round(base_radius + pulse))
            pg.draw.circle(self.surface, self.GRID, center, radius, 1)
        pg.draw.line(self.surface, self.GRID, (center[0], 0), (center[0], self.config.height), 1)
        pg.draw.line(self.surface, self.GRID, (0, center[1]), (self.config.width, center[1]), 1)
        pg.draw.rect(self.surface, self.EDGE, (4, 4, self.config.width - 8, self.config.height - 8), 2)

    def _draw_player(self, core: ArenaCore) -> None:
        pg = self.pg
        player = core.player
        angle = player.angle
        radius = player.radius
        nose = player.pos + self._unit(angle) * radius * 1.35
        left = player.pos + self._unit(angle + 2.45) * radius
        right = player.pos + self._unit(angle - 2.45) * radius
        rear = player.pos - self._unit(angle) * radius * 0.45

        glow_radius = round(radius * 1.55)
        glow = pg.Surface((glow_radius * 2 + 4, glow_radius * 2 + 4), pg.SRCALPHA)
        pg.draw.circle(glow, (*self.CYAN, 52), (glow_radius + 2, glow_radius + 2), glow_radius)
        self.surface.blit(
            glow,
            (
                round(float(player.pos[0]) - glow_radius - 2),
                round(float(player.pos[1]) - glow_radius - 2),
            ),
        )
        pg.draw.polygon(self.surface, self.CYAN, [self._point(nose), self._point(left), self._point(rear), self._point(right)])
        pg.draw.polygon(
            self.surface,
            (220, 251, 255),
            [self._point(nose), self._point(left), self._point(rear), self._point(right)],
            2,
        )
        pg.draw.circle(self.surface, self.BLUE, self._point(player.pos), 4)

    def _draw_enemy(self, enemy) -> None:
        pg = self.pg
        if length(enemy.vel) <= EPSILON:
            angle = 0.0
        else:
            angle = math.atan2(float(enemy.vel[1]), float(enemy.vel[0]))
        tip = enemy.pos + self._unit(angle) * enemy.radius * 1.15
        left = enemy.pos + self._unit(angle + 2.35) * enemy.radius
        right = enemy.pos + self._unit(angle - 2.35) * enemy.radius
        pg.draw.polygon(self.surface, self.CORAL, [self._point(tip), self._point(left), self._point(right)])
        pg.draw.circle(self.surface, (255, 181, 188), self._point(enemy.pos), 3)
        self._draw_health_bar(enemy.pos, enemy.radius + 9, enemy.health, enemy.max_health, self.CORAL)

    def _draw_spawner(self, spawner, elapsed: float) -> None:
        pg = self.pg
        center = self._point(spawner.pos)
        pulse = 1.0 + 0.07 * math.sin(elapsed * 4.0 + spawner.entity_id)
        outer_radius = spawner.radius * pulse
        rotation = elapsed * 0.8 + spawner.entity_id
        points = [
            (
                spawner.pos[0] + math.cos(rotation + index * math.tau / 6.0) * outer_radius,
                spawner.pos[1] + math.sin(rotation + index * math.tau / 6.0) * outer_radius,
            )
            for index in range(6)
        ]
        pg.draw.circle(self.surface, (72, 42, 101), center, round(spawner.radius * 1.35))
        pg.draw.polygon(self.surface, self.VIOLET, [self._point(point) for point in points], 4)
        pg.draw.circle(self.surface, (64, 29, 94), center, round(spawner.radius * 0.62))
        pg.draw.circle(self.surface, self.GOLD, center, 5)
        self._draw_health_bar(spawner.pos, spawner.radius + 12, spawner.health, spawner.max_health, self.VIOLET)

    def _draw_health_bar(self, pos, y_offset: float, value: float, maximum: float, color) -> None:
        pg = self.pg
        width, height = 36, 5
        x = round(float(pos[0]) - width / 2)
        y = round(float(pos[1]) - y_offset)
        pg.draw.rect(self.surface, (37, 43, 54), (x, y, width, height), border_radius=2)
        ratio = max(0.0, min(1.0, value / max(maximum, 1e-6)))
        pg.draw.rect(self.surface, color, (x, y, round(width * ratio), height), border_radius=2)

    def _draw_hud(self, core: ArenaCore, style: ControlStyle) -> None:
        pg = self.pg
        panel = pg.Surface((286, 166), pg.SRCALPHA)
        pg.draw.rect(panel, (9, 15, 26, 222), panel.get_rect(), border_radius=12)
        pg.draw.rect(panel, (*self.EDGE, 220), panel.get_rect(), 2, border_radius=12)
        self.surface.blit(panel, (16, 16))

        self._text("NEON PURSUIT ARENA", 31, 28, self.TEXT, self.font)
        self._text(f"PHASE {core.phase}", 31, 55, self.LIME, self.font)
        self._text(f"HP {core.player.health:5.1f} / {core.player.max_health:.0f}", 31, 81, self.CYAN, self.font)
        self._text(f"Enemies {len(core.enemies):02d}   Spawners {len(core.spawners):02d}", 31, 107, self.TEXT, self.small_font)
        self._text(f"Step {core.step_count:04d}   Time {core.elapsed_seconds:6.1f}s", 31, 130, self.MUTED, self.small_font)
        self._text(f"Control: {style.value.upper()}", 31, 151, self.MUTED, self.small_font)

        if core.last_events:
            latest = core.last_events[-1].name.replace("_", " ").upper()
            self._text(latest, self.config.width - 20, 24, self.GOLD, self.small_font, align="right")

        controls = (
            "W/UP thrust | A/D turn | SPACE shoot"
            if style is ControlStyle.ROTATION
            else "WASD/arrows move | SPACE shoot"
        )
        self._text(controls, 18, self.config.height - 42, self.TEXT, self.small_font)
        self._text(
            "P pause | F3 debug | R reset | ESC exit",
            18,
            self.config.height - 22,
            self.MUTED,
            self.small_font,
        )

    def _draw_debug_overlays(self, core: ArenaCore) -> None:
        """Draw exact simulation geometry without changing the world state."""

        pg = self.pg
        player = core.player

        # Collision circles and player steering vectors.
        pg.draw.circle(self.surface, self.CYAN, self._point(player.pos), round(player.radius), 1)
        self._draw_vector(player.pos, player.vel, self.CYAN, scale=0.34)
        self._draw_vector(player.pos, self._unit(player.angle) * 78.0, self.GOLD, scale=1.0)

        nearest_enemy = core.nearest_enemy()
        nearest_spawner = core.nearest_spawner()
        if nearest_enemy is not None:
            pg.draw.line(
                self.surface,
                self.CORAL,
                self._point(player.pos),
                self._point(nearest_enemy.pos),
                2,
            )
        if nearest_spawner is not None:
            pg.draw.line(
                self.surface,
                self.VIOLET,
                self._point(player.pos),
                self._point(nearest_spawner.pos),
                2,
            )

        # Show exact pursuit calculations for the nearest enemies. Limiting the
        # rays keeps later phases readable while retaining useful evidence.
        ordered_enemies = sorted(
            core.enemies,
            key=lambda enemy: length(enemy.pos - player.pos),
        )
        visualized_ids = {enemy.entity_id for enemy in ordered_enemies[:8]}
        for enemy in core.enemies:
            pg.draw.circle(
                self.surface,
                (255, 151, 163),
                self._point(enemy.pos),
                round(enemy.radius),
                1,
            )
            self._draw_vector(enemy.pos, enemy.vel, self.CORAL, scale=0.25)
            self._text(
                f"E{enemy.entity_id}",
                round(float(enemy.pos[0]) + enemy.radius + 3),
                round(float(enemy.pos[1]) + 3),
                self.CORAL,
                self.small_font,
            )
            if enemy.entity_id not in visualized_ids:
                continue
            target, _look_ahead = core.enemy_pursuit_target(enemy)
            pg.draw.line(
                self.surface,
                (116, 111, 174),
                self._point(enemy.pos),
                self._point(target),
                1,
            )
            self._draw_cross(target, self.LIME, radius=5)

        for spawner in core.spawners:
            pg.draw.circle(
                self.surface,
                (207, 180, 255),
                self._point(spawner.pos),
                round(spawner.radius),
                1,
            )
            spawn_in = max(0.0, spawner.spawn_timer)
            self._text(
                f"S{spawner.entity_id} {spawn_in:.1f}s",
                round(float(spawner.pos[0]) + spawner.radius + 4),
                round(float(spawner.pos[1]) + 5),
                self.VIOLET,
                self.small_font,
            )

        for projectile in core.projectiles:
            pg.draw.circle(
                self.surface,
                (255, 240, 168),
                self._point(projectile.pos),
                max(2, round(projectile.radius)),
                1,
            )
            self._draw_vector(projectile.pos, projectile.vel, self.GOLD, scale=0.08)

    def _draw_debug_panel(self, core: ArenaCore, action_label: str) -> None:
        pg = self.pg
        width, height = 346, 322
        x = self.config.width - width - 16
        y = 48
        panel = pg.Surface((width, height), pg.SRCALPHA)
        pg.draw.rect(panel, (9, 15, 26, 232), panel.get_rect(), border_radius=12)
        pg.draw.rect(panel, (*self.LIME, 225), panel.get_rect(), 2, border_radius=12)
        self.surface.blit(panel, (x, y))

        player = core.player
        speed = length(player.vel)
        heading_degrees = math.degrees(player.angle) % 360.0
        enemy = core.nearest_enemy()
        spawner = core.nearest_spawner()

        lines: list[tuple[str, tuple[int, int, int]]] = [
            ("DEBUG MODE  /  F3 TOGGLE", self.LIME),
            (f"Action       {action_label}", self.GOLD),
            (f"Player pos   ({player.pos[0]:6.1f}, {player.pos[1]:6.1f})", self.TEXT),
            (f"Velocity     ({player.vel[0]:6.1f}, {player.vel[1]:6.1f})", self.TEXT),
            (f"Speed        {speed:6.1f} px/s", self.CYAN),
            (f"Heading      {heading_degrees:6.1f} deg", self.CYAN),
            (f"Fire CD      {player.shoot_cooldown:6.2f} s", self.TEXT),
        ]

        if enemy is None:
            lines.extend(
                (
                    ("Nearest E    none", self.MUTED),
                    ("Pursuit      waiting for spawn", self.MUTED),
                )
            )
        else:
            target, look_ahead = core.enemy_pursuit_target(enemy)
            enemy_distance = length(enemy.pos - player.pos)
            lines.extend(
                (
                    (
                        f"Nearest E{enemy.entity_id:<3} d={enemy_distance:6.1f} hp={enemy.health:4.0f}",
                        self.CORAL,
                    ),
                    (
                        f"Pursuit      +{look_ahead:.2f}s -> ({target[0]:.0f},{target[1]:.0f})",
                        self.LIME,
                    ),
                )
            )

        if spawner is None:
            lines.append(("Nearest S    none / phase transition", self.MUTED))
        else:
            spawner_distance = length(spawner.pos - player.pos)
            lines.append(
                (
                    f"Nearest S{spawner.entity_id:<3} d={spawner_distance:6.1f} hp={spawner.health:4.0f}",
                    self.VIOLET,
                )
            )

        lines.extend(
            (
                (
                    f"Entities     E:{len(core.enemies):02d} S:{len(core.spawners):02d} "
                    f"P:{len(core.projectiles):02d}",
                    self.TEXT,
                ),
                (f"Episode      step {core.step_count}/{core.config.max_steps}", self.MUTED),
                ("Rings=colliders  arrows=velocity", self.MUTED),
                ("Gold=heading  green=pursuit target", self.MUTED),
            )
        )

        for index, (message, color) in enumerate(lines):
            font = self.font if index == 0 else self.small_font
            self._text(message, x + 15, y + 13 + index * 21, color, font)

    def _draw_vector(self, origin, vector, color, scale: float) -> None:
        vector_length = length(vector)
        if vector_length <= EPSILON:
            return
        pg = self.pg
        end = origin + vector * scale
        pg.draw.line(self.surface, color, self._point(origin), self._point(end), 2)
        angle = math.atan2(float(vector[1]), float(vector[0]))
        left = end - self._unit(angle - 0.55) * 7.0
        right = end - self._unit(angle + 0.55) * 7.0
        pg.draw.line(self.surface, color, self._point(end), self._point(left), 2)
        pg.draw.line(self.surface, color, self._point(end), self._point(right), 2)

    def _draw_cross(self, position, color, radius: int) -> None:
        pg = self.pg
        x, y = self._point(position)
        pg.draw.line(self.surface, color, (x - radius, y), (x + radius, y), 2)
        pg.draw.line(self.surface, color, (x, y - radius), (x, y + radius), 2)

    def _draw_status(self, status: str) -> None:
        pg = self.pg
        veil = pg.Surface((self.config.width, self.config.height), pg.SRCALPHA)
        veil.fill((4, 8, 15, 145))
        self.surface.blit(veil, (0, 0))
        image = self.large_font.render(status, True, self.TEXT)
        rect = image.get_rect(center=(self.config.width // 2, self.config.height // 2))
        self.surface.blit(image, rect)

    def _text(self, text, x, y, color, font, align: str = "left") -> None:
        image = font.render(str(text), True, color)
        rect = image.get_rect()
        if align == "right":
            rect.topright = (x, y)
        else:
            rect.topleft = (x, y)
        self.surface.blit(image, rect)

    @staticmethod
    def _unit(angle: float) -> np.ndarray:
        return np.array((math.cos(angle), math.sin(angle)), dtype=np.float32)

    @staticmethod
    def _point(value) -> tuple[int, int]:
        return round(float(value[0])), round(float(value[1]))

    def close(self) -> None:
        if self.render_mode == "human":
            self.pg.display.quit()
