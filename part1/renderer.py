"""Interactive Pygame presentation for trained Part I policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from shared import Theme, get_theme

from .agents import BaseTabularAgent
from .config import GridworldSettings
from .environment import ACTION_NAMES, GridWorld


SPRITE_FILES = {
    "agent": "agent.png",
    "rock": "rock.png",
    "fire": "lava.png",
    "apple": "apple.png",
    "key": "key.png",
    "chest_closed": "chest_locked.png",
    "chest_open": "chest_open.png",
    "monster": "monster.png",
}


@dataclass
class DemoState:
    algorithm: str
    intrinsic: bool
    action: int | None = None
    episode: int = 1
    environment_return: float = 0.0
    step: int = 0
    status: str = "LEARNED POLICY"
    event: str = "reset"
    paused: bool = False
    detail: bool = True


class GridworldRenderer:
    PANEL_WIDTH = 430
    MIN_HEIGHT = 720

    def __init__(self, environment: GridWorld, settings: GridworldSettings, theme: str = "neon") -> None:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - setup dependent
            raise RuntimeError("Install pygame with: python -m pip install -r requirements.txt") from exc
        self.pg = pygame
        self.environment = environment
        self.settings = settings
        self.tile = settings.tile_size
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.map_width = environment.width * self.tile
        self.width = self.map_width + self.PANEL_WIDTH
        self.height = max(environment.height * self.tile, self.MIN_HEIGHT)
        pygame.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("GAIT A3 - Part I Classical RL Gridworld")
        self.title_font = pygame.font.SysFont("consolas", 28, bold=True)
        self.font = pygame.font.SysFont("consolas", 17)
        self.small = pygame.font.SysFont("consolas", 14)
        self.sprites = self._load_sprites()

    def set_theme(self, name: str) -> None:
        self.theme_name = name
        self.theme = get_theme(name)

    def close(self) -> None:
        self.pg.display.quit()

    def draw(
        self,
        agent: BaseTabularAgent,
        demo: DemoState,
        trail: list[tuple[int, int]],
    ) -> None:
        pg = self.pg
        theme = self.theme
        self.surface.fill(theme.background)
        self._draw_map(trail)
        self._draw_panel(agent, demo)
        # A terminal result is more important than the paused state that keeps
        # it on screen.  The result remains visible until R/RESET is pressed.
        if demo.status != "LEARNED POLICY":
            color = theme.success if demo.status == "OBJECTIVES COMPLETE" else theme.danger
            self._status_badge(demo.status, color)
        elif demo.paused:
            self._status_badge("PAUSED - N: SINGLE STEP", theme.warning)
        pg.display.flip()

    def control_at(self, position: tuple[int, int]) -> str | None:
        """Return the clicked visual-control ID, if any."""

        for control, _label, rect in self._control_buttons():
            if rect.collidepoint(position):
                return control
        return None

    def draw_training_progress(
        self,
        current: int,
        total: int,
        algorithm: str,
        level: int,
        success: int,
    ) -> None:
        pg = self.pg
        theme = self.theme
        self.surface.fill(theme.background)
        self._text("TRAINING TABULAR AGENT", 60, 90, theme.text, self.title_font)
        self._text(
            f"Level {level}  |  {algorithm.replace('_', ' ').upper()}",
            60, 145, theme.primary, self.font,
        )
        ratio = current / max(total, 1)
        box = pg.Rect(60, 215, self.width - 120, 34)
        pg.draw.rect(self.surface, theme.panel, box, border_radius=10)
        fill = box.copy()
        fill.width = round(box.width * ratio)
        pg.draw.rect(self.surface, theme.success, fill, border_radius=10)
        pg.draw.rect(self.surface, theme.edge, box, 2, border_radius=10)
        self._text(f"{current:,} / {total:,} episodes ({ratio:.0%})", 60, 270, theme.text, self.font)
        self._text(f"Latest episode success: {'YES' if success else 'NO'}", 60, 310, theme.muted, self.font)
        self._text("Training runs headless for speed; the learned greedy policy is animated next.", 60, 380, theme.muted, self.small)
        pg.display.flip()
        pg.event.pump()

    def _draw_map(self, trail: list[tuple[int, int]]) -> None:
        pg = self.pg
        env = self.environment
        theme = self.theme
        map_rect = pg.Rect(0, 0, self.map_width, self.height)
        pg.draw.rect(self.surface, theme.background_alt, map_rect)

        for y in range(env.height):
            for x in range(env.width):
                rect = pg.Rect(x * self.tile, y * self.tile, self.tile, self.tile)
                if (x + y) % 2 == 0:
                    pg.draw.rect(self.surface, (*theme.panel,), rect)
                pg.draw.rect(self.surface, theme.grid, rect, 1)

        if len(trail) > 1:
            points = [self._center(position) for position in trail]
            pg.draw.lines(self.surface, (*theme.primary,), False, points, 4)

        for position in env.rocks:
            if self._blit_sprite("rock", position):
                continue
            rect = self._tile_rect(position, inset=5)
            pg.draw.rect(self.surface, theme.muted, rect, border_radius=7)
            pg.draw.line(self.surface, theme.edge, rect.topleft, rect.bottomright, 3)
            pg.draw.line(self.surface, theme.edge, rect.topright, rect.bottomleft, 3)
        for position in env.fires:
            if self._blit_sprite("fire", position):
                continue
            cx, cy = self._center(position)
            points = [
                (cx, cy - self.tile * 0.34),
                (cx + self.tile * 0.26, cy + self.tile * 0.30),
                (cx, cy + self.tile * 0.16),
                (cx - self.tile * 0.26, cy + self.tile * 0.30),
            ]
            pg.draw.polygon(self.surface, theme.danger, points)
            pg.draw.circle(self.surface, theme.warning, (cx, cy + 7), max(3, self.tile // 10))

        for position, index in env.apple_index.items():
            if (env.apple_mask >> index) & 1:
                if self._blit_sprite("apple", position):
                    continue
                center = self._center(position)
                pg.draw.circle(self.surface, theme.success, center, self.tile // 4)
                pg.draw.line(
                    self.surface,
                    theme.warning,
                    (center[0], center[1] - 13),
                    (center[0] + 5, center[1] - 20),
                    3,
                )
        if env.key_pos is not None and not env.has_key:
            if not self._blit_sprite("key", env.key_pos):
                cx, cy = self._center(env.key_pos)
                pg.draw.circle(self.surface, theme.warning, (cx - 7, cy), self.tile // 7, 4)
                pg.draw.line(self.surface, theme.warning, (cx, cy), (cx + 18, cy), 5)
                pg.draw.line(self.surface, theme.warning, (cx + 12, cy), (cx + 12, cy + 8), 4)
        if env.chest_pos is not None:
            sprite_name = "chest_open" if env.chest_opened else "chest_closed"
            if not self._blit_sprite(sprite_name, env.chest_pos):
                color = theme.muted if env.chest_opened else theme.accent
                rect = self._tile_rect(env.chest_pos, inset=10)
                pg.draw.rect(self.surface, color, rect, border_radius=5)
                pg.draw.line(self.surface, theme.warning, rect.midleft, rect.midright, 3)

        for mx, my in env.monsters:
            if self._blit_sprite("monster", (mx, my)):
                continue
            cx, cy = self._center((mx, my))
            radius = self.tile // 3
            pg.draw.circle(self.surface, theme.danger, (cx, cy), radius)
            pg.draw.circle(self.surface, theme.text, (cx - 9, cy - 5), 4)
            pg.draw.circle(self.surface, theme.text, (cx + 9, cy - 5), 4)
            pg.draw.line(
                self.surface,
                theme.background,
                (cx - 11, cy + 10),
                (cx + 11, cy + 10),
                3,
            )

        bob = -2 if (pg.time.get_ticks() // 220) % 2 else 0
        if not self._blit_sprite("agent", env.agent, y_offset=bob):
            cx, cy = self._center(env.agent)
            cy += bob
            pg.draw.circle(self.surface, theme.primary, (cx, cy), self.tile // 3)
            pg.draw.circle(self.surface, theme.text, (cx - 9, cy - 5), 5)
            pg.draw.circle(self.surface, theme.text, (cx + 9, cy - 5), 5)
            pg.draw.arc(
                self.surface,
                theme.background,
                pg.Rect(cx - 13, cy - 2, 26, 18),
                math.radians(10),
                math.radians(170),
                3,
            )
        pg.draw.rect(self.surface, theme.edge, (0, 0, self.map_width, self.height), 3)

    def _draw_panel(self, agent: BaseTabularAgent, demo: DemoState) -> None:
        pg = self.pg
        env = self.environment
        theme = self.theme
        left = self.map_width
        panel = pg.Rect(left, 0, self.PANEL_WIDTH, self.height)
        pg.draw.rect(self.surface, theme.panel, panel)
        pg.draw.line(self.surface, theme.edge, (left, 0), (left, self.height), 3)
        x = left + 24
        self._text("PART I / GRIDWORLD", x, 25, theme.text, self.title_font)
        self._text(env.level.task.upper(), x, 68, theme.primary, self.small)
        self._text(f"LEVEL {env.level.number}: {env.level.name}", x, 98, theme.success, self.font)
        difficulty = "BEGINNER" if env.level.number <= 1 else "INTERMEDIATE" if env.level.number <= 3 else "ADVANCED"
        self._text(f"{demo.algorithm.replace('_', ' ').upper()}  |  {difficulty}", x, 127, theme.text, self.font)

        self._section("LIVE EVIDENCE", x, 170)
        self._text(f"Policy: GREEDY / epsilon = 0.000", x, 198, theme.warning, self.small)
        action_name = "--" if demo.action is None else ACTION_NAMES[demo.action]
        self._text(f"Action: {action_name}", x, 222, theme.primary, self.font)
        self._text(f"Step: {demo.step:03d}    Env return: {demo.environment_return:.1f}", x, 249, theme.text, self.small)
        self._text(f"Event: {demo.event.replace('_', ' ').upper()}", x, 273, theme.muted, self.small)
        self._text(f"Apples left: {env.remaining_apples()}    Key: {env.has_key}    Chest: {env.chest_opened}", x, 297, theme.text, self.small)
        self._text(f"Monster move chance: {env.monster_move_chance:.0%}", x, 321, theme.danger, self.small)

        if demo.detail:
            state_values = agent.qtable.values(env.encode_state())
            self._section("CURRENT Q(s, a)", x, 358)
            for index, (name, value) in enumerate(zip(ACTION_NAMES, state_values)):
                color = theme.success if demo.action == index else theme.muted
                self._text(f"{index} {name:<5}  {value:8.4f}", x, 386 + index * 22, color, self.small)

            y = 490
            self._section("ALGORITHM", x, y)
            if demo.algorithm == "q_learning":
                lines = ("Off-policy target:", "r + gamma max Q(s', a')")
            else:
                lines = ("On-policy target:", "r + gamma Q(s', chosen a')")
            if demo.intrinsic:
                lines += ("Intrinsic update bonus:", "strength / sqrt(n(s) + 1)")
            for index, line in enumerate(lines):
                self._text(line, x, y + 28 + index * 20, theme.muted, self.small)
        else:
            self._section("COMPACT VIEW", x, 358)
            for index, line in enumerate(
                (env.level.description, "Press D to restore Q-values and equations.")
            ):
                # Wrap the long level description at a predictable character count.
                words = line.split()
                rows: list[str] = []
                current = ""
                for word in words:
                    candidate = f"{current} {word}".strip()
                    if len(candidate) > 42 and current:
                        rows.append(current)
                        current = word
                    else:
                        current = candidate
                if current:
                    rows.append(current)
                for row_index, row in enumerate(rows):
                    self._text(row, x, 390 + index * 82 + row_index * 20, theme.muted, self.small)

        if env.level.number == 6:
            message = (
                "COUNT BONUS ACTIVE: sparse exploration aid"
                if demo.intrinsic
                else "BASELINE: first external reward is 61+ steps away"
            )
            self._text(message, x, self.height - 108, theme.warning, self.small)

        pointer = pg.mouse.get_pos()
        for control, label, rect in self._control_buttons():
            active = (control == "pause" and demo.paused) or rect.collidepoint(pointer)
            background = theme.primary if active else theme.background_alt
            foreground = theme.background if active else theme.text
            pg.draw.rect(self.surface, background, rect, border_radius=7)
            pg.draw.rect(self.surface, theme.edge, rect, 1, border_radius=7)
            rendered = self.small.render(label, True, foreground)
            self.surface.blit(rendered, rendered.get_rect(center=rect.center))

    def _control_buttons(self):
        left = self.map_width + 18
        gap = 7
        width = (self.PANEL_WIDTH - 36 - gap * 3) // 4
        height = 31
        rows = (
            (("pause", "P PAUSE"), ("step", "N STEP"), ("replay", "R RESET"), ("theme", "T THEME")),
            (("details", "D INFO"), ("slower", "- SLOW"), ("faster", "+ FAST"), ("menu", "ESC MENU")),
        )
        buttons = []
        for row_index, row in enumerate(rows):
            y = self.height - 79 + row_index * 39
            for column, (control, label) in enumerate(row):
                rect = self.pg.Rect(left + column * (width + gap), y, width, height)
                buttons.append((control, label, rect))
        return buttons

    def _load_sprites(self):
        """Load the user-supplied pixel artwork with safe primitive fallbacks."""

        directory = Path(__file__).resolve().parent / "assets" / "sprites"
        atlas = {}
        maximum = max(8, self.tile - 6)
        for name, filename in SPRITE_FILES.items():
            path = directory / filename
            try:
                source = self.pg.image.load(str(path)).convert_alpha()
            except (FileNotFoundError, self.pg.error):
                atlas[name] = None
                continue
            width, height = source.get_size()
            scale = min(maximum / max(width, 1), maximum / max(height, 1))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            rendered = self.pg.transform.scale(source, size)
            tile_surface = self.pg.Surface((self.tile, self.tile), self.pg.SRCALPHA)
            tile_surface.blit(rendered, rendered.get_rect(center=(self.tile // 2, self.tile // 2)))
            atlas[name] = tile_surface
        return atlas

    def _blit_sprite(
        self,
        name: str,
        position: tuple[int, int],
        *,
        y_offset: int = 0,
    ) -> bool:
        sprite = self.sprites.get(name)
        if sprite is None:
            return False
        self.surface.blit(
            sprite,
            (position[0] * self.tile, position[1] * self.tile + y_offset),
        )
        return True

    def _section(self, text: str, x: int, y: int) -> None:
        self._text(text, x, y, self.theme.primary, self.small)
        self.pg.draw.line(self.surface, self.theme.grid, (x, y + 19), (self.width - 20, y + 19), 1)

    def _status_badge(self, text: str, color) -> None:
        pg = self.pg
        surface = self.title_font.render(text, True, self.theme.background)
        rect = surface.get_rect(center=(self.map_width // 2, self.height - 38))
        background = rect.inflate(36, 18)
        pg.draw.rect(self.surface, color, background, border_radius=10)
        self.surface.blit(surface, rect)

    def _tile_rect(self, position: tuple[int, int], inset: int = 0):
        return self.pg.Rect(
            position[0] * self.tile + inset,
            position[1] * self.tile + inset,
            self.tile - inset * 2,
            self.tile - inset * 2,
        )

    def _center(self, position: tuple[int, int]) -> tuple[int, int]:
        return (
            position[0] * self.tile + self.tile // 2,
            position[1] * self.tile + self.tile // 2,
        )

    def _text(self, text: str, x: int, y: int, color, font) -> None:
        self.surface.blit(font.render(str(text), True, color), (x, y))
