#!/usr/bin/env python3
"""Unified visual launcher for the two independent GAIT A3 games."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from shared import get_theme, theme_names

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MenuAction:
    title: str
    subtitle: str
    command: tuple[str, ...] | None = None
    submenu: str | None = None


MAIN_ACTIONS = (
    MenuAction(
        "PART I  /  GRIDWORLD",
        "Q-learning, SARSA, monsters, and intrinsic exploration",
        command=("part1.py",),
    ),
    MenuAction(
        "PART II /  PURSUIT ARENA",
        "Continuous combat arena with two PPO control agents",
        submenu="arena",
    ),
)

ARENA_ACTIONS = (
    MenuAction("PLAY ROTATION + THRUST", "Manual Discrete(5) control contract", ("run_manual.py", "--style", "rotation")),
    MenuAction("PLAY DIRECT MOVEMENT", "Manual Discrete(6) control contract", ("run_manual.py", "--style", "direct")),
    MenuAction("WATCH PPO / ROTATION", "Evaluate the saved rotation_agent.zip", ("evaluate_rotation.py", "--episodes", "3")),
    MenuAction("WATCH PPO / DIRECT", "Evaluate the saved direct_agent.zip", ("evaluate_direct.py", "--episodes", "3")),
    MenuAction("BACK", "Return to the two-game selection", submenu="main"),
)


def _launch(command: tuple[str, ...], theme: str) -> None:
    script, *arguments = command
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), *arguments, "--theme", theme],
        cwd=PROJECT_ROOT,
        check=False,
    )


def _visual_menu(initial_theme: str) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Install pygame first: python -m pip install -r requirements.txt") from exc

    themes = list(theme_names())
    theme_index = themes.index(initial_theme)
    page = "main"
    selected = 0
    running = True

    while running:
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((1060, 700))
        pygame.display.set_caption("GAIT A3 - Reinforcement Learning Arcade")
        title_font = pygame.font.SysFont("consolas", 38, bold=True)
        heading_font = pygame.font.SysFont("consolas", 24, bold=True)
        font = pygame.font.SysFont("consolas", 17)
        small = pygame.font.SysFont("consolas", 14)
        clock = pygame.time.Clock()
        relaunch = False

        while running and not relaunch:
            palette = get_theme(themes[theme_index])
            actions = MAIN_ACTIONS if page == "main" else ARENA_ACTIONS
            screen.fill(palette.background)

            # Decorative horizon keeps the launcher visually related to both games.
            for y in range(0, 700, 70):
                color = palette.background_alt if (y // 70) % 2 == 0 else palette.background
                pygame.draw.rect(screen, color, (0, y, 1060, 70))
            for x in range(0, 1060, 70):
                pygame.draw.line(screen, palette.grid, (x, 0), (x, 700), 1)

            screen.blit(title_font.render("GAIT A3 / RL ARCADE", True, palette.text), (54, 38))
            subtitle = (
                "Two separate environments, one evidence-backed submission"
                if page == "main"
                else "PART II / CHOOSE MANUAL OR TRAINED AGENT"
            )
            screen.blit(font.render(subtitle, True, palette.primary), (57, 91))

            top = 150
            card_height = 158 if page == "main" else 82
            gap = 30 if page == "main" else 18
            card_rects = []
            for index, action in enumerate(actions):
                y = top + index * (card_height + gap)
                rect = pygame.Rect(54, y, 952, card_height)
                card_rects.append(rect)
                active = index == selected
                background = palette.primary if active else palette.panel
                foreground = palette.background if active else palette.text
                pygame.draw.rect(screen, background, rect, border_radius=15)
                pygame.draw.rect(screen, palette.edge, rect, 2, border_radius=15)
                screen.blit(heading_font.render(action.title, True, foreground), (rect.x + 28, rect.y + 22))
                screen.blit(font.render(action.subtitle, True, foreground), (rect.x + 30, rect.y + 63))
                if page == "main":
                    detail = (
                        "7 levels  |  4 cardinal actions  |  tabular policies"
                        if index == 0
                        else "real-time physics  |  25 observations  |  PPO"
                    )
                    screen.blit(small.render(detail, True, foreground), (rect.x + 30, rect.y + 108))

            help_text = f"UP/DOWN select  |  ENTER open  |  T colour theme: {palette.name}  |  ESC exit/back"
            screen.blit(small.render(help_text, True, palette.muted), (55, 665))
            pygame.display.flip()

            def activate(index: int) -> None:
                nonlocal page, selected, relaunch, running
                action = actions[index]
                if action.submenu == "arena":
                    page, selected = "arena", 0
                elif action.submenu == "main":
                    page, selected = "main", 0
                elif action.command:
                    pygame.quit()
                    _launch(action.command, themes[theme_index])
                    relaunch = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if page == "arena":
                            page, selected = "main", 0
                        else:
                            running = False
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        selected = (selected + 1) % len(actions)
                    elif event.key in (pygame.K_UP, pygame.K_w):
                        selected = (selected - 1) % len(actions)
                    elif event.key == pygame.K_t:
                        theme_index = (theme_index + 1) % len(themes)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        activate(selected)
                elif event.type == pygame.MOUSEMOTION:
                    for index, rect in enumerate(card_rects):
                        if rect.collidepoint(event.pos):
                            selected = index
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for index, rect in enumerate(card_rects):
                        if rect.collidepoint(event.pos):
                            selected = index
                            activate(index)
                            break
            clock.tick(30)
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=theme_names(), default="neon")
    parser.add_argument("--part", choices=("1", "2-rotation", "2-direct"))
    args = parser.parse_args()
    if args.part == "1":
        _launch(("part1.py",), args.theme)
    elif args.part == "2-rotation":
        _launch(("run_manual.py", "--style", "rotation"), args.theme)
    elif args.part == "2-direct":
        _launch(("run_manual.py", "--style", "direct"), args.theme)
    else:
        _visual_menu(args.theme)


if __name__ == "__main__":
    main()
