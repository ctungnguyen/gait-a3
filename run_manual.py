#!/usr/bin/env python3
"""Human-controlled visual test for the Arena and its Block 4 action adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from arena import ArenaConfig, ArenaEnv, ControlStyle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the GAIT Neon Pursuit Arena")
    parser.add_argument("--style", choices=("rotation", "direct"), default="direct")
    parser.add_argument("--seed", type=int, default=42)
    default_config = Path(__file__).resolve().parent / "config" / "arena.json"
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def selected_action(keys, style, pygame, shoot_ready: bool = True) -> int:
    shoot_held = bool(keys[pygame.K_SPACE])
    shoot_action = 4 if style is ControlStyle.ROTATION else 5
    if shoot_held and shoot_ready:
        return shoot_action

    if style is ControlStyle.ROTATION:
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            return 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            return 2
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            return 3
        return shoot_action if shoot_held else 0

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return 2
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return 3
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return 4
    return shoot_action if shoot_held else 0


def main() -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt") from exc

    args = parse_args()
    config = ArenaConfig.from_json(args.config)
    style = ControlStyle(args.style)
    env = ArenaEnv(config=config, control_style=style, render_mode="human", seed=args.seed)
    env.reset(seed=args.seed)
    debug_enabled = False

    # ArenaEnv creates Pygame lazily so headless training never opens a window.
    # Manual mode must render once before reading the event queue; otherwise
    # pygame.event.get() raises "video system not initialized" on Windows.
    env.render(debug=debug_enabled)
    clock = pygame.time.Clock()
    paused = False
    episode_done = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_F3:
                    debug_enabled = not debug_enabled
                elif event.key == pygame.K_r:
                    env.reset()
                    episode_done = False
                    paused = False
                    debug_enabled = False

        if not running:
            break

        if not paused and not episode_done:
            action = selected_action(
                pygame.key.get_pressed(),
                style,
                pygame,
                shoot_ready=env.core.player.shoot_cooldown <= 0.0,
            )
            _obs, _reward, terminated, truncated, _info = env.step(action)
            episode_done = terminated or truncated

        status = "PAUSED" if paused else "EPISODE ENDED - PRESS R" if episode_done else None
        env.render(status=status, debug=debug_enabled)
        clock.tick(round(1.0 / config.fixed_dt))

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
