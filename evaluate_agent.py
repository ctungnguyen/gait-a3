#!/usr/bin/env python3
"""Shared visual evaluator used by the two Block 4 entry-point scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from arena import ArenaConfig, ControlStyle, make_direct_env, make_rotation_env


PROJECT_ROOT = Path(__file__).resolve().parent


def _model_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _load_model(algorithm: str, path: Path, device: str):
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:
        raise SystemExit(
            "Stable Baselines3 is required for model evaluation. Install it with "
            "'python -m pip install -r requirements-rl.txt'."
        ) from exc

    algorithm_class = {"dqn": DQN, "ppo": PPO}[algorithm]
    try:
        return algorithm_class.load(str(path), device=device)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Model not found: {path}\n"
            "Train and save the matching style model before evaluation."
        ) from exc


def _build_parser(style: ControlStyle, default_model: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Visually evaluate the trained {style.value} control agent"
    )
    parser.add_argument("--model", type=_model_path, default=default_model)
    parser.add_argument("--algorithm", choices=("ppo", "dqn"), default="ppo")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--debug", action="store_true")
    default_config = PROJECT_ROOT / "config" / "arena.json"
    parser.add_argument("--config", type=Path, default=default_config)
    return parser


def evaluate_style(
    style: ControlStyle,
    default_model: Path,
    argv: Sequence[str] | None = None,
) -> None:
    """Load exactly one style's model and show real Pygame gameplay."""

    parser = _build_parser(style, default_model)
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    model = _load_model(args.algorithm, args.model, args.device)
    config = ArenaConfig.from_json(args.config)
    factory = make_rotation_env if style is ControlStyle.ROTATION else make_direct_env
    env = factory(config=config, render_mode="human", seed=args.seed)

    trained_action_count = getattr(model.action_space, "n", None)
    if trained_action_count != env.action_space.n:
        env.close()
        raise SystemExit(
            f"Wrong model for {style.value}: model uses Discrete({trained_action_count}), "
            f"but this style requires Discrete({env.action_space.n})."
        )

    try:
        import pygame
    except ImportError as exc:
        env.close()
        raise SystemExit("Pygame is required for visual evaluation.") from exc

    debug_enabled = bool(args.debug)
    running = True
    clock = None
    completed = 0

    try:
        for episode_index in range(args.episodes):
            if not running:
                break
            observation, _info = env.reset(seed=args.seed + episode_index)
            total_reward = 0.0
            paused = False

            # Rendering initializes Pygame before the first event queue read.
            env.render(debug=debug_enabled)
            if clock is None:
                clock = pygame.time.Clock()

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

                if not running:
                    break
                if paused:
                    env.render(status="MODEL EVALUATION PAUSED", debug=debug_enabled)
                    clock.tick(args.fps)
                    continue

                prediction, _state = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                action = int(np.asarray(prediction).item())
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                env.render(debug=debug_enabled)
                clock.tick(args.fps)

                if terminated or truncated:
                    completed += 1
                    reason = "player died" if terminated else "time limit"
                    print(
                        f"{style.value} episode {episode_index + 1}: {reason}, "
                        f"phase={info['phase']}, steps={info['step']}, "
                        f"reward={total_reward:.2f}"
                    )
                    break
    finally:
        env.close()
        pygame.quit()

    print(f"Completed {completed}/{args.episodes} visual {style.value} evaluation episode(s).")


def default_model_path(style: ControlStyle) -> Path:
    return PROJECT_ROOT / "models" / f"{style.value}_agent.zip"
