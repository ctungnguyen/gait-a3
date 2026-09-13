"""Part I command-line entry point and interactive learned-policy demo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from shared import theme_names

from .agents import QTable, make_agent
from .config import DEFAULT_CONFIG_PATH, PART1_ROOT, load_gridworld_settings
from .environment import GridWorld
from .evidence import generate_all
from .levels import LEVELS, get_level
from .renderer import DemoState, GridworldRenderer
from .training import TrainingResult, save_evaluation, save_training_result, evaluate, train


DEMO_OPTIONS = (
    (0, "q_learning", False),
    (1, "q_learning", False),
    (1, "sarsa", False),
    (2, "q_learning", False),
    (2, "sarsa", False),
    (3, "q_learning", False),
    (3, "sarsa", False),
    (4, "q_learning", False),
    (4, "sarsa", False),
    (5, "q_learning", False),
    (5, "sarsa", False),
    (6, "q_learning", False),
    (6, "q_learning", True),
)


def _run_name(level: int, algorithm: str, intrinsic: bool) -> str:
    return f"level{level}_{algorithm}{'_intrinsic' if intrinsic else ''}"


def _select_demo(theme_name: str) -> tuple[tuple[int, str, bool], str] | None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((980, 700))
    pygame.display.set_caption("GAIT A3 - Part I Demo Selection")
    title_font = pygame.font.SysFont("consolas", 31, bold=True)
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 14)
    from shared import get_theme

    themes = list(theme_names())
    theme_index = themes.index(theme_name)
    selected = 0
    clock = pygame.time.Clock()
    while True:
        palette = get_theme(themes[theme_index])
        screen.fill(palette.background)
        screen.blit(title_font.render("PART I / CLASSICAL RL GRIDWORLD", True, palette.text), (42, 30))
        screen.blit(small.render("Choose a saved policy or train it on first launch", True, palette.muted), (44, 76))
        for index, (level, algorithm, intrinsic) in enumerate(DEMO_OPTIONS):
            column = index // 7
            row = index % 7
            x = 42 + column * 450
            y = 125 + row * 66
            rect = pygame.Rect(x, y, 420, 52)
            if index == selected:
                pygame.draw.rect(screen, palette.primary, rect, border_radius=8)
                color = palette.background
            else:
                pygame.draw.rect(screen, palette.panel, rect, border_radius=8)
                pygame.draw.rect(screen, palette.grid, rect, 1, border_radius=8)
                color = palette.text
            suffix = " + INTRINSIC" if intrinsic else ""
            label = f"L{level}  {algorithm.replace('_', ' ').upper()}{suffix}"
            screen.blit(font.render(label, True, color), (x + 14, y + 8))
            screen.blit(small.render(LEVELS[level].task, True, color), (x + 14, y + 30))
        help_line = f"Arrows/WASD select | ENTER run | T theme ({palette.name}) | ESC main menu"
        screen.blit(small.render(help_line, True, palette.muted), (42, 650))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = min(len(DEMO_OPTIONS) - 1, selected + 1)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    selected = max(0, selected - 1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    selected = min(len(DEMO_OPTIONS) - 1, selected + 7)
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    selected = max(0, selected - 7)
                elif event.key == pygame.K_t:
                    theme_index = (theme_index + 1) % len(themes)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    choice = DEMO_OPTIONS[selected]
                    pygame.display.quit()
                    return choice, themes[theme_index]
        clock.tick(30)


def _load_or_train(
    level: int,
    algorithm: str,
    intrinsic: bool,
    config_path: Path,
    episodes: int | None,
    retrain: bool,
    theme: str,
) -> TrainingResult:
    run_name = _run_name(level, algorithm, intrinsic)
    model_path = PART1_ROOT / "models" / f"{run_name}.json"
    settings = load_gridworld_settings(config_path, level)
    if episodes is not None:
        settings = settings.with_overrides(
            episodes=episodes,
            epsilon_decay_episodes=min(settings.epsilon_decay_episodes, max(1, int(episodes * 0.75))),
        )
    definition = get_level(level)

    if model_path.exists() and not retrain and episodes is None:
        table, metadata = QTable.load(model_path)
        if metadata.get("level") != level or metadata.get("algorithm") != algorithm:
            raise ValueError(f"Model metadata does not match requested demo: {model_path}")
        agent = make_agent(algorithm, settings)
        agent.qtable = table
        return TrainingResult(definition, algorithm, intrinsic, settings, agent, [])

    preview_env = GridWorld(definition, settings.monster_move_chance, seed=settings.seed)
    renderer = GridworldRenderer(preview_env, settings, theme)
    next_update = 0

    def progress(current, total, record):
        nonlocal next_update
        interval = max(1, total // 100)
        if current >= next_update or current == total:
            renderer.draw_training_progress(current, total, algorithm, level, record.success)
            next_update = current + interval

    result = train(
        level,
        algorithm,
        config_path=config_path,
        episodes=episodes,
        intrinsic=intrinsic,
        progress=progress,
    )
    renderer.close()
    save_training_result(result)
    save_evaluation(result, evaluate(result))
    return result


def _visual_policy(result: TrainingResult, theme: str) -> None:
    import pygame

    settings = result.settings
    themes = list(theme_names())
    theme_index = themes.index(theme)
    seed_counter = 0
    environment = GridWorld(
        result.level,
        settings.monster_move_chance,
        seed=settings.seed + 20_000,
    )
    renderer = GridworldRenderer(environment, settings, theme)
    clock = pygame.time.Clock()
    demo = DemoState(result.algorithm, result.intrinsic_enabled)
    state = environment.reset()
    trail = [environment.agent]
    running = True
    frames_per_step = max(1, round(30 / settings.fps_visual))
    frame_counter = 0
    single_step = False
    terminal_at: float | None = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    demo.paused = not demo.paused
                elif event.key == pygame.K_n:
                    single_step = True
                    demo.paused = True
                elif event.key == pygame.K_r:
                    seed_counter += 1
                    state = environment.reset(seed=settings.seed + 20_000 + seed_counter)
                    trail = [environment.agent]
                    demo = DemoState(result.algorithm, result.intrinsic_enabled, episode=demo.episode + 1)
                    terminal_at = None
                elif event.key == pygame.K_t:
                    theme_index = (theme_index + 1) % len(themes)
                    renderer.set_theme(themes[theme_index])
                elif event.key == pygame.K_d:
                    demo.detail = not demo.detail
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    frames_per_step = max(1, frames_per_step - 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    frames_per_step = min(30, frames_per_step + 1)

        should_step = (
            terminal_at is None
            and ((not demo.paused and frame_counter % frames_per_step == 0) or single_step)
        )
        if should_step:
            action = result.agent.choose_action(state, 0.0, evaluate=True)
            step_result = environment.step(action)
            state = step_result.next_state
            demo.action = action
            demo.step += 1
            demo.environment_return += step_result.reward
            demo.event = step_result.info.get("event", "moved")
            trail.append(environment.agent)
            single_step = False
            if step_result.done or demo.step >= settings.max_steps_per_episode:
                won = "win" in step_result.info.get("events", ())
                demo.status = "OBJECTIVES COMPLETE" if won else "AGENT DIED"
                terminal_at = time.monotonic()

        if terminal_at is not None and time.monotonic() - terminal_at >= 1.4:
            seed_counter += 1
            state = environment.reset(seed=settings.seed + 20_000 + seed_counter)
            trail = [environment.agent]
            demo = DemoState(result.algorithm, result.intrinsic_enabled, episode=demo.episode + 1)
            terminal_at = None

        renderer.draw(result.agent, demo, trail)
        clock.tick(30)
        frame_counter += 1

    renderer.close()
    pygame.quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAIT A3 Part I Gridworld")
    parser.add_argument("--level", type=int, choices=sorted(LEVELS))
    parser.add_argument("--algorithm", choices=("q_learning", "sarsa"), default="q_learning")
    parser.add_argument("--intrinsic", action="store_true", help="Enable Level 6 count bonus")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--generate-evidence", action="store_true")
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--theme", choices=theme_names(), default="neon")
    args = parser.parse_args(argv)
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.intrinsic and args.level not in (None, 6):
        parser.error("Intrinsic reward evidence is defined for Level 6")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.generate_evidence:
        generate_all(
            config_path=args.config,
            episodes=args.episodes,
            evaluation_episodes=args.evaluation_episodes,
        )
        return

    if args.level is None:
        selected = _select_demo(args.theme)
        if selected is None:
            return
        (level, algorithm, intrinsic), args.theme = selected
    else:
        level, algorithm, intrinsic = args.level, args.algorithm, args.intrinsic

    result = _load_or_train(
        level,
        algorithm,
        intrinsic,
        args.config,
        args.episodes,
        args.retrain,
        args.theme,
    )
    if not args.train_only:
        _visual_policy(result, args.theme)


if __name__ == "__main__":
    main()
