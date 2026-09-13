"""Part I command-line entry point and interactive learned-policy demo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from shared import theme_names

from .agents import QTable, make_agent
from .config import DEFAULT_CONFIG_PATH, PART1_ROOT, load_gridworld_settings
from .environment import GridWorld
from .evidence import generate_all
from .levels import LEVELS, get_level, level_signature
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
    screen = pygame.display.set_mode((980, 720))
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
        card_rects = []
        for index, (level, algorithm, intrinsic) in enumerate(DEMO_OPTIONS):
            column = index // 7
            row = index % 7
            x = 42 + column * 450
            y = 125 + row * 66
            rect = pygame.Rect(x, y, 420, 52)
            card_rects.append(rect)
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
        theme_rect = pygame.Rect(42, 626, 270, 38)
        back_rect = pygame.Rect(326, 626, 185, 38)
        for rect, label in ((theme_rect, f"T THEME: {palette.name.upper()}"), (back_rect, "ESC BACK")):
            hovered = rect.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(screen, palette.primary if hovered else palette.panel, rect, border_radius=8)
            pygame.draw.rect(screen, palette.edge, rect, 1, border_radius=8)
            color = palette.background if hovered else palette.text
            rendered = small.render(label, True, color)
            screen.blit(rendered, rendered.get_rect(center=rect.center))
        help_line = "Arrows/WASD select | ENTER or click to run | all controls support mouse"
        screen.blit(small.render(help_line, True, palette.muted), (42, 683))
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
            elif event.type == pygame.MOUSEMOTION:
                for index, rect in enumerate(card_rects):
                    if rect.collidepoint(event.pos):
                        selected = index
                        break
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if theme_rect.collidepoint(event.pos):
                    theme_index = (theme_index + 1) % len(themes)
                elif back_rect.collidepoint(event.pos):
                    pygame.quit()
                    return None
                else:
                    for index, rect in enumerate(card_rects):
                        if rect.collidepoint(event.pos):
                            choice = DEMO_OPTIONS[index]
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
        compatible = (
            metadata.get("level") == level
            and metadata.get("algorithm") == algorithm
            and bool(metadata.get("intrinsic_enabled", False)) == intrinsic
            and metadata.get("level_signature") == level_signature(definition)
            and metadata.get("settings") == asdict(settings)
        )
        if compatible:
            agent = make_agent(algorithm, settings)
            agent.qtable = table
            return TrainingResult(definition, algorithm, intrinsic, settings, agent, [])
        print(f"Saved policy is stale for the current layout; retraining {run_name}.")

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


def _visual_policy(result: TrainingResult, theme: str) -> str:
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
    episode_finished = False

    def reset_episode() -> None:
        nonlocal state, trail, demo, episode_finished, seed_counter
        seed_counter += 1
        state = environment.reset(seed=settings.seed + 20_000 + seed_counter)
        trail = [environment.agent]
        demo = DemoState(result.algorithm, result.intrinsic_enabled, episode=demo.episode + 1)
        episode_finished = False

    def activate(control: str) -> None:
        nonlocal running, single_step, frames_per_step, theme_index
        if control == "pause":
            demo.paused = not demo.paused
        elif control == "step":
            single_step = True
            demo.paused = True
        elif control == "replay":
            reset_episode()
        elif control == "theme":
            theme_index = (theme_index + 1) % len(themes)
            renderer.set_theme(themes[theme_index])
        elif control == "details":
            demo.detail = not demo.detail
        elif control == "faster":
            frames_per_step = max(1, frames_per_step - 1)
        elif control == "slower":
            frames_per_step = min(30, frames_per_step + 1)
        elif control == "menu":
            running = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    activate("menu")
                elif event.key == pygame.K_p:
                    activate("pause")
                elif event.key == pygame.K_n:
                    activate("step")
                elif event.key == pygame.K_r:
                    activate("replay")
                elif event.key == pygame.K_t:
                    activate("theme")
                elif event.key == pygame.K_d:
                    activate("details")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    activate("faster")
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    activate("slower")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                control = renderer.control_at(event.pos)
                if control is not None:
                    activate(control)

        should_step = (
            not episode_finished
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
                demo.paused = True
                episode_finished = True

        renderer.draw(result.agent, demo, trail)
        clock.tick(30)
        frame_counter += 1

    renderer.close()
    pygame.quit()
    return themes[theme_index]


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

    if args.level is not None:
        result = _load_or_train(
            args.level,
            args.algorithm,
            args.intrinsic,
            args.config,
            args.episodes,
            args.retrain,
            args.theme,
        )
        if not args.train_only:
            _visual_policy(result, args.theme)
        return

    # When launched interactively, leaving a policy returns to this selector.
    # Esc from the selector exits Part I and reveals the unified parent menu.
    while True:
        selected = _select_demo(args.theme)
        if selected is None:
            return
        (level, algorithm, intrinsic), args.theme = selected
        result = _load_or_train(
            level,
            algorithm,
            intrinsic,
            args.config,
            args.episodes,
            args.retrain,
            args.theme,
        )
        if args.train_only:
            return
        args.theme = _visual_policy(result, args.theme)


if __name__ == "__main__":
    main()
