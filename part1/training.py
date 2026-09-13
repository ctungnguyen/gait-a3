"""Headless tabular training plus reproducible evaluation and evidence logs."""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Iterable

from .agents import BaseTabularAgent, QLearningAgent, SARSAAgent, make_agent
from .config import DEFAULT_CONFIG_PATH, GridworldSettings, PART1_ROOT, load_gridworld_settings
from .environment import ACTIONS, GridWorld, State
from .levels import LevelDefinition, get_level


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    environment_return: float
    learning_return: float
    intrinsic_return: float
    steps: int
    epsilon: float
    success: int
    died: int
    truncated: int
    monster_moves: int


@dataclass
class TrainingResult:
    level: LevelDefinition
    algorithm: str
    intrinsic_enabled: bool
    settings: GridworldSettings
    agent: BaseTabularAgent
    episodes: list[EpisodeRecord]

    @property
    def run_name(self) -> str:
        suffix = "_intrinsic" if self.intrinsic_enabled else ""
        return f"level{self.level.number}_{self.algorithm}{suffix}"


@dataclass(frozen=True)
class EvaluationRecord:
    episode: int
    environment_return: float
    steps: int
    success: int
    died: int
    truncated: int
    monster_moves: int


ProgressCallback = Callable[[int, int, EpisodeRecord], None]


def train(
    level_number: int,
    algorithm: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    episodes: int | None = None,
    intrinsic: bool = False,
    seed: int | None = None,
    progress: ProgressCallback | None = None,
) -> TrainingResult:
    """Train without rendering; final gameplay is rendered separately."""

    level = get_level(level_number)
    settings = load_gridworld_settings(config_path, level_number)
    overrides = {}
    if episodes is not None:
        overrides["episodes"] = int(episodes)
        # Preserve a genuinely linear schedule even for a shorter demo run.
        overrides["epsilon_decay_episodes"] = min(
            settings.epsilon_decay_episodes,
            max(1, int(episodes * 0.75)),
        )
    if seed is not None:
        overrides["seed"] = int(seed)
    if overrides:
        settings = settings.with_overrides(**overrides)

    normalized = algorithm.lower().replace("-", "_")
    if normalized in {"q", "qlearning"}:
        normalized = "q_learning"
    agent = make_agent(normalized, settings, seed=settings.seed + 17)
    environment = GridWorld(
        level,
        monster_move_chance=settings.monster_move_chance,
        seed=settings.seed + 31,
    )
    records: list[EpisodeRecord] = []

    for episode_index in range(settings.episodes):
        state = environment.reset()
        agent.start_episode(state)
        epsilon = agent.epsilon(episode_index)
        action = agent.choose_action(state, epsilon)
        environment_return = 0.0
        learning_return = 0.0
        intrinsic_return = 0.0
        monster_moves = 0
        final_result = None
        truncated = False

        for step_index in range(settings.max_steps_per_episode):
            if isinstance(agent, QLearningAgent):
                action = agent.choose_action(state, epsilon)

            result = environment.step(action)
            final_result = result
            next_state = result.next_state
            environment_return += result.reward
            monster_moves += int(result.info.get("monster_moves", 0))
            bonus = agent.intrinsic_reward(next_state) if intrinsic else 0.0
            intrinsic_return += bonus
            learning_reward = result.reward + bonus
            learning_return += learning_reward

            reached_limit = step_index + 1 >= settings.max_steps_per_episode
            truncated = bool(reached_limit and not result.done)
            terminal_for_update = result.done or truncated

            if isinstance(agent, QLearningAgent):
                agent.update(state, action, learning_reward, next_state, terminal_for_update)
            elif isinstance(agent, SARSAAgent):
                next_action = (
                    agent.choose_action(next_state, epsilon)
                    if not terminal_for_update
                    else 0
                )
                agent.update(
                    state,
                    action,
                    learning_reward,
                    next_state,
                    next_action,
                    terminal_for_update,
                )
                action = next_action
            else:  # pragma: no cover - guarded by make_agent
                raise TypeError(f"Unsupported agent {type(agent)!r}")

            state = next_state
            if terminal_for_update:
                break

        events = () if final_result is None else final_result.info.get("events", ())
        record = EpisodeRecord(
            episode=episode_index + 1,
            environment_return=environment_return,
            learning_return=learning_return,
            intrinsic_return=intrinsic_return,
            steps=step_index + 1,
            epsilon=epsilon,
            success=int("win" in events),
            died=int("death" in events),
            truncated=int(truncated),
            monster_moves=monster_moves,
        )
        records.append(record)
        if progress is not None:
            progress(episode_index + 1, settings.episodes, record)

    return TrainingResult(level, normalized, intrinsic, settings, agent, records)


def evaluate(
    result: TrainingResult,
    *,
    episodes: int | None = None,
    seed: int | None = None,
) -> list[EvaluationRecord]:
    count = result.settings.evaluation_episodes if episodes is None else int(episodes)
    if count < 1:
        raise ValueError("evaluation episodes must be positive")
    first_seed = result.settings.seed + 10_000 if seed is None else int(seed)
    records: list[EvaluationRecord] = []

    for episode_index in range(count):
        environment = GridWorld(
            result.level,
            monster_move_chance=result.settings.monster_move_chance,
            seed=first_seed + episode_index,
        )
        state = environment.reset()
        episode_return = 0.0
        monster_moves = 0
        final_result = None
        truncated = False
        for step_index in range(result.settings.max_steps_per_episode):
            action = result.agent.choose_action(state, 0.0, evaluate=True)
            final_result = environment.step(action)
            state = final_result.next_state
            episode_return += final_result.reward
            monster_moves += int(final_result.info.get("monster_moves", 0))
            if final_result.done:
                break
        else:
            truncated = True

        events = () if final_result is None else final_result.info.get("events", ())
        records.append(
            EvaluationRecord(
                episode=episode_index + 1,
                environment_return=episode_return,
                steps=step_index + 1,
                success=int("win" in events),
                died=int("death" in events),
                truncated=int(truncated),
                monster_moves=monster_moves,
            )
        )
    return records


def evaluation_summary(records: list[EvaluationRecord]) -> dict[str, float | int]:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    returns = [record.environment_return for record in records]
    successful_steps = [record.steps for record in records if record.success]
    return {
        "episodes": len(records),
        "mean_environment_return": mean(returns),
        "return_std": pstdev(returns),
        "success_rate": mean(record.success for record in records),
        "death_rate": mean(record.died for record in records),
        "truncation_rate": mean(record.truncated for record in records),
        "mean_steps": mean(record.steps for record in records),
        "mean_success_steps": mean(successful_steps) if successful_steps else 0.0,
        "mean_monster_moves": mean(record.monster_moves for record in records),
    }


def save_training_result(
    result: TrainingResult,
    *,
    logs_dir: str | Path = PART1_ROOT / "logs",
    models_dir: str | Path = PART1_ROOT / "models",
) -> tuple[Path, Path]:
    logs_path = Path(logs_dir)
    models_path = Path(models_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    csv_path = logs_path / f"{result.run_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EpisodeRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in result.episodes)

    metadata = {
        "level": result.level.number,
        "level_name": result.level.name,
        "task": result.level.task,
        "algorithm": result.algorithm,
        "intrinsic_enabled": result.intrinsic_enabled,
        "settings": asdict(result.settings),
        "q_learning_rule": "Q(s,a) <- Q(s,a) + alpha[r + gamma max_a' Q(s',a') - Q(s,a)]",
        "sarsa_rule": "Q(s,a) <- Q(s,a) + alpha[r + gamma Q(s',a') - Q(s,a)]",
        "intrinsic_rule": "r_i = intrinsicRewardStrength / sqrt(n(s) + 1)",
        "environment_rewards_unchanged": True,
    }
    model_path = result.agent.qtable.save(
        models_path / f"{result.run_name}.json",
        metadata=metadata,
    )
    return csv_path, model_path


def save_evaluation(
    result: TrainingResult,
    records: list[EvaluationRecord],
    *,
    reports_dir: str | Path = PART1_ROOT / "reports",
) -> tuple[Path, Path]:
    output = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{result.run_name}_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvaluationRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    summary_path = output / f"{result.run_name}_summary.json"
    payload = {
        "run_name": result.run_name,
        "level": result.level.number,
        "algorithm": result.algorithm,
        "intrinsic_enabled": result.intrinsic_enabled,
        **evaluation_summary(records),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, summary_path


def shortest_collectible_steps(level: LevelDefinition) -> int | None:
    """Exact BFS baseline for static levels; used to prove Level 0 efficiency."""

    environment = GridWorld(level, monster_move_chance=0.0, seed=0)
    if environment.monsters:
        return None
    assert environment.start is not None
    start = (environment.start, (1 << len(environment.apples)) - 1, 0, 0)
    frontier = deque([(start, 0)])
    seen = {start}
    while frontier:
        (position, apple_mask, has_key, chest_opened), distance = frontier.popleft()
        for dx, dy in ACTIONS:
            candidate = (position[0] + dx, position[1] + dy)
            if not environment.legal_destination(candidate) or candidate in environment.fires:
                continue
            next_mask = apple_mask
            apple_index = environment.apple_index.get(candidate)
            if apple_index is not None:
                next_mask &= ~(1 << apple_index)
            next_key = int(has_key or candidate == environment.key_pos)
            next_chest = int(
                chest_opened
                or (candidate == environment.chest_pos and next_key)
            )
            state = (candidate, next_mask, next_key, next_chest)
            complete = next_mask == 0 and (
                environment.chest_pos is None or next_chest == 1
            )
            if complete:
                return distance + 1
            if state not in seen:
                seen.add(state)
                frontier.append((state, distance + 1))
    return None


def moving_average(values: Iterable[float], window: int) -> list[float]:
    items = list(values)
    if not items:
        return []
    window = max(1, min(int(window), len(items)))
    queue: deque[float] = deque()
    total = 0.0
    output: list[float] = []
    for value in items:
        queue.append(float(value))
        total += float(value)
        if len(queue) > window:
            total -= queue.popleft()
        output.append(total / len(queue))
    return output


def greedy_policy_trace(
    result: TrainingResult,
    *,
    seed: int = 30_001,
) -> dict:
    """Capture one inspectable epsilon-zero route for report/demo evidence."""

    environment = GridWorld(
        result.level,
        monster_move_chance=result.settings.monster_move_chance,
        seed=seed,
    )
    result.agent.rng.seed(seed + 1)
    state = environment.reset()
    positions = [environment.agent]
    actions: list[int] = []
    environment_return = 0.0
    final_events: tuple[str, ...] = ()
    for _step in range(result.settings.max_steps_per_episode):
        action = result.agent.choose_action(state, 0.0, evaluate=True)
        step_result = environment.step(action)
        actions.append(action)
        positions.append(environment.agent)
        environment_return += step_result.reward
        state = step_result.next_state
        final_events = step_result.info.get("events", ())
        if step_result.done:
            break
    fire_clearance = None
    if environment.fires:
        fire_clearance = min(
            abs(position[0] - fire[0]) + abs(position[1] - fire[1])
            for position in positions
            for fire in environment.fires
        )
    return {
        "positions": [list(position) for position in positions],
        "actions": actions,
        "steps": len(actions),
        "environment_return": environment_return,
        "success": "win" in final_events,
        "died": "death" in final_events,
        "minimum_manhattan_fire_clearance": fire_clearance,
    }
