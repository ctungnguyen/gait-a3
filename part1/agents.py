"""Tabular Q-learning/SARSA with the exact required exploration behaviour."""

from __future__ import annotations

import ast
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .config import GridworldSettings
from .environment import ALL_ACTIONS, State


class QTable:
    def __init__(self) -> None:
        self.q: defaultdict[State, list[float]] = defaultdict(
            lambda: [0.0] * len(ALL_ACTIONS)
        )

    def values(self, state: State) -> list[float]:
        return self.q[state]

    def get(self, state: State, action: int) -> float:
        return self.q[state][action]

    def set(self, state: State, action: int, value: float) -> None:
        self.q[state][action] = float(value)

    def best_value(self, state: State) -> float:
        return max(self.q[state])

    def best_actions(self, state: State) -> list[int]:
        values = self.q[state]
        maximum = max(values)
        return [
            action
            for action, value in enumerate(values)
            if math.isclose(value, maximum, rel_tol=1e-9, abs_tol=1e-9)
        ]

    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "gait-tabular-q-v1",
            "metadata": metadata or {},
            "states": {repr(state): values for state, values in self.q.items()},
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> tuple["QTable", dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "gait-tabular-q-v1":
            raise ValueError("Unsupported Q-table file")
        table = cls()
        for encoded_state, values in payload["states"].items():
            state = ast.literal_eval(encoded_state)
            if len(values) != len(ALL_ACTIONS):
                raise ValueError("Saved Q-table has the wrong action count")
            table.q[state] = [float(value) for value in values]
        return table, dict(payload.get("metadata", {}))


class BaseTabularAgent:
    algorithm = "base"

    def __init__(self, settings: GridworldSettings, seed: int | None = None) -> None:
        self.settings = settings
        self.alpha = settings.alpha
        self.gamma = settings.gamma
        self.qtable = QTable()
        self.rng = random.Random(settings.seed if seed is None else seed)
        self.visit_counts: dict[State, int] = {}

    def start_episode(self, initial_state: State) -> None:
        # n(s) counts the initial occupancy before any action is taken.
        self.visit_counts = {initial_state: 1}

    def intrinsic_reward(self, state: State, strength: float | None = None) -> float:
        """Return strength/sqrt(n(s)+1), then record this visit."""

        n_state = self.visit_counts.get(state, 0)
        self.visit_counts[state] = n_state + 1
        coefficient = (
            self.settings.intrinsic_reward_strength
            if strength is None
            else float(strength)
        )
        return coefficient / math.sqrt(n_state + 1)

    def epsilon(self, episode_index: int) -> float:
        decay = self.settings.epsilon_decay_episodes
        if decay <= 0:
            return self.settings.epsilon_end
        if episode_index >= decay:
            return self.settings.epsilon_end
        progress = min(max(episode_index, 0) / float(decay), 1.0)
        return self.settings.epsilon_start + progress * (
            self.settings.epsilon_end - self.settings.epsilon_start
        )

    # Legacy spelling retained for teammate code/tests.
    get_epsilon = epsilon

    def choose_action(self, state: State, epsilon: float, evaluate: bool = False) -> int:
        if not evaluate and self.rng.random() < epsilon:
            return self.rng.choice(ALL_ACTIONS)
        # Random choice is intentional and required when Q-values tie.
        return self.rng.choice(self.qtable.best_actions(state))


class QLearningAgent(BaseTabularAgent):
    algorithm = "q_learning"

    def update(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        current = self.qtable.get(state, action)
        target = reward if done else reward + self.gamma * self.qtable.best_value(next_state)
        self.qtable.set(state, action, current + self.alpha * (target - current))


class SARSAAgent(BaseTabularAgent):
    algorithm = "sarsa"

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        done: bool,
    ) -> None:
        current = self.qtable.get(state, action)
        target = reward if done else reward + self.gamma * self.qtable.get(next_state, next_action)
        self.qtable.set(state, action, current + self.alpha * (target - current))


def make_agent(
    algorithm: str,
    settings: GridworldSettings,
    seed: int | None = None,
) -> BaseTabularAgent:
    normalized = algorithm.lower().replace("-", "_")
    if normalized in {"q", "qlearning", "q_learning"}:
        return QLearningAgent(settings, seed=seed)
    if normalized == "sarsa":
        return SARSAAgent(settings, seed=seed)
    raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
