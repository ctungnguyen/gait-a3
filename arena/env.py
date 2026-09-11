"""Gymnasium and assignment-compatible APIs for Part II Block 2."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .adapters import (
    OBSERVATION_SIZE,
    ObservationFunction,
    RewardFunction,
    build_baseline_observation,
    neutral_reward,
)
from .config import ArenaConfig
from .controls import ControlStyle, action_count, action_name, action_names, decode_action
from .core import ArenaCore
from .gym_compat import BaseEnv, GYMNASIUM_AVAILABLE, spaces


class ArenaEnv(BaseEnv):
    """Current Gymnasium API used by modern Stable Baselines3 releases.

    ``reset`` returns ``(observation, info)`` and ``step`` returns the modern
    five-value tuple. Use :class:`LegacyArenaAdapter` for the four-value API
    printed in the assignment sheet.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: ArenaConfig | None = None,
        control_style: ControlStyle | str = ControlStyle.DIRECT,
        render_mode: str | None = None,
        observation_fn: ObservationFunction = build_baseline_observation,
        reward_fn: RewardFunction = neutral_reward,
        seed: int | None = None,
    ):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.config = config or ArenaConfig()
        self.control_style = ControlStyle(control_style)
        self.render_mode = render_mode
        self.observation_fn = observation_fn
        self.reward_fn = reward_fn
        self.core = ArenaCore(self.config, seed=seed)
        self.action_space = spaces.Discrete(action_count(self.control_style))
        if seed is not None:
            self.action_space.seed(seed)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._renderer = None
        self.last_action_index: int | None = None
        self.last_action_name = "NO ACTION YET"
        self.episode_metrics: dict[str, float | int] = {}
        self._reset_episode_metrics()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if GYMNASIUM_AVAILABLE:
            super().reset(seed=seed)
        del options
        if seed is not None:
            self.action_space.seed(seed)
        self.core.reset(seed=seed)
        reset_reward = getattr(self.reward_fn, "reset", None)
        if callable(reset_reward):
            reset_reward(self.core)
        self._reset_episode_metrics()
        self.last_action_index = None
        self.last_action_name = "NO ACTION YET"
        observation = self.observation_fn(self.core)
        info = self.core.info()
        info.update(
            {
                "control_style": self.control_style.value,
                "action_meanings": self.get_action_meanings(),
                **self.episode_metrics,
            }
        )
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for {self.control_style.value}")
        self.last_action_index = int(action)
        command = decode_action(self.control_style, self.last_action_index)
        self.last_action_name = action_name(self.control_style, self.last_action_index)
        outcome = self.core.step(command, self.control_style)
        observation = self.observation_fn(self.core)
        reward = float(self.reward_fn(self.core, outcome))
        self._update_episode_metrics(outcome)
        info = self.core.info(outcome)
        info.update(
            {
                "control_style": self.control_style.value,
                "action_index": self.last_action_index,
                "action_name": self.last_action_name,
                "terminated": outcome.terminated,
                "truncated": outcome.truncated,
                **self.episode_metrics,
            }
        )
        reward_components = getattr(self.reward_fn, "last_components", None)
        if isinstance(reward_components, dict):
            info["reward_components"] = dict(reward_components)
        return observation, reward, outcome.terminated, outcome.truncated, info

    def _reset_episode_metrics(self) -> None:
        self.episode_metrics = {
            "episode_max_phase": int(self.core.phase),
            "episode_phases_advanced": 0,
            "episode_enemies_destroyed": 0,
            "episode_spawners_destroyed": 0,
            "episode_shots_fired": 0,
            "episode_damage_dealt": 0.0,
            "episode_damage_taken": 0.0,
            "is_success": False,
        }

    def _update_episode_metrics(self, outcome) -> None:
        for event in outcome.events:
            if event.name == "phase_advanced":
                self.episode_metrics["episode_phases_advanced"] += 1
            elif event.name == "enemy_destroyed":
                self.episode_metrics["episode_enemies_destroyed"] += 1
            elif event.name == "spawner_destroyed":
                self.episode_metrics["episode_spawners_destroyed"] += 1
            elif event.name == "projectile_fired":
                self.episode_metrics["episode_shots_fired"] += 1
            elif event.name in ("enemy_damaged", "spawner_damaged"):
                self.episode_metrics["episode_damage_dealt"] += float(event.amount)
            elif event.name == "player_damaged":
                self.episode_metrics["episode_damage_taken"] += float(event.amount)

        self.episode_metrics["episode_max_phase"] = max(
            int(self.episode_metrics["episode_max_phase"]),
            int(self.core.phase),
        )
        self.episode_metrics["is_success"] = bool(
            self.episode_metrics["episode_phases_advanced"] > 0
        )

    def get_action_meanings(self) -> tuple[str, ...]:
        """Expose the exact action order used to train or evaluate a model."""

        return action_names(self.control_style)

    def render(self, status: str | None = None, debug: bool = False):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .renderer import ArenaRenderer

            self._renderer = ArenaRenderer(self.config, self.render_mode)
        return self._renderer.draw(
            self.core,
            self.control_style,
            status=status,
            debug=debug,
            action_label=self.last_action_name,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class LegacyArenaAdapter:
    """Exact ``obs, reward, done, info`` API stated in the assessment brief."""

    def __init__(self, env: ArenaEnv | None = None, **env_kwargs):
        self.env = env or ArenaEnv(**env_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    @property
    def core(self) -> ArenaCore:
        return self.env.core

    def reset(self, seed: int | None = None) -> np.ndarray:
        observation, _info = self.env.reset(seed=seed)
        return observation

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated or truncated, info

    def render(self, status: str | None = None, debug: bool = False):
        return self.env.render(status=status, debug=debug)

    def close(self) -> None:
        self.env.close()
