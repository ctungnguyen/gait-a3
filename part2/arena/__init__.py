"""Continuous Pygame arena and Gym-style environment for GAIT Part II."""

from .config import ArenaConfig
from .controls import (
    ControlStyle,
    DirectAction,
    RotationAction,
    action_names,
    control_scheme,
)
from .core import ArenaCore
from .env import ArenaEnv, LegacyArenaAdapter
from .factories import make_direct_env, make_rotation_env
from .rewards import ProgressionReward, RewardConfig, make_progression_reward

__version__ = "2.0.0"

__all__ = [
    "ArenaConfig",
    "ArenaCore",
    "ArenaEnv",
    "ControlStyle",
    "DirectAction",
    "LegacyArenaAdapter",
    "ProgressionReward",
    "RewardConfig",
    "RotationAction",
    "__version__",
    "action_names",
    "control_scheme",
    "make_direct_env",
    "make_progression_reward",
    "make_rotation_env",
]
