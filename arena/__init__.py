"""Continuous Pygame arena and Gym-style environment for GAIT Part II."""

from .config import ArenaConfig
from .controls import ControlStyle
from .core import ArenaCore
from .env import ArenaEnv, LegacyArenaAdapter

__version__ = "1.2.2"

__all__ = [
    "ArenaConfig",
    "ArenaCore",
    "ArenaEnv",
    "ControlStyle",
    "LegacyArenaAdapter",
    "__version__",
]
