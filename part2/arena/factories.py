"""Style-locked environment factories for separate Part II agents.

Training and evaluation code should call these helpers instead of duplicating
the Arena.  Both models therefore receive identical physics, observations,
rewards, and level progression; only the required action space changes.
"""

from __future__ import annotations

from typing import Any

from .controls import ControlStyle
from .env import ArenaEnv


def _style_locked_env(style: ControlStyle, **env_kwargs: Any) -> ArenaEnv:
    if "control_style" in env_kwargs:
        raise TypeError("control_style is fixed by this factory")
    return ArenaEnv(control_style=style, **env_kwargs)


def make_rotation_env(**env_kwargs: Any) -> ArenaEnv:
    """Build the required rotation/thrust ``Discrete(5)`` environment."""

    return _style_locked_env(ControlStyle.ROTATION, **env_kwargs)


def make_direct_env(**env_kwargs: Any) -> ArenaEnv:
    """Build the required direct-direction ``Discrete(6)`` environment."""

    return _style_locked_env(ControlStyle.DIRECT, **env_kwargs)
