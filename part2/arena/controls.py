"""Authoritative discrete-action contract for both Part II control styles.

The assessment requires two agents to share one Arena while using different
action spaces. This module is the single source of truth for those spaces, so
manual play, Gymnasium, training, evaluation, the HUD, and tests cannot drift
into different mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from numbers import Integral
from typing import TypeAlias

from .math2d import Vec, vec


class ControlStyle(str, Enum):
    """The two control schemes named in the assignment specification."""

    ROTATION = "rotation"
    DIRECT = "direct"


class RotationAction(IntEnum):
    """Rotation-and-thrust action space: ``Discrete(5)``."""

    NOOP = 0
    THRUST_FORWARD = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    SHOOT = 4


class DirectAction(IntEnum):
    """World-direction action space: ``Discrete(6)``."""

    NOOP = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    SHOOT = 5


ActionEnum: TypeAlias = type[RotationAction] | type[DirectAction]


@dataclass(frozen=True, slots=True)
class ControlScheme:
    """Immutable metadata used by environments, UI, and evaluation scripts."""

    style: ControlStyle
    action_enum: ActionEnum
    display_name: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(action.name for action in self.action_enum)

    @property
    def action_count(self) -> int:
        return len(self.action_enum)


CONTROL_SCHEMES = {
    ControlStyle.ROTATION: ControlScheme(
        style=ControlStyle.ROTATION,
        action_enum=RotationAction,
        display_name="Rotation + thrust",
    ),
    ControlStyle.DIRECT: ControlScheme(
        style=ControlStyle.DIRECT,
        action_enum=DirectAction,
        display_name="Direct directional",
    ),
}

# Backward-compatible constants used by earlier code and documentation.
ROTATION_ACTIONS = CONTROL_SCHEMES[ControlStyle.ROTATION].names
DIRECT_ACTIONS = CONTROL_SCHEMES[ControlStyle.DIRECT].names


@dataclass(slots=True)
class ControlCommand:
    """Continuous command consumed by :class:`arena.core.ArenaCore`."""

    thrust: float = 0.0
    turn: float = 0.0
    move: Vec = field(default_factory=vec)
    shoot: bool = False
    preserve_momentum: bool = False


def control_scheme(style: ControlStyle | str) -> ControlScheme:
    """Return the immutable specification for ``style``."""

    return CONTROL_SCHEMES[ControlStyle(style)]


def action_names(style: ControlStyle | str) -> tuple[str, ...]:
    """Return action meanings ordered by their exact integer IDs."""

    return control_scheme(style).names


def action_count(style: ControlStyle | str) -> int:
    """Return 5 for rotation/thrust and 6 for direct movement."""

    return control_scheme(style).action_count


def action_name(style: ControlStyle | str, action: int) -> str:
    """Return one action label and reject IDs outside that style's space."""

    scheme = control_scheme(style)
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise ValueError(f"Action {action!r} is invalid for {scheme.style.value}")
    try:
        return scheme.action_enum(int(action)).name
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action {action!r} is invalid for {scheme.style.value}") from exc


def decode_action(style: ControlStyle | str, action: int) -> ControlCommand:
    """Translate one rubric-defined discrete action into Arena movement.

    Only one discrete action is applied per environment step. ``SHOOT`` keeps
    the ship's current velocity so firing does not act as an accidental brake.
    """

    scheme = control_scheme(style)
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise ValueError(f"Action {action!r} is invalid for {scheme.style.value}")
    try:
        selected = scheme.action_enum(int(action))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action {action!r} is invalid for {scheme.style.value}") from exc

    if scheme.style is ControlStyle.ROTATION:
        if selected is RotationAction.NOOP:
            return ControlCommand()
        if selected is RotationAction.THRUST_FORWARD:
            return ControlCommand(thrust=1.0)
        if selected is RotationAction.ROTATE_LEFT:
            return ControlCommand(turn=-1.0)
        if selected is RotationAction.ROTATE_RIGHT:
            return ControlCommand(turn=1.0)
        return ControlCommand(shoot=True, preserve_momentum=True)

    if selected is DirectAction.NOOP:
        return ControlCommand()
    if selected is DirectAction.MOVE_UP:
        return ControlCommand(move=vec(0.0, -1.0))
    if selected is DirectAction.MOVE_DOWN:
        return ControlCommand(move=vec(0.0, 1.0))
    if selected is DirectAction.MOVE_LEFT:
        return ControlCommand(move=vec(-1.0, 0.0))
    if selected is DirectAction.MOVE_RIGHT:
        return ControlCommand(move=vec(1.0, 0.0))
    return ControlCommand(shoot=True, preserve_momentum=True)
