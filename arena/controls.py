"""Action-to-command adapters.

The Arena core accepts a continuous ``ControlCommand``. These small adapters
provide the two discrete action sets required by the assessment. Teammates can
tune the mappings without touching collision, spawning, or rendering code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .math2d import Vec, vec


class ControlStyle(str, Enum):
    ROTATION = "rotation"
    DIRECT = "direct"


ROTATION_ACTIONS = ("NOOP", "THRUST", "ROTATE_LEFT", "ROTATE_RIGHT", "SHOOT")
DIRECT_ACTIONS = ("NOOP", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "SHOOT")


@dataclass(slots=True)
class ControlCommand:
    thrust: float = 0.0
    turn: float = 0.0
    move: Vec = field(default_factory=vec)
    shoot: bool = False
    preserve_momentum: bool = False


def action_count(style: ControlStyle | str) -> int:
    style = ControlStyle(style)
    return len(ROTATION_ACTIONS if style is ControlStyle.ROTATION else DIRECT_ACTIONS)


def action_name(style: ControlStyle | str, action: int) -> str:
    style = ControlStyle(style)
    names = ROTATION_ACTIONS if style is ControlStyle.ROTATION else DIRECT_ACTIONS
    if not 0 <= int(action) < len(names):
        raise ValueError(f"Action {action} is invalid for {style.value}")
    return names[int(action)]


def decode_action(style: ControlStyle | str, action: int) -> ControlCommand:
    style = ControlStyle(style)
    action = int(action)
    action_name(style, action)  # validates the range

    if style is ControlStyle.ROTATION:
        return (
            ControlCommand()
            if action == 0
            else ControlCommand(thrust=1.0)
            if action == 1
            else ControlCommand(turn=-1.0)
            if action == 2
            else ControlCommand(turn=1.0)
            if action == 3
            else ControlCommand(shoot=True, preserve_momentum=True)
        )

    return (
        ControlCommand()
        if action == 0
        else ControlCommand(move=vec(0.0, -1.0))
        if action == 1
        else ControlCommand(move=vec(0.0, 1.0))
        if action == 2
        else ControlCommand(move=vec(-1.0, 0.0))
        if action == 3
        else ControlCommand(move=vec(1.0, 0.0))
        if action == 4
        else ControlCommand(shoot=True, preserve_momentum=True)
    )
