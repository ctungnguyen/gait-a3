"""Small NumPy vector helpers shared by the headless simulation."""

from __future__ import annotations

import math
import numpy as np


Vec = np.ndarray
EPSILON = 1e-8


def vec(x: float = 0.0, y: float = 0.0) -> Vec:
    return np.array((x, y), dtype=np.float32)


def length(value: Vec) -> float:
    return float(np.linalg.norm(value))


def normalized(value: Vec) -> Vec:
    magnitude = length(value)
    if magnitude <= EPSILON:
        return vec()
    return np.asarray(value, dtype=np.float32) / magnitude


def limited(value: Vec, maximum: float) -> Vec:
    result = np.asarray(value, dtype=np.float32).copy()
    magnitude = length(result)
    if maximum >= 0.0 and magnitude > maximum and magnitude > EPSILON:
        result *= maximum / magnitude
    return result


def forward(angle_radians: float) -> Vec:
    return vec(math.cos(angle_radians), math.sin(angle_radians))


def circle_overlap(first_pos: Vec, first_radius: float, second_pos: Vec, second_radius: float) -> bool:
    total = first_radius + second_radius
    delta = first_pos - second_pos
    return float(np.dot(delta, delta)) <= total * total


def clamp_to_arena(position: Vec, radius: float, width: float, height: float) -> Vec:
    position[0] = float(np.clip(position[0], radius, width - radius))
    position[1] = float(np.clip(position[1], radius, height - radius))
    return position
