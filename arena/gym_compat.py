"""Gymnasium import plus a tiny fallback for dependency-free core tests."""

from __future__ import annotations

import numpy as np


try:  # pragma: no cover - exercised when the real dependency is installed
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
    BaseEnv = gym.Env
except ImportError:  # pragma: no cover - the fallback itself is tested
    GYMNASIUM_AVAILABLE = False

    class BaseEnv:
        metadata: dict = {}

        def reset(self, *, seed=None, options=None):
            del seed, options

    class Discrete:
        def __init__(self, n: int):
            self.n = int(n)
            self._rng = np.random.default_rng()

        def sample(self) -> int:
            return int(self._rng.integers(self.n))

        def contains(self, value) -> bool:
            return isinstance(value, (int, np.integer)) and 0 <= int(value) < self.n

    class Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)
            self.low = np.full(self.shape, low, dtype=self.dtype)
            self.high = np.full(self.shape, high, dtype=self.dtype)

        def contains(self, value) -> bool:
            array = np.asarray(value)
            return (
                array.shape == self.shape
                and np.all(array >= self.low)
                and np.all(array <= self.high)
            )

        def sample(self):
            return np.random.default_rng().uniform(self.low, self.high).astype(self.dtype)

    class _Spaces:
        Discrete = Discrete
        Box = Box

    spaces = _Spaces()
