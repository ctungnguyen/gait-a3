from __future__ import annotations

import unittest

import numpy as np

from arena.adapters import OBSERVATION_SIZE
from arena.controls import ControlStyle
from arena.env import ArenaEnv, LegacyArenaAdapter


class ArenaEnvironmentApiTests(unittest.TestCase):
    def test_modern_reset_and_step_signatures(self):
        env = ArenaEnv(control_style=ControlStyle.DIRECT, seed=9)
        observation, info = env.reset(seed=9)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, 6)
        self.assertEqual(info["phase"], 1)

        result = env.step(0)
        self.assertEqual(len(result), 5)
        next_observation, reward, terminated, truncated, step_info = result
        self.assertTrue(env.observation_space.contains(next_observation))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertEqual(step_info["action_name"], "NOOP")
        env.close()

    def test_rotation_action_space_has_five_actions(self):
        env = ArenaEnv(control_style=ControlStyle.ROTATION)
        self.assertEqual(env.action_space.n, 5)
        env.step(4)
        self.assertEqual(len(env.core.projectiles), 1)
        env.close()

    def test_legacy_adapter_matches_assignment_four_value_api(self):
        env = LegacyArenaAdapter(control_style=ControlStyle.DIRECT, seed=4)
        observation = env.reset(seed=4)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        result = env.step(0)
        self.assertEqual(len(result), 4)
        _observation, reward, done, info = result
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertIn("terminated", info)
        self.assertIn("truncated", info)
        env.close()

    def test_invalid_action_is_rejected(self):
        env = ArenaEnv(control_style=ControlStyle.ROTATION)
        with self.assertRaises(ValueError):
            env.step(5)
        env.close()

    def test_debug_render_forwards_live_action_without_changing_api(self):
        calls = []

        class RendererSpy:
            def draw(self, core, style, **kwargs):
                calls.append((core, style, kwargs))
                return "frame"

            def close(self):
                pass

        env = ArenaEnv(control_style=ControlStyle.DIRECT, render_mode="rgb_array")
        env.step(4)
        env._renderer = RendererSpy()
        result = env.render(status="PAUSED", debug=True)

        self.assertEqual(result, "frame")
        self.assertEqual(calls[0][1], ControlStyle.DIRECT)
        self.assertEqual(calls[0][2]["status"], "PAUSED")
        self.assertTrue(calls[0][2]["debug"])
        self.assertEqual(calls[0][2]["action_label"], "MOVE_RIGHT")
        env.close()


if __name__ == "__main__":
    unittest.main()
