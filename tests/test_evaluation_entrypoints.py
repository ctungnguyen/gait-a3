from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from arena import ControlStyle
import evaluate_agent


class SeparateEvaluationEntrypointTests(unittest.TestCase):
    def test_default_model_outputs_are_separate(self):
        rotation = evaluate_agent.default_model_path(ControlStyle.ROTATION)
        direct = evaluate_agent.default_model_path(ControlStyle.DIRECT)
        self.assertEqual(rotation.name, "rotation_agent.zip")
        self.assertEqual(direct.name, "direct_agent.zip")
        self.assertNotEqual(rotation, direct)

    def test_rotation_evaluator_rejects_direct_action_space_model(self):
        fake_model = SimpleNamespace(action_space=SimpleNamespace(n=6))

        class FakeEnv:
            action_space = SimpleNamespace(n=5)

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        fake_env = FakeEnv()
        with (
            patch.object(evaluate_agent, "_load_model", return_value=fake_model),
            patch.object(evaluate_agent.ArenaConfig, "from_json", return_value=object()),
            patch.object(evaluate_agent, "make_rotation_env", return_value=fake_env),
        ):
            with self.assertRaisesRegex(SystemExit, "Wrong model for rotation"):
                evaluate_agent.evaluate_style(
                    ControlStyle.ROTATION,
                    default_model=Path("models/rotation_agent.zip"),
                    argv=["--episodes", "1"],
                )

        self.assertTrue(fake_env.closed)


if __name__ == "__main__":
    unittest.main()
