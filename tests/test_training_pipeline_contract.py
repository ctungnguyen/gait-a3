from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from part2.arena import ArenaConfig, ControlStyle
from part2.training import artifacts
from part2.training.config import load_training_settings
from part2.training.pipeline import train_style
import part2.training.pipeline as pipeline


PROJECT_ROOT = Path(__file__).parents[1]


class FakeMonitor:
    def __init__(self, env, **_kwargs):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def close(self):
        self.env.close()


class FakeDummyVecEnv:
    def __init__(self, factories):
        self.envs = [factory() for factory in factories]
        self.action_space = self.envs[0].action_space

    def close(self):
        for env in self.envs:
            env.close()


class FakePPO:
    def __init__(self, _policy, env, **_kwargs):
        self.action_space = env.action_space
        self.num_timesteps = 0

    def learn(self, total_timesteps, **_kwargs):
        self.num_timesteps += int(total_timesteps)
        return self

    def save(self, path):
        destination = Path(path)
        if destination.suffix != ".zip":
            destination = destination.with_suffix(".zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(self.action_space.n), encoding="utf-8")

    @classmethod
    def load(cls, path, env=None, **_kwargs):
        source = Path(path)
        obj = cls.__new__(cls)
        obj.action_space = SimpleNamespace(n=int(source.read_text(encoding="utf-8")))
        obj.num_timesteps = 0
        if env is not None and env.action_space.n != obj.action_space.n:
            raise ValueError("action space mismatch")
        return obj

    def predict(self, _observation, deterministic=True):
        del deterministic
        return np.asarray(0), None


class FakeCallback:
    def __init__(self, *_args, **_kwargs):
        pass


class FakeBaseCallback:
    def __init__(self, verbose=0):
        self.verbose = verbose


class TrainingPipelineContractTests(unittest.TestCase):
    def test_pipeline_writes_separate_model_metadata_and_evaluation(self):
        dependencies = {
            "gymnasium": SimpleNamespace(__version__="test"),
            "stable_baselines3": SimpleNamespace(__version__="test"),
            "torch": SimpleNamespace(
                __version__="test",
                nn=SimpleNamespace(Tanh=object, ReLU=object),
            ),
            "PPO": FakePPO,
            "BaseCallback": FakeBaseCallback,
            "CallbackList": FakeCallback,
            "CheckpointCallback": FakeCallback,
            "EvalCallback": FakeCallback,
            "Monitor": FakeMonitor,
            "set_random_seed": lambda _seed: None,
            "DummyVecEnv": FakeDummyVecEnv,
        }
        settings = load_training_settings(
            PROJECT_ROOT / "part2" / "config" / "training.json",
            total_timesteps=8,
            n_envs=1,
            final_eval_episodes=2,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(artifacts, "PROJECT_ROOT", root),
                patch.object(pipeline, "_require_training_dependencies", return_value=dependencies),
            ):
                summary = train_style(
                    style=ControlStyle.ROTATION,
                    arena_config=ArenaConfig(max_steps=5),
                    settings=settings,
                    run_name="contract",
                    promote=True,
                    progress_bar=False,
                )

            model_path = root / "models" / "rotation_agent.zip"
            metadata_path = root / "models" / "rotation_agent.metadata.json"
            self.assertTrue(model_path.exists())
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["algorithm"], "PPO")
            self.assertEqual(metadata["control_style"], "rotation")
            self.assertEqual(metadata["action_count"], 5)
            self.assertEqual(summary["final_evaluation"]["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
