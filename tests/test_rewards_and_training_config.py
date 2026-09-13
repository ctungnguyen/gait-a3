from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from part2.arena import ControlStyle, make_direct_env
from part2.arena.core import ArenaCore
from part2.arena.entities import ArenaEvent, Projectile, StepOutcome
from part2.arena.math2d import normalized, vec
from part2.arena.rewards import ProgressionReward
from part2.arena.adapters import OBSERVATION_SIZE
from part2.training.artifacts import ArtifactPaths, sha256_file
from part2.training.config import load_training_settings
from part2.training.evaluation import summarize_episodes, validate_model_contract


PROJECT_ROOT = Path(__file__).parents[1]


class ProgressionRewardTests(unittest.TestCase):
    def test_event_rewards_are_auditable_components(self):
        core = ArenaCore(seed=4)
        reward = ProgressionReward()
        reward.reset(core)
        outcome = StepOutcome(
            events=[
                ArenaEvent("enemy_damaged", amount=10.0),
                ArenaEvent("enemy_destroyed"),
                ArenaEvent("spawner_damaged", amount=10.0),
                ArenaEvent("spawner_destroyed"),
                ArenaEvent("phase_advanced"),
                ArenaEvent("player_damaged", amount=10.0),
                ArenaEvent("player_died"),
                ArenaEvent("time_limit_reached"),
            ],
            terminated=True,
            truncated=False,
        )
        result = reward(core, outcome)

        self.assertAlmostEqual(reward.last_components["enemy_damage"], 0.2)
        self.assertAlmostEqual(reward.last_components["spawner_damage"], 0.5)
        self.assertAlmostEqual(reward.last_components["player_damage"], -1.5)
        self.assertAlmostEqual(result, sum(reward.last_components.values()))

    def test_approaching_spawner_is_better_than_retreating(self):
        toward_core = ArenaCore(seed=9)
        toward_reward = ProgressionReward()
        toward_reward.reset(toward_core)
        target = toward_core.nearest_spawner()
        direction = normalized(target.pos - toward_core.player.pos)
        toward_core.player.pos += direction * 5.0
        toward_value = toward_reward(toward_core, StepOutcome([], False, False))

        away_core = ArenaCore(seed=9)
        away_reward = ProgressionReward()
        away_reward.reset(away_core)
        target = away_core.nearest_spawner()
        direction = normalized(target.pos - away_core.player.pos)
        away_core.player.pos -= direction * 5.0
        away_value = away_reward(away_core, StepOutcome([], False, False))

        self.assertGreater(toward_value, away_value)
        self.assertGreater(toward_reward.last_components["approach"], 0.0)
        self.assertLess(away_reward.last_components["approach"], 0.0)

    def test_aligned_shot_scores_better_than_wasted_shot(self):
        aligned_core = ArenaCore(seed=2)
        target = aligned_core.nearest_spawner()
        relative = target.pos - aligned_core.player.pos
        aligned_core.player.angle = math.atan2(float(relative[1]), float(relative[0]))
        aligned_reward = ProgressionReward()
        aligned_reward.reset(aligned_core)
        aligned_reward(
            aligned_core,
            StepOutcome([ArenaEvent("projectile_fired")], False, False),
        )

        wasted_core = ArenaCore(seed=2)
        target = wasted_core.nearest_spawner()
        relative = target.pos - wasted_core.player.pos
        wasted_core.player.angle = math.atan2(float(relative[1]), float(relative[0])) + math.pi
        wasted_reward = ProgressionReward()
        wasted_reward.reset(wasted_core)
        wasted_reward(
            wasted_core,
            StepOutcome([ArenaEvent("projectile_fired")], False, False),
        )

        self.assertGreater(
            aligned_reward.last_components["shot_quality"],
            wasted_reward.last_components["shot_quality"],
        )

    def test_reward_adapter_does_not_mutate_game_mechanics(self):
        core = ArenaCore(seed=7)
        reward = ProgressionReward()
        reward.reset(core)
        position = core.player.pos.copy()
        health = core.player.health
        spawner_health = [spawner.health for spawner in core.spawners]
        reward(core, StepOutcome([], False, False))
        np.testing.assert_array_equal(core.player.pos, position)
        self.assertEqual(core.player.health, health)
        self.assertEqual([spawner.health for spawner in core.spawners], spawner_health)

    def test_environment_tracks_terminal_evidence_metrics(self):
        env = make_direct_env(reward_fn=ProgressionReward(), seed=14)
        target = env.core.spawners[0]
        env.core.spawners = [target]
        target.health = env.core.config.projectile_damage
        target.spawn_timer = 999.0
        env.core.projectiles.append(
            Projectile(
                entity_id=env.core._id(),
                pos=target.pos.copy(),
                vel=vec(),
                damage=env.core.config.projectile_damage,
                lifetime=1.0,
            )
        )
        _obs, reward, _terminated, _truncated, info = env.step(0)
        self.assertEqual(info["episode_spawners_destroyed"], 1)
        self.assertEqual(info["episode_phases_advanced"], 1)
        self.assertEqual(info["episode_max_phase"], 2)
        self.assertTrue(info["is_success"])
        self.assertIn("spawner_destroyed", info["reward_components"])
        self.assertGreater(reward, 0.0)
        env.close()


class TrainingConfigurationTests(unittest.TestCase):
    config_path = PROJECT_ROOT / "part2" / "config" / "training.json"

    def test_default_ppo_configuration_has_two_hidden_layers(self):
        settings = load_training_settings(self.config_path)
        self.assertEqual(settings.algorithm, "PPO")
        self.assertEqual(settings.net_arch, (128, 128))
        self.assertEqual(settings.n_envs, 4)
        rollout_size = settings.ppo["n_steps"] * settings.n_envs
        self.assertEqual(rollout_size % settings.ppo["batch_size"], 0)

    def test_named_presets_make_meaningful_changes(self):
        baseline = load_training_settings(self.config_path, "baseline")
        exploratory = load_training_settings(self.config_path, "exploratory")
        stable = load_training_settings(self.config_path, "stable")
        self.assertGreater(exploratory.ppo["ent_coef"], baseline.ppo["ent_coef"])
        self.assertLess(stable.ppo["learning_rate"], baseline.ppo["learning_rate"])
        self.assertNotEqual(stable.net_arch, baseline.net_arch)

    def test_cli_overrides_are_validated(self):
        settings = load_training_settings(
            self.config_path,
            total_timesteps=12345,
            seed=88,
            final_eval_episodes=3,
        )
        self.assertEqual(settings.total_timesteps, 12345)
        self.assertEqual(settings.seed, 88)
        self.assertEqual(settings.final_eval_episodes, 3)
        with self.assertRaises(ValueError):
            load_training_settings(self.config_path, "missing")

    def test_artifact_paths_keep_models_separate(self):
        rotation = ArtifactPaths(ControlStyle.ROTATION, "final")
        direct = ArtifactPaths(ControlStyle.DIRECT, "final")
        self.assertEqual(rotation.output_model.name, "rotation_agent.zip")
        self.assertEqual(direct.output_model.name, "direct_agent.zip")
        self.assertNotEqual(rotation.output_model, direct.output_model)

    def test_model_metadata_checksum_rejects_a_replaced_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "rotation_agent.zip"
            model_path.write_bytes(b"original model")
            metadata_path = model_path.with_suffix(".metadata.json")
            metadata_path.write_text(
                json.dumps(
                    {
                        "algorithm": "PPO",
                        "control_style": "rotation",
                        "action_count": 5,
                        "action_meanings": [
                            "NOOP",
                            "THRUST_FORWARD",
                            "ROTATE_LEFT",
                            "ROTATE_RIGHT",
                            "SHOOT",
                        ],
                        "observation_size": OBSERVATION_SIZE,
                        "model_sha256": sha256_file(model_path),
                    }
                ),
                encoding="utf-8",
            )
            model = SimpleNamespace(action_space=SimpleNamespace(n=5))
            validate_model_contract(model, ControlStyle.ROTATION, model_path)

            model_path.write_bytes(b"different model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_model_contract(model, ControlStyle.ROTATION, model_path)

    def test_episode_summary_reports_required_comparison_metrics(self):
        rows = [
            {
                "return": 10.0,
                "steps": 100,
                "episode_max_phase": 2,
                "is_success": True,
                "episode_enemies_destroyed": 3,
                "episode_spawners_destroyed": 2,
                "episode_damage_dealt": 50,
                "episode_damage_taken": 10,
                "episode_shots_fired": 8,
            },
            {
                "return": 0.0,
                "steps": 50,
                "episode_max_phase": 1,
                "is_success": False,
                "episode_enemies_destroyed": 1,
                "episode_spawners_destroyed": 0,
                "episode_damage_dealt": 10,
                "episode_damage_taken": 30,
                "episode_shots_fired": 4,
            },
        ]
        summary = summarize_episodes(rows)
        self.assertEqual(summary["mean_return"], 5.0)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["highest_phase"], 2)


if __name__ == "__main__":
    unittest.main()
