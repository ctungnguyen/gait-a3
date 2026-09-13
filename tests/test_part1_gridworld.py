from __future__ import annotations

from types import SimpleNamespace
import math
from pathlib import Path
import tempfile
import unittest

from part1.agents import QLearningAgent, QTable, SARSAAgent
from part1.config import GridworldSettings, load_gridworld_settings
from part1.environment import GridWorld
from part1.levels import LevelDefinition, get_level, level_signature
from part1.training import (
    evaluate,
    evaluation_summary,
    shortest_collectible_steps,
    shortest_external_reward_steps,
    train,
)


def level(rows: list[str]) -> LevelDefinition:
    return LevelDefinition(99, "test", "test", tuple(rows), "test fixture")


class GridworldMechanicsTests(unittest.TestCase):
    def test_level_five_uses_two_stochastic_monsters(self):
        env = GridWorld(get_level(5), seed=1)
        self.assertEqual(len(env.monsters), 2)

    def test_level_signature_detects_layout_changes(self):
        original = level(["S A"])
        changed = level(["S  A"])
        self.assertNotEqual(level_signature(original), level_signature(changed))

    def test_rocks_block_without_changing_environment_reward(self):
        env = GridWorld(level(["SR A"]), seed=1)
        result = env.step(1)
        self.assertEqual(env.agent, (0, 0))
        self.assertTrue(result.info["blocked"])
        self.assertEqual(result.reward, 0.0)

    def test_fire_is_immediate_terminal_death(self):
        env = GridWorld(level(["SF A"]), seed=1)
        result = env.step(1)
        self.assertTrue(result.done)
        self.assertEqual(result.info["event"], "death")
        self.assertEqual(result.reward, 0.0)

    def test_key_is_zero_chest_is_two_and_apple_is_one(self):
        env = GridWorld(level(["SKCA"]), seed=1)
        key = env.step(1)
        chest = env.step(1)
        apple = env.step(1)
        self.assertEqual((key.reward, chest.reward, apple.reward), (0.0, 2.0, 1.0))
        self.assertFalse(key.done)
        self.assertFalse(chest.done)
        self.assertTrue(apple.done)
        self.assertEqual(apple.info["event"], "win")

    def test_monster_gets_one_move_opportunity_after_player_action(self):
        env = GridWorld(level(["SM A"]), monster_move_chance=1.0, seed=1)
        env.rng = SimpleNamespace(
            random=lambda: 0.0,
            choice=lambda choices: (0, 0) if (0, 0) in choices else choices[0],
        )
        result = env.step(3)  # blocked left, then monster moves into the player
        self.assertTrue(result.done)
        self.assertIn("monster_moved", result.info["events"])
        self.assertIn("death", result.info["events"])
        self.assertEqual(result.info["monster_moves"], 1)

    def test_monster_positions_are_part_of_the_markov_state(self):
        env = GridWorld(level(["S M A"]), monster_move_chance=0.0, seed=1)
        state = env.reset()
        self.assertEqual(state[-1], ((2, 0),))


class TabularAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.settings = GridworldSettings(
            episodes=10,
            alpha=0.5,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.2,
            epsilon_decay_episodes=8,
        ).validate()
        self.state = (0, 0, 1, 0, 0, ())
        self.next_state = (1, 0, 1, 0, 0, ())

    def test_linear_epsilon_decay_has_exact_endpoints(self):
        agent = QLearningAgent(self.settings, seed=1)
        self.assertEqual(agent.epsilon(0), 1.0)
        self.assertAlmostEqual(agent.epsilon(4), 0.6)
        self.assertEqual(agent.epsilon(8), 0.2)
        self.assertEqual(agent.epsilon(80), 0.2)

    def test_random_tie_breaking_uses_all_equal_best_actions(self):
        agent = QLearningAgent(self.settings, seed=7)
        actions = {agent.choose_action(self.state, 0.0, evaluate=True) for _ in range(200)}
        self.assertEqual(actions, {0, 1, 2, 3})

    def test_saved_qtable_prunes_only_information_free_zero_rows(self):
        table = QTable()
        _ = table.values(self.state)
        table.set(self.next_state, 2, 1.25)
        with tempfile.TemporaryDirectory() as temporary:
            path = table.save(Path(temporary) / "policy.json")
            loaded, metadata = QTable.load(path)
        self.assertNotIn(self.state, loaded.q)
        self.assertEqual(loaded.get(self.state, 0), 0.0)
        self.assertEqual(loaded.get(self.next_state, 2), 1.25)
        self.assertEqual(metadata["all_zero_rows_pruned"], 1)

    def test_q_learning_uses_off_policy_maximum_target(self):
        agent = QLearningAgent(self.settings, seed=1)
        agent.qtable.q[self.next_state] = [1.0, 2.0, 3.0, 4.0]
        agent.update(self.state, 0, 1.0, self.next_state, False)
        self.assertAlmostEqual(agent.qtable.get(self.state, 0), 0.5 * (1.0 + 0.9 * 4.0))

    def test_sarsa_uses_the_selected_next_action(self):
        agent = SARSAAgent(self.settings, seed=1)
        agent.qtable.q[self.next_state] = [1.0, 2.0, 3.0, 4.0]
        agent.update(self.state, 0, 1.0, self.next_state, 1, False)
        self.assertAlmostEqual(agent.qtable.get(self.state, 0), 0.5 * (1.0 + 0.9 * 2.0))

    def test_intrinsic_counter_is_per_episode_and_uses_required_formula(self):
        agent = QLearningAgent(self.settings, seed=1)
        agent.start_episode(self.state)
        first_return = agent.intrinsic_reward(self.state, strength=0.2)
        second_return = agent.intrinsic_reward(self.state, strength=0.2)
        self.assertAlmostEqual(first_return, 0.2 / math.sqrt(2))
        self.assertAlmostEqual(second_return, 0.2 / math.sqrt(3))
        agent.start_episode(self.next_state)
        self.assertAlmostEqual(agent.intrinsic_reward(self.state, strength=0.2), 0.2)


class GridworldLearningSmokeTests(unittest.TestCase):
    def test_file_backed_config_applies_level_override(self):
        settings = load_gridworld_settings(level=4)
        self.assertEqual(settings.monster_move_chance, 0.4)
        self.assertEqual(settings.episodes, 18_000)

    def test_level_zero_learns_the_exact_shortest_collection_path(self):
        result = train(0, "q_learning", episodes=1_000)
        summary = evaluation_summary(evaluate(result, episodes=5))
        self.assertEqual(shortest_collectible_steps(get_level(0)), 15)
        self.assertEqual(summary["success_rate"], 1)
        self.assertEqual(summary["mean_success_steps"], 15)

    def test_level_six_is_solvable_but_delays_external_reward(self):
        self.assertEqual(shortest_external_reward_steps(get_level(6)), 61)
        self.assertEqual(shortest_collectible_steps(get_level(6)), 81)

    def test_q_learning_handles_stochastic_monster_transitions(self):
        result = train(4, "q_learning", episodes=1_000)
        summary = evaluation_summary(evaluate(result, episodes=10))
        self.assertGreater(summary["mean_monster_moves"], 0)
        self.assertGreaterEqual(summary["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
