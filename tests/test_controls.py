from __future__ import annotations

import math
import unittest

import numpy as np

from arena import (
    ControlStyle,
    DirectAction,
    RotationAction,
    action_names,
    make_direct_env,
    make_rotation_env,
)
from arena.controls import action_count, action_name, decode_action


class Block4ControlContractTests(unittest.TestCase):
    def test_rotation_action_contract_exactly_matches_assignment(self):
        self.assertEqual(action_count(ControlStyle.ROTATION), 5)
        self.assertEqual(
            action_names(ControlStyle.ROTATION),
            (
                "NOOP",
                "THRUST_FORWARD",
                "ROTATE_LEFT",
                "ROTATE_RIGHT",
                "SHOOT",
            ),
        )
        self.assertEqual(RotationAction.NOOP.value, 0)
        self.assertEqual(RotationAction.THRUST_FORWARD.value, 1)
        self.assertEqual(RotationAction.ROTATE_LEFT.value, 2)
        self.assertEqual(RotationAction.ROTATE_RIGHT.value, 3)
        self.assertEqual(RotationAction.SHOOT.value, 4)

    def test_direct_action_contract_exactly_matches_assignment(self):
        self.assertEqual(action_count(ControlStyle.DIRECT), 6)
        self.assertEqual(
            action_names(ControlStyle.DIRECT),
            (
                "NOOP",
                "MOVE_UP",
                "MOVE_DOWN",
                "MOVE_LEFT",
                "MOVE_RIGHT",
                "SHOOT",
            ),
        )
        self.assertEqual(DirectAction.NOOP.value, 0)
        self.assertEqual(DirectAction.MOVE_UP.value, 1)
        self.assertEqual(DirectAction.MOVE_DOWN.value, 2)
        self.assertEqual(DirectAction.MOVE_LEFT.value, 3)
        self.assertEqual(DirectAction.MOVE_RIGHT.value, 4)
        self.assertEqual(DirectAction.SHOOT.value, 5)

    def test_every_direct_action_decodes_to_the_expected_command(self):
        expected_moves = {
            DirectAction.NOOP: (0.0, 0.0),
            DirectAction.MOVE_UP: (0.0, -1.0),
            DirectAction.MOVE_DOWN: (0.0, 1.0),
            DirectAction.MOVE_LEFT: (-1.0, 0.0),
            DirectAction.MOVE_RIGHT: (1.0, 0.0),
        }
        for action, expected in expected_moves.items():
            with self.subTest(action=action):
                command = decode_action(ControlStyle.DIRECT, action.value)
                np.testing.assert_array_equal(command.move, expected)
                self.assertFalse(command.shoot)

        shoot = decode_action(ControlStyle.DIRECT, DirectAction.SHOOT.value)
        self.assertTrue(shoot.shoot)
        self.assertTrue(shoot.preserve_momentum)

    def test_every_rotation_action_decodes_to_the_expected_command(self):
        noop = decode_action(ControlStyle.ROTATION, RotationAction.NOOP.value)
        thrust = decode_action(ControlStyle.ROTATION, RotationAction.THRUST_FORWARD.value)
        left = decode_action(ControlStyle.ROTATION, RotationAction.ROTATE_LEFT.value)
        right = decode_action(ControlStyle.ROTATION, RotationAction.ROTATE_RIGHT.value)
        shoot = decode_action(ControlStyle.ROTATION, RotationAction.SHOOT.value)

        self.assertEqual((noop.thrust, noop.turn, noop.shoot), (0.0, 0.0, False))
        self.assertEqual((thrust.thrust, thrust.turn, thrust.shoot), (1.0, 0.0, False))
        self.assertEqual((left.thrust, left.turn, left.shoot), (0.0, -1.0, False))
        self.assertEqual((right.thrust, right.turn, right.shoot), (0.0, 1.0, False))
        self.assertTrue(shoot.shoot)
        self.assertTrue(shoot.preserve_momentum)

    def test_invalid_actions_are_style_specific(self):
        with self.assertRaises(ValueError):
            action_name(ControlStyle.ROTATION, 5)
        with self.assertRaises(ValueError):
            decode_action(ControlStyle.DIRECT, 6)
        with self.assertRaises(ValueError):
            decode_action(ControlStyle.DIRECT, 1.5)
        with self.assertRaises(ValueError):
            decode_action(ControlStyle.ROTATION, True)

    def test_direct_actions_move_on_world_axes_and_set_facing(self):
        cases = (
            (DirectAction.MOVE_UP, 0, -1, -math.pi / 2.0),
            (DirectAction.MOVE_DOWN, 0, 1, math.pi / 2.0),
            (DirectAction.MOVE_LEFT, -1, 0, math.pi),
            (DirectAction.MOVE_RIGHT, 1, 0, 0.0),
        )
        for action, expected_x_sign, expected_y_sign, expected_angle in cases:
            with self.subTest(action=action):
                env = make_direct_env(seed=31)
                start = env.core.player.pos.copy()
                env.step(action.value)
                delta = env.core.player.pos - start
                self.assertEqual(int(np.sign(delta[0])), expected_x_sign)
                self.assertEqual(int(np.sign(delta[1])), expected_y_sign)
                self.assertAlmostEqual(env.core.player.angle, expected_angle)
                env.close()

    def test_rotation_actions_turn_without_translation_then_thrust_forward(self):
        left_env = make_rotation_env(seed=41)
        left_start = left_env.core.player.pos.copy()
        left_angle = left_env.core.player.angle
        left_env.step(RotationAction.ROTATE_LEFT.value)
        np.testing.assert_allclose(left_env.core.player.pos, left_start)
        self.assertLess(left_env.core.player.angle, left_angle)
        left_env.close()

        right_env = make_rotation_env(seed=42)
        right_start = right_env.core.player.pos.copy()
        right_angle = right_env.core.player.angle
        right_env.step(RotationAction.ROTATE_RIGHT.value)
        np.testing.assert_allclose(right_env.core.player.pos, right_start)
        self.assertGreater(right_env.core.player.angle, right_angle)
        right_env.close()

        thrust_env = make_rotation_env(seed=43)
        thrust_start = thrust_env.core.player.pos.copy()
        thrust_env.step(RotationAction.THRUST_FORWARD.value)
        self.assertAlmostEqual(float(thrust_env.core.player.pos[0]), float(thrust_start[0]))
        self.assertLess(float(thrust_env.core.player.pos[1]), float(thrust_start[1]))
        thrust_env.close()

    def test_style_locked_factories_create_separate_spaces(self):
        rotation = make_rotation_env(seed=8)
        direct = make_direct_env(seed=8)
        self.assertEqual(rotation.control_style, ControlStyle.ROTATION)
        self.assertEqual(direct.control_style, ControlStyle.DIRECT)
        self.assertEqual(rotation.action_space.n, 5)
        self.assertEqual(direct.action_space.n, 6)
        self.assertEqual(rotation.observation_space.shape, direct.observation_space.shape)
        rotation.close()
        direct.close()

        with self.assertRaises(TypeError):
            make_rotation_env(control_style=ControlStyle.DIRECT)


if __name__ == "__main__":
    unittest.main()
