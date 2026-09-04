from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import run_manual


class ManualStartupOrderTests(unittest.TestCase):
    def test_first_render_happens_before_event_queue_read(self):
        order: list[str] = []

        class FakeEnv:
            def __init__(self, **_kwargs):
                pass

            def reset(self, **_kwargs):
                order.append("reset")

            def render(self, **_kwargs):
                order.append("render")

            def close(self):
                order.append("close")

        class FakeEventQueue:
            def get(self):
                order.append("event_get")
                return [SimpleNamespace(type=1)]

        fake_pygame = SimpleNamespace(
            QUIT=1,
            KEYDOWN=2,
            event=FakeEventQueue(),
            time=SimpleNamespace(
                Clock=lambda: SimpleNamespace(tick=lambda _fps: None),
            ),
            quit=lambda: order.append("pygame_quit"),
        )
        fake_config = SimpleNamespace(fixed_dt=1.0 / 30.0)

        with (
            patch.dict(sys.modules, {"pygame": fake_pygame}),
            patch.object(run_manual, "ArenaEnv", FakeEnv),
            patch.object(run_manual.ArenaConfig, "from_json", return_value=fake_config),
            patch.object(sys, "argv", ["run_manual.py"]),
        ):
            run_manual.main()

        self.assertLess(order.index("render"), order.index("event_get"))
        self.assertEqual(order[-2:], ["close", "pygame_quit"])

    def test_f3_key_enables_debug_overlay(self):
        debug_values: list[bool] = []

        class FakeEnv:
            def __init__(self, **_kwargs):
                self.core = SimpleNamespace(
                    player=SimpleNamespace(shoot_cooldown=0.0),
                )

            def reset(self, **_kwargs):
                pass

            def render(self, **kwargs):
                debug_values.append(bool(kwargs.get("debug", False)))

            def step(self, _action):
                return None, 0.0, False, False, {}

            def close(self):
                pass

        class FakeEventQueue:
            def __init__(self):
                self.frames = [
                    [SimpleNamespace(type=2, key=12)],
                    [SimpleNamespace(type=1)],
                ]

            def get(self):
                return self.frames.pop(0)

        fake_pygame = SimpleNamespace(
            QUIT=1,
            KEYDOWN=2,
            K_F3=12,
            K_d=24,
            K_ESCAPE=13,
            K_p=14,
            K_r=15,
            K_SPACE=16,
            K_w=17,
            K_UP=18,
            K_s=19,
            K_DOWN=20,
            K_a=21,
            K_LEFT=22,
            K_RIGHT=23,
            event=FakeEventQueue(),
            key=SimpleNamespace(get_pressed=lambda: defaultdict(bool)),
            time=SimpleNamespace(Clock=lambda: SimpleNamespace(tick=lambda _fps: None)),
            quit=lambda: None,
        )
        fake_config = SimpleNamespace(fixed_dt=1.0 / 30.0)

        with (
            patch.dict(sys.modules, {"pygame": fake_pygame}),
            patch.object(run_manual, "ArenaEnv", FakeEnv),
            patch.object(run_manual.ArenaConfig, "from_json", return_value=fake_config),
            patch.object(sys, "argv", ["run_manual.py"]),
        ):
            run_manual.main()

        self.assertEqual(debug_values, [False, True])

    def test_d_key_remains_direct_move_right(self):
        fake_pygame = SimpleNamespace(
            K_SPACE=1,
            K_w=2,
            K_UP=3,
            K_s=4,
            K_DOWN=5,
            K_a=6,
            K_LEFT=7,
            K_d=8,
            K_RIGHT=9,
        )
        keys = defaultdict(bool)
        keys[fake_pygame.K_d] = True

        action = run_manual.selected_action(
            keys,
            run_manual.ControlStyle.DIRECT,
            fake_pygame,
        )

        self.assertEqual(action, 4)  # Direct action 4 is MOVE_RIGHT.

    def test_held_space_interleaves_movement_during_fire_cooldown(self):
        fake_pygame = SimpleNamespace(
            K_SPACE=1,
            K_w=2,
            K_UP=3,
            K_s=4,
            K_DOWN=5,
            K_a=6,
            K_LEFT=7,
            K_d=8,
            K_RIGHT=9,
        )
        keys = defaultdict(bool)
        keys[fake_pygame.K_SPACE] = True
        keys[fake_pygame.K_w] = True

        self.assertEqual(
            run_manual.selected_action(
                keys,
                run_manual.ControlStyle.DIRECT,
                fake_pygame,
                shoot_ready=True,
            ),
            5,
        )
        self.assertEqual(
            run_manual.selected_action(
                keys,
                run_manual.ControlStyle.DIRECT,
                fake_pygame,
                shoot_ready=False,
            ),
            1,
        )
        self.assertEqual(
            run_manual.selected_action(
                keys,
                run_manual.ControlStyle.ROTATION,
                fake_pygame,
                shoot_ready=True,
            ),
            4,
        )
        self.assertEqual(
            run_manual.selected_action(
                keys,
                run_manual.ControlStyle.ROTATION,
                fake_pygame,
                shoot_ready=False,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
