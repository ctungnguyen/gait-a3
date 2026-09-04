from __future__ import annotations

import unittest

from arena.config import ArenaConfig
from arena.controls import ControlCommand, ControlStyle, decode_action
from arena.core import ArenaCore
from arena.entities import Enemy, Projectile
from arena.math2d import length, vec


class ArenaCoreTests(unittest.TestCase):
    def setUp(self):
        self.core = ArenaCore(seed=11)

    def step_noop(self):
        return self.core.step(ControlCommand(), ControlStyle.DIRECT)

    def test_reset_builds_player_and_phase_spawners(self):
        self.assertEqual(self.core.phase, 1)
        self.assertEqual(len(self.core.spawners), self.core.config.initial_spawner_count)
        self.assertEqual(self.core.player.health, self.core.player.max_health)
        self.core.assert_invariants()

    def test_spawners_periodically_create_enemies(self):
        observed = []
        for _ in range(40):
            outcome = self.step_noop()
            observed.extend(event.name for event in outcome.events)
            if self.core.enemies:
                break
        self.assertTrue(self.core.enemies)
        self.assertIn("enemy_spawned", observed)

    def test_enemy_pursuit_reduces_distance(self):
        for spawner in self.core.spawners:
            spawner.spawn_timer = 999.0
        enemy = Enemy(
            entity_id=self.core._id(),
            pos=vec(100.0, self.core.config.height / 2.0),
            max_speed=self.core.config.enemy_max_speed,
        )
        self.core.enemies.append(enemy)
        before = length(enemy.pos - self.core.player.pos)
        self.step_noop()
        after = length(enemy.pos - self.core.player.pos)
        self.assertLess(after, before)

    def test_projectile_can_destroy_last_spawner_and_advance_phase(self):
        target = self.core.spawners[0]
        self.core.spawners = [target]
        target.health = self.core.config.projectile_damage
        target.spawn_timer = 999.0
        projectile = Projectile(
            entity_id=self.core._id(),
            pos=target.pos.copy(),
            vel=vec(),
            damage=self.core.config.projectile_damage,
            lifetime=1.0,
        )
        self.core.projectiles.append(projectile)

        outcome = self.step_noop()
        names = [event.name for event in outcome.events]
        self.assertIn("spawner_destroyed", names)
        self.assertIn("phase_advanced", names)
        self.assertEqual(self.core.phase, 2)
        self.assertGreater(len(self.core.spawners), 0)

    def test_player_death_terminates_episode(self):
        self.core.player.health = 0.0
        outcome = self.step_noop()
        self.assertTrue(outcome.terminated)
        self.assertIn("player_died", [event.name for event in outcome.events])

    def test_maximum_step_count_truncates_episode(self):
        core = ArenaCore(ArenaConfig(max_steps=2), seed=5)
        first = core.step(ControlCommand(), ControlStyle.DIRECT)
        second = core.step(ControlCommand(), ControlStyle.DIRECT)
        self.assertFalse(first.truncated)
        self.assertTrue(second.truncated)

    def test_seeded_enemy_spawn_is_reproducible(self):
        def first_spawn(seed):
            core = ArenaCore(seed=seed)
            for _ in range(40):
                core.step(ControlCommand(), ControlStyle.DIRECT)
                if core.enemies:
                    return tuple(float(value) for value in core.enemies[0].pos)
            self.fail("Spawner never produced an enemy")

        self.assertEqual(first_spawn(73), first_spawn(73))

    def test_shoot_preserves_momentum_in_both_control_styles(self):
        for style, shoot_action in (
            (ControlStyle.ROTATION, 4),
            (ControlStyle.DIRECT, 5),
        ):
            with self.subTest(style=style.value):
                core = ArenaCore(seed=12)
                for spawner in core.spawners:
                    spawner.spawn_timer = 999.0
                core.player.vel = vec(125.0, -40.0)
                before = core.player.vel.copy()

                core.step(decode_action(style, shoot_action), style)

                self.assertAlmostEqual(float(core.player.vel[0]), float(before[0]))
                self.assertAlmostEqual(float(core.player.vel[1]), float(before[1]))
                self.assertEqual(len(core.projectiles), 1)


if __name__ == "__main__":
    unittest.main()
