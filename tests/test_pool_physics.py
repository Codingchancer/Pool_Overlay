import unittest

from pool_physics import Ball, simulate_shot


class PoolPhysicsTests(unittest.TestCase):
    def test_head_on_hit_moves_object_and_recoils_cue(self):
        balls = [Ball(0, 100, 250, 12), Ball(1, 300, 250, 12)]
        result = simulate_shot(balls, (0, 0, 1000, 500), 0, 1.0)
        self.assertGreaterEqual(result.ball_collisions, 1)
        cue_x = [frame[0].x for frame in result.frames]
        self.assertGreater(max(cue_x), 250)
        self.assertLess(cue_x[-1], 288)
        object_x = [frame[1].x for frame in result.frames]
        self.assertGreater(max(object_x), 350)

    def test_glancing_hit_deflects_both_balls(self):
        balls = [Ball(0, 100, 180, 12), Ball(1, 260, 195, 12)]
        result = simulate_shot(balls, (0, 0, 1000, 500), 0, .8)
        self.assertGreaterEqual(result.ball_collisions, 1)
        cue_y = [frame[0].y for frame in result.frames]
        object_x = [frame[1].x for frame in result.frames]
        self.assertGreater(max(cue_y) - min(cue_y), 10)
        self.assertGreater(max(object_x), 270)

    def test_cushion_reverses_ball_direction(self):
        result = simulate_shot([Ball(0, 100, 250, 12)], (0, 0, 500, 500), 0, 1.0)
        xs = [frame[0].x for frame in result.frames]
        self.assertGreaterEqual(result.cushion_collisions, 1)
        self.assertGreater(max(xs), 450)
        self.assertLess(xs[-1], 488)

    def test_center_pocket_captures_ball(self):
        result = simulate_shot([Ball(0, 250, 100, 12)], (0, 0, 500, 500), -90, 1.0)
        self.assertIn(0, result.pocketed)
        self.assertTrue(result.frames[-1][0].pocketed)

    def test_simulation_is_deterministic_and_does_not_mutate_inputs(self):
        balls = [Ball(0, 100, 250, 12), Ball(1, 300, 250, 12)]
        first = simulate_shot(balls, (0, 0, 1000, 500), 2, .65)
        second = simulate_shot(balls, (0, 0, 1000, 500), 2, .65)
        self.assertEqual(first, second)
        self.assertEqual(balls[0], Ball(0, 100, 250, 12))


if __name__ == "__main__":
    unittest.main()
