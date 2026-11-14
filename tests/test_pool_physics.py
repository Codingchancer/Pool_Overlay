import unittest

from pool_physics import Ball, simulate_shot


class PoolPhysicsTests(unittest.TestCase):
    def test_head_on_hit_moves_object_and_stops_the_cue_near_contact(self):
        # A centre hit slides, so most of its speed goes into the object ball and
        # the cue ball only rolls on a little (contact is at x = 276).
        balls = [Ball(0, 100, 250, 12), Ball(1, 300, 250, 12)]
        result = simulate_shot(balls, (0, 0, 1000, 500), 0, 1.0)
        self.assertGreaterEqual(result.ball_collisions, 1)
        cue_x = [frame[0].x for frame in result.frames]
        self.assertGreater(max(cue_x), 250)
        self.assertLess(cue_x[-1], 360)
        object_x = [frame[1].x for frame in result.frames]
        self.assertGreater(max(object_x), 350)

    def test_draw_comes_back_and_follow_runs_through(self):
        balls = [Ball(0, 100, 250, 12), Ball(1, 300, 250, 12)]
        end = {top: simulate_shot(balls, (0, 0, 1000, 500), 0, 1.0, spin=(0, top)).frames[-1][0].x
               for top in (-1, 0, 1)}
        self.assertLess(end[-1], 276 - 40)          # draw: back from the contact point
        self.assertGreater(end[0], 276)             # centre: rolls on a little
        self.assertGreater(end[1], end[0] + 80)     # follow: well through
        for top in (-1, 0, 1):
            frames = simulate_shot(balls, (0, 0, 1000, 500), 0, 1.0, spin=(0, top)).frames
            self.assertTrue(all(abs(frame[0].y - 250) < 1e-6 for frame in frames), "straight shot must stay straight")

    def test_english_changes_the_angle_off_a_cushion(self):
        # Straight into the right cushion: right english (shooter facing right,
        # so their right is down the screen) comes off downwards, left english up.
        def end_y(side):
            return simulate_shot([Ball(0, 100, 250, 12)], (0, 0, 500, 500), 0, 1.0, spin=(side, 0)).frames[-1][0].y
        self.assertAlmostEqual(end_y(0), 250, delta=1e-6)
        self.assertGreater(end_y(1), 250 + 50)
        self.assertLess(end_y(-1), 250 - 50)
        self.assertAlmostEqual(end_y(1) - 250, 250 - end_y(-1), delta=1)

    def test_a_rolling_ball_comes_to_rest(self):
        result = simulate_shot([Ball(0, 100, 250, 12)], (0, 0, 2000, 500), 0, .3, spin=(0, .5))
        self.assertLess(result.duration, 7.5, "simulation ran to the time limit instead of stopping")

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
