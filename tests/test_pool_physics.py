import math
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
        # 80% power: the strongest normal shot (above that is break power).
        balls = [Ball(0, 100, 250, 12), Ball(1, 300, 250, 12)]
        end = {top: simulate_shot(balls, (0, 0, 1000, 500), 0, .8, spin=(0, top)).frames[-1][0].x
               for top in (-1, 0, 1)}
        self.assertLess(end[-1], 276 - 30)          # draw: back from the contact point
        self.assertGreater(end[0], 276)             # centre: rolls on a little
        self.assertGreater(end[1], end[0] + 80)     # follow: well through
        for top in (-1, 0, 1):
            frames = simulate_shot(balls, (0, 0, 1000, 500), 0, .8, spin=(0, top)).frames
            self.assertTrue(all(abs(frame[0].y - 250) < 1e-6 for frame in frames), "straight shot must stay straight")

    def test_english_changes_the_angle_off_a_cushion(self):
        # Straight into the right cushion: right english (shooter facing right,
        # so their right is down the screen) comes off downwards, left english up.
        def end_y(side):
            return simulate_shot([Ball(0, 100, 250, 12)], (0, 0, 500, 500), 0, .8, spin=(side, 0)).frames[-1][0].y
        self.assertAlmostEqual(end_y(0), 250, delta=1e-6)
        self.assertGreater(end_y(1), 250 + 50)
        self.assertLess(end_y(-1), 250 - 50)
        self.assertAlmostEqual(end_y(1) - 250, 250 - end_y(-1), delta=1)

    def test_power_up_to_80_percent_is_unchanged_and_above_it_breaks(self):
        def speed(power):
            # Distance a lone ball needs to stop is a clean proxy for its start speed.
            frames = simulate_shot([Ball(0, 50, 250, 10)], (0, 0, 100000, 500), 0, power).frames
            return frames[-1][0].x - 50
        legacy = lambda power: 150 + 480 * power
        for power in (.1, .5, .8):
            ratio = speed(power) / speed(.5)
            self.assertAlmostEqual(ratio, (legacy(power) / legacy(.5)) ** 2, delta=.03)
        self.assertGreater(speed(1.0), speed(.8) * 10)

    def test_full_power_break_scatters_the_rack(self):
        r = 14
        rack = []
        number = 1
        for row in range(5):
            for slot in range(row + 1):
                rack.append(Ball(number, 680 + row * r * 1.78, 250 + (slot - row / 2) * r * 2.04, r))
                number += 1
        balls = [Ball(0, 200, 250, r)] + rack
        for angle in (0, .5, -.7):
            result = simulate_shot(balls, (0, 0, 924, 462), angle, 1.0)
            final = result.frames[-1]
            moved = sum(1 for f, b in zip(final[1:], balls[1:]) if ((f.x - b.x) ** 2 + (f.y - b.y) ** 2) ** .5 > 2 * r)
            self.assertGreaterEqual(moved, 8, f"break at {angle} deg moved only {moved} balls")
            for frame in result.frames:
                live = [(s.x, s.y) for s in frame if not s.pocketed]
                for i in range(len(live)):
                    for j in range(i + 1, len(live)):
                        gap = ((live[i][0] - live[j][0]) ** 2 + (live[i][1] - live[j][1]) ** 2) ** .5
                        self.assertGreater(gap, 2 * r - 2, "balls passed through each other")

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

    def test_ball_along_the_rail_drops_into_the_corner(self):
        # Shallow off the top rail into the top-left corner: it used to bounce
        # off the left rail beside the mouth and stop in the jaws.
        result = simulate_shot([Ball(0, 400, 60, 12)], (0, 0, 800, 400), -171.5, .5)
        self.assertIn(0, result.pocketed)
        # Rolled straight down the top rail.
        result = simulate_shot([Ball(0, 300, 12, 12)], (0, 0, 800, 400), 180, .6)
        self.assertIn(0, result.pocketed)

    def test_rail_hit_away_from_the_pockets_still_bounces(self):
        result = simulate_shot([Ball(0, 200, 200, 12)], (0, 0, 800, 400), -90, .5)
        self.assertNotIn(0, result.pocketed)
        self.assertGreaterEqual(result.cushion_collisions, 1)

    def test_struck_ball_direction_is_exact_at_any_power(self):
        # Contacts are resolved at the moment the balls touch, in sub-steps at
        # speed: at break power the struck ball was 13 degrees off.
        for power in (.5, .9, 1.0):
            for offset in (-15.0, -6.5, 0.0, 4.0, 11.5):
                balls = [Ball(0, 100, 250, 12), Ball(1, 300, 250 + offset, 12)]
                result = simulate_shot(balls, (0, 0, 3000, 500), 0, power, max_seconds=2)
                end = next(f[1] for f in result.frames if math.hypot(f[1].x - 300, f[1].y - 250 - offset) > 24)
                got = math.degrees(math.atan2(end.y - 250 - offset, end.x - 300))
                want = math.degrees(math.asin(offset / 24))
                self.assertAlmostEqual(got, want, delta=.3, msg=f"power {power} offset {offset}")

    def test_simulation_is_deterministic_and_does_not_mutate_inputs(self):
        balls = [Ball(0, 100, 250, 12), Ball(1, 300, 250, 12)]
        first = simulate_shot(balls, (0, 0, 1000, 500), 2, .65)
        second = simulate_shot(balls, (0, 0, 1000, 500), 2, .65)
        self.assertEqual(first, second)
        self.assertEqual(balls[0], Ball(0, 100, 250, 12))


if __name__ == "__main__":
    unittest.main()
