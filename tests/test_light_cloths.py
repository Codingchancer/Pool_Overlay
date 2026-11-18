"""Light grey and white cloths.

On a light grey cloth the brightly lit middle passes the white-stroke test
itself, so the guide's lines were read in a flood of cloth; the cue ball's
rim is no darker than the cloth (no dip, as on a white one) and no tint (as
on beige); a striped ball's white caps blend into the cloth, and a second
circle on it stayed as a false ball or was taken for the struck ball. On a
white cloth the cue ball itself looks like cloth to circle finding.
Synthetic seeds 40000+; each failed before."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import detector
import fake_game as fg
import stress_test as st


def setUpModule():
    # These frames were chosen on the old flat look (number up, straight
    # band); balls turned as in the game are checked in test_turned_balls
    # and by the sweeps.
    fg.TURN_BALLS = False


def tearDownModule():
    fg.TURN_BALLS = True


def problems(seed, kind, cloth):
    return st.evaluate(seed, kind, cloth)[4]


class LineTests(unittest.TestCase):
    def test_lines_read_on_light_grey(self):
        # The lit cloth passed the white test: the game's lines were read in
        # a flood of it, 3-12 degrees off, or the ghost was not found.
        for seed, kind in ((40015, "scatter"), (40007, "pot"), (40020, "cushion"), (40017, "scatter")):
            self.assertFalse([p for p in problems(seed, kind, "light grey") if "path" in p or "ghost" in p],
                             f"{seed} {kind}")

    def test_line_fit_ignores_other_white_in_its_cone(self):
        # The cushion's lit edge on the table's border (40008) and a
        # second circle's white cap (40033 near) tilted one fit through all
        # the white in the cone by 5-6 degrees.
        for seed, kind in ((40008, "scatter"), (40033, "near")):
            self.assertFalse([p for p in problems(seed, kind, "light grey") if "path" in p], f"{seed} {kind}")

    def test_line_along_its_own_ray(self):
        # A line with a bar of other white beside it in the cone: read along
        # the ray with the most stroke pixels (one axis through both lost
        # the line).
        white = np.zeros((200, 200), np.uint8)
        white[100:103, 120:190] = 255          # the line, from a ball at (100, 101) to the right
        white[60:63, 150:185] = 255            # other white higher up in the cone
        found = detector.game_line_direction(white, (100.0, 101.0), (1.0, -.2), 12.0, cone_degrees=30.0)
        self.assertIsNotNone(found)
        (vx, vy), _ = found
        self.assertLess(abs(np.degrees(np.arctan2(vy, vx))), .5)


class SecondCircleTests(unittest.TestCase):
    def test_one_circle_per_striped_ball(self):
        # A second circle on a striped ball, 0.5-0.7 radii off, stayed as a
        # false ball.
        for seed, kind, cloth in ((40009, "scatter", "light grey"), (40005, "rack", "light grey"),
                                  (40009, "miss", "white"), (40020, "scatter", "orange"), (40063, "near", "pink")):
            self.assertFalse([p for p in problems(seed, kind, cloth) if p.startswith("balls")],
                             f"{seed} {kind} {cloth}")

    def test_struck_ball_from_the_right_circle(self):
        # The object line was looked for from the second circle 7 px off the
        # struck ball's centre, missed, and the fallback was 5 degrees off.
        self.assertFalse(problems(40001, "scatter", "light grey"))

    def test_better_rim_stays(self):
        distance = np.zeros((100, 100), np.float32)
        yy, xx = np.mgrid[:100, :100]
        distance[np.hypot(xx - 50, yy - 50) <= 10] = 40.0
        balls = [(50, 50, 10, "ball"), (50, 56, 10, "ball"), (20, 20, 10, "cue")]
        kept = detector.drop_second_circles(balls, distance, (0, 0))
        self.assertEqual(kept, [(50, 50, 10, "ball"), (20, 20, 10, "cue")])
        # Two touching balls both stay.
        distance[np.hypot(xx - 70, yy - 50) <= 10] = 40.0
        both = [(50, 50, 10, "ball"), (70, 50, 10, "ball")]
        self.assertEqual(detector.drop_second_circles(both, distance, (0, 0)), both)


class LightGreyCueTests(unittest.TestCase):
    def test_cue_ball_beside_the_ring(self):
        # The ghost ring on the cue ball: the rim read on the lit side with
        # the ring painted out (4-5 px off before, the usual fit sliding
        # towards the shadow).
        for seed in (40011, 40015, 40055, 40089):
            frame, truth = fg.render(fg.random_scene(seed, "near", "light grey"))
            d = detector.detect(frame)
            self.assertLess(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]), 2.5, seed)

    def test_white_ball_under_the_ring_is_the_cue_ball(self):
        # The stick's light tip and rings showed as a line leaving the cue
        # ball, so it was taken for the struck ball (40050); white bits of
        # the struck ball and the ring's arc beside it passed for a ghost a
        # short ray away (40096).
        for seed in (40050, 40096):
            self.assertFalse(problems(seed, "near", "light grey"), seed)


class WhiteClothTests(unittest.TestCase):
    def test_cue_ball_found_on_white(self):
        # Circle finding took the white cue ball on the white cloth for
        # cloth (no number badge), and the aim was lost with it.
        for seed, kind in ((40019, "cushion"), (40002, "cushion")):
            self.assertFalse(problems(seed, kind, "white"), f"{seed} {kind}")

    def test_nothing_added_beside_a_ball(self):
        # A circle on the cloth right beside a ball fits that ball's rim and
        # shadow: never rescued.
        for seed, kind in ((40005, "scatter"), (40014, "scatter"), (40010, "rack")):
            self.assertFalse([p for p in problems(seed, kind, "white") if "false" in p], f"{seed} {kind}")


if __name__ == "__main__":
    unittest.main()
