"""Striped balls on light cloths.

A striped ball's white caps are close to a light cloth (white, light grey,
beige, sky), so circle finding slid along the band or towards the drop
shadow. The band is coloured right up to the rim at both its ends, and the
ball is fitted on those two opposite arcs (``stripe_centres``). Synthetic
seeds 40000+; each failed before."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
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


class StripeCentreTests(unittest.TestCase):
    def test_stripes_on_light_cloths(self):
        # 0.3-0.47 radii off before (towards the shadow on white, along the
        # band on beige).
        for seed, kind, cloth, number in ((40012, "scatter", "white", 13), (40016, "scatter", "white", 11),
                                          (40023, "scatter", "white", 10), (40004, "scatter", "beige", 10),
                                          (40011, "scatter", "beige", 15), (40017, "scatter", "beige", 13)):
            frame, truth = fg.render(fg.random_scene(seed, kind, cloth))
            d = detector.detect(frame)
            px, py = truth.balls[number]
            err = min(np.hypot(bx - px, by - py) for bx, by, _, _ in d.balls) / truth.radius
            self.assertLess(err, .1, f"{seed} {kind} {cloth} #{number}")

    def test_fit_on_the_band_arcs(self):
        # A stripe drawn on a light grey cloth, its circle started 0.45
        # radii off: the fit lands on its centre.
        canvas = np.full((130, 130, 3), 182.0, np.float32)
        fg.draw_ball(canvas, 64.3, 61.7, 15.0, 11, np.random.default_rng(3))
        roi = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
        moved = detector.stripe_centres(roi, [(70, 65, 15, "ball")], (0, 0), (-500, -500))
        self.assertLess(np.hypot(moved[0][0] - 64.3, moved[0][1] - 61.7), 1.0)

    def test_darker_cloth_keeps_its_centres(self):
        # On a darker cloth circle finding places stripes well already; the
        # fit (its level pulled by the drop shadow) only confirms the ball.
        frame, truth = fg.render(fg.random_scene(40001, "scatter", "red"))
        d = detector.detect(frame)
        x, y, w, h = d.table
        confirmed = set()
        moved = detector.stripe_centres(frame[y:y + h, x:x + w], d.balls, (x, y), d.cue[:2], confirmed=confirmed)
        self.assertEqual([b[:2] for b in moved], [b[:2] for b in d.balls])
        self.assertTrue(confirmed)


class StripeSafetyTests(unittest.TestCase):
    def test_struck_ball_crossed_by_the_object_line(self):
        # The game's object line is drawn from the struck ball's centre
        # across its band: fitted, it was 3-6 px off and the object path
        # 8-11 degrees.
        for seed, kind in ((40015, "scatter"), (40020, "pot"), (40022, "pot")):
            self.assertFalse(problems(seed, kind, "white"), f"{seed} {kind}")

    def test_struck_ball_fitted_when_its_circle_misses_the_ghost(self):
        # Its circle was 10 px off (not a diameter from the ghost): fitted,
        # 0.7 px.
        self.assertFalse(problems(40004, "pot", "sky"))

    def test_stripes_under_the_stick_stay(self):
        # Moved onto the stick's axis, a stripe under the stick read as one
        # of its rings (the fit now runs after the stick filter, a ball the
        # stick lies across keeps its centre, and a fitted ball is a ball).
        for seed, kind, cloth in ((40014, "pot", "purple"), (40004, "cushion", "light grey"),
                                  (40017, "near", "grey"), (40077, "near", "blue")):
            self.assertFalse([p for p in problems(seed, kind, cloth) if p.startswith("balls")],
                             f"{seed} {kind} {cloth}")

    def test_second_circle_on_the_ring_dropped_after_the_refit(self):
        # With the "cue ball" first fitted on the ring, the ghost was only
        # found after the cue ball was fitted again beside it, and a second
        # circle on the ring stayed as a ball.
        self.assertFalse(problems(40014, "near", "sky"))


if __name__ == "__main__":
    unittest.main()
