"""The detector works on every table cloth colour, not just blue.

White, light grey, beige and other pale cloths used to fail: the cloth itself
passed the "white guide stroke" test, the table's darker lettering was taken
for balls, and the cue ball (barely brighter than white cloth) was lost."""
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

SEEDS = range(9000, 9004)


def setUpModule():
    # These frames were chosen on the old flat look (number up, straight
    # band); balls turned as in the game are checked in test_turned_balls
    # and by the sweeps.
    fg.TURN_BALLS = False


def tearDownModule():
    fg.TURN_BALLS = True


class ClothColourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {name: [st.evaluate(seed, None, name) for seed in SEEDS] for name in fg.CLOTH_HSV}

    def test_table_found_on_every_cloth(self):
        for name, results in self.results.items():
            for seed, (*_, problems, _stats) in zip(SEEDS, results):
                self.assertFalse([p for p in problems if p.startswith("table")], f"{name} seed {seed}")

    def test_cue_ball_found_on_every_cloth(self):
        for name, results in self.results.items():
            # The right ball (on white cloth its centre can still be a few
            # pixels off: the cue ball is barely brighter than the cloth).
            found = sum(r[-1].get("cue", 99) <= 8 for r in results)
            self.assertGreaterEqual(found, 3, name)

    def test_whole_shot_read_on_every_cloth(self):
        for name, results in self.results.items():
            passed = sum(not r[-2] for r in results)
            self.assertGreaterEqual(passed, 2, f"{name}: {[r[-2] for r in results]}")


class WhiteTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.truth = fg.render(fg.random_scene(9001, None, "white"))
        cls.detection = detector.detect(cls.frame)

    def test_lettering_is_not_a_ball(self):
        r = self.truth.radius
        points = list(self.truth.balls.values()) + [self.truth.cue]
        for x, y, _, _ in self.detection.balls:
            self.assertTrue(any(np.hypot(px - x, py - y) <= r * .5 for px, py in points), (x, y))

    def test_cue_ball_and_ghost(self):
        self.assertLessEqual(np.hypot(self.detection.cue[0] - self.truth.cue[0],
                                      self.detection.cue[1] - self.truth.cue[1]), st.CUE_PX)
        self.assertIsNotNone(self.detection.ghost)
        self.assertLessEqual(np.hypot(self.detection.ghost[0] - self.truth.ghost[0],
                                      self.detection.ghost[1] - self.truth.ghost[1]), st.GHOST_PX)

    def test_lettering_shade_is_measured(self):
        x, y, w, h = self.detection.table
        roi = self.frame[y:y + h, x:x + w]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        light = detector.local_lightness(lab[:, :, 0])
        shade = detector.lettering_shade(lab, light, detector._felt_lab(roi))
        # The synthetic lettering is the cloth at 72% brightness.
        self.assertIsNotNone(shade)
        self.assertLess(shade, -30)
        self.assertGreater(shade, -90)

    def test_brightness_rule_only_on_light_cloth(self):
        for cloth, light in (("blue", False), ("white", True), ("beige", True)):
            frame, truth = fg.render(fg.random_scene(9001, None, cloth))
            x, y, w, h = (int(v) for v in truth.table)
            hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
            self.assertEqual(detector._light_cloth(hsv), light, cloth)


if __name__ == "__main__":
    unittest.main()
