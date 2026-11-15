"""Live tracking keeps going while the cue stick passes over balls.

The stick crossing a ball used to look like the ball moving, so every such
frame fell back to a full detection (~115 ms) and the guide lagged while
aiming. A ball that really moves must still trigger a full detection."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import fake_game as fg
from tracking import ShotTracker


class StickCrossingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.truth = fg.render(fg.random_scene(9002))
        cls.tracker = ShotTracker()
        cls.tracker.detect(cls.frame, now=0.0)
        cls.ball = next(b for b in cls.tracker.base.balls if b[3] != "cue")

    def still(self, frame):
        return self.tracker._ball_still(frame, self.ball)

    def test_unchanged_ball_is_still(self):
        self.assertTrue(self.still(self.frame))

    def test_stick_crossing_the_ball_is_not_movement(self):
        x, y, r = self.ball[0], self.ball[1], self.ball[2]
        for angle in (0, 35, 90, 150):
            for width in (.4, .6, .8):
                frame = self.frame.copy()
                a = np.radians(angle)
                dx, dy = np.cos(a) * r * 4, np.sin(a) * r * 4
                cv2.line(frame, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), (40, 60, 30),
                         max(2, int(round(r * width))))
                self.assertTrue(self.still(frame), f"stick at {angle} deg, width {width} r")

    def test_ball_that_moved_still_triggers_a_full_detection(self):
        x, y, r = self.ball[0], self.ball[1], self.ball[2]
        for shift in (.6, 1.0, 2.5):
            frame = self.frame.copy()
            patch = self.frame[y - 2 * r:y + 2 * r + 1, x - 2 * r:x + 2 * r + 1].copy()
            # Cover the ball with plain cloth and draw it a little further on.
            cloth = self.frame[y, x - 3 * r].tolist()
            cv2.circle(frame, (x, y), int(r * 1.2), cloth, -1)
            s = int(round(shift * r))
            h, w = patch.shape[:2]
            frame[y - 2 * r:y - 2 * r + h, x - 2 * r + s:x - 2 * r + s + w] = patch
            self.assertFalse(self.still(frame), f"ball moved by {shift} r")


if __name__ == "__main__":
    unittest.main()
