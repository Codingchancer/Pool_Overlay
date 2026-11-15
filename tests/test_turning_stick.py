"""The guide follows a stick being turned round the cue ball.

The stabilizer held the drawn guide while each frame's aim changed by less
than its 0.3 deg tolerance, so while the player slowly lined up a shot the
guide trailed the game's own line and then jumped. A steady turn is now
drawn every frame, and a still shot still never redraws."""
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

import fake_game as fg
from overlay import GuideStabilizer
from tracking import ShotTracker


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def sweep(speed_deg_s, frames=24, step_s=1 / 30, seed=9004):
    """Drawn-aim error against the game's aim, one analysis per game frame."""
    base = replace(fg.random_scene(seed), jpeg_quality=85, blur=.5)
    tracker, stabilizer = ShotTracker(), GuideStabilizer()
    errors, redraws, shown_before = [], 0, None
    for i in range(frames):
        now = i * step_s
        angle = base.aim_degrees + speed_deg_s * now
        frame, _ = fg.render(replace(base, aim_degrees=angle))
        shown = stabilizer.update(tracker.detect(frame, now=now), now=now)
        if i >= 2 and shown.aim_dir is not None:   # after the first full detection
            drawn = float(np.degrees(np.arctan2(shown.aim_dir[1], shown.aim_dir[0])))
            errors.append(angle_diff(drawn, angle))
            redraws += shown is not shown_before
        shown_before = shown
    return np.array(errors), redraws


class TurningStickTests(unittest.TestCase):
    def test_slow_turn_is_followed(self):
        errors, _ = sweep(5.0)
        # Same frame, no analysis delay: what is left is the reading itself.
        self.assertLessEqual(np.median(errors), .1, errors)
        self.assertLessEqual(errors.max(), .3, errors)

    def test_still_shot_is_not_redrawn(self):
        errors, redraws = sweep(0.0)
        self.assertEqual(redraws, 0)
        self.assertLessEqual(errors.max(), .1)


if __name__ == "__main__":
    unittest.main()
