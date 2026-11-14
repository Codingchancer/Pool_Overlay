"""Real-time overlay: frames analysed back to back without a settle wait.

* The capture is cleaned with whichever recent guide drawing the screen
  actually shows (the compositor can lag a repaint by a frame or two).
* ``ShotTracker`` re-reads only the aim graphics while the balls stay still,
  and falls back to a full detection when they move.
"""
import dataclasses
import os
import time
import types
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

import fake_game
from capture import DEFAULT_STYLE, composite_overlay, render_guide_layer, shown_layer
from detector import Detection, detect
from overlay import GrabPlan, grab_clean
from tracking import ShotTracker

APP = QApplication.instance() or QApplication([])
W, H = 640, 360


def guide(aim_end):
    return Detection(table=(20, 20, 600, 320), cue=(120, 180, 12), aim_segments=[((120, 180), aim_end)],
                     ghost=(*aim_end, 12), note="shot")


def layer_of(detection):
    return render_guide_layer(detection, W, H, adjusting=False, active=True, style=DEFAULT_STYLE)


def paint_of(detection):
    """The tuple ``Overlay.paintEvent`` records for a drawing."""
    return (detection, False, True, W, H, 1.0, DEFAULT_STYLE)


def table_frame():
    clean, _ = fake_game.render(fake_game.random_scene(7))
    return np.ascontiguousarray(clean[:H, :W])


class FakeCapture:
    """Stands in for mss: returns a fixed BGRA screen."""
    def __init__(self, bgr):
        self.screen = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])

    def grab(self, region):
        part = np.ascontiguousarray(self.screen[region["top"]:region["top"] + region["height"],
                                                region["left"]:region["left"] + region["width"]])
        return types.SimpleNamespace(bgra=part.tobytes(), width=part.shape[1], height=part.shape[0])


class ShownLayerTests(unittest.TestCase):
    def setUp(self):
        self.background = table_frame()
        self.new = layer_of(guide((450, 120)))
        self.old = layer_of(guide((430, 260)))

    def test_finds_the_drawing_on_screen_new_or_old(self):
        for shown, expected in ((self.new, 0), (self.old, 1)):
            captured = composite_overlay(self.background, shown)
            self.assertEqual(shown_layer(captured, [self.new, self.old]), expected)

    def test_an_empty_drawing_still_on_screen_is_recognised(self):
        empty = layer_of(Detection(note="no shot"))
        self.assertEqual(shown_layer(composite_overlay(self.background, empty), [self.new, empty]), 1)
        self.assertEqual(shown_layer(composite_overlay(self.background, self.new), [self.new, empty]), 0)

    def test_a_capture_showing_none_of_them_is_rejected(self):
        third = layer_of(guide((300, 60)))
        captured = composite_overlay(self.background, third)
        self.assertIsNone(shown_layer(captured, [self.new, self.old]))


class GrabCleanTests(unittest.TestCase):
    def test_lagging_screen_is_cleaned_with_the_old_drawing(self):
        background = table_frame()
        old, new = guide((430, 260)), guide((450, 120))
        # The screen still shows the old drawing although the new one was painted.
        capture = FakeCapture(composite_overlay(background, layer_of(old)))
        region = {"left": 0, "top": 0, "width": W, "height": H}
        plan = GrabPlan(region, dict(region), None, [paint_of(new), paint_of(old)])
        cleaned = grab_clean(capture, plan)
        self.assertIsNotNone(cleaned)
        error = np.abs(cleaned.astype(int) - background.astype(int))
        self.assertLessEqual(np.percentile(error, 99.9), 3)

    def test_overlay_resized_since_the_drawing_is_skipped(self):
        capture = FakeCapture(table_frame())
        region = {"left": 0, "top": 0, "width": W - 40, "height": H}
        plan = GrabPlan(region, dict(region), None, [paint_of(guide((450, 120)))])
        self.assertIsNone(grab_clean(capture, plan))


class ShotTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = fake_game.random_scene(2003)
        cls.first, _ = fake_game.render(cls.scene)

    def turned(self, degrees):
        s = dataclasses.replace(self.scene, aim_degrees=self.scene.aim_degrees + degrees)
        return fake_game.render(s)

    def test_aiming_reuses_the_balls_and_follows_the_aim(self):
        tracker = ShotTracker()
        tracker.detect(self.first, now=0)
        self.assertTrue(tracker.last_was_full)
        frame, truth = self.turned(1.5)
        start = time.perf_counter()
        tracked = tracker.detect(frame, now=.05)
        tracked_seconds = time.perf_counter() - start
        self.assertFalse(tracker.last_was_full)
        full = detect(frame)
        a = np.radians(truth.aim_degrees)
        for d in (tracked, full):
            self.assertLess(np.degrees(np.arccos(np.clip(d.aim_dir[0] * np.cos(a) + d.aim_dir[1] * np.sin(a), -1, 1))),
                            .3)
        self.assertEqual(tracked.balls, tracker.base.balls)
        # Generous bound (CI machines vary); typically ~30 ms against ~200 ms.
        self.assertLess(tracked_seconds, .15)

    def test_a_moved_ball_means_a_full_detection(self):
        tracker = ShotTracker()
        tracker.detect(self.first, now=0)
        number, (bx, by) = next(iter(self.scene.balls.items()))
        moved = dataclasses.replace(self.scene, balls={**self.scene.balls,
                                                       number: (bx + self.scene.radius * 1.5, by)})
        frame, _ = fake_game.render(moved)
        tracker.detect(frame, now=.05)
        self.assertTrue(tracker.last_was_full)

    def test_full_detection_again_after_a_few_seconds(self):
        tracker = ShotTracker()
        tracker.detect(self.first, now=0)
        tracker.detect(self.first, now=tracker.REFRESH_SECONDS + .1)
        self.assertTrue(tracker.last_was_full)

    def test_repeated_misses_trigger_a_full_detection(self):
        calls = []

        def counting_detect(frame):
            calls.append(1)
            return detect(frame)

        tracker = ShotTracker(full_detect=counting_detect)
        tracker.detect(self.first, now=0)
        no_guide, _ = fake_game.render(dataclasses.replace(self.scene, show_guide=False))
        for k in range(tracker.MISSES_BEFORE_FULL):
            tracker.detect(no_guide, now=.05 * (k + 1))
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
