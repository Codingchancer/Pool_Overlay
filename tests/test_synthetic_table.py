"""Detector accuracy on synthetic tables with exactly known geometry.

Frames come from fake_game.py (blue cloth, pockets, shaded numbered balls,
decorated cue stick, white aim ray / ghost ball / object and cue lines, JPEG
and blur). Each regression test below is a case that failed before a fix.
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6.QtWidgets import QApplication

import fake_game as fg
import main
import stress_test as st

APP = QApplication.instance() or QApplication([])


def run(seed, kind=None):
    return st.evaluate(seed, kind)


class SyntheticAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = [run(seed) for seed in range(9000, 9040)]

    def test_most_frames_pass_every_check(self):
        passed = sum(1 for *_, problems, _ in self.results if not problems)
        self.assertGreaterEqual(passed, 34, f"only {passed}/40 synthetic frames passed")

    def test_table_edges_within_two_pixels(self):
        worst = max(stats["table"] for *_, stats in self.results)
        self.assertLessEqual(worst, 2.0)

    def test_aim_and_paths_are_accurate(self):
        aim = [s["aim"] for *_, s in self.results if "aim" in s]
        obj = [s["object"] for *_, s in self.results if "object" in s]
        offsets = [s["aim_px"] for *_, s in self.results if "aim_px" in s]
        self.assertLessEqual(np.percentile(aim, 95), .5)
        self.assertLessEqual(max(aim), 1.2)
        # Where the aim meets the ghost ball it is never off by a pixel.
        self.assertLessEqual(max(offsets), 1.0)
        self.assertLessEqual(np.percentile(obj, 95), 1.5)

    def test_balls_are_found(self):
        missing = sum(s["missing"] for *_, s in self.results)
        extra = sum(s["extra"] for *_, s in self.results)
        self.assertLessEqual(missing, 4, "balls missed across 40 frames (600 balls)")
        self.assertLessEqual(extra, 4, "false balls across 40 frames")


class SyntheticRegressionTests(unittest.TestCase):
    def assertClean(self, seed, kind=None, allowed=()):
        *_, problems, _ = run(seed, kind)
        problems = [p for p in problems if not p.startswith(allowed)] if allowed else problems
        self.assertEqual(problems, [], f"seed {seed}")

    def test_ball_frozen_on_cushion_does_not_crash_path_tracing(self):
        # Target detected a pixel past the cushion limit used to give NaN.
        frame, truth = fg.render(fg.random_scene(4))
        main.detect(frame)
        path = main.trace_bounces((1001.0, 50.0), (0.0, 1.0), (0, 0, 1000, 400), 10)
        self.assertTrue(all(np.isfinite(v) for seg in path for pt in seg for v in pt))

    def test_long_ray_split_into_pieces_is_still_read(self):
        self.assertClean(70, allowed=("balls",))

    def test_cue_ball_next_to_rail_with_stick_off_table(self):
        self.assertClean(109, allowed=("balls",))

    def test_short_shot_uses_ray_that_ends_in_ghost_ball(self):
        self.assertClean(165, allowed=("balls",))

    def test_ball_beside_ghost_on_cushion_is_not_the_target(self):
        *_, d, problems, _ = run(179)
        self.assertFalse(any(p.startswith("target") for p in problems), problems)

    def test_blue_balls_on_blue_cloth_are_found(self):
        missed = 0
        for seed in range(9100, 9115):
            frame, truth = fg.render(fg.random_scene(seed))
            d = main.detect(frame)
            for number in (2, 10):
                if number not in truth.balls:
                    continue
                bx, by = truth.balls[number]
                if not any(np.hypot(bx - x, by - y) <= truth.radius * .5 for x, y, _, _ in d.balls or []):
                    missed += 1
        self.assertLessEqual(missed, 1)

    def test_aim_at_empty_cushion_reports_no_target(self):
        for seed in range(9200, 9210):
            *_, truth, d, problems, _ = run(seed, "miss")
            self.assertIsNone(d.target, f"seed {seed}: {problems}")
            self.assertIsNotNone(d.aim_dir, f"seed {seed}")


class SyntheticWindowTests(unittest.TestCase):
    def test_window_shows_a_new_shot_on_demand(self):
        window = fg.make_table_window(seed=3, interval_ms=0, width=900)
        try:
            first = window.pixmap().toImage()
            window.show_next()
            self.assertNotEqual(first, window.pixmap().toImage())
            self.assertEqual(window.width(), 900)
        finally:
            window.close()

    def test_live_scenes_keep_the_table_in_place(self):
        tables = {fg.live_scene(seed).table for seed in range(5)}
        self.assertEqual(len(tables), 1)


if __name__ == "__main__":
    unittest.main()
