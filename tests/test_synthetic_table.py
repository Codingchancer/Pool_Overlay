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

    def test_aim_at_empty_cushion_continues_off_the_cushion(self):
        # The guide used to stop at the ghost ball when it sat on a cushion;
        # the cue ball rebounds there, mirrored off that cushion.
        checked = 0
        for seed in range(9200, 9210):
            *_, truth, d, problems, _ = run(seed, "miss")
            if truth.ghost is None or d.ghost is None:
                continue
            left, top, width, height = truth.table
            r, (gx, gy) = truth.radius, truth.ghost
            a = np.radians(truth.aim_degrees)
            ux, uy = np.cos(a), np.sin(a)
            near = r * .2
            if abs(gx - (left + r)) < near or abs(left + width - r - gx) < near:
                ux = -ux
            if abs(gy - (top + r)) < near or abs(top + height - r - gy) < near:
                uy = -uy
            if (ux, uy) == (np.cos(a), np.sin(a)):
                continue
            self.assertTrue(d.cue_segments, f"seed {seed}: no rebound · {d.note}")
            (x1, y1), (x2, y2) = d.cue_segments[0]
            self.assertLess(np.hypot(x1 - gx, y1 - gy), r * .5, f"seed {seed}")
            self.assertLess(st.angle_diff(st.seg_angle(d.cue_segments[0]), np.degrees(np.arctan2(uy, ux))),
                            2.0, f"seed {seed}")
            checked += 1
        self.assertGreaterEqual(checked, 5)


class BounceTests(unittest.TestCase):
    def test_tracer_matches_exact_bounces_through_three_cushions(self):
        import detector

        rng = np.random.default_rng(5)
        table, r = (100.0, 80.0, 1200.0, 600.0), 15.0
        bounds = (115.0, 95.0, 1285.0, 665.0)
        checked = 0
        for _ in range(200):
            start = (rng.uniform(200, 1200), rng.uniform(150, 600))
            a = rng.uniform(-np.pi, np.pi)
            direction = (float(np.cos(a)), float(np.sin(a)))
            path = detector.trace_bounces(start, direction, bounds, r, max_bounces=3)
            exact, stop = fg.truth_path(start, direction, table, r, bounces=3)
            # Same number of cushions and the same pocket stops.
            self.assertEqual(len(path), len(exact) - 1, f"angle {np.degrees(a):.1f}")
            for (_, end), point in zip(path, exact[1:]):
                self.assertLess(np.hypot(end[0] - point[0], end[1] - point[1]), 1.5)
            checked += stop == "bounces"
        self.assertGreater(checked, 100)

    def test_live_paths_follow_three_bounces_like_the_truth(self):
        bounce_px, bounce_deg, most = [], [], 0
        for seed in range(9000, 9040):
            *_, stats = run(seed)
            if "bounce_px" in stats:
                bounce_px.append(stats["bounce_px"])
                most = max(most, stats["bounces"])
            if "bounce_deg" in stats:
                bounce_deg.append(stats["bounce_deg"])
        self.assertEqual(most, 3)
        self.assertLessEqual(max(bounce_px), 4.0)
        self.assertLessEqual(np.percentile(bounce_deg, 95), 2.0)

    def test_simulator_paths_keep_one_bounce(self):
        import detector

        _, cue_path, object_path, _ = detector.collision_paths(
            (300, 300, 15), (500, 310, 15), (1.0, 0.0), (100, 80, 1200, 600))
        self.assertLessEqual(len(object_path), 2)
        self.assertLessEqual(len(cue_path), 2)

    def test_cue_path_is_drawn_along_the_reported_cue_direction(self):
        # Seed 11: the reported cue direction was 90 deg to the game's object
        # line, but the drawn path used the raw ball centres and ran 7 deg off.
        *_, d, problems, _ = run(11)
        (x1, y1), (x2, y2) = d.cue_segments[0]
        drawn = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        self.assertLess(st.angle_diff(drawn, st.angle_of(d.cue_dir)), 2.0)


class GameLineReaderTests(unittest.TestCase):
    def test_one_pixel_guide_thread_is_not_read_as_the_game_line(self):
        # Real frame: only a short stub of Miniclip's thick object line shows
        # past the ball, while the anti-aliased edge of our own yellow guide
        # left a 1 px pale thread from the same centre, 13 deg off. The reader
        # took the thread and the guide locked onto its own wrong angle.
        import cv2
        import detector

        r, ox, oy = 18.0, 200.0, 200.0
        white = np.zeros((400, 400), np.uint8)
        game, thread = np.radians(-110.0), np.radians(-123.0)
        stub = [(ox + t * np.cos(game), oy + t * np.sin(game)) for t in (r * 1.18, r * 2.9)]
        cv2.line(white, *[(int(round(x)), int(round(y))) for x, y in stub], 255, 4)
        end = (int(round(ox + r * 9 * np.cos(thread))), int(round(oy + r * 9 * np.sin(thread))))
        cv2.line(white, (int(ox), int(oy)), end, 255, 1)
        cv2.circle(white, (int(ox), int(oy)), int(r * 1.18), 0, -1)
        found = detector.game_line_direction(white, (ox, oy), (np.cos(game), np.sin(game)), r)
        self.assertIsNotNone(found)
        angle = float(np.degrees(np.arctan2(found[0][1], found[0][0])))
        self.assertAlmostEqual(angle, -110.0, delta=2.0)


class StickButtTests(unittest.TestCase):
    def test_round_ring_on_a_ball_wide_butt_is_not_a_ball(self):
        # Real frame: the butt's end rings are full ball size with clean rims,
        # but the stick around them is as wide as they are. A ball under a
        # thin part of the stick (as in fake_game) must stay.
        import cv2
        import detector

        r = 18
        cue = (900, 300, r)
        aim = (1.0, 0.0)  # stick lies to the left of the cue ball
        for stick_half, expect_kept in ((r * .8, False), (r * .38, True)):
            distance = np.zeros((600, 1000), np.float32)
            cv2.rectangle(distance, (300, int(300 - stick_half)), (900, int(300 + stick_half)), 50.0, -1)
            ring = (400, 301, 20)
            cv2.circle(distance, ring[:2], ring[2], 50.0, -1)
            balls = [(*cue, "cue"), (*ring, "ball"), (600, 500, r, "ball")]
            kept = detector.separate_cue_stick_detections(
                balls, cue, aim, rim_open={ring[:2]: .8, (600, 500): 1.0}, distance=distance)
            self.assertEqual((*ring, "ball") in kept, expect_kept, f"stick half-width {stick_half}")
            self.assertIn((600, 500, r, "ball"), kept)


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
