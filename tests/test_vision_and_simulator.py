import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

import main


APP = QApplication.instance() or QApplication([])
SAMPLE = Path(__file__).resolve().parents[1] / "outputs" / "example_pool_screenshot.png"
MINICLIP_SAMPLE = Path(__file__).resolve().parents[1] / "outputs" / "miniclip_example_aiming.png"
needs_screenshots = unittest.skipUnless(SAMPLE.exists() and MINICLIP_SAMPLE.exists(), "real screenshots in outputs/ are not included in this copy")


class CueSeparationTests(unittest.TestCase):
    def test_stick_artifacts_are_removed_but_cue_and_offset_ball_remain(self):
        balls = [
            (500, 500, 15, "cue"),
            (400, 500, 12, "ball"),  # cue-stick ornament
            (400, 524, 15, "ball"),  # real ball beside the stick
            (600, 500, 15, "ball"),  # target in front of cue
        ]
        kept = main.separate_cue_stick_detections(balls, (500, 500, 15), (1.0, 0.0))
        self.assertIn(balls[0], kept)
        self.assertNotIn(balls[1], kept)
        self.assertIn(balls[2], kept)
        self.assertIn(balls[3], kept)

    def test_cue_face_graphic_is_not_counted_as_a_second_ball(self):
        balls = [(401, 341, 21, "cue"), (420, 354, 18, "ball"), (946, 343, 17, "ball")]
        kept = main.separate_cue_overlap_detections(balls, (401, 341, 21))
        self.assertIn(balls[0], kept)
        self.assertNotIn(balls[1], kept)
        self.assertIn(balls[2], kept)

    def test_cue_tracker_holds_only_brief_occlusion(self):
        tracker = main.CueTracker()
        geometry = QRect(0, 0, 500, 250)
        found = main.Detection(cue=(210, 120, 12), balls=[(210, 120, 12, "cue")])
        self.assertEqual(tracker.update(found, geometry).cue, (210, 120, 12))
        missing = tracker.update(main.Detection(note="Cue missing"), geometry)
        self.assertEqual(missing.cue, (210, 120, 12))
        self.assertIn("briefly obscured", missing.note)
        expired = tracker.update(main.Detection(), QRect(0, 0, 600, 300))
        self.assertIsNone(expired.cue)

    @needs_screenshots
    def test_saved_miniclip_frame_finds_cue_and_game_aim_target(self):
        frame = cv2.imread(str(SAMPLE))
        self.assertIsNotNone(frame)
        result = main.detect(frame)
        self.assertIsNotNone(result.cue)
        self.assertLessEqual(abs(result.cue[0] - 1123), 4)
        self.assertLessEqual(abs(result.cue[1] - 605), 4)
        self.assertIsNotNone(result.target)
        self.assertLessEqual(abs(result.target[0] - 1233), 6)
        self.assertLessEqual(abs(result.target[1] - 529), 6)
        object_balls = [ball for ball in result.balls or [] if ball[3] != "cue"]
        self.assertEqual(len(object_balls), 15)
        self.assertTrue(any(abs(bx - 1259) <= 5 and abs(by - 727) <= 5
                            for bx, by, _, _ in object_balls), "blue-striped ball blended into the felt")
        self.assertIn("Aim ray matched", result.note)

    @needs_screenshots
    def test_saved_miniclip_frame_does_not_treat_cue_stick_joints_as_balls(self):
        frame = cv2.imread(str(SAMPLE))
        result = main.detect(frame)
        stick_joints = ((934, 779), (912, 796), (903, 806), (873, 830), (859, 836))
        for x, y in stick_joints:
            self.assertFalse(any((bx - x) ** 2 + (by - y) ** 2 < 18 ** 2
                                 for bx, by, _, kind in result.balls or [] if kind != "cue"),
                             f"cue-stick detail at {(x, y)} was counted as a ball")

    @needs_screenshots
    def test_official_aiming_sample_tracks_cue_without_overlapping_cue_tip_marker(self):
        result = main.detect(cv2.imread(str(MINICLIP_SAMPLE)))
        self.assertIsNotNone(result.cue)
        self.assertIsNotNone(result.target)
        self.assertIn("Aim ray matched", result.note)
        for bx, by, br, kind in result.balls or []:
            if kind != "cue":
                self.assertGreaterEqual(np.hypot(bx-result.cue[0], by-result.cue[1]),
                                        (br+result.cue[2])*.70)

    def test_simulator_starts_with_all_fifteen_object_balls(self):
        canvas = main.SimCanvas()
        canvas.resize(900, 500)
        self.assertEqual(len(canvas.balls), 15)
        self.assertEqual(len({ball[4] for ball in canvas.balls}), 15)
        self.assertTrue(any(ball[4] == 8 for ball in canvas.balls))
        self.assertEqual(canvas.cue[4], 0)
        left, top, width, height = canvas.table()
        for x, y, radius, *_ in canvas.balls + [canvas.cue]:
            self.assertGreaterEqual(x, left + radius)
            self.assertLessEqual(x, left + width - radius)
            self.assertGreaterEqual(y, top + radius)
            self.assertLessEqual(y, top + height - radius)

    def test_simulator_can_shoot_the_full_rack(self):
        canvas = main.SimCanvas()
        canvas.resize(900, 500)
        canvas.shoot()
        try:
            self.assertGreater(len(canvas.result.frames), 1)
            self.assertGreaterEqual(canvas.result.ball_collisions, 1)
            self.assertEqual(len(canvas._physics_ball_order), 15)
        finally:
            canvas.timer.stop()

    @needs_screenshots
    def test_screenshot_tester_loads_example_without_airplay(self):
        tester = main.ScreenshotTester()
        self.assertIsNotNone(tester.canvas.image)
        self.assertIn("Aim ray matched", tester.canvas.detection.note)
        self.assertIsNotNone(tester.canvas.detection.target)
        self.assertGreaterEqual(len(tester.canvas.detection.balls or []), 12)

    def test_simulator_states_its_physics_limits(self):
        window = main.ShotSimulator()
        texts = " ".join(label.text() for label in window.findChildren(main.QLabel))
        self.assertIn("no spin", texts)
        self.assertIn("8-ball rules", texts)
        window.close()

    @needs_screenshots
    def test_screenshot_tester_can_switch_to_earlier_screenshot(self):
        tester = main.ScreenshotTester()
        tester.load_sample("example_pool_screenshot.png", "Earlier table screenshot")
        self.assertIsNotNone(tester.canvas.image)
        self.assertIsNotNone(tester.canvas.detection.target)
        self.assertIn("Earlier table screenshot", tester.status.text())


if __name__ == "__main__":
    unittest.main()
