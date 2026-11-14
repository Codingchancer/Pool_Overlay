"""Alignment with Miniclip's own aim graphics, table fitting, and overlay feedback.

Reference values were measured by hand on the bundled frames (zoomed crops):
user screenshot: ghost-ball reticle centre (1200.5, 535), target centre
(1233.5, 530), game object line about -9 to -11 deg, cushion noses about
x 364..1434 and y 341..877, middle pockets at x = 900.
"""
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

import main


APP = QApplication.instance() or QApplication([])
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
USER = cv2.imread(str(OUTPUTS / "example_pool_screenshot.png"))
OFFICIAL = cv2.imread(str(OUTPUTS / "miniclip_example_aiming.png"))
needs_screenshots = unittest.skipIf(USER is None or OFFICIAL is None, "real screenshots in outputs/ are not included in this copy")


def angle(segment):
    (x1, y1), (x2, y2) = segment
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def guide_cycles(frame, cycles, subtract):
    """Simulate the live loop: draw the guide, capture it on screen, analyse."""
    h, w = frame.shape[:2]
    detection = main.detect(frame)
    results = []
    for _ in range(cycles):
        layer = main.render_guide_layer(detection, w, h, adjusting=False, active=True)
        captured = main.composite_overlay(frame, layer)
        if subtract:
            captured = main.remove_overlay(captured, layer)
        detection = main.detect(captured)
        results.append(detection)
    return results


@needs_screenshots
class TableFitTests(unittest.TestCase):
    def test_pockets_fit_the_cushion_noses_symmetrically(self):
        x, y, w, h = main.find_table_bounds(USER)
        self.assertLessEqual(abs(x - 364), 3)
        self.assertLessEqual(abs(y - 341), 3)
        self.assertLessEqual(abs(x + w - 1434), 3)
        self.assertLessEqual(abs(y + h - 877), 3)
        # The old felt-colour fit was centred 13 px right of the real middle pockets.
        self.assertLessEqual(abs(x + w / 2 - 900), 2)

    def test_pockets_are_found_on_a_dark_multi_monitor_desktop(self):
        # Fit table captures every monitor; dark wallpaper and panels add many
        # large near-black blobs that once crowded the real pockets out.
        desktop = np.full((1440, 4480, 3), 28, np.uint8)
        rng = np.random.default_rng(3)
        for _ in range(60):
            x, y = int(rng.integers(1900, 4400)), int(rng.integers(0, 1360))
            size = int(rng.integers(60, 140))
            cv2.rectangle(desktop, (x, y), (x + size, y + size), (0, 0, 0), -1)
        oy, ox = 200, 40
        desktop[oy:oy + USER.shape[0], ox:ox + USER.shape[1]] = USER
        x, y, w, h = main.find_table_bounds(USER)
        self.assertEqual(main.find_table_bounds(desktop), (x + ox, y + oy, w, h))

    def test_official_sample_fit_is_two_to_one(self):
        x, y, w, h = main.find_table_bounds(OFFICIAL)
        self.assertAlmostEqual(w / h, 2.0, delta=.05)
        self.assertLessEqual(abs(x + w / 2 - 671), 3)

    def test_live_capture_crop_maps_back_to_the_same_table_and_shot(self):
        # Fit table places the overlay on the table plus a 6% pocket margin and
        # each live frame is a capture of exactly that rectangle.
        x, y, w, h = main.find_table_bounds(USER)
        margin = round(w * .06)
        left, top = x - margin, y - margin
        crop = USER[top:y + h + margin, left:x + w + margin].copy()
        full, live = main.detect(USER), main.detect(crop)
        self.assertEqual(live.table, (x - left, y - top, w, h))
        self.assertLessEqual(np.hypot(live.ghost[0] - (full.ghost[0] - left),
                                      live.ghost[1] - (full.ghost[1] - top)), 1.5)
        self.assertLessEqual(abs(live.target[0] + left - full.target[0]), 2)
        self.assertLessEqual(abs(live.target[1] + top - full.target[1]), 2)
        self.assertLess(abs(angle(live.aim_segments[0]) - angle(full.aim_segments[0])), .5)


@needs_screenshots
class AimAlignmentTests(unittest.TestCase):
    def test_ray_ends_on_the_games_ghost_ball(self):
        result = main.detect(USER)
        self.assertIsNotNone(result.ghost)
        self.assertLessEqual(abs(result.ghost[0] - 1200.5), 2)
        self.assertLessEqual(abs(result.ghost[1] - 535), 2)
        start, end = result.aim_segments[0]
        self.assertEqual(start, result.cue[:2])
        self.assertLessEqual(np.hypot(end[0] - result.ghost[0], end[1] - result.ghost[1]), 1.5)
        # Through the reticle centre, the aim is -42.1 deg (the old Hough-only
        # estimate reported -44 deg).
        self.assertAlmostEqual(angle(result.aim_segments[0]), -42.1, delta=1.0)

    def test_object_and_cue_paths_follow_the_games_white_lines(self):
        result = main.detect(USER)
        self.assertIn("game object line", result.note)
        self.assertIn("game cue line", result.note)
        object_start = result.object_segments[0][0]
        self.assertLessEqual(np.hypot(object_start[0] - 1233.5, object_start[1] - 530), 4)
        # The game's own object line runs at about -9..-11 deg; the old contact
        # estimate (3 px off) produced -3 deg.
        self.assertAlmostEqual(angle(result.object_segments[0]), -10, delta=1.5)
        # The cue ball leaves the ghost-ball centre at roughly 90 deg to the
        # object ball (no spin), like the game's short deflection line.
        cue_start = result.cue_segments[0][0]
        self.assertLessEqual(np.hypot(cue_start[0] - result.ghost[0], cue_start[1] - result.ghost[1]), 1.5)
        between = abs(angle(result.cue_segments[0]) - angle(result.object_segments[0]))
        self.assertAlmostEqual(between, 90, delta=6)

    def test_official_sample_uses_the_reticle_and_stops_at_the_rack(self):
        result = main.detect(OFFICIAL)
        self.assertLessEqual(abs(result.ghost[0] - 913), 2)
        self.assertLessEqual(abs(result.ghost[1] - 341), 2)
        self.assertAlmostEqual(angle(result.aim_segments[0]), 0, delta=1.0)
        self.assertLessEqual(np.hypot(result.target[0] - 946, result.target[1] - 343), 2)
        # The struck 14 ball runs straight into the rack, so its path ends there.
        self.assertTrue(result.blockers)

    def test_object_angle_is_stable_under_pixel_noise(self):
        rng = np.random.default_rng(7)
        angles = []
        for _ in range(4):
            noisy = np.clip(USER.astype(int) + rng.integers(-2, 3, USER.shape), 0, 255).astype(np.uint8)
            angles.append(angle(main.detect(noisy).object_segments[0]))
        self.assertLess(max(angles) - min(angles), 1.0)


@needs_screenshots
class OverlayFeedbackTests(unittest.TestCase):
    def test_guide_stays_below_full_opacity(self):
        detection = main.detect(USER)
        h, w = USER.shape[:2]
        layer = main.render_guide_layer(detection, w, h, adjusting=True, active=True)
        self.assertGreater(int((layer[:, :, 3] > 0).sum()), 1000, "guide drew nothing")
        self.assertLessEqual(int(layer[:, :, 3].max()), 239)

    def test_remove_overlay_recovers_the_game_pixels(self):
        detection = main.detect(USER)
        h, w = USER.shape[:2]
        for adjusting in (False, True):
            layer = main.render_guide_layer(detection, w, h, adjusting=adjusting, active=True)
            recovered = main.remove_overlay(main.composite_overlay(USER, layer), layer)
            self.assertLessEqual(int(np.abs(recovered.astype(int) - USER).max()), 4)

    def test_capture_visible_guide_without_subtraction_corrupts_the_next_frame(self):
        # This is the feedback the capture-visible overlay causes when its own
        # drawing is analysed. It documents why subtraction is required.
        reference = main.detect(USER)
        broken = guide_cycles(USER, 2, subtract=False)
        self.assertTrue(any(d.ghost is None or d.target != reference.target or d.cue != reference.cue
                            for d in broken))

    def test_subtracted_guide_keeps_the_shot_stable_over_repeated_frames(self):
        reference = main.detect(USER)
        ref_aim = angle(reference.aim_segments[0])
        ref_obj = angle(reference.object_segments[0])
        for detection in guide_cycles(USER, 3, subtract=True):
            self.assertEqual(detection.table, reference.table)
            self.assertEqual(len(detection.balls), len(reference.balls))
            self.assertLessEqual(abs(detection.cue[0] - reference.cue[0]), 2)
            self.assertLessEqual(abs(detection.cue[1] - reference.cue[1]), 2)
            self.assertLessEqual(np.hypot(detection.ghost[0] - reference.ghost[0],
                                          detection.ghost[1] - reference.ghost[1]), 2)
            self.assertLess(abs(angle(detection.aim_segments[0]) - ref_aim), .6)
            self.assertLess(abs(angle(detection.object_segments[0]) - ref_obj), 1.0)

    def test_fit_table_ignores_the_guide_drawn_over_the_pockets(self):
        detection = main.detect(USER)
        h, w = USER.shape[:2]
        layer = main.render_guide_layer(detection, w, h, adjusting=True, active=True)
        captured = main.remove_overlay(main.composite_overlay(USER, layer), layer)
        self.assertEqual(main.find_table_bounds(captured), main.find_table_bounds(USER))

    def test_logical_and_physical_rects_round_trip_at_unit_scale(self):
        screen = QApplication.primaryScreen()
        rect = main.QRect(100, 120, 640, 360)
        physical = main.logical_to_physical(rect, screen)
        back = main.physical_to_logical(physical["left"], physical["top"], physical["width"], physical["height"])
        if screen.devicePixelRatio() == 1:
            self.assertEqual(physical, {"left": 100, "top": 120, "width": 640, "height": 360})
        self.assertEqual(back, rect)


class PathTracingTests(unittest.TestCase):
    def test_path_stops_at_the_first_ball_in_the_way(self):
        blocked = []
        path = main.trace_bounces((100, 100), (1, 0), (0, 0, 1000, 500), 10,
                                  obstacles=[(300, 105, 10), (200, 140, 10)], blocked=blocked)
        self.assertEqual(len(path), 1)
        self.assertEqual(blocked, [(300, 105, 10)])
        end_x = path[0][1][0]
        self.assertAlmostEqual(end_x, 300 - np.sqrt(20 ** 2 - 5 ** 2), delta=1)

    def test_path_bounces_off_a_cushion_without_obstacles(self):
        path = main.trace_bounces((100, 100), (1, 1), (0, 0, 1000, 400), 10, max_bounces=1)
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0][1], (400, 400))
        self.assertGreater(path[1][1][0], 400)
        self.assertLess(path[1][1][1], 400)


if __name__ == "__main__":
    unittest.main()
