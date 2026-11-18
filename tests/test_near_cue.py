"""The struck ball nearly touching the cue ball ("near" shots).

Then Miniclip's ghost ring lies on top of the cue ball and hides the aim
ray. The ring is found on its own anywhere on the table (a thin bright
stroke between dark outlines all the way round); only the cue ball can
overlap it, so the ball under it is the cue ball, and the ring places the
ghost exactly. Synthetic seeds 40000+; each failed before."""
import dataclasses
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


def table_roi(frame):
    x, y, w, h = detector.find_table_bounds(frame)
    return frame[y:y + h, x:x + w], (x, y)


class GhostRingTests(unittest.TestCase):
    def test_ring_found_anywhere_on_every_kind_of_cloth(self):
        for seed, kind, cloth in ((40018, "near", "blue"), (40023, "near", "lime"), (40001, "near", "white"),
                                  (40002, "scatter", "light grey"), (40013, "pot", "black"), (40004, "rack", "yellow")):
            frame, truth = fg.render(fg.random_scene(seed, kind, cloth))
            roi, (x, y) = table_roi(frame)
            ring = detector.find_ghost_ring(roi, float(round(truth.radius)))
            self.assertIsNotNone(ring, f"{seed} {kind} {cloth}")
            self.assertLess(np.hypot(ring[0] + x - truth.ghost[0], ring[1] + y - truth.ghost[1]), .8,
                            f"{seed} {kind} {cloth}")

    def test_striped_balls_and_stick_joints_are_not_rings(self):
        # A striped ball's white caps (40022) and the stick's metal joint
        # (40017) make ring-like ridges; the real ring is the one returned.
        for seed, kind, cloth in ((40022, "miss", "blue"), (40017, "near", "dark green"), (40015, "near", "blue")):
            frame, truth = fg.render(fg.random_scene(seed, kind, cloth))
            roi, (x, y) = table_roi(frame)
            ring = detector.find_ghost_ring(roi, float(round(truth.radius)))
            self.assertLess(np.hypot(ring[0] + x - truth.ghost[0], ring[1] + y - truth.ghost[1]), .8, f"{seed} {kind}")

    def test_no_ring_without_the_guide(self):
        scene = dataclasses.replace(fg.random_scene(40015, "near", "blue"), show_guide=False)
        frame, truth = fg.render(scene)
        roi, _ = table_roi(frame)
        self.assertIsNone(detector.find_ghost_ring(roi, float(round(truth.radius))))

    def test_search_around_a_spot(self):
        frame, truth = fg.render(fg.random_scene(40000, "near", "blue"))
        roi, (x, y) = table_roi(frame)
        r = float(round(truth.radius))
        cx, cy = truth.cue[0] - x, truth.cue[1] - y
        ring = detector.find_ghost_ring(roi, r, around=(cx, cy, r * detector.RING_OVERLAP))
        self.assertLess(np.hypot(ring[0] + x - truth.ghost[0], ring[1] + y - truth.ghost[1]), .8)
        # Far from it: nothing.
        self.assertIsNone(detector.find_ghost_ring(roi, r, around=(cx + 12 * r, cy, r)))

    def test_vectorised_stroke_share_matches_the_ray_by_ray_one(self):
        frame, truth = fg.render(fg.random_scene(40001, "near", "white"))
        roi, (x, y) = table_roi(frame)
        lightness = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        offsets = np.arange(-5.0, 5.01, .5)
        near = np.nonzero(np.abs(offsets) <= 2.5)[0]
        rng = np.random.default_rng(3)
        gx, gy = truth.ghost[0] - x, truth.ghost[1] - y
        for _ in range(40):
            fx, fy, fr = gx + rng.uniform(-6, 6), gy + rng.uniform(-6, 6), rng.uniform(10, 20)
            xs = (fx + np.outer(np.cos(detector._STROKE_ANGLES), fr + offsets)).astype(np.float32)
            ys = (fy + np.outer(np.sin(detector._STROKE_ANGLES), fr + offsets)).astype(np.float32)
            profiles = cv2.remap(lightness, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            hits = sum(any(p[k] - p[max(0, k - 6):k].min() >= 15 and p[k] - p[k + 1:k + 7].min() >= 15 for k in near)
                       for p in profiles)
            self.assertAlmostEqual(detector._thin_stroke_share(lightness, fx, fy, fr), hits / len(profiles))


class NearCueTests(unittest.TestCase):
    def test_cue_ball_is_the_ball_under_the_ring(self):
        # A striped ball (its own object line ending at a "ghost"), the
        # struck ball (the ring touching it) or a stick ring was taken.
        for seed in (40004, 40024, 40029, 40037, 40053, 40075, 40076, 40086):
            frame, truth = fg.render(fg.random_scene(seed, "near", "blue"))
            d = detector.detect(frame)
            self.assertLess(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]), 3, f"{seed}")

    def test_cue_ball_missed_by_circle_finding(self):
        # Under the ring no circle was found at all: found again with the
        # ring painted out.
        for seed, cloth in ((40085, "blue"), (40002, "white")):
            frame, truth = fg.render(fg.random_scene(seed, "near", cloth))
            d = detector.detect(frame)
            self.assertLess(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]), 3, f"{seed} {cloth}")

    def test_ghost_placed_on_the_ring(self):
        # The object line was not read (no ghost), or the ghost placed one
        # diameter back along it was 3 px off.
        for seed in (40000, 40070, 40084, 40039, 40042, 40061):
            *_, problems, stats = st.evaluate(seed, "near", "blue")
            self.assertFalse([p for p in problems if p.startswith("ghost")], f"{seed}")
            self.assertLess(stats["ghost"], 1.0, f"{seed}")

    def test_white_cloth_cue_ball_beside_the_ring_stays(self):
        # Fitted again beside the ring on the cloth-distance image, which is
        # weak on a white cloth, it moved 5-19 px off.
        for seed in (40000, 40001, 40003, 40004, 40007):
            frame, truth = fg.render(fg.random_scene(seed, "near", "white"))
            d = detector.detect(frame)
            self.assertLess(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]), 1.5, f"{seed}")

    def test_stick_rings_filtered_along_the_corrected_aim(self):
        # The aim read before the cue ball was fitted beside the ring was far
        # off, so one of the stick's rings stayed as a ball.
        for seed in (40003, 40014, 40027, 40038, 40069):
            *_, problems, _ = st.evaluate(seed, "near", "blue")
            self.assertFalse([p for p in problems if "false" in p], f"{seed}")

    def test_near_shots_pass(self):
        for seed, cloth in ((40001, "blue"), (40010, "green"), (40020, "red"), (40030, "navy"), (40003, "white")):
            *_, problems, _ = st.evaluate(seed, "near", cloth)
            self.assertEqual(problems, [], f"{seed} {cloth}")


class NearShotAimTests(unittest.TestCase):
    """With the ghost on top of the cue ball, the aim from the cue ball's
    centre to it turned by degrees per pixel; the stick gives it."""

    def test_aim_from_the_stick(self):
        # Each failed only on the aim before (1-2 px off at the ghost).
        for seed, cloth in ((40033, "blue"), (40084, "blue"), (40054, "green"), (40021, "teal"),
                            (40012, "white"), (40010, "light grey"), (40036, "beige"), (40052, "sky")):
            *_, problems, stats = st.evaluate(seed, "near", cloth)
            self.assertEqual(problems, [], f"{seed} {cloth}")

    def test_stick_direction(self):
        for seed, cloth in ((40010, "green"), (40002, "white"), (40030, "navy")):
            scene = fg.random_scene(seed, "near", cloth)
            for shown in (True, False):
                frame, truth = fg.render(dataclasses.replace(scene, show_stick=shown))
                roi, (x, y) = table_roi(frame)
                u = detector.stick_direction_near(roi, (truth.cue[0] - x, truth.cue[1] - y),
                                                  (truth.ghost[0] - x, truth.ghost[1] - y), float(round(truth.radius)))
                if not shown:
                    self.assertIsNone(u, f"{seed} {cloth}: a stick without one")
                    continue
                a = np.radians(truth.aim_degrees)
                self.assertLess(np.degrees(np.arccos(np.clip(u[0] * np.cos(a) + u[1] * np.sin(a), -1, 1))), .3,
                                f"{seed} {cloth}")

    def test_stick_found_from_a_rough_direction(self):
        # The ghost's direction from the cue ball is rough here (20 degrees
        # off is common): the stick is still found and read exactly.
        frame, truth = fg.render(fg.random_scene(40030, "near", "navy"))
        roi, (x, y) = table_roi(frame)
        cx, cy = truth.cue[0] - x, truth.cue[1] - y
        a = np.radians(truth.aim_degrees)
        for turn in (-25, 25):
            b = a + np.radians(turn)
            u = detector.stick_direction_near(roi, (cx, cy), (cx + 8 * np.cos(b), cy + 8 * np.sin(b)),
                                              float(round(truth.radius)))
            self.assertLess(np.degrees(np.arccos(np.clip(u[0] * np.cos(a) + u[1] * np.sin(a), -1, 1))), .3, turn)


if __name__ == "__main__":
    unittest.main()
