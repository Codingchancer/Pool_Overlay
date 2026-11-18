"""White tables and tight racks.

On white cloth the cue ball's shaded rim and its drop shadow are equally
dark, so a full-circle fit slid 3-7 px towards the shadow; it is now fitted
on its lit side. In a tight rack, balls that circle finding missed are
found on the packing grid their touching neighbours form."""
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


class WhiteTableCueTests(unittest.TestCase):
    def test_cue_ball_on_white_cloth(self):
        # Each failed before (cue ball 3-7 px off towards its shadow).
        for seed, kind in ((30000, "scatter"), (30001, "scatter"), (30004, "scatter"),
                           (30002, "miss"), (30001, "pot"), (30004, "pot")):
            *_, problems, stats = st.evaluate(seed, kind, "white")
            self.assertEqual(problems, [], f"white {seed} {kind}")

    def test_shadow_direction_is_measured(self):
        # The synthetic shadows fall down-right (offset 0.2, 0.25 radii).
        frame, truth = fg.render(fg.random_scene(30001, "scatter", "white"))
        d = detector.detect(frame)
        x, y, w, h = d.table
        lightness = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        balls = [(bx - x, by - y, br, kind) for bx, by, br, kind in d.balls]
        sdir = detector.shadow_direction(lightness, detector.local_lightness(lightness), balls, d.balls[0][2])
        self.assertIsNotNone(sdir)
        angle = np.degrees(np.arctan2(sdir[1], sdir[0]))
        self.assertLess(abs(angle - np.degrees(np.arctan2(.25, .2))), 25)

    def test_light_grey_keeps_the_usual_fit(self):
        # There the cue ball is far brighter than the cloth: no lit-side fit.
        frame, truth = fg.render(fg.random_scene(30001, "scatter", "light grey"))
        d = detector.detect(frame)
        x, y, w, h = d.table
        self.assertIsNone(detector.white_cloth_cue_centre(frame[y:y + h, x:x + w], (d.cue[0] - x, d.cue[1] - y),
                                                          d.balls, (x, y)))


class WhiteTableTests(unittest.TestCase):
    """White cloth, synthetic seeds 40000+: each failed before."""

    def test_cue_centre_from_the_lit_side_rim_dip(self):
        # The rim is shaded darker than the cloth; the step fit slid 4-8 px
        # towards the shadow (here 5-6 px).
        for seed, kind in ((40012, "scatter"), (40020, "scatter"), (40024, "miss"), (40005, "scatter")):
            frame, truth = fg.render(fg.random_scene(seed, kind, "white"))
            d = detector.detect(frame)
            err = float(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]))
            self.assertLess(err, 1.5, f"{seed} {kind}")

    def test_shadow_direction_robust_to_false_candidates_and_racks(self):
        # Five false candidates pulled the plain average apart (seed 40001);
        # in a rack every ring touched a neighbour (seed 40001 rack).
        for kind in ("pot", "rack"):
            frame, truth = fg.render(fg.random_scene(40001, kind, "white"))
            d = detector.detect(frame)
            x, y, w, h = d.table
            lightness = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            balls = [(bx - x, by - y, br, k) for bx, by, br, k in d.balls]
            sdir = detector.shadow_direction(lightness, detector.local_lightness(lightness), balls,
                                             float(np.median([b[2] for b in balls])))
            self.assertIsNotNone(sdir, kind)
            angle = np.degrees(np.arctan2(sdir[1], sdir[0]))
            self.assertLess(abs(angle - np.degrees(np.arctan2(.25, .2))), 15, kind)

    def test_aim_ray_found_by_its_outlines(self):
        # On white cloth the ray's white core is no brighter than the cloth;
        # only its dark outlines show. No ray, so no ghost, before.
        for seed, kind in ((40012, "scatter"), (40020, "miss"), (40024, "pot")):
            *_, problems, stats = st.evaluate(seed, kind, "white")
            self.assertFalse([p for p in problems if p.startswith(("ghost", "aim"))], f"{seed} {kind}")

    def test_whitest_ball_is_the_cue_ball_on_white_cloth(self):
        # A striped ball's white half is as unsaturated as the cue ball there.
        for seed, kind in ((40003, "rack"), (40007, "scatter")):
            frame, truth = fg.render(fg.random_scene(seed, kind, "white"))
            d = detector.detect(frame)
            self.assertLess(float(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1])), 3, f"{seed} {kind}")

    def test_ghost_ring_over_the_cue_ball_is_not_fitted_as_it(self):
        # A fitted circle with a thin bright stroke along it is the ring.
        frame, truth = fg.render(fg.random_scene(40001, "near", "white"))
        d = detector.detect(frame)
        x, y, w, h = d.table
        lightness = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        gx, gy = truth.ghost[0] - x, truth.ghost[1] - y
        cx, cy = truth.cue[0] - x, truth.cue[1] - y
        self.assertGreater(detector._thin_stroke_share(lightness, gx, gy, truth.radius), detector.WHITE_RING_STROKE)
        self.assertLess(detector._thin_stroke_share(lightness, cx, cy, truth.radius * .95), detector.WHITE_RING_STROKE)


class WhiteTableLinesAndStripesTests(unittest.TestCase):
    """Striped balls and the game's short lines on white cloth."""

    def test_stripe_centres_from_the_rim(self):
        # Circle finding put stripes 3-6 px towards their shadow there.
        errors = []
        for seed in (40001, 40002):
            frame, truth = fg.render(fg.random_scene(seed, "scatter", "white"))
            d = detector.detect(frame)
            r = truth.radius
            for n, (tx, ty) in truth.balls.items():
                if n < 9 or any(np.hypot(ox - tx, oy - ty) < 2.6 * r for m, (ox, oy) in truth.balls.items() if m != n):
                    continue
                near = min(d.balls, key=lambda b: np.hypot(b[0] - tx, b[1] - ty))
                errors.append(float(np.hypot(near[0] - tx, near[1] - ty)))
        self.assertGreater(len(errors), 5)
        self.assertLess(float(np.median(errors)), 1.5)

    def test_object_line_read_by_its_outlines(self):
        # Never read before (its white core is as bright as the cloth), so
        # the struck ball was fitted on the ghost, 4-5 px off: 9 deg.
        for seed in (40001, 40022, 40024):
            *_, problems, _ = st.evaluate(seed, "scatter", "white")
            self.assertFalse([p for p in problems if p.startswith(("object", "cue path"))], f"{seed}")

    def test_ghost_ring_found_on_its_ridges(self):
        # The ring itself was missing from the plain strokes there.
        for kind in ("scatter", "pot"):
            *_, problems, _ = st.evaluate(40020, kind, "white")
            self.assertFalse([p for p in problems if p.startswith("ghost")], kind)

    def test_stripe_on_the_sticks_line_is_kept(self):
        # Its white caps look like cloth beside the stick, so it read as one
        # of the stick's rings once its centre was refitted.
        *_, problems, _ = st.evaluate(40022, "miss", "white")
        self.assertFalse([p for p in problems if p.startswith("balls")])

    def test_second_circle_on_a_stripe_is_merged(self):
        *_, problems, _ = st.evaluate(40007, "rack", "white")
        self.assertFalse([p for p in problems if "false" in p])

    def test_thin_ridges_need_dark_on_both_sides(self):
        v = np.full((40, 60), 200, np.uint8)
        v[10, 5:55] = 205                      # a stroke's core...
        v[8:10, 5:55] = 120                    # ...between its dark outlines
        v[11:13, 5:55] = 120
        v[25:27, 5:55] = 120                   # a single dark line: cloth beside it is no ridge
        ridges = detector.thin_ridges(v, 40)
        self.assertTrue(ridges[10, 10:50].all())
        self.assertFalse(ridges[22:31, :].any())

    def test_boxed_in_ball_has_no_rim_rays(self):
        # Every lit direction led into a neighbour: an empty map crashed
        # OpenCV; a duplicate detection at the same spot divided by zero.
        lightness = np.full((200, 200), 200, np.float32)
        ring = [(100 + 25 * np.cos(a), 100 + 25 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        self.assertEqual(detector._lit_rim_dips(lightness, 100.0, 100.0, 12.0, (1.0, 0.0), .4, 1.9, ring).shape, (0, 2))
        detector._lit_rim_dips(lightness, 100.0, 100.0, 12.0, (1.0, 0.0), .4, 1.9, [(100.0, 100.0)])


class TightRackTests(unittest.TestCase):
    def test_packed_balls_are_found(self):
        # Each missed packed balls (or misplaced the struck one) before.
        for seed, cloth in ((30003, "blue"), (30003, "lime"), (30004, "yellow"), (30003, "maroon"),
                            (30003, "sky"), (30006, "grey")):
            *_, problems, _ = st.evaluate(seed, "rack", cloth)
            self.assertEqual(problems, [], f"rack {seed} {cloth}")

    def test_grid_spot_and_gaps(self):
        r = 10.0
        # A row of three touching balls with one gap above the middle pair.
        pts = [(100.0, 100.0), (120.0, 100.0), (140.0, 100.0), (130.0, 100.0 - 20 * np.sqrt(3) / 2)]
        gaps = detector.rack_gaps(pts, r, 400, 400, [(-50, -50)])
        spots = [(round(x), round(y)) for x, y, _ in gaps]
        self.assertIn((110, round(100 - 20 * np.sqrt(3) / 2)), spots)
        # A ball 3 px off its grid spot is placed back on it.
        moved = list(pts) + [(113.0, 100 - 20 * np.sqrt(3) / 2 + 2)]
        spot = detector.grid_spot(moved, 4, r)
        self.assertIsNotNone(spot)
        self.assertLess(np.hypot(spot[0] - 110, spot[1] - (100 - 20 * np.sqrt(3) / 2)), .6)

    def test_lone_balls_get_no_gaps(self):
        # Two balls far apart: nothing to fill.
        self.assertEqual(detector.rack_gaps([(100.0, 100.0), (300.0, 100.0), (200.0, 300.0)], 10.0,
                                            400, 400, []), [])


if __name__ == "__main__":
    unittest.main()
