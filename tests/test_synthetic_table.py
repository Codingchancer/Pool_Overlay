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


class ClothColourTests(unittest.TestCase):
    """Miniclip tables come in many colours; the detector used to assume blue
    and found no table at all on green, red or purple cloth (0/40 shots)."""

    def test_every_cloth_colour_works(self):
        # Purple (hue 140) is the worst case: exactly the colour of ball 4,
        # and the stick's grey rings stand out as ball-sized circles on it.
        for cloth, least in (("green", 12), ("red", 12), ("purple", 8), ("brown", 12), ("teal", 12)):
            results = [st.evaluate(seed, None, cloth) for seed in range(9000, 9016)]
            passed = sum(1 for *_, problems, _ in results if not problems)
            self.assertGreaterEqual(passed, least, f"{cloth}: only {passed}/16 frames passed")
            worst_table = max(stats.get("table", 99.0) for *_, stats in results)
            self.assertLessEqual(worst_table, 2.0, cloth)
            aims = [stats["aim"] for *_, stats in results if "aim" in stats]
            self.assertGreaterEqual(len(aims), 14, cloth)
            self.assertLessEqual(np.median(aims), .3, cloth)

    def test_cloth_hue_is_found_across_the_red_wrap(self):
        import cv2
        import detector

        for hue in (2, 176, 60, 140, 98):
            frame, _ = fg.render(fg.random_scene(9001, None, hue))
            found = detector.cloth_hue(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), step=4)
            self.assertLess(abs(float(detector.hue_delta(found, hue))), 3.0, f"hue {hue}: {found}")

    def test_same_colour_ball_is_kept_by_its_number_badge(self):
        # Purple ball 4 on a purple table is cloth-coloured inside.
        found = 0
        for seed in range(9000, 9012):
            frame, truth = fg.render(fg.random_scene(seed, None, "purple"))
            if 4 not in truth.balls:
                continue
            bx, by = truth.balls[4]
            d = main.detect(frame)
            found += any(np.hypot(bx - x, by - y) <= truth.radius * .5 for x, y, _, _ in d.balls or [])
        self.assertGreaterEqual(found, 8)


class PotTests(unittest.TestCase):
    def test_pocket_outcome_geometry(self):
        import detector

        table, r = (100, 100, 1000, 500), 13.5
        left, top = 100 + r, 100 + r
        holes = detector.pocket_holes(table)
        px, py, _ = holes[0]  # top-left corner pocket

        def path(start, toward):
            ux, uy = toward[0] - start[0], toward[1] - start[1]
            return detector.trace_bounces(start, (ux, uy), (left, top, 1100 - r, 600 - r), r, max_bounces=3)

        straight = path((600, 400), (px, py))
        self.assertEqual(detector.pocket_outcome(straight, table, r), ("pot", (int(round(px)), int(round(py)))))
        # Along the diagonal moved sideways by the hole radius: hits the jaw.
        # (At a shallow angle the same offset meets the cushion several radii
        # before the pocket and is a plain bounce.)
        hole = holes[0][2]
        side = hole / np.sqrt(2)
        jaw = path((px + 300 + side, py + 300 - side), (px + side, py - side))
        self.assertEqual(detector.pocket_outcome(jaw, table, r)[0], "jaw")
        # Straight at the middle of the long cushion between pockets: no pocket.
        rail = path((350, 400), (350, 0))
        self.assertEqual(detector.pocket_outcome(rail, table, r), (None, None))
        # A ball in the way stops it before the pocket.
        blocked = detector.trace_bounces((600, 400), (px - 600, py - 400), (left, top, 1100 - r, 600 - r), r,
                                         max_bounces=3, obstacles=[(400, 287, 13)])
        self.assertEqual(detector.pocket_outcome(blocked, table, r), (None, None))

    def test_pot_calls_match_the_truth(self):
        judged = wrong = 0
        outcomes = set()
        for seed in range(9300, 9340):
            *_, truth, d, problems, stats = run(seed, "pot")
            if "pot_agree" in stats:
                judged += 1
                outcomes.add(truth.pot)
                wrong += any(p.startswith("pot") for p in problems)
        self.assertGreaterEqual(judged, 25)
        self.assertEqual(outcomes >= {"pot", None}, True)
        self.assertLessEqual(wrong, 1)


class StickPullTests(unittest.TestCase):
    def test_stick_pull_matches_the_truth(self):
        errors, missing = [], 0
        for seed in range(9000, 9040):
            scene = fg.random_scene(seed)
            frame, _ = fg.render(scene)
            d = main.detect(frame)
            if d.cue is None or d.aim_dir is None:
                continue
            if d.stick_pull is None:
                missing += 1
                continue
            errors.append(abs(d.stick_pull - (scene.stick_pull + fg.STICK_REST)))
        self.assertLessEqual(missing, 6)
        self.assertLessEqual(np.median(errors), .12)
        self.assertLessEqual(np.percentile(errors, 90), .4)

    def test_no_stick_reads_none(self):
        from dataclasses import replace

        quiet = 0
        for seed in range(9000, 9010):
            frame, _ = fg.render(replace(fg.random_scene(seed), show_stick=False))
            d = main.detect(frame)
            quiet += d.stick_pull is None
        self.assertGreaterEqual(quiet, 9)

    def test_pull_grows_with_power(self):
        from dataclasses import replace

        scene = fg.random_scene(9004)
        readings = []
        for pull in (0.5, 1.5, 2.5, 3.5):
            frame, _ = fg.render(replace(scene, stick_pull=pull))
            readings.append(main.detect(frame).stick_pull)
        self.assertTrue(all(v is not None for v in readings), readings)
        self.assertTrue(all(b > a for a, b in zip(readings, readings[1:])), readings)


class PowerSliderTests(unittest.TestCase):
    def test_slider_level_matches_the_truth(self):
        from dataclasses import replace

        good = total = 0
        for seed in range(9000, 9016):
            base = fg.random_scene(seed)
            for power in (0.0, .3, .6, .9):
                frame, _ = fg.render(replace(base, power=power))
                percent = main.detect(frame).power_percent
                total += 1
                good += percent is not None and abs(percent - power * 100) <= 2
        # A shot whose cue stick runs across the slider can spoil a reading.
        self.assertGreaterEqual(good, total - 4)

    def test_no_slider_reads_none(self):
        for seed in range(9000, 9010):
            frame, _ = fg.render(fg.random_scene(seed))
            d = main.detect(frame)
            self.assertIsNone(d.power_slider)
            self.assertIsNone(d.power_percent)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "outputs/bad/stick-and-cushion/snip.png").exists(),
                         "real screenshot not present")
    def test_real_slider_at_rest(self):
        import cv2

        frame = cv2.imread(str(Path(__file__).resolve().parents[1] / "outputs/bad/stick-and-cushion/snip.png"))
        d = main.detect(frame)
        self.assertIsNotNone(d.power_slider)
        self.assertLess(d.power_percent, 3)


class BlackClothTests(unittest.TestCase):
    """Colourless near-black cloth (Miniclip's Lucky Shot table): the table
    used to be "not found" because pockets and cloth were judged by hue and
    a darkness limit the cloth itself was under."""

    def test_black_table_and_aim_are_found(self):
        results = [st.evaluate(seed, None, "black") for seed in range(9000, 9016)]
        tables = [stats.get("table", 99.0) for *_, stats in results]
        self.assertLessEqual(sorted(tables)[-2], 2.0, tables)
        aims = [stats["aim"] for *_, stats in results if "aim" in stats]
        self.assertGreaterEqual(len(aims), 14)
        self.assertLessEqual(np.median(aims), .3)
        passed = sum(1 for *_, problems, _ in results if not problems)
        # The black 8 ball on black cloth is the weak spot (only its white
        # badge shows).
        self.assertGreaterEqual(passed, 7)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "outputs/bad/lucky-shot/window.png").exists(),
                         "real screenshot not present")
    def test_real_lucky_shot_table(self):
        import cv2

        frame = cv2.imread(str(Path(__file__).resolve().parents[1] / "outputs/bad/lucky-shot/window.png"))
        d = main.detect(frame)
        self.assertIsNotNone(d.table)
        for got, want in zip(d.table, (369, 231, 1070, 535)):
            self.assertLessEqual(abs(got - want), 4, d.table)
        self.assertIsNotNone(d.cue)
        self.assertIsNotNone(d.target)
        self.assertIsNotNone(d.ghost)


class BallSizeTests(unittest.TestCase):
    def test_every_ball_is_drawn_at_one_true_size(self):
        # A ball's drop shadow used to make its circle grow (up to 22% too
        # big on synthetic frames), so the overlay drew a bigger ring.
        for seed in range(9000, 9012):
            frame, truth = fg.render(fg.random_scene(seed))
            d = main.detect(frame)
            radii = {r for _, _, r, _ in d.balls}
            self.assertEqual(len(radii), 1, f"seed {seed}: {sorted(radii)}")
            self.assertLessEqual(abs(radii.pop() - truth.radius), max(1.5, truth.radius * .1), f"seed {seed}")

    def test_cue_ball_centre_not_pulled_by_its_shadow(self):
        # Synthetic seeds 13 and 55: the cue ball's circle grew onto its
        # shadow and its centre moved 3-4 px towards it.
        for seed in (13, 55):
            *_, truth, d, problems, stats = run(seed)
            self.assertLessEqual(stats["cue"], 2.0, f"seed {seed}: {problems}")


class CoveredPocketTests(unittest.TestCase):
    """The cue ball or the stick on a corner pocket used to make the pocket
    fit give up; on a real frame the colour fallback then returned a table
    too small, and Fit table resized the overlay to it."""

    def test_pocket_fit_works_with_one_corner_covered(self):
        import cv2
        import detector

        for seed in range(9000, 9010):
            frame, truth = fg.render(fg.random_scene(seed))
            holes = fg.pocket_holes(truth.table)
            for corner in (0, 2, 3, 5):
                covered = frame.copy()
                px, py, hole = holes[corner]
                cv2.circle(covered, (int(px), int(py)), int(hole * 1.3), (230, 232, 232), -1)
                found = detector.table_from_pockets(covered)
                self.assertIsNotNone(found, f"seed {seed}, corner {corner}")
                err = max(abs(a - b) for a, b in zip(found, truth.table))
                self.assertLessEqual(err, 3.0, f"seed {seed}, corner {corner}: {found}")

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "outputs/bad/resize-stick-top/window.png").exists(),
                         "real screenshot not present")
    def test_real_frame_with_cue_ball_on_a_corner_pocket(self):
        import cv2
        import detector

        frame = cv2.imread(str(Path(__file__).resolve().parents[1] / "outputs/bad/resize-stick-top/window.png"))
        table = detector.find_table_bounds(frame)
        for got, want in zip(table, (331, 219, 966, 484)):
            self.assertLessEqual(abs(got - want), 4, table)


class KnockOnTests(unittest.TestCase):
    """A predicted path used to stop at the first ball in its way; now the
    hit ball's path and the moving ball's deflection are predicted too."""

    def test_knocked_ball_direction_matches_the_truth(self):
        errors = []
        for kind in (None, "pot"):
            for seed in range(9000, 9080):
                *_, truth, d, problems, stats = run(seed, kind)
                if truth.target is None or d.target is None:
                    continue
                tx, ty = truth.balls[truth.target]
                if np.hypot(d.target[0] - tx, d.target[1] - ty) > truth.radius * .5:
                    continue
                others = [p for n, p in truth.balls.items() if n != truth.target]
                knock = fg.truth_knock((tx, ty), truth.object_dir, truth.table, truth.radius, others)
                if knock is None or knock[3] < truth.radius * .4:
                    continue
                hit, direction = knock[0], knock[1]
                path = d.knock_paths[0] if d.knock_paths else []
                if not path or np.hypot(path[0][0][0] - hit[0], path[0][0][1] - hit[1]) > truth.radius * .6:
                    continue
                contact = d.object_segments[-1][1]
                got = np.degrees(np.arctan2(path[0][0][1] - contact[1], path[0][0][0] - contact[0]))
                errors.append(st.angle_diff(got, np.degrees(np.arctan2(direction[1], direction[0]))))
        self.assertGreaterEqual(len(errors), 20)
        self.assertLessEqual(np.median(errors), 2.5)
        self.assertLessEqual(np.percentile(errors, 90), 6.0)

    def test_lists_are_object_then_cue_and_empty_when_unblocked(self):
        for seed in range(9000, 9010):
            *_, truth, d, problems, stats = run(seed)
            if d.target is None:
                continue
            self.assertEqual(len(d.knock_paths), 2)
            self.assertEqual(len(d.deflect_paths), 2)
            for path, knock in ((d.object_segments, d.knock_paths[0]), (d.cue_segments, d.knock_paths[1])):
                if knock:
                    # It starts on a blocker the path ends against.
                    self.assertIn(tuple(knock[0][0]), [tuple(b[:2]) for b in d.blockers])

    def test_knock_on_geometry(self):
        import detector

        bounds, r = (10.0, 10.0, 990.0, 490.0), 10.0
        # Moving right along y=250 into a ball offset half a radius below.
        path = [((300, 250), (480, 250))]
        hit = (499, 255, 10)
        knocked, deflected = detector.knock_on(path, [hit], [hit], bounds, r, 1)
        (ax, ay), (bx, by) = knocked[0]
        self.assertEqual((ax, ay), (499, 255))
        self.assertGreater(bx, ax)
        self.assertGreater(by, ay)          # pushed down-right, along the line of centres
        (cx, cy), (dx_, dy_) = deflected[0]
        self.assertLess(dy_, cy)            # the moving ball glances off upwards


class NeonTableTests(unittest.TestCase):
    """Miniclip's neon table has dark bands along the rails that touch the
    corner pockets and joined them into long blobs: "Table not found"."""

    def test_pockets_found_when_dark_bands_touch_them(self):
        import cv2
        import detector

        for seed in range(9000, 9008):
            frame, truth = fg.render(fg.random_scene(seed))
            x, y, w, h = (int(round(v)) for v in truth.table)
            for yy in (y - 6, y + h + 6):
                cv2.line(frame, (x - 40, yy), (x + w + 40, yy), (8, 8, 8), 6)
            for xx in (x - 6, x + w + 6):
                cv2.line(frame, (xx, y - 40), (xx, y + h + 40), (8, 8, 8), 6)
            found = detector.table_from_pockets(frame)
            self.assertIsNotNone(found, f"seed {seed}")
            self.assertLessEqual(max(abs(a - b) for a, b in zip(found, truth.table)), 3.0, f"seed {seed}")

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "outputs/bad/batch3/shot9.png").exists(),
                         "real screenshot not present")
    def test_real_neon_table(self):
        import cv2

        frame = cv2.imread(str(Path(__file__).resolve().parents[1] / "outputs/bad/batch3/shot9.png"))
        d = main.detect(frame)
        for got, want in zip(d.table, (578, 340, 1017, 510)):
            self.assertLessEqual(abs(got - want), 4, d.table)
        self.assertGreaterEqual(len(d.balls), 15)
        self.assertIsNotNone(d.ghost)


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
