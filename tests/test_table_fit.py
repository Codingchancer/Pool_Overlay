"""Table fit when the corner pockets or the cushion edges do not look as usual.

A real frame (a dark grey table with both right corner pockets lit red, the
cue ball against the right cushion) came out ~40 px short on the right: with
two corner pockets lit the pocket fit gave up, and the cushion edge was
snapped onto a thin bright line inside the cushion (the guide's own table
border, still in the capture), so the border, once drawn there, held the
table there.
"""
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import detector
import fake_game as fg

REAL_LIT = Path(__file__).resolve().parents[1] / "outputs/bad/lit-pockets/frame.png"
# The guide's table border as capture.py draws it: QColor(60, 255, 180, 110), 2 px.
BORDER_BGR = np.array([180, 255, 60], np.float32)
BORDER_ALPHA = 110 / 255
SEEDS = range(9100, 9106)
CLOTHS = ("blue", "grey", "black", "green")


def light_pocket(frame, px, py, hole):
    """A pocket lit as the game does it: a bright red ring round a small dark core."""
    cv2.circle(frame, (int(round(px)), int(round(py))), int(round(hole * 1.2)), (40, 30, 225), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(round(px)), int(round(py))), int(round(hole * .5)), (8, 6, 20), -1, cv2.LINE_AA)


def draw_border(frame, table):
    """The guide's table border blended into the frame, as a capture shows it."""
    layer = np.zeros(frame.shape[:2], np.uint8)
    x, y, w, h = (int(round(v)) for v in table)
    cv2.rectangle(layer, (x, y), (x + w, y + h), 255, 2)
    a = layer.astype(np.float32)[:, :, None] / 255 * BORDER_ALPHA
    frame[:] = (frame.astype(np.float32) * (1 - a) + BORDER_BGR * a).round().astype(np.uint8)


def scene_against_cushion(seed, cloth, side):
    """A random scene with the cue ball and one object ball against the cushion of ``side``."""
    scene = fg.random_scene(seed, "scatter", cloth)
    x, y, w, h = scene.table
    r = scene.radius
    nose = x + w - r - .5 if side == "right" else x + r + .5
    cue = (nose, y + h * .62)
    balls = {n: p for n, p in scene.balls.items()
             if np.hypot(p[0] - cue[0], p[1] - cue[1]) > 4 * r and abs(p[0] - nose) > 3 * r}
    balls[3] = (nose, y + h * .38)
    aim = 270.0  # along the cushion, as in the real frame
    return replace(scene, cue=cue, balls=balls, aim_degrees=aim)


def error(found, truth):
    return max(abs(a - b) for a, b in zip(found, truth))


def edges(table):
    x, y, w, h = table
    return {"left": x, "top": y, "right": x + w, "bottom": y + h}


def check_edges(case, found, truth, side, side_px=2.0, other_px=3.0):
    """``side`` within ``side_px`` of the truth. The others within ``other_px``:
    a border lying right on a cushion nose moves that edge 1-2.5 px inwards
    (as before this fix)."""
    case.assertIsNotNone(found)
    got, want = edges(found), edges(truth)
    for name in want:
        limit = side_px if name == side else other_px
        case.assertLessEqual(abs(got[name] - want[name]), limit, f"{name}: {found} vs {truth}")


class LitPocketTests(unittest.TestCase):
    def test_both_corners_of_a_short_side_lit(self):
        for cloth in CLOTHS:
            for seed in SEEDS:
                for side in ("left", "right"):
                    frame, truth = fg.render(scene_against_cushion(seed, cloth, side))
                    holes = fg.pocket_holes(truth.table)
                    for corner in ((0, 3) if side == "left" else (2, 5)):
                        light_pocket(frame, *holes[corner])
                    found = detector.find_table_bounds(frame)
                    self.assertIsNotNone(found, f"{cloth} seed {seed} {side}")
                    self.assertLessEqual(error(found, truth.table), 2.0,
                                         f"{cloth} seed {seed} {side}: {found} vs {truth.table}")


class GuideBorderTests(unittest.TestCase):
    def test_border_inside_the_cushion_is_not_the_edge(self):
        """The border drawn ~2% of the table inside a cushion (a table fitted
        too small once) must not hold the next fit there."""
        for cloth in CLOTHS:
            for seed in SEEDS:
                side = "right" if seed % 2 else "left"
                frame, truth = fg.render(scene_against_cushion(seed, cloth, side))
                x, y, w, h = truth.table
                short = w * .022
                drawn = (x, y, w - short, h) if side == "right" else (x + short, y, w - short, h)
                draw_border(frame, drawn)
                with self.subTest(cloth=cloth, seed=seed, side=side):
                    check_edges(self, detector.find_table_bounds(frame), truth.table, side)

    def test_lit_pockets_and_border_together(self):
        for seed in SEEDS:
            frame, truth = fg.render(scene_against_cushion(seed, "grey", "right"))
            holes = fg.pocket_holes(truth.table)
            for corner in (2, 5):
                light_pocket(frame, *holes[corner])
            x, y, w, h = truth.table
            draw_border(frame, (x, y, w * .973, h))
            with self.subTest(seed=seed):
                check_edges(self, detector.find_table_bounds(frame), truth.table, "right")

    def test_cushion_step_and_thin_line(self):
        # Outside (cushion) to inside (cloth).
        nose = np.array([65, 65, 67, 71, 82, 100, 125, 147, 138, 99, 52, 24, 18, 21, 22, 22, 23], float)
        line = np.array([44, 44, 45, 46, 44, 45, 72, 107, 75, 48, 47, 46, 44, 44, 43, 43, 42], float)
        self.assertTrue(detector._edge_is_step(nose, 8, 11))
        self.assertFalse(detector._edge_is_step(line, 7, 8))


@unittest.skipUnless(REAL_LIT.exists(), "real screenshot not present")
class RealLitPocketTests(unittest.TestCase):
    def test_real_frame_with_lit_pockets_and_drawn_border(self):
        """Both right corner pockets lit red, the cue ball and the 3 against the
        right cushion, the guide's border drawn at x=1473 (40 px short)."""
        frame = cv2.imread(str(REAL_LIT))
        found = detector.find_table_bounds(frame)
        self.assertIsNotNone(found)
        x, y, w, h = found
        for got, want in zip((x, y, x + w, y + h), (77, 267, 1511, 985)):
            self.assertLessEqual(abs(got - want), 3, found)


if __name__ == "__main__":
    unittest.main()
