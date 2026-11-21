"""Cue ball against a cushion beside a corner pocket, aimed into it: the game
draws the ghost ring in the pocket's mouth, past the cushion line, and the
aim ray lies almost all on the cue ball. Before: the ring (more than half
outside the table crop) was not found, another ball was taken for the cue
ball and no aim was read (real frame resize-stick-top)."""
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import detector  # noqa: E402
import fake_game as fg  # noqa: E402


def pocket_ghost_scene(seed: int, cloth: str, corner: int):
    """Cue ball frozen on the long cushion 1.5-2.2 radii from a corner
    (0 top right, 1 top left, 2 bottom right, 3 bottom left), aimed 3-10
    degrees into the table at the end cushion; the ghost's centre -1 to 3 px
    past the end cushion's line, as the real game shows it."""
    base = fg.random_scene(seed, "scatter", cloth)
    rng = np.random.default_rng(seed * 7 + corner)
    x, y, w, h = base.table
    r = base.radius
    sx = 1 if corner in (0, 2) else -1
    top = corner in (0, 1)
    k = rng.uniform(1.5, 2.2)
    cx = x + w - k * r if sx > 0 else x + k * r
    cy = y + r if top else y + h - r
    a = np.radians(rng.uniform(3, 10))
    ux, uy = sx * np.cos(a), (np.sin(a) if top else -np.sin(a))
    e = rng.uniform(-1, 3)
    edge = x + w + e if sx > 0 else x - e
    t = (edge - cx) / ux
    ghost = (cx + ux * t, cy + uy * t)
    balls = {n: p for n, p in base.balls.items()
             if np.hypot(p[0] - cx, p[1] - cy) > 3.5 * r and np.hypot(p[0] - ghost[0], p[1] - ghost[1]) > 3.5 * r}
    return replace(base, cue=(float(cx), float(cy)), aim_degrees=float(np.degrees(np.arctan2(uy, ux))),
                   ghost_at=(float(ghost[0]), float(ghost[1])), balls=balls)


class PocketGhost(unittest.TestCase):
    def test_ring_past_the_crop_edge(self):
        """The ring in the pocket's mouth is found on the table with a margin
        round it, not on the table crop alone."""
        scene = pocket_ghost_scene(50001, "green", 0)
        frame, truth = fg.render(scene)
        x, y, w, h = (int(round(v)) for v in truth.table)
        roi = frame[y:y + h, x:x + w]
        gx, gy = truth.ghost[0] - x, truth.ghost[1] - y
        around = (gx, gy, truth.radius * .5)
        self.assertIsNone(detector.find_ring_in_view(detector.ring_view(None, (x, y, w, h), truth.radius, roi),
                                                     truth.radius, around=around))
        ring = detector.find_ring_in_view(detector.ring_view(frame, (x, y, w, h), truth.radius, roi),
                                          truth.radius, around=around)
        self.assertIsNotNone(ring)
        self.assertLess(np.hypot(ring[0] - gx, ring[1] - gy), .5)

    def test_cue_ball_ghost_and_aim(self):
        """All four corners, 6 cloths (main: cue ball 61/80, ghost 6/80 on
        the 80-scene version of this)."""
        cue = ghost = aimed = total = 0
        for i, cloth in enumerate(["blue", "green", "red", "purple", "beige", "navy"]):
            for corner in range(4):
                frame, truth = fg.render(pocket_ghost_scene(50000 + i, cloth, corner))
                d = detector.detect(frame)
                total += 1
                cue_ok = d.cue is not None and np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]) <= 3
                cue += cue_ok
                ghost += d.ghost is not None and np.hypot(d.ghost[0] - truth.ghost[0],
                                                          d.ghost[1] - truth.ghost[1]) <= 2.5
                if cue_ok and d.aim_dir is not None:
                    err = abs((np.degrees(np.arctan2(d.aim_dir[1], d.aim_dir[0])) - truth.aim_degrees + 180) % 360 - 180)
                    aimed += err <= 10
        self.assertGreaterEqual(cue, total - 3)
        self.assertGreaterEqual(ghost, total - 2)
        self.assertGreaterEqual(aimed, total - 3)

    def test_stick_along_the_cushion_not_misread(self):
        """With the stick lying along the cushion behind the cue ball it is
        not read, rather than read off the cushion's straight edge (without
        the check: 3 of these 40 scenes more than a degree off)."""
        for i in range(10):
            for corner in range(4):
                frame, truth = fg.render(pocket_ghost_scene(50000 + i, ["blue", "green", "red", "purple", "beige",
                                                                         "navy"][i % 6], corner))
                x, y, w, h = (int(round(v)) for v in truth.table)
                stick = detector.stick_direction_near(frame[y:y + h, x:x + w], (truth.cue[0] - x, truth.cue[1] - y),
                                                      (truth.ghost[0] - x, truth.ghost[1] - y), truth.radius)
                if stick is not None:
                    err = abs((np.degrees(np.arctan2(stick[1], stick[0])) - truth.aim_degrees + 180) % 360 - 180)
                    self.assertLessEqual(err, 1.0, (i, corner))


REAL = ROOT / "outputs" / "bad" / "resize-stick-top" / "window.png"


@unittest.skipUnless(REAL.is_file(), "real frame not here (outputs/ is not in git)")
class RealFrame(unittest.TestCase):
    def test_cue_ball_beside_the_pocket(self):
        """Cue ball against the top cushion beside the right corner pocket,
        ring in the pocket's mouth: the cue ball (centre from its outline
        about (1273.2, 232.7)) and an aim along the ray (cue ball to ring:
        7.2 degrees; the stick, on the cushion, read 1.8 before)."""
        d = detector.detect(cv2.imread(str(REAL)))
        self.assertLess(np.hypot(d.cue[0] - 1273.2, d.cue[1] - 232.7), 1.5)
        self.assertIsNotNone(d.ghost)
        self.assertLess(np.hypot(d.ghost[0] - 1298.3, d.ghost[1] - 235.2), 1.5)
        self.assertLess(abs(np.degrees(np.arctan2(d.aim_dir[1], d.aim_dir[0])) - 7.2), 1.5)


if __name__ == "__main__":
    unittest.main()
