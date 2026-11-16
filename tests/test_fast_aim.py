"""Fast aim reader: ``predict_shot`` with last frame's aim and ghost.

While aiming only the angle changes, a little per frame, so the ghost ring
is looked for near where it was instead of scanning every ray direction.
The reading must be the same as without the hint, and the hint must be
dropped when the ghost jumps (the aim moved onto another ball). Checked
against the exact truth: the hinted reading is never worse than the plain
one (it can be better: its search corridor is wider, so it does not clip a
ring that a slightly-off candidate ray would)."""
import os
import sys
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import detector
import sim_game as sg


def angle(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def sweep(seed, cloth, shot, step, count=25):
    """(frames, truths, detection) of an aiming sweep in a test game."""
    game = sg.SimGame(seed, cloth=cloth)
    while len(game.shots) < shot:
        game.shoot(game.plan_shot())
    plan = game.plan_shot()
    d = detector.detect(game.render_aim(plan)[0])
    half = count // 2
    rendered = [game.render_aim(plan, aim_degrees=plan.aim_degrees + k * step) for k in range(-half, half + 1)]
    return [f for f, _ in rendered], [t for _, t in rendered], d


class FastAimTests(unittest.TestCase):
    def read_both(self, frames, truths, d):
        x, y, w, h = d.table
        hint, out = None, []
        for frame, truth in zip(frames, truths):
            roi = frame[y:y + h, x:x + w]
            plain = detector.predict_shot(roi, d.table, d.balls, d.cue)
            fast = detector.predict_shot(roi, d.table, d.balls, d.cue, hint=hint)
            hint = fast["hint"]
            out.append((plain, fast, truth))
        return out

    def assert_as_good(self, pairs, label):
        for k, (plain, fast, truth) in enumerate(pairs):
            where = f"{label} frame {k}"
            self.assertEqual(plain["target"] is None, fast["target"] is None, where)
            if plain["target"] is not None:
                self.assertLess(np.hypot(plain["target"][0] - fast["target"][0],
                                         plain["target"][1] - fast["target"][1]), 3, where)
            self.assertEqual(plain["pot"], fast["pot"], where)
            self.assertEqual(plain["aim_dir"] is None, fast["aim_dir"] is None, where)
            if plain["aim_dir"] is not None:
                def err(v):
                    return abs((angle(v) - truth.aim_degrees + 180) % 360 - 180)
                self.assertLessEqual(err(fast["aim_dir"]), err(plain["aim_dir"]) + .05, where)

    def test_same_reading_with_the_hint(self):
        # Aims that hit a ball, turning slowly to fast.
        for seed, shot, step in ((1, 1, .3), (4, 1, 1.0), (5, 2, 2.0)):
            pairs = self.read_both(*sweep(seed, "blue", shot, step))
            self.assert_as_good(pairs, f"seed {seed} step {step}")
            self.assertGreater(sum(f["fast"] for _, f, _ in pairs), len(pairs) * .5)

    def test_hint_dropped_when_the_ghost_jumps_nearer(self):
        # Turning 2 deg per frame moves the aim onto a nearer ball: something
        # ring-like near the old ghost (score 0.45, half the real ring's) was
        # taken and kept for the rest of the sweep.
        self.assert_as_good(self.read_both(*sweep(3, "green", 1, 2.0)), "green")

    def test_close_ghost_with_a_stale_aim(self):
        # The ghost three cue-ball radii away: last frame's 2-deg-old ray put
        # its peak at the edge of a 4 px wide corridor (0.56 deg off).
        frames, truths, d = sweep(3, "blue", 4, 2.0, count=5)
        self.assert_as_good(self.read_both(frames, truths, d), "close ghost")

    def test_hinted_read_is_faster(self):
        frames, _, d = sweep(1, "blue", 1, .3, count=13)
        x, y, w, h = d.table
        rois = [f[y:y + h, x:x + w] for f in frames]
        hints = [None]
        for roi in rois[:-1]:
            hints.append(detector.predict_shot(roi, d.table, d.balls, d.cue, hint=hints[-1])["hint"])
        start = time.perf_counter()
        for roi in rois[1:]:
            detector.predict_shot(roi, d.table, d.balls, d.cue)
        plain = time.perf_counter() - start
        start = time.perf_counter()
        for roi, hint in zip(rois[1:], hints[1:]):
            detector.predict_shot(roi, d.table, d.balls, d.cue, hint=hint)
        fast = time.perf_counter() - start
        # Typically ~2.5x; generous for busy machines.
        self.assertLess(fast, plain * .8)

    def test_ray_reaches(self):
        white = np.zeros((100, 300), np.uint8)
        white[50, 20:150] = 255
        self.assertTrue(detector.ray_reaches(white, (10.0, 50.0, 8.0), (160.0, 50.0), 8.0))
        # A ring further on, with no ray leading to it.
        self.assertFalse(detector.ray_reaches(white, (10.0, 50.0, 8.0), (260.0, 50.0), 8.0))


if __name__ == "__main__":
    unittest.main()
