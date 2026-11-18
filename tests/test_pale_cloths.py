"""Grey and pale tinted cloths.

On grey cloth the table's watermark lettering is the cloth in one darker
shade, and the bowls of its letters were taken for balls. On a light tinted
cloth (beige, sky) the cue ball's shaded rim is as light as the cloth and its
drop shadow loses colour like the white ball, so the usual fit slid towards
the shadow; the rim is read by the cloth's tint on the lit side instead.
Synthetic seeds 40000+; each failed before."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


class GreyClothTests(unittest.TestCase):
    def test_lettering_is_not_a_ball(self):
        # The bowl of a "P", "R" or "C" was a ball (on its stroke, the ring
        # round it made the cloth inside look foreign).
        for seed, kind in ((40005, "scatter"), (40011, "scatter"), (40015, "scatter"), (40019, "pot")):
            *_, problems, _ = st.evaluate(seed, kind, "grey")
            self.assertFalse([p for p in problems if "false" in p], f"{seed} {kind}")

    def test_black_cloth_keeps_its_rule(self):
        # The near-black Lucky Shot cloth has no lettering shade to go by
        # (with the grey rule the 8 ball there read as lettering).
        *_, problems, _ = st.evaluate(40015, "rack", "black")
        self.assertFalse([p for p in problems if p.startswith("balls")])


class TintedClothCueTests(unittest.TestCase):
    def test_cue_ball_on_sky_and_beige(self):
        # 3-4.4 px towards the shadow before.
        for seed, kind, cloth in ((40003, "pot", "sky"), (40011, "scatter", "sky"), (40014, "scatter", "sky"),
                                  (40020, "scatter", "sky"), (40011, "miss", "beige"), (40014, "scatter", "beige"),
                                  (40020, "scatter", "beige")):
            frame, truth = fg.render(fg.random_scene(seed, kind, cloth))
            d = detector.detect(frame)
            self.assertLess(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]), 1.5, f"{seed} {kind} {cloth}")

    def test_only_on_light_tinted_or_light_grey_cloth(self):
        # A darker cloth, or a white one (its dip fit), keeps its own fits.
        for cloth in ("blue", "white", "grey", "pink"):
            frame, truth = fg.render(fg.random_scene(40001, "scatter", cloth))
            x, y, w, h = detector.find_table_bounds(frame)
            d = detector.detect(frame)
            self.assertIsNone(detector.pale_cloth_cue_centre(frame[y:y + h, x:x + w],
                                                             (d.cue[0] - x, d.cue[1] - y), d.balls, (x, y)), cloth)

    def test_cue_ball_on_light_grey(self):
        # 3-5 px towards the shadow before: no tint, and no rim dip either.
        for seed, kind in ((40020, "scatter"), (40014, "pot"), (40020, "miss"), (40007, "cushion"), (40019, "cushion")):
            frame, truth = fg.render(fg.random_scene(seed, kind, "light grey"))
            d = detector.detect(frame)
            self.assertLess(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]), 1.5, f"{seed} {kind}")


class StickRingTests(unittest.TestCase):
    def test_ring_filtered_along_the_aim_read_from_the_ghost(self):
        # The first rough aim (0.5 degree steps) put a metal joint 17 radii
        # back 2.4 px off the stick's axis: it stayed as a ball.
        *_, problems, _ = st.evaluate(40025, "miss", "beige")
        self.assertFalse([p for p in problems if "false" in p])

    def test_ball_under_the_stick_stays(self):
        # The green 6 lies under the stick on green cloth, grey under one of
        # its joints: re-filtered along the exact aim it read as a joint.
        for kind in ("scatter", "pot"):
            *_, problems, _ = st.evaluate(40022, kind, "dark green")
            self.assertFalse([p for p in problems if p.startswith("balls")], kind)


if __name__ == "__main__":
    unittest.main()
