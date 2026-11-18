"""Balls turned as in the real game.

Miniclip's balls roll: a solid ball's one number badge is mostly at the
rim or round the back, a stripe is a belt round the ball. The synthetic
table draws them so (``fake_game.draw_ball`` with a turn), and the detector
no longer needs a ball the colour of the cloth (purple 4 on a purple table)
to show its number: it finds it by the highlight every ball shows at the
same spot (``highlight_offset``, ``cloth_coloured_ball``). Synthetic seeds
40000+; each frame failed before."""
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
import stress_test as st


def problems(seed, kind, cloth):
    return st.evaluate(seed, kind, cloth)[4]


def badge_in_middle(frame, x, y, r):
    """A white number badge covers the ball's middle (as on the flat look)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    middle = np.hypot(xx - x, yy - y) <= r * .25
    white = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 170)
    return float(white[middle].mean()) > .5


class TurnedLookTests(unittest.TestCase):
    def test_turned_by_default_flat_on_request(self):
        scene = fg.random_scene(40003, "scatter", "blue")
        self.assertTrue(scene.turned)
        flat = replace(scene, turned=False, show_guide=False, show_stick=False, blur=0.0, jpeg_quality=None)
        turned = replace(flat, turned=True)
        r = scene.radius
        frame_flat, truth = fg.render(flat)
        frame_turned, _ = fg.render(turned)
        solids = [n for n in truth.balls if n <= 7]
        flat_badges = sum(badge_in_middle(frame_flat, *truth.balls[n], r) for n in solids)
        turned_badges = sum(badge_in_middle(frame_turned, *truth.balls[n], r) for n in solids)
        # Flat: every solid shows its number in the middle; turned: few do.
        self.assertEqual(flat_badges, len(solids))
        self.assertLessEqual(turned_badges, len(solids) // 2)

    def test_a_ball_that_stays_keeps_its_look(self):
        scene = fg.random_scene(40003, "scatter", "blue")
        a = fg.ball_turn(scene, 4, *scene.balls[4])
        b = fg.ball_turn(scene, 4, *scene.balls[4])
        c = fg.ball_turn(scene, 4, scene.balls[4][0] + 3, scene.balls[4][1])
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.allclose(a, c))
        self.assertIsNone(fg.ball_turn(replace(scene, turned=False), 4, *scene.balls[4]))

    def test_rolling(self):
        turn = fg.random_turn(np.random.default_rng(5))
        self.assertTrue(np.allclose(turn @ turn.T, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(turn)), 1.0)
        r = 12.0
        # One full turn rolled (2 pi r) brings it back.
        rolled = turn
        for _ in range(100):
            rolled = fg.roll(rolled, 2 * np.pi * r / 100 * .6, 2 * np.pi * r / 100 * .8, r)
        self.assertTrue(np.allclose(rolled, turn, atol=1e-9))
        # Rolling right a quarter turn, the point facing the camera goes to
        # the leading side (camera: x right, y down, z into the table).
        quarter = fg.roll(np.eye(3), np.pi * r / 2, 0.0, r)
        self.assertTrue(np.allclose(quarter @ np.array([0.0, 0.0, -1.0]), [1.0, 0.0, 0.0], atol=1e-9))

    def test_sim_game_balls_roll(self):
        import sim_game as sg
        game = sg.SimGame(3, "blue")
        start = {n: game._turn_of(n, p) for n, p in [(0, game.cue), *game.balls.items()]}
        record = game.shoot(game.plan_shot())
        end = record.turns[-1]
        moved = [n for n, p in record.positions[-1][1].items() if np.hypot(p[0] - record.start_balls[n][0],
                                                                            p[1] - record.start_balls[n][1]) > 1]
        still = [n for n, p in record.positions[-1][1].items() if p == record.start_balls[n]]
        self.assertTrue(moved)
        for n in moved:
            self.assertFalse(np.allclose(np.reshape(end[n], (3, 3)), start[n]))
        for n in still:
            self.assertTrue(np.allclose(np.reshape(end[n], (3, 3)), start[n]))


class ClothColouredBallTests(unittest.TestCase):
    def test_balls_the_colour_of_the_cloth(self):
        # Missed before: the ball looks like cloth inside and its number
        # faces away.
        for seed, cloth, number in ((40001, "purple", 4), (40000, "red", 7), (40002, "maroon", 3),
                                    (40000, "dark green", 6), (40007, "orange", 5), (40000, "yellow", 1),
                                    (40007, "black", 8)):
            frame, truth = fg.render(fg.random_scene(seed, "scatter", cloth))
            d = detector.detect(frame)
            px, py = truth.balls[number]
            err = min(np.hypot(bx - px, by - py) for bx, by, _, _ in d.balls) / truth.radius
            self.assertLess(err, .3, f"{seed} {cloth} #{number}")
            self.assertEqual(problems(seed, "scatter", cloth), [], f"{seed} {cloth}")

    def test_striped_ball_turned_end_on_is_not_the_cue_ball(self):
        # White cloth: a stripe seen end-on (a white face ringed by its
        # colour) is as white as the cue ball and was taken for it when the
        # cue ball's own ray was not confirmed.
        for seed, kind in ((40001, "scatter"), (40010, "scatter"), (40001, "pot")):
            frame, truth = fg.render(fg.random_scene(seed, kind, "white"))
            d = detector.detect(frame)
            self.assertIsNotNone(d.cue, f"{seed} {kind}")
            self.assertLess(float(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1])), 3, f"{seed} {kind}")
        # A cue ball's rim is white all round; a stripe's is coloured along
        # a good part of it, however it is turned.
        scene = fg.random_scene(40003, "scatter", "blue")
        frame, truth = fg.render(scene)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        self.assertLess(detector.coloured_rim_share(lab, *truth.cue, truth.radius), detector.STRIPE_RIM)
        stripes = [n for n in truth.balls if n >= 9]
        shares = [detector.coloured_rim_share(lab, *truth.balls[n], truth.radius) for n in stripes]
        self.assertTrue(all(s >= detector.STRIPE_RIM for s in shares), shares)

    def test_every_cloth_with_turned_balls(self):
        # Whole frames (scatter, seeds 40000-40005) passing every check, at
        # most one fewer than now. Before the highlight test, with the same
        # turned balls: red, purple, orange and dark green 0, maroon 1,
        # yellow 3, white 4, black 2.
        minimum = {"blue": 5, "green": 5, "red": 4, "purple": 2, "brown": 5, "teal": 5, "white": 4, "light grey": 5,
                   "grey": 5, "beige": 5, "pink": 5, "yellow": 4, "orange": 4, "navy": 4, "lime": 5,
                   "dark green": 4, "maroon": 3, "sky": 5, "black": 1}
        for cloth, least in minimum.items():
            passed = sum(not problems(seed, "scatter", cloth) for seed in range(40000, 40006))
            self.assertGreaterEqual(passed, least, cloth)

    def test_highlight_found_where_the_balls_show_it(self):
        # The synthetic table puts the highlight 0.35 radii left of and 0.38
        # above the centre; it is read from the balls themselves.
        frame, truth = fg.render(fg.random_scene(40002, "scatter", "green"))
        lightness = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        smooth = cv2.GaussianBlur(lightness, (0, 0), max(1.0, truth.radius * .1))
        ox, oy = detector.highlight_offset(smooth, list(truth.balls.values()), truth.radius)
        self.assertLess(np.hypot(ox + .35, oy + .38), .1)
        # Too few balls: the measured default.
        self.assertEqual(detector.highlight_offset(smooth, list(truth.balls.values())[:2], truth.radius),
                         detector.HIGHLIGHT)

    def test_no_ball_on_the_ghost_ring_or_the_cloth(self):
        frame, truth = fg.render(fg.random_scene(40001, "rack", "purple"))
        r = truth.radius
        look = detector.BallLook(frame, [], list(truth.balls.values()), r)
        gx, gy = truth.ghost
        self.assertFalse(look.cloth_coloured_ball(gx, gy))
        x, y, w, h = truth.table
        empty = next((px, py) for px, py in ((x + w * a, y + h * b) for a in np.linspace(.1, .9, 9)
                                             for b in np.linspace(.15, .85, 5))
                     if min(np.hypot(px - bx, py - by) for bx, by in list(truth.balls.values()) + [truth.cue]) > 4 * r
                     and np.hypot(px - gx, py - gy) > 4 * r)
        self.assertFalse(look.cloth_coloured_ball(*empty))
        d = detector.detect(frame)
        self.assertFalse(any(np.hypot(bx - gx, by - gy) < 1.3 * r for bx, by, _, kind in d.balls if kind != "cue"))


if __name__ == "__main__":
    unittest.main()
