"""Shot recorder: logs real shots (power at release, ball tracks, where they
stopped) for fitting the game's power and friction. Checked on synthetic
shots streamed as the mirror would show them (``SimGame.shot_stream``), the
balls turning as they roll as the game's do."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import detector
import fake_game as fg
import recorder_check as rc
import shot_recorder as sr
import sim_game as sg


class ReleasePowerTests(unittest.TestCase):
    def test_held_then_snapped_back(self):
        history = [(0.0, 10.0), (.1, 30.0), (.2, 55.0), (.3, 55.0), (.4, 55.0), (.45, 0.0)]
        self.assertEqual(sr.release_power(history, .5), 55.0)

    def test_snap_back_over_a_few_frames(self):
        history = [(0.0, 60.0), (.1, 60.0), (.133, 40.0), (.167, 20.0), (.2, 0.0)]
        self.assertEqual(sr.release_power(history, .25), 60.0)

    def test_released_while_still_pulling(self):
        history = [(0.0, 20.0), (.033, 24.0), (.067, 28.0), (.1, 0.0)]
        self.assertEqual(sr.release_power(history, .12), 28.0)

    def test_never_pulled_or_never_seen(self):
        self.assertIsNone(sr.release_power([(0.0, 0.0), (.1, 0.0)], .2))
        self.assertIsNone(sr.release_power([(0.0, None), (.1, None)], .2))


class HelperTests(unittest.TestCase):
    def test_rim_fit_on_synthetic_balls(self):
        # Turned balls: a stripe's band or a number badge at the rim makes an
        # edge just inside it, so the outermost strong edge is fitted where
        # that lies better on a circle than the strongest.
        for turned, typical in ((False, .3), (True, .35)):
            game = sg.SimGame(3, turned=turned)
            game.shoot(game.plan_shot())
            frame, _ = game.render_motion(game.cue, game.balls)
            d = detector.detect(frame)
            truth = {0: game.cue, **game.balls}
            before, after = [], []
            for bx, by, br, _ in d.balls:
                n = min(truth, key=lambda k: np.hypot(truth[k][0] - bx, truth[k][1] - by))
                if np.hypot(truth[n][0] - bx, truth[n][1] - by) > br:
                    continue
                rx, ry = sr.refine_ball_centre(frame, bx, by, br)
                before.append(np.hypot(truth[n][0] - bx, truth[n][1] - by))
                after.append(np.hypot(truth[n][0] - rx, truth[n][1] - ry))
            self.assertLess(np.median(after), typical, turned)
            self.assertLess(np.median(after), np.median(before), turned)
            # (Turned: one ball in this frame is 2.3 px off with either edge.)
            self.assertLess(np.percentile(after, 90) if turned else max(after), 1.0, turned)

    def test_duplicates_and_pockets(self):
        balls = [(100, 100, 10, "ball"), (108, 102, 10, "ball"), (130, 100, 10, "ball")]
        self.assertEqual(len(sr.distinct_balls(balls)), 2)
        table = (0, 0, 1000, 500)
        self.assertTrue(sr.near_pocket(15, 20, table, 10))
        self.assertTrue(sr.near_pocket(500, 490, table, 10))
        self.assertFalse(sr.near_pocket(250, 250, table, 10))


class RecorderGameTests(unittest.TestCase):
    def run_game(self, seed, cloth, shots, fps=30.0, snap=2, hide=(), turned=None):
        game = sg.SimGame(seed, cloth=cloth, turned=turned)
        rec = sr.ShotRecorder(None)
        truth_at, results, t = {}, [], 0.0
        for k in range(shots):
            plan = game.plan_shot()
            start_cue, start_balls = game.cue, dict(game.balls)
            slider = k not in hide
            logs, roll = [], []
            for ft, frame, phase in game.shot_stream(plan, fps=fps, t0=t, snap_frames=snap, slider=slider):
                t = ft + 1.0 / fps
                if phase == "roll":
                    roll.append(round(ft, 6))
                log = rec.feed(frame, ft)
                if log is not None:
                    logs.append(log)
            record = game.shots[-1]
            for i, rt in enumerate(roll):
                truth_at[rt] = record.positions[min(int(i * 60.0 / fps), len(record.positions) - 1)]
            results.append(rc.score_shot(plan, record, start_cue, start_balls, logs, truth_at, game, slider, fps))
            truth_at.clear()
        return results, rec

    def test_a_game_is_logged_against_the_truth(self):
        # Break, pots, a scratch: power, tracks, potted balls, end spots.
        results, _ = self.run_game(1, "blue", 4)
        for k, r in enumerate(results):
            self.assertEqual(r["problems"], [], f"shot {k}")
            self.assertLess(r.get("cue_move_max", 0), rc.MOVE_TOL)

    def test_other_cloths(self):
        # Everything the recorder measures itself. (Where the balls it does
        # not follow stopped comes from the detector, which on a light cloth
        # can be a few pixels off.)
        for seed, cloth in ((3, "red"), (2, "light grey")):
            results, _ = self.run_game(seed, cloth, 2)
            for k, r in enumerate(results):
                own = [p for p in r["problems"] if "end spot off" not in p]
                self.assertEqual(own, [], f"{cloth} shot {k}")

    def test_flat_balls_too(self):
        # The old look (number up, straight band, no turning).
        results, _ = self.run_game(1, "blue", 3, turned=False)
        for k, r in enumerate(results):
            self.assertEqual(r["problems"], [], f"shot {k}")

    def test_hidden_slider_logs_no_power(self):
        # The opponent's turn: no slider. Logged with power null and a note;
        # the next shot (slider back) reads its power again.
        results, _ = self.run_game(1, "blue", 3, hide=(1,))
        self.assertIsNone(results[1]["power"])
        self.assertEqual(results[1]["problems"], [])
        self.assertEqual(results[2]["problems"], [])

    def test_ball_in_hand_is_not_a_shot(self):
        game = sg.SimGame(1)
        game.shoot(game.plan_shot())
        rec = sr.ShotRecorder(None)
        cue = game.cue
        x0, y0, w, h = game.table
        target = (x0 + w * .3, y0 + h * .5)
        t = 0.0
        frames = [game.render_motion(cue, game.balls)[0]] * 20
        for k in range(1, 31):
            p = (cue[0] + (target[0] - cue[0]) * k / 30, cue[1] + (target[1] - cue[1]) * k / 30)
            frames.append(fg.render(game.scene(cue=p, power=0.0))[0])
        frames += [fg.render(game.scene(cue=target, power=0.0))[0]] * 30
        logs = []
        for frame in frames:
            log = rec.feed(frame, t)
            t += 1 / 30
            if log is not None:
                logs.append(log)
        self.assertEqual(logs, [])
        self.assertTrue(any("ball in hand" in n for n in rec.notes))


class BreakSliderTests(unittest.TestCase):
    def test_full_power_with_the_stick_across_the_slider(self):
        # On a break the stick lies across the slider's top: it cut the frame
        # into two pieces that each failed the size test (read None from 80%).
        game = sg.SimGame(2)
        plan = game.plan_shot()
        readings = [detector.detect(f).power_percent for _, f, phase in game.shot_stream(plan, fps=10)
                    if phase == "hold"]
        self.assertTrue(readings)
        for value in readings:
            self.assertIsNotNone(value)
            self.assertGreaterEqual(value, 99)


def rolling_path(cloth, seed=40003, steps=12):
    """A coloured ball rolled along a free straight path on a synthetic table
    (turning as it rolls), rendered as the mirror shows it, and the table
    without it: (frames, truth positions, empty table, radius, others)."""
    from dataclasses import replace
    rng = np.random.default_rng(seed)
    base = replace(fg.random_scene(seed, "scatter", cloth), show_guide=False, show_stick=False)
    r = base.radius
    x, y, w, h = base.table
    for n in [k for k in base.balls if k != 8]:
        others = {k: v for k, v in base.balls.items() if k != n}
        x0, y0 = base.balls[n]
        for _ in range(20):
            heading = rng.random() * 2 * np.pi
            path = [(x0 + np.cos(heading) * s * .4 * r, y0 + np.sin(heading) * s * .4 * r) for s in range(steps)]
            if all(x + 1.5 * r < px < x + w - 1.5 * r and y + 1.5 * r < py < y + h - 1.5 * r
                   and min(np.hypot(px - ox, py - oy) for ox, oy in others.values()) > 2.6 * r for px, py in path):
                break
        else:
            continue
        turn = fg.random_turn(rng)
        frames = [fg.render(replace(base, balls={**others, n: p},
                                    turns={n: tuple(fg.roll(turn, p[0] - x0, p[1] - y0, r).ravel())}))[0]
                  for p in path]
        return frames, path, fg.render(replace(base, balls=others))[0], r, list(others.values())
    raise AssertionError("no free path")


class RollingBallTests(unittest.TestCase):
    def test_a_rolling_ball_is_followed_whatever_it_shows(self):
        # Its number and stripe turn as it rolls; the disc on the cloth and
        # its rim do not. (A picture of the ball lost a stripe within half a
        # radius of rolling.)
        for cloth in ("blue", "red", "white"):
            frames, path, empty, r, others = rolling_path(cloth)
            follower = sr.RollingFollower(frames[0], *path[0], r, empty)
            follower.others = others
            follower.calibrate(frames[0], *path[0])
            self.assertTrue(follower.usable, cloth)
            errors = []
            for frame, (px, py) in zip(frames, path):
                got = follower.locate(frame, 0.0, reach=r, centre=(px + 2, py - 1))
                self.assertIsNotNone(got, cloth)
                self.assertLessEqual(got[2], 1.0, cloth)
                errors.append(np.hypot(got[0] - px, got[1] - py))
            self.assertLess(np.median(errors), .35, cloth)
            self.assertLess(max(errors), 1.2, cloth)

    def test_the_cloth_picture_keeps_the_table_and_drops_the_balls(self):
        game = sg.SimGame(1)
        game.shoot(game.plan_shot())
        plan = game.plan_shot()
        aim, _ = game.render_aim(plan)
        d = detector.detect(aim)
        r = float(np.median([b[2] for b in d.balls]))
        cloth = sr.cloth_picture(aim, d.table, [b[:2] for b in d.balls], r, sr.game_graphics(d, r))
        empty = fg.render(game.scene(balls={}, cue=(-100.0, -100.0)))[0]
        x, y, w, h = d.table
        diff = np.abs(cloth.astype(int) - empty.astype(int)).max(axis=2)[y + 2:y + h - 2, x + 2:x + w - 2]
        # Balls, the game's lines, ghost ring and stick gone: the cloth within
        # a few levels nearly everywhere.
        self.assertGreater(float(np.mean(diff <= 6)), .99)


class TellingBallsApartTests(unittest.TestCase):
    def test_solids_and_stripes_however_turned(self):
        wrong, unsure, n = 0, 0, 0
        for cloth in ("blue", "red", "white", "light grey"):
            for seed in range(40000, 40004):
                frame, truth = fg.render(fg.random_scene(seed, "scatter", cloth))
                r = truth.radius
                for k, (x, y) in truth.balls.items():
                    if k == 8 or min(np.hypot(x - ox, y - oy) for m, (ox, oy) in truth.balls.items() if m != k) < 2.2 * r:
                        continue
                    stripe = sr.ball_signature(frame, x, y, r)["stripe"]
                    n += 1
                    unsure += stripe is None
                    wrong += stripe is not None and stripe != (k >= 9)
        self.assertEqual(wrong, 0)
        # Unsure (about a sixth): matched by colour and position only.
        self.assertLess(unsure, .25 * n)

    def test_matching_after_the_shot(self):
        def sig(x, lab, white, stripe):
            return {"x": x, "y": 100.0, "lab": lab, "white": white, "stripe": stripe}
        yellow = [183.0, 134.0, 196.0]
        # A solid never takes a stripe's place, and an unsure ball (in a
        # rack) does not take a sure one's.
        before = [sig(100.0, yellow, .18, None), sig(130.0, yellow, .05, False)]
        after = [sig(400.0, yellow, .0, False), sig(300.0, yellow, .5, True)]
        self.assertEqual(sr.match_balls(before, after, 12.0), [1, 0])
        # A red ball does not become a maroon one (it was potted).
        before = [sig(100.0, [107.0, 179.0, 158.0], 0.0, False)]
        after = [sig(110.0, [78.0, 156.0, 148.0], 0.0, False)]
        self.assertEqual(sr.match_balls(before, after, 12.0), [None])
        # A stripe turned end-on shows little of its colour: no colour bound.
        before = [sig(100.0, [79.5, 147.0, 111.5], .8, True)]
        after = [sig(400.0, [100.0, 162.0, 92.0], .35, True)]
        self.assertEqual(sr.match_balls(before, after, 12.0), [0])


if __name__ == "__main__":
    unittest.main()
