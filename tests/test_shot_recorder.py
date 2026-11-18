"""Shot recorder: logs real shots (power at release, ball tracks, where they
stopped) for fitting the game's power and friction. Checked on synthetic
shots streamed as the mirror would show them (``SimGame.shot_stream``)."""
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


def setUpModule():
    # These frames were chosen on the old flat look (number up, straight
    # band); balls turned as in the game are checked in test_turned_balls
    # and by the sweeps.
    fg.TURN_BALLS = False


def tearDownModule():
    fg.TURN_BALLS = True


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
        game = sg.SimGame(3)
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
        self.assertLess(np.median(after), .3)
        self.assertLess(np.median(after), np.median(before))
        self.assertLess(max(after), 1.0)

    def test_duplicates_and_pockets(self):
        balls = [(100, 100, 10, "ball"), (108, 102, 10, "ball"), (130, 100, 10, "ball")]
        self.assertEqual(len(sr.distinct_balls(balls)), 2)
        table = (0, 0, 1000, 500)
        self.assertTrue(sr.near_pocket(15, 20, table, 10))
        self.assertTrue(sr.near_pocket(500, 490, table, 10))
        self.assertFalse(sr.near_pocket(250, 250, table, 10))


class RecorderGameTests(unittest.TestCase):
    def run_game(self, seed, cloth, shots, fps=30.0, snap=2, hide=()):
        game = sg.SimGame(seed, cloth=cloth)
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


if __name__ == "__main__":
    unittest.main()
