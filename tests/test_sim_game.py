"""The headless 8-ball test game (``sim_game``) and what it found.

The game racks, breaks, plans and shoots real shots through the physics and
the 8-ball rules, and renders every frame with exact truth, so the overlay
can be checked on whole games (``play_test.py``)."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import detector
import fake_game as fg
import sim_game as sg


class SimGameTests(unittest.TestCase):
    def test_a_game_plays_by_the_rules(self):
        game = sg.SimGame(1)
        self.assertEqual(len(game.balls), 15)
        shots = 0
        while not game.over and shots < 80:
            plan = game.plan_shot()
            if shots == 0:
                self.assertEqual(plan.kind, "break")
            record = game.shoot(plan)
            shots += 1
            self.assertTrue(record.message)
            # Balls never overlap and stay on the table.
            pts = list(game.balls.values()) + [game.cue]
            x0, y0, w, h = game.table
            for i, (ax, ay) in enumerate(pts):
                self.assertTrue(x0 - 1 <= ax <= x0 + w + 1 and y0 - 1 <= ay <= y0 + h + 1)
                for bx, by in pts[i + 1:]:
                    self.assertGreater(np.hypot(ax - bx, ay - by), game.r * 1.6)
        self.assertTrue(game.over, "game 1 is played to the end within 80 shots")

    def test_planned_pot_goes_in(self):
        # A shot at a pocket, planned by the game with no aim error: it drops.
        game = sg.SimGame(3, aim_error=0)
        x0, y0, w, h = game.table
        game.balls = {1: (x0 + w * .82, y0 + h * .7)}
        game.cue = (x0 + w * .62, y0 + h * .5)
        game.rules.break_shot = False
        game.rules.ball_in_hand = False
        plan = game.plan_shot()
        self.assertEqual(plan.kind, "pot")
        record = game.shoot(plan)
        self.assertIn(1, record.pocketed)

    def test_frames_match_the_physics(self):
        game = sg.SimGame(2)
        record = game.shoot(game.plan_shot(), every=3)
        self.assertGreater(len(record.positions), len(record.frames))
        self.assertEqual(record.frames[-1], record.positions[-1])
        frame, truth = game.render_motion(*record.frames[len(record.frames) // 2])
        self.assertEqual(frame.shape[:2], (game.base.height, game.base.width))
        self.assertIsNone(truth.ghost)


class SliderUnderStickTests(unittest.TestCase):
    def test_power_read_with_the_stick_across_the_slider(self):
        # Aiming from near the left cushion puts the stick across the slider;
        # the reading stopped at the stick (48% read as 74%).
        found = 0
        for seed in range(1, 7):
            game = sg.SimGame(seed)
            while len(game.shots) < 12 and not game.over:
                plan = game.plan_shot()
                if plan.kind != "break":
                    frame, _ = game.render_aim(plan)
                    d = detector.detect(frame)
                    if d.power_percent is not None:
                        found += 1
                        self.assertLessEqual(abs(d.power_percent - round(min(1, plan.power) * 100)), 3,
                                             f"seed {seed} shot {len(game.shots)}")
                game.shoot(plan)
        self.assertGreater(found, 40)


if __name__ == "__main__":
    unittest.main()
