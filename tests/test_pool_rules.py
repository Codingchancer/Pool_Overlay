import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import main
from pool_physics import Ball, simulate_shot
from pool_rules import SOLIDS, STRIPES, EightBallGame, ShotFacts

APP = QApplication.instance() or QApplication([])


def shot(first, pocketed=(), rail=True):
    return ShotFacts(first, tuple(pocketed), rail)


def after_break(potted=()):
    """A game just after a clean break by Player 1 that potted nothing (or ``potted``)."""
    game = EightBallGame()
    game.apply(shot(1, potted))
    return game


class EightBallRuleTests(unittest.TestCase):
    def test_break_starts_with_ball_in_hand_behind_the_line(self):
        game = EightBallGame()
        self.assertEqual(game.turn, 0)
        self.assertTrue(game.ball_in_hand and game.kitchen_only)
        self.assertIn("break", game.status())

    def test_dry_break_passes_the_turn(self):
        game = after_break()
        self.assertEqual(game.turn, 1)
        self.assertFalse(game.ball_in_hand)

    def test_potting_on_the_break_keeps_the_turn_and_the_table_open(self):
        game = after_break([3, 11])
        self.assertEqual(game.turn, 0)
        self.assertEqual(game.groups, [None, None])
        self.assertNotIn(3, game.on_table)

    def test_eight_on_the_break_is_spotted(self):
        game = EightBallGame()
        outcome = game.apply(shot(1, [8]))
        self.assertTrue(outcome.respot_eight)
        self.assertIn(8, game.on_table)
        self.assertIsNone(game.winner)

    def test_scratch_on_the_break_gives_ball_in_hand(self):
        game = EightBallGame()
        outcome = game.apply(shot(1, [0]))
        self.assertIn("cue ball potted", outcome.fouls)
        self.assertEqual(game.turn, 1)
        self.assertTrue(game.ball_in_hand)
        self.assertFalse(game.kitchen_only)

    def test_first_legal_pot_decides_the_groups(self):
        game = after_break()
        game.apply(shot(12, [12]))
        self.assertEqual(game.groups[1], STRIPES)
        self.assertEqual(game.groups[0], SOLIDS)
        self.assertEqual(game.turn, 1)
        self.assertIn("stripes", game.status())

    def test_first_pot_order_decides_when_both_groups_drop(self):
        game = after_break()
        game.apply(shot(4, [4, 13]))
        self.assertEqual(game.groups[1], SOLIDS)

    def test_hitting_the_eight_first_on_an_open_table_is_a_foul(self):
        game = after_break()
        outcome = game.apply(shot(8, [2]))
        self.assertTrue(outcome.fouls)
        self.assertEqual(game.groups, [None, None])
        self.assertEqual(game.turn, 0)
        self.assertTrue(game.ball_in_hand)

    def test_wrong_group_first_is_a_foul(self):
        game = after_break()
        game.apply(shot(12, [12]))          # Player 2: stripes, keeps the turn
        outcome = game.apply(shot(3, [11]))  # hits a solid first
        self.assertIn("hit the 3 first", outcome.fouls)
        self.assertEqual(game.turn, 0)
        self.assertTrue(game.ball_in_hand)

    def test_no_rail_after_contact_is_a_foul(self):
        game = after_break()
        outcome = game.apply(shot(5, [], rail=False))
        self.assertIn("no ball reached a cushion", outcome.fouls)

    def test_missing_everything_is_a_foul(self):
        game = after_break()
        outcome = game.apply(shot(None))
        self.assertIn("no ball hit", outcome.fouls)

    def test_potting_an_opponent_ball_only_passes_the_turn(self):
        game = after_break()
        game.apply(shot(12, [12]))          # Player 2 stripes
        outcome = game.apply(shot(10, [2]))  # legal hit, potted a solid
        self.assertFalse(outcome.fouls)
        self.assertFalse(outcome.keeps_turn)
        self.assertEqual(game.turn, 0)
        self.assertNotIn(2, game.on_table)

    def test_potting_the_eight_after_clearing_wins(self):
        game = after_break()
        game.apply(shot(12, [12]))
        game.on_table -= STRIPES
        self.assertEqual(game.targets(1), {8})
        self.assertIn("on the 8", game.status())
        outcome = game.apply(shot(8, [8]))
        self.assertEqual(game.winner, 1)
        self.assertIn("wins", outcome.message)

    def test_potting_the_eight_early_loses(self):
        game = after_break()
        game.apply(shot(12, [12]))
        game.apply(shot(10, [10, 8]))
        self.assertEqual(game.winner, 0)

    def test_potting_the_eight_with_a_scratch_loses(self):
        game = after_break()
        game.apply(shot(12, [12]))
        game.on_table -= STRIPES
        game.apply(shot(8, [8, 0]))
        self.assertEqual(game.winner, 0)

    def test_no_shots_after_the_game_is_over(self):
        game = after_break()
        game.apply(shot(12, [12]))
        game.apply(shot(10, [8]))
        with self.assertRaises(ValueError):
            game.apply(shot(1))


class ShotFactTests(unittest.TestCase):
    def test_physics_reports_first_contact_rail_and_pocket_order(self):
        result = simulate_shot([Ball(0, 100, 250, 12), Ball(5, 300, 250, 12)], (0, 0, 1000, 500), 0, 1.0)
        self.assertEqual(result.first_contact, 5)
        self.assertTrue(result.rail_after_contact)
        scratch = simulate_shot([Ball(0, 250, 100, 12)], (0, 0, 500, 500), -90, 1.0)
        self.assertIsNone(scratch.first_contact)
        self.assertEqual(scratch.pocket_order, (0,))


class SimulatorGameTests(unittest.TestCase):
    def setUp(self):
        self.window = main.ShotSimulator()
        self.window.rules_box.setChecked(True)
        self.canvas = self.window.canvas

    def tearDown(self):
        self.canvas.timer.stop()
        self.window.close()

    def play_out(self):
        self.window.shoot()
        self.canvas._jump_to_end()

    def test_game_mode_shows_whose_turn_it_is(self):
        self.assertTrue(self.window.game_text.isVisibleTo(self.window))
        self.assertIn("Player 1 to break", self.window.game_text.text())

    def test_a_break_shot_moves_on_to_the_next_turn(self):
        self.play_out()
        game = self.canvas.game
        self.assertFalse(game.break_shot)
        self.assertIsNotNone(game.last)
        self.assertIn(game.last.message.split(".")[0], self.window.game_text.text())
        self.assertIsNone(self.canvas.result, "aiming preview should be back after the shot")
        self.assertEqual(game.on_table, set(range(1, 16)) - self.canvas.pocketed)

    def test_only_the_cue_ball_can_be_moved_and_only_with_ball_in_hand(self):
        game = self.canvas.game
        game.ball_in_hand = False
        self.canvas.dragging = None
        ball = self.canvas.balls[0]
        before = (ball[0], ball[1])

        class Event:
            def __init__(self, x, y):
                self._p = type("P", (), {"x": lambda s: x, "y": lambda s: y})()

            def position(self):
                return self._p

        self.canvas.mousePressEvent(Event(ball[0], ball[1]))
        self.assertIsNone(self.canvas.dragging)
        self.canvas.mousePressEvent(Event(self.canvas.cue[0], self.canvas.cue[1]))
        self.assertIsNone(self.canvas.dragging)
        game.ball_in_hand = True
        game.kitchen_only = True
        self.canvas.mousePressEvent(Event(self.canvas.cue[0], self.canvas.cue[1]))
        self.assertEqual(self.canvas.dragging, "cue")
        self.canvas.mouseMoveEvent(Event(self.canvas.head_string_x() + 200, self.canvas.cue[1]))
        self.assertLessEqual(self.canvas.cue[0], self.canvas.head_string_x())
        self.assertEqual((ball[0], ball[1]), before)

    def test_a_scratch_puts_the_cue_ball_back_on_the_table(self):
        left, top, width, height = self.canvas.table()
        self.canvas.cue[0], self.canvas.cue[1] = left + width / 2, top + 40
        self.window.slider.setValue(-90)
        self.window.power_slider.setValue(100)
        self.play_out()
        self.assertIn("cue ball potted", self.canvas.game.last.fouls)
        self.assertTrue(left < self.canvas.cue[0] < left + width and top < self.canvas.cue[1] < top + height)
        self.assertTrue(self.canvas.game.ball_in_hand)

    def test_turning_the_game_off_returns_to_free_practice(self):
        self.window.rules_box.setChecked(False)
        self.assertIsNone(self.canvas.game)
        self.assertFalse(self.window.game_text.isVisibleTo(self.window))


if __name__ == "__main__":
    unittest.main()
