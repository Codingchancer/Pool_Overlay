"""Guide style settings: colours, thickness, bounce, saving, and that any style
is still removed from captures before analysis (no feedback into detection)."""
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import fake_game
from capture import (DEFAULT_STYLE, GUIDE_ALPHA, GUIDE_COLOURS, GUIDE_THICKNESS, GuideStyle, composite_overlay,
                     load_guide_style, remove_overlay, render_guide_layer, save_guide_style)
from detector import Detection, detect
from overlay import GuideSettings

APP = QApplication.instance() or QApplication([])
THICKEST = GuideStyle("orange", "green", "red", max(GUIDE_THICKNESS.values()), True)


def angle(segment):
    (x1, y1), (x2, y2) = segment
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


class StyleStorageTests(unittest.TestCase):
    def settings(self):
        folder = tempfile.mkdtemp()
        return QSettings(os.path.join(folder, "guide.ini"), QSettings.Format.IniFormat)

    def test_defaults_when_nothing_is_saved(self):
        self.assertEqual(load_guide_style(self.settings()), DEFAULT_STYLE)

    def test_saved_style_comes_back(self):
        settings = self.settings()
        style = GuideStyle("green", "purple", "blue", 1.5, False)
        save_guide_style(settings, style)
        settings.sync()
        self.assertEqual(load_guide_style(settings), style)

    def test_broken_values_fall_back_to_defaults(self):
        settings = self.settings()
        settings.setValue("guide/aim_colour", "white")
        settings.setValue("guide/thickness", "huge")
        settings.setValue("guide/show_bounce", "false")
        style = load_guide_style(settings)
        self.assertEqual(style.aim_colour, DEFAULT_STYLE.aim_colour)
        self.assertEqual(style.thickness, DEFAULT_STYLE.thickness)
        self.assertFalse(style.show_bounce)

    def test_no_colour_looks_like_the_games_white_lines(self):
        for name, rgb in GUIDE_COLOURS.items():
            hsv = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2HSV)[0, 0]
            self.assertGreater(int(hsv[1]), 100, f"{name} is too pale")


class StyleDrawingTests(unittest.TestCase):
    detection = Detection(table=(20, 20, 560, 280), cue=(100, 150, 10), balls=[(100, 150, 10, "cue")],
                          aim_segments=[((100, 150), (300, 100))],
                          object_segments=[((320, 95), (560, 40)), ((560, 40), (500, 25))],
                          cue_segments=[((300, 100), (380, 300)), ((380, 300), (420, 290))],
                          target=(320, 95, 10))

    def layer(self, style):
        return render_guide_layer(self.detection, 600, 320, adjusting=False, active=True, style=style)

    def test_opacity_stays_at_the_fixed_limit_in_every_style(self):
        # A single stroke is never more opaque than GUIDE_ALPHA; only crossings add
        # up (the same 239 bound as the default style; remove_overlay inpaints those).
        layer = self.layer(THICKEST)
        self.assertLessEqual(int(layer[:, :, 3].max()), 239)
        self.assertLessEqual(int(np.percentile(layer[:, :, 3][layer[:, :, 3] > 0], 90)), GUIDE_ALPHA)

    def test_thicker_lines_cover_more_pixels(self):
        thin = (self.layer(GuideStyle(thickness=.75))[:, :, 3] > 0).sum()
        thick = (self.layer(GuideStyle(thickness=2.0))[:, :, 3] > 0).sum()
        # The table outline and pocket dots keep their size, so the gain is under 2x.
        self.assertGreater(thick, thin * 1.3)

    def test_hiding_the_bounce_drops_the_segments_after_the_cushion(self):
        with_bounce = self.layer(GuideStyle(show_bounce=True))
        without = self.layer(GuideStyle(show_bounce=False))
        # Around the end of the object ball's bounce segment only the full style draws.
        self.assertGreater(int(with_bounce[25:32, 495:510, 3].max()), 0)
        self.assertEqual(int(without[25:32, 495:510, 3].max()), 0)

    def test_chosen_colours_are_used(self):
        layer = self.layer(GuideStyle(aim_colour="green"))
        b, g, r, a = (int(v) for v in layer[125, 200])  # on the aim line
        self.assertGreater(a, 0)
        self.assertGreater(g, r)
        self.assertGreater(g, b)


class StyleFeedbackTests(unittest.TestCase):
    """Detect -> draw in the chosen style -> capture -> remove -> detect again."""

    def test_any_style_is_removed_before_analysis(self):
        for seed in (3, 11, 27):
            frame, _ = fake_game.render(fake_game.random_scene(seed))
            reference = detect(frame)
            self.assertIsNotNone(reference.ghost, f"seed {seed}: synthetic shot not detected")
            h, w = frame.shape[:2]
            for style in (DEFAULT_STYLE, THICKEST, GuideStyle(show_bounce=False, thickness=.75)):
                layer = render_guide_layer(reference, w, h, adjusting=False, active=True, style=style)
                captured = composite_overlay(frame, layer)
                cleaned = remove_overlay(captured, layer)
                self.assertLessEqual(np.percentile(np.abs(cleaned.astype(int) - frame), 99), 2)
                again = detect(cleaned)
                self.assertEqual(again.table, reference.table)
                self.assertLessEqual(np.hypot(again.cue[0] - reference.cue[0], again.cue[1] - reference.cue[1]), 2)
                self.assertLessEqual(np.hypot(again.ghost[0] - reference.ghost[0],
                                              again.ghost[1] - reference.ghost[1]), 1.5)
                self.assertLess(abs(angle(again.aim_segments[0]) - angle(reference.aim_segments[0])), .6)


class FakeOverlay:
    def __init__(self):
        self.guide_style = DEFAULT_STYLE
        self.applied = []

    def set_style(self, style):
        self.guide_style = style
        self.applied.append(style)


class GuideSettingsDialogTests(unittest.TestCase):
    def test_dialog_shows_current_style_and_applies_changes_at_once(self):
        overlay = FakeOverlay()
        overlay.guide_style = GuideStyle("blue", "yellow", "magenta", 1.5, False)
        changes = []
        dialog = GuideSettings(overlay, on_change=lambda: changes.append(1))
        try:
            self.assertEqual(dialog.current_style(), overlay.guide_style)
            self.assertEqual(overlay.applied, [], "opening the dialog must not change anything")
            dialog.colour_boxes["aim"].setCurrentIndex(dialog.colour_boxes["aim"].findData("red"))
            self.assertEqual(overlay.guide_style.aim_colour, "red")
            dialog.thickness.setCurrentIndex(dialog.thickness.findData(2.0))
            dialog.bounce.setChecked(True)
            self.assertEqual(overlay.guide_style, GuideStyle("red", "yellow", "magenta", 2.0, True))
            self.assertEqual(len(changes), 3)
            dialog.show_style(DEFAULT_STYLE, apply=True)
            self.assertEqual(overlay.guide_style, DEFAULT_STYLE)
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
