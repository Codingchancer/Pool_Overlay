"""Ball outlines where a drawn path meets a cushion: the paths follow the
ball's centre (one radius inside the cushion), the outline shows the ball
touching it. Drawing them must keep the capture-feedback loop working."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6.QtWidgets import QApplication

import fake_game as fg
import main
from capture import DEFAULT_STYLE, GuideStyle, cushion_contacts

APP = QApplication.instance() or QApplication([])


def setUpModule():
    # These frames were chosen on the old flat look (number up, straight
    # band); balls turned as in the game are checked in test_turned_balls
    # and by the sweeps.
    fg.TURN_BALLS = False


def tearDownModule():
    fg.TURN_BALLS = True


class CushionOutlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, _ = fg.render(fg.random_scene(9002))
        cls.detection = main.detect(cls.frame)
        cls.radius = cls.detection.balls[0][2]

    def layer(self, detection, style=DEFAULT_STYLE):
        h, w = self.frame.shape[:2]
        return main.render_guide_layer(detection, w, h, adjusting=False, active=True, style=style)

    def test_contacts_are_the_bounce_points_on_the_cushion_line(self):
        x, y, w, h = self.detection.table
        r = self.radius
        contacts = cushion_contacts(self.detection.object_segments, self.detection.table, r)
        self.assertTrue(contacts)
        for cx, cy in contacts:
            self.assertLessEqual(min(abs(cx - (x + r)), abs(cy - (y + r)), abs(cx - (x + w - r)),
                                     abs(cy - (y + h - r))), 1.5)
        self.assertEqual(cushion_contacts(None, self.detection.table, r), [])

    def test_outline_touches_the_cushion(self):
        from dataclasses import replace

        with_outline = self.layer(self.detection)
        segments = self.detection.object_segments
        cx, cy = cushion_contacts(segments, self.detection.table, self.radius)[0]
        x, y, w, h = self.detection.table
        # The outline's far edge reaches the cushion nose: a point just inside
        # the nose, straight out from the contact, is drawn.
        if abs(cx - (x + self.radius)) <= 1.5:
            px, py = x + 1, cy
        elif abs(cx - (x + w - self.radius)) <= 1.5:
            px, py = x + w - 1, cy
        elif abs(cy - (y + self.radius)) <= 1.5:
            px, py = cx, y + 1
        else:
            px, py = cx, y + h - 1
        without = self.layer(replace(self.detection, object_segments=None, cue_segments=None,
                                     aim_segments=None, knock_paths=None, deflect_paths=None))
        added = with_outline[:, :, 3].astype(int) - without[:, :, 3]
        self.assertTrue((added[int(py) - 2:int(py) + 3, int(px) - 2:int(px) + 3] > 0).any())

    def test_opacity_limit_and_removal(self):
        for style in (DEFAULT_STYLE, GuideStyle(thickness=2.0), GuideStyle(show_bounce=False)):
            layer = self.layer(self.detection, style)
            self.assertLessEqual(int(layer[:, :, 3].max()), 239)
            recovered = main.remove_overlay(main.composite_overlay(self.frame, layer), layer)
            error = np.abs(recovered.astype(int) - self.frame)
            self.assertLessEqual(np.percentile(error, 99), 2)
            # The thickest style already reaches 7 on this frame without the
            # outlines (overlapping main strokes); the outlines add nothing.
            self.assertLessEqual(int(error.max()), 8)

    def test_detect_draw_capture_loop_is_stable(self):
        layer = self.layer(self.detection)
        cleaned = main.remove_overlay(main.composite_overlay(self.frame, layer), layer)
        again = main.detect(cleaned)
        self.assertEqual(again.cue, self.detection.cue)
        self.assertEqual(again.object_segments, self.detection.object_segments)


if __name__ == "__main__":
    unittest.main()
