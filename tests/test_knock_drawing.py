"""Drawing the paths through other balls (knock_paths / deflect_paths) keeps
the capture-feedback loop working: drawn within the opacity limit, removed
exactly before analysis, and the next detection is unchanged."""
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
from capture import DEFAULT_STYLE, GuideStyle

APP = QApplication.instance() or QApplication([])


def blocked_shot():
    """A synthetic frame whose predicted object path runs into another ball."""
    for kind in (None, "pot"):
        for seed in range(9000, 9200):
            frame, _ = fg.render(fg.random_scene(seed, kind))
            d = main.detect(frame)
            if d.knock_paths and d.knock_paths[0] and d.deflect_paths and d.deflect_paths[0]:
                return frame, d
    raise unittest.SkipTest("no synthetic shot with a blocked object path")


class KnockDrawingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.detection = blocked_shot()

    def layer(self, detection, style=DEFAULT_STYLE):
        h, w = self.frame.shape[:2]
        return main.render_guide_layer(detection, w, h, adjusting=False, active=True, style=style)

    def test_knock_and_deflection_paths_are_drawn(self):
        from dataclasses import replace

        with_paths = self.layer(self.detection)
        without = self.layer(replace(self.detection, knock_paths=None, deflect_paths=None))
        added = (with_paths[:, :, 3].astype(int) - without[:, :, 3]) > 0
        (x0, y0), (x1, y1) = self.detection.knock_paths[0][0]
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        self.assertTrue(added[max(0, my - 3):my + 4, max(0, mx - 3):mx + 4].any(), "knock path not drawn")
        self.assertGreater(int(added.sum()), 200)

    def test_opacity_limit_and_exact_removal(self):
        for style in (DEFAULT_STYLE, GuideStyle(thickness=2.0)):
            layer = self.layer(self.detection, style)
            self.assertLessEqual(int(layer[:, :, 3].max()), 239)
            recovered = main.remove_overlay(main.composite_overlay(self.frame, layer), layer)
            error = np.abs(recovered.astype(int) - self.frame)
            # Same standard as the guide-style tests (test_guide_settings).
            self.assertLessEqual(np.percentile(error, 99), 2)
            self.assertLessEqual(int(error.max()), 6)

    def test_detect_draw_capture_loop_is_stable(self):
        layer = self.layer(self.detection)
        cleaned = main.remove_overlay(main.composite_overlay(self.frame, layer), layer)
        again = main.detect(cleaned)
        self.assertEqual(again.cue, self.detection.cue)
        # (The target's Hough radius can differ by a pixel after any guide
        # cycle, with or without these lines; its centre must not move.)
        self.assertEqual(again.target[:2], self.detection.target[:2])
        self.assertEqual(again.knock_paths, self.detection.knock_paths)

    def test_bounce_setting_limits_the_drawn_paths_to_one_segment(self):
        from dataclasses import replace

        style = GuideStyle(show_bounce=False)
        knock = [[((100, 100), (300, 100)), ((300, 100), (300, 300))], []]
        d = replace(self.detection, knock_paths=knock, deflect_paths=[[], []], aim_segments=None,
                    object_segments=None, cue_segments=None, balls=None, blockers=None, target=None, table=None)
        layer = self.layer(d, style)
        self.assertTrue((layer[95:106, 190:211, 3] > 0).any())
        self.assertFalse((layer[195:206, 295:306, 3] > 0).any())


if __name__ == "__main__":
    unittest.main()
