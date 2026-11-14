"""The guide stays still while the shot is unchanged and follows real changes at once."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

import fake_game
from detector import Detection, detect
from overlay import GuideStabilizer


def shot(cue=(100, 150, 12), ghost=(300, 100, 12), target=(324, 94, 12), aim_deg=-14.0, table=(20, 20, 600, 300)):
    a = np.radians(aim_deg)
    return Detection(table=table, cue=cue, balls=[(*cue, "cue"), (*target, "ball")],
                     aim_segments=[(cue[:2], ghost[:2])], ghost=ghost, target=target,
                     aim_dir=(float(np.cos(a)), float(np.sin(a))), note="shot")


def noisy(frame, rng):
    out = np.clip(frame.astype(int) + rng.integers(-4, 5, frame.shape), 0, 255).astype(np.uint8)
    ok, data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 90))])
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class StabilizerRuleTests(unittest.TestCase):
    def setUp(self):
        self.s = GuideStabilizer()
        self.first = self.s.update(shot(), now=0)

    def test_noise_keeps_the_same_guide(self):
        jitter = shot(cue=(101, 151, 12), ghost=(301, 101, 12), target=(326, 95, 12), aim_deg=-13.85)
        self.assertIs(self.s.update(jitter, now=.1), self.first)

    def test_a_real_aim_change_is_shown_at_once(self):
        turned = shot(aim_deg=-15.0, ghost=(300, 104, 12))
        self.assertIs(self.s.update(turned, now=.1), turned)

    def test_moving_the_cue_ball_ghost_or_target_is_shown(self):
        for change in (dict(cue=(106, 150, 12)), dict(ghost=(305, 100, 12)), dict(target=(330, 99, 12)),
                       dict(table=(25, 20, 600, 300))):
            s = GuideStabilizer()
            first = s.update(shot(), now=0)
            moved = shot(**change)
            self.assertIsNot(s.update(moved, now=.1), first, change)

    def test_a_short_gap_without_aim_line_keeps_the_guide(self):
        gap = Detection(table=(20, 20, 600, 300), cue=(100, 150, 12), note="no aim")
        self.assertIs(self.s.update(gap, now=.3), self.first)
        # ...but not for long.
        self.assertIs(self.s.update(gap, now=.9), gap)

    def test_the_hold_counts_from_the_last_confirmed_frame(self):
        self.s.update(shot(), now=.4)  # same shot again confirms it
        gap = Detection(table=(20, 20, 600, 300), note="no aim")
        self.assertIs(self.s.update(gap, now=.8), self.first)

    def test_no_hold_once_the_cue_ball_has_moved(self):
        gap = Detection(table=(20, 20, 600, 300), cue=(160, 150, 12), note="cue ball rolling")
        self.assertIs(self.s.update(gap, now=.1), gap)

    def test_reset_forgets_the_shown_guide(self):
        self.s.reset()
        again = shot()
        self.assertIs(self.s.update(again, now=.1), again)

    def test_a_noisy_frame_does_not_drift_the_reference(self):
        # Small changes are compared with what is on screen, not the last frame,
        # so a slow turn still shows up once it adds up.
        for k, deg in enumerate((-13.9, -13.8, -13.75)):
            self.assertIs(self.s.update(shot(aim_deg=deg), now=.1 * (k + 1)), self.first)
        turned = shot(aim_deg=-13.6)
        self.assertIs(self.s.update(turned, now=.5), turned)


class StabilizerOnSyntheticFramesTests(unittest.TestCase):
    def test_still_shots_stop_flickering_and_new_shots_appear_at_once(self):
        s = GuideStabilizer()
        rng = np.random.default_rng(1)
        now, redraws_still, previous = 0.0, 0, None
        for seed in (40, 41, 42, 43):
            clean, _ = fake_game.render(fake_game.random_scene(seed))
            for k in range(5):
                now += .125
                shown = s.update(detect(noisy(clean, rng)), now=now)
                if k == 0:
                    # A new shot replaces the old guide on its first frame.
                    self.assertIsNot(shown, previous)
                    self.assertTrue(shown.aim_segments)
                elif shown is not previous:
                    redraws_still += 1
                previous = shown
        self.assertEqual(redraws_still, 0)

    def test_without_the_stabilizer_the_same_frames_do_jitter(self):
        # Documents the problem: raw detections of a still shot differ frame to frame.
        rng = np.random.default_rng(2)
        clean, _ = fake_game.render(fake_game.random_scene(40))
        pictures = {tuple(map(tuple, detect(noisy(clean, rng)).balls)) for _ in range(5)}
        self.assertGreater(len(pictures), 1)


if __name__ == "__main__":
    unittest.main()
