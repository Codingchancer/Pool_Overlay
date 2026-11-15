"""Wide stress-test fixes: real-mirror picture effects, pale coloured cloths,
the cue ball under the ghost ring, and no invented aim on the opponent's turn.

Each case below failed before (see the PR for the sweep numbers)."""
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import detector
import fake_game as fg
import stress_test as st


class MirrorEffectTests(unittest.TestCase):
    def test_effects_off_by_default(self):
        scene = fg.random_scene(9001)
        frame, _ = fg.render(scene)
        same, _ = fg.render(replace(scene, mirror_scale=1.0, chroma_420=False, tone=(1.0, 0.0, 1.0)))
        self.assertTrue(np.array_equal(frame, same))

    def test_harsh_changes_the_picture_not_the_layout(self):
        scene = fg.random_scene(9001)
        harsh = fg.harsh(scene)
        self.assertEqual(harsh.balls, scene.balls)
        self.assertEqual(harsh.cue, scene.cue)
        self.assertTrue(harsh.chroma_420 or harsh.mirror_scale < 1 or harsh.tone != (1.0, 0.0, 1.0))
        a, truth_a = fg.render(scene)
        b, truth_b = fg.render(harsh)
        self.assertEqual(truth_a.ghost, truth_b.ghost)
        self.assertFalse(np.array_equal(a, b))

    def test_harsh_stream_scale_is_realistic(self):
        # An iPhone mirror streams ~1600-2000 px wide: never scaled below that.
        for seed in range(20):
            scene = fg.random_scene(seed)
            harsh = fg.harsh(scene)
            self.assertGreaterEqual(harsh.mirror_scale * scene.width, min(scene.width, 1600) - 1)

    def test_table_found_on_a_darker_display(self):
        # The pocket's shaded rim fell under the darkness limit and moved the
        # table by up to 6 px; pockets are now centred on their darkest core.
        for cloth in ("orange", "blue", "white"):
            for seed in range(30000, 30004):
                scene = replace(fg.random_scene(seed, "scatter", cloth), tone=(1.1, -12.0, 1.12))
                frame, truth = fg.render(scene)
                found = detector.find_table_bounds(frame)
                x, y, w, h = truth.table
                edges = np.abs(np.array([found[0] - x, found[1] - y, found[0] + found[2] - x - w,
                                         found[1] + found[3] - y - h]))
                self.assertLessEqual(edges.max(), st.TABLE_PX, f"{cloth} {seed}")


class PaleColouredClothTests(unittest.TestCase):
    def test_sky_cloth_guide_and_balls(self):
        # The thin white lines, blurred with their dark outline, are no
        # brighter than sky cloth (they are far less saturated), and the
        # lettering passed the number-badge test.
        for seed, kind in ((30003, "scatter"), (30006, "scatter"), (30005, "pot"), (30001, "cushion")):
            *_, problems, _ = st.evaluate(seed, kind, "sky")
            self.assertEqual(problems, [], f"sky {seed} {kind}")

    def test_pink_lettering_is_not_a_ball(self):
        for seed, kind in ((30006, "scatter"), (30001, "scatter"), (30004, "miss")):
            *_, problems, _ = st.evaluate(seed, kind, "pink")
            self.assertEqual(problems, [], f"pink {seed} {kind}")


class GhostOnCueBallTests(unittest.TestCase):
    def test_cue_ball_not_locked_onto_the_ring(self):
        # 30001: the found "cue ball" was the ghost ring itself (10 px off);
        # 30005: the ring pulled it 6 px.
        for seed in (30001, 30005):
            frame, truth = fg.render(fg.random_scene(seed, "near", "blue"))
            d = detector.detect(frame)
            self.assertLessEqual(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]), st.CUE_PX, seed)


class NoGuideTests(unittest.TestCase):
    def test_no_aim_without_the_games_guide(self):
        # Opponent's turn: no guide, maybe no stick. White bits of packed or
        # neighbouring balls added up to a "ghost ring" on 11 of 100 frames.
        for cloth in ("blue", "green", "black"):
            for seed in (30000, 30005, 30008, 30009):
                for stick in (True, False):
                    scene = replace(fg.random_scene(seed, None, cloth), show_guide=False, show_stick=stick)
                    d = detector.detect(fg.render(scene)[0])
                    self.assertFalse(d.aim_segments, f"{cloth} {seed} stick={stick}: {d.note}")

    def test_guided_shots_still_read(self):
        for seed in (30000, 30005, 30008):
            *_, d, problems, _ = st.evaluate(seed, None, "blue")
            self.assertTrue(d.aim_segments, seed)


if __name__ == "__main__":
    unittest.main()
