"""Neural-network ball proposals (``ball_net``): coordinate mapping, peak
decoding, and the trained model on synthetic frames it never saw."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ball_net as bn  # noqa: E402
import fake_game as fg  # noqa: E402


class Mapping(unittest.TestCase):
    def test_roundtrip(self):
        for scale in (.8, 1.0, 1.75, 2.9):
            for v in (0.0, 3.3, 100.0, 511.5):
                self.assertAlmostEqual(bn.to_roi(bn.to_input(v, scale), scale), v, places=9)

    def test_decode_places_peak(self):
        h, w = bn.INPUT_H // bn.STRIDE, bn.INPUT_W // bn.STRIDE
        heat = np.zeros((len(bn.CLASSES), h, w), np.float32)
        offset = np.zeros((2, h, w), np.float32)
        heat[3, 10, 20] = .9
        heat[3, 10, 21] = .5          # neighbour, not a second peak
        heat[0, 30, 5] = .7
        offset[:, 10, 20] = (.25, .75)
        found = sorted(bn.decode(heat, offset))
        self.assertEqual(len(found), 2)
        (x0, y0, s0, c0), (x1, y1, s1, c1) = found
        self.assertEqual((c0, c1), (0, 3))
        self.assertAlmostEqual(x1, 20.25 * bn.STRIDE)
        self.assertAlmostEqual(y1, 10.75 * bn.STRIDE)

    def test_classes(self):
        self.assertEqual([bn.class_of(n) for n in (0, 1, 7, 8, 9, 15)], [0, 1, 1, 2, 3, 3])


@unittest.skipUnless(bn.available(), "no trained model (train_ball_net.py)")
class Model(unittest.TestCase):
    def test_recall_on_unseen_frames(self):
        """Every ball is proposed within half a radius on the sweep seeds
        (never trained on), all but a few in a hundred."""
        found = total = false = 0
        for i, cloth in enumerate(["blue", "white", "black", "purple", "beige"]):
            for kind in ("scatter", "rack", "cushion"):
                scene = fg.random_scene(40000 + i, kind, cloth)
                frame, truth = fg.render(scene)
                x, y, w, h = (int(round(v)) for v in truth.table)
                props = bn.propose(frame[y:y + h, x:x + w])
                balls = [truth.cue] + list(truth.balls.values())
                used = set()
                for bx, by in balls:
                    d = [np.hypot(px + x - bx, py + y - by) for px, py, _, _ in props]
                    k = int(np.argmin(d)) if d else -1
                    if k >= 0 and d[k] <= truth.radius * .5:
                        used.add(k)
                        found += 1
                total += len(balls)
                false += len(props) - len(used)
        self.assertGreaterEqual(found / total, .97)
        self.assertLessEqual(false, total * .03)

    def test_black_eight_on_black_cloth(self):
        """With the switch on, detect() finds the black 8 on the near-black
        cloth, where circle finding proposes nothing (main: 4 of 12)."""
        import detector
        old = detector.NN_BALLS
        detector.NN_BALLS = "only+trust"
        try:
            found = 0
            for seed in range(40000, 40012):
                frame, truth = fg.render(fg.random_scene(seed, "scatter", "black"))
                d = detector.detect(frame)
                bx, by = truth.balls[8]
                found += any(np.hypot(x - bx, y - by) <= truth.radius * .5 for x, y, _, _ in d.balls or [])
        finally:
            detector.NN_BALLS = old
        self.assertGreaterEqual(found, 11)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "outputs" / "nn" / "data" / "real-8bd-scenes-test"
                          / "frames.json").is_file(), "real test frames not imported (train_ball_net.py import-real)")
    def test_real_frames(self):
        """Held-out real screenshots (public set, scenes never trained on):
        with the switch on, the first 30 frames' balls are found (main:
        288 of 305, 7 false; network: 298, 1 false)."""
        import json
        import train_ball_net as tbn
        frames = json.loads((tbn.DATA / "real-8bd-scenes-test" / "frames.json").read_text())[:30]
        if not all(Path(f["file"]).is_file() for f in frames):
            self.skipTest("downloaded dataset not in outputs/datasets")
        import cv2
        import detector
        old, threads = detector.NN_BALLS, cv2.getNumThreads()
        try:
            res = [tbn._eval_real_frame(("only+trust", f)) for f in frames]
        finally:
            detector.NN_BALLS = old
            cv2.setNumThreads(threads)
        total, hit, false = (sum(r[i] for r in res) for i in range(3))
        self.assertGreaterEqual(hit, total - 10)
        self.assertLessEqual(false, 3)

    def test_off_by_default(self):
        import detector
        if not os.environ.get("POOL_OVERLAY_NN_BALLS"):
            self.assertEqual(detector.NN_BALLS, "")


if __name__ == "__main__":
    unittest.main()
