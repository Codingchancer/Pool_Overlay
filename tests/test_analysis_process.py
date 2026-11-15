"""The capture-and-analyse loop that runs in its own process (analysis_process.py)."""
import dataclasses
import os
import queue
import sys
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fake_game
from analysis_process import AnalysisProcess, run_loop


class FakeWindow:
    """Stands in for WindowCapture: shows prepared frames, one per grab."""
    instances = []

    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.frames = FRAMES.get(hwnd, [])
        self.index = 0
        self.closed = False
        FakeWindow.instances.append(self)

    def grab(self, region):
        if not self.frames:
            return None
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def close(self):
        self.closed = True


SCENE = fake_game.random_scene(2003)
AIMING = [fake_game.render(dataclasses.replace(SCENE, aim_degrees=SCENE.aim_degrees + .5 * k))[0]
          for k in range(4)]
FRAMES = {1: AIMING, 2: [AIMING[0]]}


class LoopTests(unittest.TestCase):
    def setUp(self):
        FakeWindow.instances.clear()
        self.requests, self.results = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=run_loop, args=(self.requests, self.results, FakeWindow),
                                       daemon=True)
        self.thread.start()

    def tearDown(self):
        self.requests.put(("stop",))
        self.thread.join(10)

    def messages(self, count, timeout=20):
        out, end = [], time.time() + timeout
        while len(out) < count and time.time() < end:
            try:
                out.append(self.results.get(timeout=.2))
            except queue.Empty:
                pass
        return out

    def test_every_new_frame_is_analysed_and_tagged_with_its_generation(self):
        self.requests.put(("job", {"hwnd": 1, "region": {}, "generation": 5}))
        got = self.messages(4)
        self.assertEqual([m[0] for m in got], ["detection"] * 4)
        self.assertTrue(all(m[1] == 5 for m in got))
        self.assertTrue(all(m[2].aim_segments for m in got))

    def test_an_unchanged_frame_is_not_analysed_again(self):
        self.requests.put(("job", {"hwnd": 2, "region": {}, "generation": 1}))
        self.assertEqual(len(self.messages(1)), 1)
        self.assertEqual(self.messages(1, timeout=1), [])

    def test_no_picture_is_reported_and_the_loop_waits(self):
        self.requests.put(("job", {"hwnd": 3, "region": {}, "generation": 2}))
        self.assertEqual(self.messages(1), [("no picture", 2)])
        self.assertEqual(self.messages(1, timeout=.6), [])

    def test_stop_closes_the_capture(self):
        self.requests.put(("job", {"hwnd": 2, "region": {}, "generation": 1}))
        self.messages(1)
        self.requests.put(("stop",))
        self.thread.join(10)
        self.assertFalse(self.thread.is_alive())
        self.assertTrue(all(c.closed for c in FakeWindow.instances))

    def test_another_window_gets_a_new_capture_and_the_old_one_is_closed(self):
        self.requests.put(("job", {"hwnd": 2, "region": {}, "generation": 1}))
        self.messages(1)
        self.requests.put(("job", {"hwnd": 1, "region": {}, "generation": 1}))
        self.messages(1)
        self.assertEqual([c.hwnd for c in FakeWindow.instances], [2, 1])
        self.assertTrue(FakeWindow.instances[0].closed)


class RealProcessTests(unittest.TestCase):
    def test_process_starts_answers_and_stops(self):
        received = []
        process = AnalysisProcess(received.append)
        try:
            # No such window: the process answers that it has no picture.
            process.send_job(0, {"left": 0, "top": 0, "width": 50, "height": 50}, 7)
            end = time.time() + 30
            while not received and time.time() < end:
                time.sleep(.1)
            self.assertEqual(received[:1], [("no picture", 7)])
        finally:
            process.stop()
        self.assertFalse(process.alive())


if __name__ == "__main__":
    unittest.main()
