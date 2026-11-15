"""The screen capture must be reused across analysis frames (issue #13).

PySide clears a threading.local after every QRunnable, so keeping the mss
capture there opened a new one for each frame without closing it. Its GDI
objects leaked until Windows' limit of 10,000 and the guide froze after a
few minutes.
"""
import os
import sys
import threading
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QRunnable, QThreadPool
from PySide6.QtWidgets import QApplication

import overlay

APP = QApplication.instance() or QApplication([])


class _Grab(QRunnable):
    def __init__(self, seen):
        super().__init__()
        self.seen = seen
        self.done = threading.Event()

    def run(self):
        self.seen.append(id(overlay._capture_for_this_thread()))
        self.done.set()


def run_frames(pool, count, seen):
    # Like the app: one frame after another on a live pool. (waitForDone
    # would shut the pool's threads down, giving every frame a new thread.)
    for _ in range(count):
        task = _Grab(seen)
        pool.start(task)
        assert task.done.wait(5)


class CaptureReuseTests(unittest.TestCase):
    def tearDown(self):
        overlay.close_captures()

    def test_one_capture_per_worker_thread_across_frames(self):
        pool = QThreadPool()
        pool.setMaxThreadCount(1)
        seen = []
        run_frames(pool, 30, seen)
        pool.waitForDone()
        self.assertEqual(len(seen), 30)
        self.assertEqual(len(set(seen)), 1, "a new capture was opened for some frames")

    def test_captures_are_closed_on_exit(self):
        pool = QThreadPool()
        pool.setMaxThreadCount(2)
        run_frames(pool, 4, [])
        pool.waitForDone()
        self.assertGreaterEqual(len(overlay._captures), 1)
        overlay.close_captures()
        self.assertEqual(overlay._captures, {})


if __name__ == "__main__":
    unittest.main()
