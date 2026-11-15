"""Window capture: the mirror window's own picture, without the guide on top.

Opens real windows in a helper process (Windows only): a synthetic table, and
a red always-on-top window covering part of it like the guide does.
"""
import ctypes
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import fake_game

WINDOWS = sys.platform == "win32"
TABLE_TITLE = "Synthetic window-capture test table"
BLACK_TITLE = "window-capture test black"
X, Y = 60, 90

HELPER = r'''
import sys
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QWidget
app = QApplication([])
frame = cv2.imread(sys.argv[1])
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
table = QLabel()
table.setWindowTitle(sys.argv[2])
table.setWindowFlags(Qt.WindowType.FramelessWindowHint)
table.setPixmap(QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], int(rgb.strides[0]),
                                         QImage.Format.Format_RGB888).copy()))
table.setFixedSize(rgb.shape[1], rgb.shape[0])
table.move(int(sys.argv[4]), int(sys.argv[5]))
table.show()
cover = QWidget()
cover.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
cover.setStyleSheet("background: rgb(255, 0, 0)")
cover.setGeometry(int(sys.argv[4]) + 100, int(sys.argv[5]) + 80, 200, 120)
cover.show()
black = QWidget()
black.setWindowTitle(sys.argv[3])
black.setWindowFlags(Qt.WindowType.FramelessWindowHint)
black.setStyleSheet("background: black")
black.setGeometry(int(sys.argv[4]) + 700, int(sys.argv[5]), 160, 120)
black.show()
app.exec()
'''


def find_window(title, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(.2)
    return None


def gdi_objects():
    process = ctypes.windll.kernel32.GetCurrentProcess()
    return ctypes.windll.user32.GetGuiResources(process, 0)


@unittest.skipUnless(WINDOWS, "window capture is Windows only")
class WindowCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from window_capture import WindowCapture
        cls.WindowCapture = WindowCapture
        frame, _ = fake_game.render(fake_game.random_scene(11))
        cls.frame = np.ascontiguousarray(frame[:420, :640])
        cls.folder = tempfile.mkdtemp()
        path = os.path.join(cls.folder, "table.png")
        cv2.imwrite(path, cls.frame)
        env = dict(os.environ)
        env.pop("QT_QPA_PLATFORM", None)
        env["QT_ENABLE_HIGHDPI_SCALING"] = "0"
        cls.helper = subprocess.Popen([sys.executable, "-c", HELPER, path, TABLE_TITLE, BLACK_TITLE, str(X), str(Y)],
                                      env=env)
        cls.table = find_window(TABLE_TITLE)
        cls.black = find_window(BLACK_TITLE)
        time.sleep(.8)  # first paint
        if not cls.table:
            cls.helper.kill()
            raise unittest.SkipTest("test window did not open")

    @classmethod
    def tearDownClass(cls):
        cls.helper.kill()
        cls.helper.wait(5)

    def client_region(self, hwnd):
        from window_capture import _client_rect
        left, top, width, height = _client_rect(hwnd)
        return {"left": left, "top": top, "width": width, "height": height}

    def test_picture_is_the_window_without_what_covers_it(self):
        capture = self.WindowCapture(self.table)
        try:
            picture = capture.grab(self.client_region(self.table))
        finally:
            capture.close()
        self.assertIsNotNone(picture)
        if picture.shape != self.frame.shape:
            self.skipTest(f"window scaled by Windows display scaling: {picture.shape}")
        # The red cover window is not in it: the picture is the table itself.
        self.assertLessEqual(int(np.abs(picture.astype(int) - self.frame).max()), 2)

    def test_area_outside_the_window_is_neutral_grey(self):
        region = self.client_region(self.table)
        region = {**region, "left": region["left"] - 40, "width": 100}
        capture = self.WindowCapture(self.table)
        try:
            picture = capture.grab(region)
        finally:
            capture.close()
        self.assertTrue((picture[:, :40] == 128).all())
        self.assertFalse((picture[:, 40:] == 128).all())

    def test_gdi_objects_are_reused_and_released(self):
        # Issue #13: per-frame GDI objects froze the guide after a few minutes.
        result = {}

        def worker():
            before = gdi_objects()
            capture = self.WindowCapture(self.table)
            region = self.client_region(self.table)
            capture.grab(region)
            opened = gdi_objects()
            for _ in range(300):
                capture.grab(region)
            result["growth"] = gdi_objects() - opened
            capture.close()
            result["left"] = gdi_objects() - before

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(30)
        self.assertEqual(result.get("growth"), 0)
        self.assertLessEqual(result.get("left"), 0)

    def test_a_window_with_no_picture_gives_none(self):
        if not self.black:
            self.skipTest("black test window did not open")
        capture = self.WindowCapture(self.black)
        try:
            self.assertIsNone(capture.grab(self.client_region(self.black)))
            # ...and it does not try again on every frame (screen capture is used meanwhile).
            start = time.perf_counter()
            for _ in range(20):
                capture.grab(self.client_region(self.black))
            self.assertLess(time.perf_counter() - start, .1)
        finally:
            capture.close()

    def test_the_mirror_is_found_even_under_another_window(self):
        from window_capture import window_under
        # The middle of this area is covered by the red always-on-top window;
        # the table under it has a mirror-style title, so it is chosen.
        covered = {"left": X + 150, "top": Y + 100, "width": 100, "height": 80}
        self.assertEqual(window_under(covered), self.table)

    def test_closed_capture_gives_none(self):
        capture = self.WindowCapture(self.table)
        capture.close()
        self.assertIsNone(capture.grab(self.client_region(self.table)))


if __name__ == "__main__":
    unittest.main()
