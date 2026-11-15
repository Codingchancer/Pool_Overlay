"""Capture and analysis of the mirror window in a process of its own.

With window capture the guide is not in the frames, so a frame no longer
depends on what was drawn last. The capture-and-analyse loop can then run
on its own, frame after frame, in a separate process: Python runs only one
thread at a time per process, so in the app's own process the guide's
drawing and the analysis slowed each other down (~25 ms per frame alone,
~35 ms inside the app). Here the app only sends what to capture (the mirror
window and the guide's area) and draws whatever result arrives last.

Messages to the process: ``("job", {"hwnd", "region", "generation"})``,
``("pause",)`` and ``("stop",)``. A new generation starts a fresh
``ShotTracker`` (Re-detect, guide moved or switched on again).
Messages back: ``("detection", generation, Detection, seconds)`` and
``("no picture", generation)`` when the window gives no picture (the app then
uses screen capture for a while).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time


def _dump(folder: str, name: str, image):
    """Write a diagnostics image atomically (same files as the in-app path)."""
    import cv2
    try:
        temporary = os.path.join(folder, "~" + name)
        if cv2.imwrite(temporary, image):
            os.replace(temporary, os.path.join(folder, name))
    except Exception:
        pass


def run_loop(requests, results, make_capture=None, idle_sleep: float = .003):
    """The process body (also runs in a thread for tests, with a fake capture)."""
    from tracking import ShotTracker, change_thumbnail, screen_changed
    if make_capture is None:
        from window_capture import WindowCapture as make_capture
    job = capture = None
    tracker = ShotTracker()
    last = None
    debug_dir = os.environ.get("POOL_OVERLAY_DEBUG_DIR")
    # In the real process: stop when the app is gone, even if it was killed
    # without sending "stop" (a daemon process is not ended then).
    parent = mp.parent_process()
    checked_parent = time.monotonic()
    try:
        while True:
            if parent is not None and time.monotonic() - checked_parent > .25:
                checked_parent = time.monotonic()
                if not parent.is_alive():
                    break
            try:
                message = requests.get(timeout=.25) if job is None else requests.get_nowait()
            except queue.Empty:
                message = None
            if message is not None:
                kind = message[0]
                if kind == "stop":
                    break
                if kind == "pause":
                    job = None
                elif kind == "job":
                    new = message[1]
                    if capture is None or job is None or new["hwnd"] != job["hwnd"]:
                        if capture is not None:
                            capture.close()
                        capture = make_capture(new["hwnd"])
                    if job is None or new["generation"] != job["generation"]:
                        tracker, last = ShotTracker(), None
                    job = new
                continue  # read every waiting message before the next frame
            if job is None:
                continue
            frame = capture.grab(job["region"])
            if frame is None:
                results.put(("no picture", job["generation"]))
                job = None
                continue
            thumbnail = change_thumbnail(frame)
            if not screen_changed(thumbnail, last):
                time.sleep(idle_sleep)
                continue
            last = thumbnail
            started = time.perf_counter()
            try:
                detection = tracker.detect(frame)
            except Exception as exc:  # keep running; report like the in-app path
                results.put(("failed", job["generation"], str(exc)))
                continue
            if debug_dir:
                _dump(debug_dir, "capture.png", frame)
                _dump(debug_dir, "cleaned.png", frame)
            results.put(("detection", job["generation"], detection, time.perf_counter() - started))
    finally:
        if capture is not None:
            capture.close()


class AnalysisProcess:
    """Starts the loop in its own process and hands the newest result to the app.

    ``on_result`` is called from a reader thread with the newest message only
    (older ones that piled up are dropped); the app forwards it to the window
    thread with a Qt signal.
    """

    def __init__(self, on_result):
        self.on_result = on_result
        context = mp.get_context("spawn")
        self.requests = context.Queue()
        self.results = context.Queue()
        self.process = context.Process(target=run_loop, args=(self.requests, self.results), daemon=True,
                                       name="pool-overlay-analysis")
        self.process.start()
        self._stopping = False
        self._reader = threading.Thread(target=self._read, daemon=True, name="pool-overlay-results")
        self._reader.start()

    def _read(self):
        while not self._stopping:
            try:
                message = self.results.get(timeout=.25)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            # Only the newest result matters for drawing.
            try:
                while True:
                    message = self.results.get_nowait()
            except queue.Empty:
                pass
            except (EOFError, OSError):
                break
            try:
                self.on_result(message)
            except Exception:
                pass

    def alive(self) -> bool:
        return self.process.is_alive()

    def send_job(self, hwnd: int, region: dict, generation: int):
        self.requests.put(("job", {"hwnd": int(hwnd), "region": dict(region), "generation": generation}))

    def pause(self):
        self.requests.put(("pause",))

    def stop(self, timeout: float = 2.0):
        self._stopping = True
        try:
            self.requests.put(("stop",))
        except Exception:
            pass
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)
        self._reader.join(1)
