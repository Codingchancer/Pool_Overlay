"""Fast frame-to-frame detection for the live overlay.

A full ``detect`` finds the table and every ball, which takes 150-250 ms. While
the player is aiming, none of that changes: only the cue stick and the game's
white aim graphics move. ``ShotTracker`` therefore keeps the table and balls
from the last full detection and, on each new frame, only checks that the
balls are still where they were (a few pixels per ball) and re-reads the aim
graphics with ``predict_shot`` (20-35 ms). Anything else (a ball moved,
potted or covered, the overlay moved, no aim found for a few frames, or a
few seconds passed) falls back to a full detection.
"""
from __future__ import annotations

import time

import numpy as np

from detector import Detection, detect, predict_shot


class ShotTracker:
    # Inside each ball (70% of its radius, clear of the stick tip and the
    # game's lines that end at its rim), this share of pixels changing by more
    # than CHANGE_LEVEL means the ball moved or something covers it.
    INNER = .7
    CHANGE_LEVEL = 40
    CHANGED_SHARE = .35
    # A full detection now and then catches what the per-ball check cannot see
    # (a ball appearing where there was none, e.g. a respot).
    REFRESH_SECONDS = 3.0
    # Frames in a row without an aim line before the balls are found again
    # (in case the cue ball was misidentified).
    MISSES_BEFORE_FULL = 3

    def __init__(self, full_detect=detect):
        self.full_detect = full_detect
        self.reset()

    def reset(self):
        self.reference: np.ndarray | None = None
        self.base: Detection | None = None
        self.detected_at = 0.0
        self.misses = 0
        self.last_was_full = True

    def detect(self, frame: np.ndarray, now: float | None = None) -> Detection:
        now = time.monotonic() if now is None else now
        if self._can_track(frame, now):
            detection = self._track(frame)
            self.misses = 0 if detection.aim_segments else self.misses + 1
            if self.misses < self.MISSES_BEFORE_FULL:
                self.last_was_full = False
                return detection
        return self._full(frame, now)

    def _full(self, frame: np.ndarray, now: float) -> Detection:
        detection = self.full_detect(frame)
        self.last_was_full = True
        self.misses = 0
        if detection.table is not None and detection.cue is not None and detection.balls:
            self.reference, self.base, self.detected_at = frame.copy(), detection, now
        else:
            self.reference = self.base = None
        return detection

    def _can_track(self, frame: np.ndarray, now: float) -> bool:
        if self.reference is None or frame.shape != self.reference.shape:
            return False
        if now - self.detected_at > self.REFRESH_SECONDS:
            return False
        return all(self._ball_still(frame, ball) for ball in self.base.balls)

    def _ball_still(self, frame: np.ndarray, ball) -> bool:
        x, y, r = int(ball[0]), int(ball[1]), float(ball[2])
        inner = max(2, int(r * self.INNER))
        h, w = frame.shape[:2]
        x0, x1, y0, y1 = max(0, x - inner), min(w, x + inner + 1), max(0, y - inner), min(h, y + inner + 1)
        if x1 <= x0 or y1 <= y0:
            return False
        now_patch = frame[y0:y1, x0:x1].astype(np.int16)
        then_patch = self.reference[y0:y1, x0:x1].astype(np.int16)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx - x) ** 2 + (yy - y) ** 2 <= inner * inner
        change = np.abs(now_patch - then_patch).max(axis=2)[disc]
        return change.size > 0 and float(np.mean(change > self.CHANGE_LEVEL)) < self.CHANGED_SHARE

    def _track(self, frame: np.ndarray) -> Detection:
        base = self.base
        x, y, w, h = base.table
        roi = frame[y:y + h, x:x + w]
        shot = predict_shot(roi, base.table, base.balls, base.cue)
        return Detection(table=base.table, balls=base.balls, cue=base.cue, aim_segments=shot["aim"],
                         cue_segments=shot["cue"], object_segments=shot["object"],
                         target=shot["target"], ghost=shot["ghost"], blockers=shot["blockers"],
                         aim_dir=shot["aim_dir"], object_dir=shot["object_dir"], cue_dir=shot["cue_dir"],
                         note=f"{shot['note']} · {len(base.balls)} ball candidates")
