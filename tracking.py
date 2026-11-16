"""Fast frame-to-frame detection for the live overlay.

A full ``detect`` finds the table and every ball, which takes 150-250 ms. While
the player is aiming, none of that changes: only the cue stick and the game's
white aim graphics move. ``ShotTracker`` therefore keeps the table and balls
from the last full detection and, on each new frame, only checks that the
balls are still where they were (a few pixels per ball) and re-reads the aim
graphics with ``predict_shot``. That read is handed last frame's aim and
ghost ball (``hint``): while aiming the angle moves a little per frame, so the
ghost ring is looked for near where it was instead of scanning every ray
direction (~5 ms instead of ~15 on the synthetic table, same result). Every
``FULL_READ_EVERY`` frames, and whenever the hint does not fit, the aim is
read without it. Anything else (a ball moved,
potted or covered, the overlay moved, no aim found for a few frames, or a
few seconds passed) falls back to a full detection.
"""
from __future__ import annotations

import dataclasses
import time

import cv2
import numpy as np

from detector import Detection, detect, find_power_slider, predict_shot, read_power_slider


class ShotTracker:
    # Inside each ball (70% of its radius, clear of the stick tip and the
    # game's lines that end at its rim), this share of pixels changing by more
    # than CHANGE_LEVEL means the ball moved or something covers it.
    INNER = .7
    CHANGE_LEVEL = 40
    CHANGED_SHARE = .35
    # Above this share the ball moved, whatever the shape of the change.
    MOVED_SHARE = .8
    # A change whose spread across its own length is at most this share of
    # the inner radius is a thin stripe: the cue stick (up to ~0.8 ball radii
    # wide) crossing the ball.
    STRIPE_SPREAD = .36
    # A full detection now and then catches what the per-ball check cannot see
    # (a ball appearing where there was none, e.g. a respot).
    REFRESH_SECONDS = 3.0
    # Frames in a row without an aim line before the balls are found again
    # (in case the cue ball was misidentified).
    MISSES_BEFORE_FULL = 3
    # Hinted aim reads in a row before one without the hint (catches a hint
    # that kept fitting something else than the game's ghost ball).
    FULL_READ_EVERY = 8
    # While the last full detection had no power slider (hidden then, e.g.
    # on the opponent's turn), look for it again this often (seconds).
    SLIDER_RETRY_SECONDS = .5

    def __init__(self, full_detect=detect):
        self.full_detect = full_detect
        self.reset()

    def reset(self):
        self.reference: np.ndarray | None = None
        self.base: Detection | None = None
        self.detected_at = 0.0
        self.misses = 0
        self.last_was_full = True
        self.hint: dict | None = None
        self.hinted_reads = 0
        # Whether the last tracked frame's aim was read with the hint.
        self.last_was_fast = False
        self.slider_checked_at = 0.0

    def detect(self, frame: np.ndarray, now: float | None = None) -> Detection:
        now = time.monotonic() if now is None else now
        if self._can_track(frame, now):
            detection = self._track(frame, now)
            self.misses = 0 if detection.aim_segments else self.misses + 1
            if self.misses < self.MISSES_BEFORE_FULL:
                self.last_was_full = False
                return detection
        return self._full(frame, now)

    def _full(self, frame: np.ndarray, now: float) -> Detection:
        detection = self.full_detect(frame)
        self.last_was_full = True
        self.last_was_fast = False
        self.misses = 0
        self.hint = None
        self.hinted_reads = 0
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
        changed = (np.abs(now_patch - then_patch).max(axis=2) > self.CHANGE_LEVEL) & disc
        area = int(disc.sum())
        if area == 0:
            return False
        share = float(changed.sum()) / area
        if share < self.CHANGED_SHARE:
            return True
        if share >= self.MOVED_SHARE:
            return False
        # While aiming, the cue stick passes over balls: the change is then a
        # thin straight stripe across the disc. A ball that moved changes
        # most of it, or a crescent that is not thin. Without this, every
        # frame with the stick over a ball fell back to a full detection.
        ys, xs = np.nonzero(changed)
        if len(xs) < 3:
            return True
        cov = np.cov(np.vstack([xs, ys]).astype(np.float64))
        thinnest = float(np.sqrt(max(0.0, np.linalg.eigvalsh(cov)[0])))
        return thinnest <= inner * self.STRIPE_SPREAD

    def _track(self, frame: np.ndarray, now: float = 0.0) -> Detection:
        base = self.base
        if base.power_slider is None and now - self.slider_checked_at >= self.SLIDER_RETRY_SECONDS:
            self.slider_checked_at = now
            slider = find_power_slider(frame, base.table)
            if slider is not None:
                base = self.base = dataclasses.replace(base, power_slider=slider)
        x, y, w, h = base.table
        roi = frame[y:y + h, x:x + w]
        hint = self.hint if self.hinted_reads < self.FULL_READ_EVERY else None
        shot = predict_shot(roi, base.table, base.balls, base.cue, hint=hint)
        self.last_was_fast = bool(shot.get("fast"))
        self.hinted_reads = self.hinted_reads + 1 if self.last_was_fast else 0
        self.hint = shot.get("hint")
        return Detection(table=base.table, balls=base.balls, cue=base.cue, aim_segments=shot["aim"],
                         cue_segments=shot["cue"], object_segments=shot["object"],
                         target=shot["target"], ghost=shot["ghost"], blockers=shot["blockers"],
                         aim_dir=shot["aim_dir"], object_dir=shot["object_dir"], cue_dir=shot["cue_dir"],
                         pot=shot["pot"], pot_pocket=shot["pot_pocket"], stick_pull=shot["stick_pull"],
                         knock_paths=shot["knock"], deflect_paths=shot["deflect"],
                         power_slider=base.power_slider,
                         power_percent=read_power_slider(frame, base.power_slider),
                         note=f"{shot['note']} · {len(base.balls)} ball candidates")


def change_thumbnail(frame: np.ndarray) -> np.ndarray:
    """A quarter-size grey copy for telling whether the screen changed."""
    h, w = frame.shape[:2]
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (max(1, w // 4), max(1, h // 4)),
                      interpolation=cv2.INTER_AREA)


def screen_changed(thumbnail: np.ndarray, last: np.ndarray | None) -> bool:
    """True unless the frame is the same as the last analysed one.

    Looks for any small spot that changed, not an average over the whole
    frame: turning the aim a fraction of a degree only moves a thin line and
    the cue stick, and that must still be analysed. Stream noise averages out
    over each 4x4 block.
    """
    if last is None or last.shape != thumbnail.shape:
        return True
    return int(np.count_nonzero(cv2.absdiff(thumbnail, last) > 8)) >= 3
