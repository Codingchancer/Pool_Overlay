"""Shot recorder: logs real shots to fit the game's power and friction.

    python shot_recorder.py [--out outputs/shots] [--title "AirPlay"]

Watches the mirror window while you play (it only reads the picture; it
never clicks, aims or sends input) and writes one JSON log per shot:

* the power slider's reading when you let go (hold the slider still for a
  moment before releasing, as you normally would);
* where every ball was before the shot, the aim the game showed, and the
  struck ball;
* the cue ball's and the struck ball's positions frame by frame while they
  roll (their start speed and how they slow down);
* where everything stopped, and which balls were potted.

Plus the release and rest frames as PNG, so every number can be checked
later. Logs go to ``outputs/shots/<date-time>/`` (ignored by git: the frames
show the game and a player profile).

How a shot is found: while you aim, the normal live tracker reads the
table, balls, aim and slider every frame. The shot starts when the cue ball
itself moves: it is followed with a template cut from its own picture (on
any cloth colour), blurred so a rolling ball's number does not matter. The
shot ends when the table has not changed for ``STILL_SECONDS``; a full
detection then gives where everything stopped.

Not read: spin (where the cue tip hits) and which player is shooting; a
shot with no slider reading (the opponent's turn) is logged with power
null. Moving the cue ball by hand (ball in hand) is not a shot: it starts no
log, because no power was set.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detector import Detection, detect
from tracking import ShotTracker, change_thumbnail, screen_changed

HERE = Path(__file__).resolve().parent

# The shot has started once the cue ball is this many radii from its rest spot.
START_MOVE = .35
# The table must stay unchanged this long before the balls count as stopped.
STILL_SECONDS = .6
# A shot longer than this is ended anyway (seconds).
MAX_SHOT_SECONDS = 20.0
# Template match (normalised squared difference) worse than this: the ball is
# not there (potted, or hidden).
LOST_SCORE = .12
LOST_FRAMES = 3
# A match also counts only when it is at most this many times the ball's own
# recent scores (another ball of a similar colour matches far worse than the
# ball itself: 0.07 against 0.006 for a maroon stripe and the maroon solid),
# with this floor for a ball whose own matches are near perfect.
SCORE_RATIO = 5.0
SCORE_FLOOR = .01
# A match must also correlate with the ball's own pattern (zero-mean,
# normalised) at least this well.
PATTERN_MIN = .6
# When a ball is not near where it should be, a window this many radii
# around is searched (a break moves the cue ball ~3 radii a frame at 60 fps).
WIDE_REACH = 8.0
# The cue ball is white: at least this share of its disc (a stripe has about
# half), and at most this much less than the whitest ball on the table.
CUE_WHITE = .6
CUE_WHITE_MARGIN = .15
# While no white ball is on the still table, detect the whole table this often.
CUE_RETRY_SECONDS = .5
# A ball has left its spot when most of its inner disc changed by more than
# this (less for a ball close to the cloth's colour: ``_change_level``).
BALL_CHANGE = 40.0
# The struck ball leaves its spot next to the cue ball: within this many
# radii of the cue ball's position then (plus its travel in two frames).
STRUCK_NEAR = 3.5
# ... and the cue ball's path so far passed within this many radii of it.
STRUCK_PATH = 2.4
# Slider readings at or below this (percent) are the slider at rest.
SLIDER_REST = 1.5
# On release the slider snaps back, possibly over a few frames: consecutive
# readings falling by more than this within SNAP_GAP seconds are that snap.
SNAP_STEP = 4.0
SNAP_GAP = .07
# How much slider history (seconds) is kept and written to the log.
HISTORY_SECONDS = 3.0


def _sub_pixel(score: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """Parabola peak (of a minimum) around (iy, ix) in a score map."""
    fx = fy = 0.0
    if 0 < ix < score.shape[1] - 1:
        a, b, c = score[iy, ix - 1], score[iy, ix], score[iy, ix + 1]
        if a - 2 * b + c > 1e-12:
            fx = float(np.clip(.5 * (a - c) / (a - 2 * b + c), -.5, .5))
    if 0 < iy < score.shape[0] - 1:
        a, b, c = score[iy - 1, ix], score[iy, ix], score[iy + 1, ix]
        if a - 2 * b + c > 1e-12:
            fy = float(np.clip(.5 * (a - c) / (a - 2 * b + c), -.5, .5))
    return fy, fx


class BallFollower:
    """Follows one ball from frame to frame with a template of its own picture.

    The template and every search window are blurred (sigma a quarter
    radius) so the ball is a colour blob: its number and stripe turn as it
    rolls, its blurred colour does not. Only the ball's disc is compared,
    so the cloth around it (any colour) does not count."""

    def __init__(self, frame: np.ndarray, x: float, y: float, r: float, exclude=()):
        self.r = float(r)
        self.sigma = max(1.0, self.r * .25)
        self.half = int(np.ceil(self.r * 1.1))
        self.pos = (float(x), float(y))
        self.vel = (0.0, 0.0)
        self.t: float | None = None
        self.misses = 0
        self.lost = False
        self.offset = (0.0, 0.0)
        self.scores: deque = deque(maxlen=10)
        patch = self._blurred(frame, int(round(x)), int(round(y)), self.half)
        self.template = None if patch is None else patch
        size = 2 * self.half + 1
        yy, xx = np.mgrid[:size, :size]
        offset_x, offset_y = x - round(x), y - round(y)
        self.dx, self.dy = xx - self.half - offset_x, yy - self.half - offset_y
        mask = self.dx ** 2 + self.dy ** 2 <= (self.r * .9) ** 2
        # The game draws a white line from this ball's centre (the aim ray
        # from the cue ball, the object line from the struck ball): leave
        # that strip out, since the ball has no line on it once it moves.
        self.mask = (mask & ~self._strips(exclude)).astype(np.float32)
        self.mask3 = np.dstack([self.mask] * 3)

    def _strips(self, directions) -> np.ndarray:
        """Template pixels on a line drawn from the ball's centre along each
        of ``directions``."""
        out = np.zeros(self.dx.shape, bool)
        width = max(2.5, self.r * .18) + self.sigma
        for ux, uy in directions:
            along = self.dx * ux + self.dy * uy
            across = np.abs(self.dx * uy - self.dy * ux)
            out |= (along >= -width) & (across <= width)
        return out

    def _blurred(self, frame, cx, cy, half):
        h, w = frame.shape[:2]
        margin = int(np.ceil(self.sigma * 3))
        x0, x1 = cx - half - margin, cx + half + margin + 1
        y0, y1 = cy - half - margin, cy + half + margin + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        patch = cv2.GaussianBlur(frame[y0:y1, x0:x1].astype(np.float32), (0, 0), self.sigma)
        return patch[margin:-margin, margin:-margin]

    def locate(self, frame: np.ndarray, t: float, reach: float | None = None,
               centre: tuple[float, float] | None = None, exclude=()):
        """Best match near where the ball should be now (or near ``centre``):
        (x, y, score) or None."""
        if self.template is None:
            return None
        dt = 0.0 if self.t is None else max(0.0, t - self.t)
        px, py = self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt
        if centre is not None:
            px, py = centre
        if reach is None:
            reach = max(1.5 * self.r, 1.5 * float(np.hypot(*self.vel)) * dt + self.r)
        reach = int(np.ceil(reach))
        h, w = frame.shape[:2]
        cx, cy = int(round(px)), int(round(py))
        size = self.half + reach
        margin = int(np.ceil(self.sigma * 3))
        x0, x1 = max(0, cx - size - margin), min(w, cx + size + margin + 1)
        y0, y1 = max(0, cy - size - margin), min(h, cy + size + margin + 1)
        if x1 - x0 < self.template.shape[1] + 2 * margin + 1 or y1 - y0 < self.template.shape[0] + 2 * margin + 1:
            return None
        window = cv2.GaussianBlur(frame[y0:y1, x0:x1].astype(np.float32), (0, 0), self.sigma)
        window = window[margin:-margin, margin:-margin]
        mask3 = self.mask3
        if exclude:
            # Lines drawn over the ball in this frame (the aim ray while aiming).
            mask = self.mask * ~self._strips(exclude)
            mask3 = np.dstack([mask.astype(np.float32)] * 3)
        score = cv2.matchTemplate(window, self.template, cv2.TM_SQDIFF_NORMED, mask=mask3)
        score = np.nan_to_num(score, nan=1.0, posinf=1.0)
        # The ball's own pattern (its shading) must be there too: on a white
        # or pale cloth a white ball's colour differs too little from the
        # cloth's for the squared difference alone.
        pattern = cv2.matchTemplate(window, self.template, cv2.TM_CCOEFF_NORMED, mask=mask3)
        pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
        score = np.where(pattern >= PATTERN_MIN, score, 1.0)
        iy, ix = divmod(int(np.argmin(score)), score.shape[1])
        if not (0 < iy < score.shape[0] - 1 and 0 < ix < score.shape[1] - 1):
            # Best at the window's edge: the ball is (partly) outside it, and
            # a part match there is off by pixels. Not found here.
            return float(x0), float(y0), float("inf")
        fy, fx = _sub_pixel(score, iy, ix)
        # The template was cut around the rounded start position; ``offset``
        # (from ``calibrate``) turns the match into the ball's centre.
        x = x0 + margin + ix + fx + self.half + self.offset[0]
        y = y0 + margin + iy + fy + self.half + self.offset[1]
        return float(x), float(y), float(score[iy, ix])

    def limit(self) -> float:
        """Worst match score that still counts as this ball."""
        if not self.scores:
            return LOST_SCORE
        return min(LOST_SCORE, max(SCORE_FLOOR, SCORE_RATIO * float(np.median(self.scores))))

    def start(self, t: float):
        self.t = t

    def update(self, frame: np.ndarray, t: float, heading: tuple[float, float] | None = None,
               max_speed: float | None = None):
        """Follow the ball into ``frame``; returns (x, y) or None once lost.

        When it is not near where it should be (struck hard, or a hit
        changed its speed), a wide window is searched, centred ahead along
        ``heading`` while its speed is not known yet. With ``max_speed``
        (px/s), a match further than the ball could have gone is another
        ball (one of a similar colour, after this one was potted)."""
        if self.lost:
            return None
        found = self.locate(frame, t)
        if found is None or found[2] > LOST_SCORE:
            centre = None
            if heading is not None and np.hypot(*self.vel) < self.r:
                centre = (self.pos[0] + heading[0] * WIDE_REACH * .5 * self.r,
                          self.pos[1] + heading[1] * WIDE_REACH * .5 * self.r)
            wide = self.locate(frame, t, reach=WIDE_REACH * self.r + 2 * np.hypot(*self.vel) * (
                0.0 if self.t is None else max(0.0, t - self.t)), centre=centre)
            if wide is not None and (found is None or wide[2] < found[2]):
                found = wide
        if found is not None and max_speed is not None and self.t is not None:
            dt = max(0.0, t - self.t)
            if np.hypot(found[0] - self.pos[0], found[1] - self.pos[1]) > max_speed * dt * 1.3 + self.r:
                found = None
        if found is None or found[2] > self.limit():
            self.misses += 1
            if self.misses >= LOST_FRAMES:
                self.lost = True
            return None
        x, y, score = found
        self.scores.append(score)
        self.misses = 0
        if self.t is not None and t > self.t:
            dt = t - self.t
            # Smoothed velocity for the next search window only (positions
            # are logged as found).
            vx, vy = (x - self.pos[0]) / dt, (y - self.pos[1]) / dt
            self.vel = (.5 * self.vel[0] + .5 * vx, .5 * self.vel[1] + .5 * vy)
        self.pos, self.t = (x, y), t
        return x, y

    def calibrate(self, frame: np.ndarray, x: float, y: float):
        """Match the template on its own frame once, so a found position is
        the detector's ball centre (the blurred blob's centre can sit a
        fraction of a pixel off it)."""
        self.offset = (0.0, 0.0)
        found = self.locate(frame, 0.0, reach=2)
        if found is not None:
            self.offset = (x - found[0], y - found[1])
            self.scores.append(found[2])


# The rim fit is used only where the rim's colour edge is at least this
# strong (Lab units per quarter pixel); a ball close to the cloth's colour
# (red on red, the cue ball on light grey) keeps the detector's centre.
RIM_CONTRAST = 6.0


def refine_ball_centre(frame: np.ndarray, x: float, y: float, r: float, angles: int = 72):
    """Sub-pixel ball centre fitted on its rim: the strongest colour edge
    along rays from (x, y) between 0.8 and 1.2 radii, then a circle fit
    that drops the worst quarter of the edge points three times (the drop
    shadow, a touching ball). Synthetic tables: 0.11 px typical against the
    detector's whole-pixel 0.5 (worst 0.7 against 2.8 on blue)."""
    h, w = frame.shape[:2]
    reach = int(np.ceil(r * 1.4)) + 2
    x0, y0 = max(0, int(x) - reach), max(0, int(y) - reach)
    x1, y1 = min(w, int(x) + reach + 1), min(h, int(y) + reach + 1)
    if x1 - x0 < 2 * reach or y1 - y0 < 2 * reach:
        return x, y
    lab = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    a = np.linspace(0, 2 * np.pi, angles, endpoint=False)
    step = .25
    radii = np.arange(.7 * r, 1.3 * r, step)
    xs = (x - x0 + np.outer(np.cos(a), radii)).astype(np.float32)
    ys = (y - y0 + np.outer(np.sin(a), radii)).astype(np.float32)
    profile = cv2.remap(lab, xs, ys, cv2.INTER_LINEAR)
    grad = np.linalg.norm(np.diff(profile, axis=1), axis=2)
    mids = (radii[:-1] + radii[1:]) / 2
    inside = (mids >= .8 * r) & (mids <= 1.2 * r)
    g, m = grad[:, inside], mids[inside]
    k = np.argmax(g, axis=1)
    strength = g[np.arange(angles), k]
    if float(np.median(strength)) < RIM_CONTRAST:
        return x, y
    rad = m[k].copy()
    for i in range(angles):
        j = k[i]
        if 0 < j < g.shape[1] - 1:
            p, q, s = g[i, j - 1], g[i, j], g[i, j + 1]
            den = p - 2 * q + s
            if den < 0:
                rad[i] += .5 * (p - s) / den * step
    keep = strength >= np.percentile(strength, 30)
    px, py = (x + np.cos(a) * rad)[keep], (y + np.sin(a) * rad)[keep]
    cx, cy = x, y
    for _ in range(3):
        if len(px) < 12:
            break
        sol, *_ = np.linalg.lstsq(np.column_stack([px, py, np.ones_like(px)]), px ** 2 + py ** 2, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        rr = np.sqrt(max(1e-6, sol[2] + cx ** 2 + cy ** 2))
        res = np.abs(np.hypot(px - cx, py - cy) - rr)
        cut = np.percentile(res, 75)
        px, py = px[res <= cut], py[res <= cut]
    if not np.isfinite(cx) or np.hypot(cx - x, cy - y) > .5 * r:
        return x, y
    return float(cx), float(cy)


def refined(frame: np.ndarray, detection: Detection) -> Detection:
    """``detection`` with every ball (and the cue ball) at its rim-fitted centre."""
    balls = []
    for bx, by, br, kind in detection.balls:
        rx, ry = refine_ball_centre(frame, bx, by, br)
        balls.append((rx, ry, br, kind))
    cue = detection.cue
    if cue is not None:
        near = min(balls, key=lambda b: np.hypot(b[0] - cue[0], b[1] - cue[1]), default=None)
        if near is not None and np.hypot(near[0] - cue[0], near[1] - cue[1]) < cue[2] * .6:
            cue = (near[0], near[1], cue[2])
    return dataclasses.replace(detection, balls=balls, cue=cue)


def ball_signature(frame: np.ndarray, x: float, y: float, r: float) -> dict:
    """Colour of a ball for telling balls apart after the shot: median Lab of
    its disc and the share of white (stripes have much more than solids)."""
    h, w = frame.shape[:2]
    reach = int(np.ceil(r))
    x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 1)
    y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 1)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return {"lab": [0.0, 0.0, 0.0], "white": 0.0}
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = ((xx - x) ** 2 + (yy - y) ** 2 <= (r * .8) ** 2).reshape(-1)
    pix = lab[inside]
    if len(pix) == 0:
        return {"lab": [0.0, 0.0, 0.0], "white": 0.0}
    chroma = np.hypot(pix[:, 1] - 128, pix[:, 2] - 128)
    white = (pix[:, 0] > 190) & (chroma < 18)
    coloured = pix[~white] if (~white).sum() >= 5 else pix
    return {"lab": [round(float(v), 1) for v in np.median(coloured, axis=0)],
            "white": round(float(white.mean()), 3)}


def signature_distance(a: dict, b: dict) -> float:
    return float(np.hypot(np.linalg.norm(np.subtract(a["lab"], b["lab"])), 100 * (a["white"] - b["white"])))


def release_power(history, t_start: float) -> float | None:
    """The slider's reading when it was let go, from (time, percent) readings.

    Readings back at rest after the release are dropped, and so is a snap
    back over a few frames (steep falls between close readings)."""
    pts = [(t, p) for t, p in history if p is not None and t_start - HISTORY_SECONDS <= t < t_start]
    while pts and pts[-1][1] <= SLIDER_REST:
        pts.pop()
    if not pts:
        return None
    i = len(pts) - 1
    while i > 0 and pts[i - 1][1] - pts[i][1] > SNAP_STEP and pts[i][0] - pts[i - 1][0] <= SNAP_GAP:
        i -= 1
    return float(pts[i][1])


def near_pocket(x: float, y: float, table, radius: float, reach: float = 3.0) -> bool:
    """True when (x, y) is within ``reach`` radii of one of the six pocket
    mouths (the table's corners and the middles of its long sides)."""
    tx, ty, w, h = table
    mouths = [(tx, ty), (tx + w / 2, ty), (tx + w, ty), (tx, ty + h), (tx + w / 2, ty + h), (tx + w, ty + h)]
    return min(np.hypot(x - mx, y - my) for mx, my in mouths) <= reach * radius


def _angle(v):
    return None if v is None else round(float(np.degrees(np.arctan2(v[1], v[0]))), 3)


def distinct_balls(balls):
    """Detections closer than 1.5 radii are the same ball (two real balls'
    centres are at least two radii apart): keep the first."""
    kept = []
    for b in balls:
        if all(np.hypot(b[0] - k[0], b[1] - k[1]) >= 1.5 * max(b[2], k[2]) for k in kept):
            kept.append(b)
    return kept


def _balls_with_signatures(frame, detection: Detection, cue_xy=None):
    """Every ball but the cue ball at ``cue_xy`` (None: the cue ball is not on
    the table, so none is left out, whatever the detection named the cue
    ball), with its colour signature."""
    out = []
    for bx, by, br, kind in distinct_balls(detection.balls):
        if cue_xy is not None and np.hypot(bx - cue_xy[0], by - cue_xy[1]) < br * .6:
            continue
        out.append({"x": round(float(bx), 2), "y": round(float(by), 2), **ball_signature(frame, bx, by, br)})
    return out


def match_balls(before: list[dict], after: list[dict], radius: float) -> list[int | None]:
    """For each ball before the shot, the index of the same ball after it (by
    colour; nearer first for a tie), or None if it was potted. Greedy on the
    closest colours."""
    pairs = []
    for i, b in enumerate(before):
        for j, a in enumerate(after):
            colour = signature_distance(b, a)
            moved = float(np.hypot(b["x"] - a["x"], b["y"] - a["y"])) / max(radius, 1.0)
            pairs.append((colour + .5 * min(moved, 20.0), i, j))
    pairs.sort()
    result: list[int | None] = [None] * len(before)
    used = set()
    for cost, i, j in pairs:
        if result[i] is not None or j in used or cost > 60:
            continue
        result[i] = j
        used.add(j)
    return result


@dataclass
class ShotLog:
    index: int
    power: float | None
    data: dict = field(default_factory=dict)
    path: Path | None = None


class ShotRecorder:
    """Feed it frames (``feed``); it returns a ``ShotLog`` when a shot ends."""

    def __init__(self, out_dir: Path | None = None, full_detect=detect, save_frames: bool = True):
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.full_detect = full_detect
        self.save_frames = save_frames and out_dir is not None
        self.tracker = ShotTracker(full_detect=full_detect)
        self.history: deque = deque()
        self.recent: deque = deque(maxlen=12)     # (t, frame, detection) of the last aiming frames
        self.shots = 0
        self.cue_retry_at = -1e9
        self.t0: float | None = None
        self.notes: list[str] = []
        self._reset_rest()
        self.state = "aiming"

    # --- aiming ----------------------------------------------------------------

    def _reset_rest(self):
        self.rest_frame = None
        self.rest: Detection | None = None   # a still table: every ball, before the shot
        self.cue: tuple[float, float, float] | None = None
        self.cue_follower: BallFollower | None = None
        self.last_aim: Detection | None = None

    def _set_still(self, frame: np.ndarray, detection: Detection, cue_hint=None):
        """The table at rest (before the next shot) from a full detection."""
        self._reset_rest()
        if detection.table is None or not detection.balls:
            return
        detection = refined(frame, detection)
        self.rest_frame, self.rest = frame, detection
        self.signatures = [ball_signature(frame, bx, by, br) for bx, by, br, _ in detection.balls]
        # The game's lines drawn over balls in this frame (if its aim showed):
        # the aim ray from the cue ball, the object line from the struck ball.
        self.still_lines = {}
        if detection.aim_dir is not None and detection.cue is not None:
            self.still_lines["cue"] = (detection.cue[:2], tuple(detection.aim_dir))
            if detection.object_dir is not None and detection.target is not None:
                self.still_lines["object"] = (detection.target[:2], tuple(detection.object_dir))
        self.change_levels = [self._change_level(frame, bx, by, br) for bx, by, br, _ in detection.balls]
        self._choose_cue(cue_hint if cue_hint is not None else detection.cue)

    def _choose_cue(self, hint):
        """The cue ball among the still table's balls: the one at ``hint`` if
        it is white there, else the whitest. (With the struck ball nearly
        touching the cue ball, the aiming frame's detection can name another
        ball; on a white cloth, so can a still frame's.)"""
        balls = self.rest.balls
        whiteness = [s["white"] for s in self.signatures]
        best = int(np.argmax(whiteness))
        pick = best
        if hint is not None:
            near = min(range(len(balls)), key=lambda i: np.hypot(balls[i][0] - hint[0], balls[i][1] - hint[1]))
            if (np.hypot(balls[near][0] - hint[0], balls[near][1] - hint[1]) <= balls[near][2] * .6
                    and whiteness[near] >= CUE_WHITE and whiteness[near] >= whiteness[best] - CUE_WHITE_MARGIN):
                pick = near
        if whiteness[pick] < CUE_WHITE:
            # No white ball on this still table (the cue ball was potted and
            # not put back yet): the cue ball is unknown for now.
            self.cue = None
            self.cue_follower = None
            return
        bx, by, br, _ = balls[pick]
        if self.cue is not None and np.hypot(self.cue[0] - bx, self.cue[1] - by) < br * .5:
            return
        self.cue = (float(bx), float(by), float(br))
        follower = BallFollower(self.rest_frame, bx, by, br, exclude=self._lines_at(bx, by, br))
        follower.calibrate(self.rest_frame, bx, by)
        self.cue_follower = follower if follower.template is not None else None

    def _cue_level(self) -> float:
        balls = self.rest.balls
        i = min(range(len(balls)), key=lambda k: np.hypot(balls[k][0] - self.cue[0], balls[k][1] - self.cue[1]))
        return self.change_levels[i]

    def _lines_at(self, x: float, y: float, r: float):
        """Directions of the game's lines drawn from the ball at (x, y) in the
        still frame."""
        return [direction for (lx, ly), direction in getattr(self, "still_lines", {}).values()
                if np.hypot(lx - x, ly - y) < r * .6]

    @staticmethod
    def _has_ball_at(detection: Detection, xy) -> bool:
        return any(np.hypot(bx - xy[0], by - xy[1]) <= br * .6 for bx, by, br, _ in detection.balls)

    def _when(self, t: float) -> str:
        """Time since the first frame, for notes."""
        seconds = int(t - self.t0)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def feed(self, frame: np.ndarray, t: float) -> ShotLog | None:
        if self.t0 is None:
            self.t0 = t
        if self.state == "aiming":
            return self._aiming(frame, t)
        return self._moving(frame, t)

    @staticmethod
    def _change_level(frame: np.ndarray, x: float, y: float, r: float) -> float:
        """How much a pixel inside the ball must change for the ball to have
        left: half its typical difference from the cloth around it (a blue
        ball on a blue table differs by less than the usual 40 levels)."""
        h, w = frame.shape[:2]
        reach = int(np.ceil(r * 1.9))
        x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 1)
        y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 1)
        patch = frame[y0:y1, x0:x1].astype(np.int16)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - x, yy - y)
        ring = (d >= r * 1.4) & (d <= r * 1.8)
        inner = d <= r * .7
        if ring.sum() < 10 or inner.sum() < 10:
            return BALL_CHANGE
        cloth = np.median(patch[ring], axis=0)
        diff = np.abs(patch[inner] - cloth).max(axis=1)
        return float(np.clip(.5 * np.median(diff), 12, BALL_CHANGE))

    def _ball_left(self, frame: np.ndarray) -> int | None:
        """The still table's ball (index) other than the cue ball that left its
        spot: most of its inner disc changed (a stick across it changes a thin
        stripe; the cue ball rolling by cannot reach it). None if none did."""
        for index, (bx, by, br, _) in enumerate(self.rest.balls):
            if self.cue is not None and np.hypot(bx - self.cue[0], by - self.cue[1]) < br * .5:
                continue
            if self._spot_left(frame, bx, by, br, self.change_levels[index]):
                return index
        return None

    def _spot_left(self, frame: np.ndarray, bx: float, by: float, br: float, level: float) -> bool:
        """True when the ball is gone from its spot: most of the pixels where
        it looked unlike the cloth (in the still frame) changed by more than
        ``level``. (A striped ball's white half on a white cloth does not
        change when it leaves; a stick across the ball changes a stripe.)"""
        h, w = frame.shape[:2]
        reach = int(np.ceil(br * 1.9))
        x, y = int(bx), int(by)
        x0, x1, y0, y1 = max(0, x - reach), min(w, x + reach + 1), max(0, y - reach), min(h, y + reach + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return False
        then = self.rest_frame[y0:y1, x0:x1].astype(np.int16)
        now = frame[y0:y1, x0:x1].astype(np.int16)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - bx, yy - by)
        disc = d <= br * .7
        ring = (d >= br * 1.4) & (d <= br * 1.8)
        if disc.sum() < 10:
            return False
        own = disc
        if ring.sum() >= 10:
            cloth = np.median(then[ring], axis=0)
            unlike = (np.abs(then - cloth).max(axis=2) > max(level, 12)) & disc
            if unlike.sum() >= .15 * disc.sum():
                own = unlike
        changed = (np.abs(now - then).max(axis=2) > level) & own
        return bool(changed.sum() >= .8 * own.sum())

    def _aiming(self, frame: np.ndarray, t: float):
        # The shot starts when the cue ball moves (checked before the
        # tracker, which falls back to a slow full detection once balls
        # move), or, in case the cue ball was misnamed, any ball leaves its
        # spot. Without a known cue ball no shot can be logged.
        if self.rest is not None and self.cue is not None:
            if self.cue_follower is not None:
                aim_now = self.recent[-1][2].aim_dir if self.recent else None
                found = self.cue_follower.locate(frame, t, reach=self.cue[2] * 1.5,
                                                 exclude=[aim_now] if aim_now is not None else ())
                if found is not None and found[2] <= self.cue_follower.limit():
                    if np.hypot(found[0] - self.cue[0], found[1] - self.cue[1]) > START_MOVE * self.cue[2]:
                        return self._start(frame, t)
                elif found is not None and self._spot_left(frame, *self.cue, self._cue_level()):
                    # Gone from its spot at once (a hard shot moves it further
                    # than the search reach in one frame).
                    return self._start(frame, t)
            if self._ball_left(frame) is not None:
                return self._start(frame, t)
        if self.rest is not None and self.cue is None and t - self.cue_retry_at >= CUE_RETRY_SECONDS:
            # Look at the whole table again until the cue ball is back.
            self.cue_retry_at = t
            self.tracker.reset()
        d = self.tracker.detect(frame, now=t)
        if (self.rest is None or self.cue is None) and self.tracker.last_was_full and self.tracker.base is not None:
            self._set_still(self.tracker.reference, self.tracker.base)
        if d.aim_dir is not None and d.cue is not None:
            self.last_aim = d
            if self.rest is not None and not self._has_ball_at(self.rest, d.cue):
                # The aim starts from a ball the still table does not have:
                # the cue ball was put back (after a scratch, ball in hand).
                base = self.tracker.base
                if base is not None and self._has_ball_at(base, d.cue):
                    self._set_still(self.tracker.reference, base, cue_hint=d.cue)
                else:
                    # Detect the whole table again on the next frame.
                    self._reset_rest()
                    self.tracker.reset()
            elif self.rest is not None:
                # The game's aim line starts at the cue ball.
                self._choose_cue(d.cue)
        self.history.append((t, d.power_percent))
        while self.history and self.history[0][0] < t - HISTORY_SECONDS:
            self.history.popleft()
        self.recent.append((t, frame, d))
        return None

    # --- the shot --------------------------------------------------------------

    def _start(self, frame: np.ndarray, t: float):
        power = release_power(self.history, t)
        # No power set: the opponent's shot (no slider), or the cue ball
        # moved by hand (ball in hand), which is told apart at the end: then
        # no other ball moves.
        note = "" if power is not None else "no power slider reading (opponent's turn?)"
        rest = self.rest
        r = float(np.median([b[2] for b in rest.balls]))
        self.shot = {
            "t_start": t, "power": power, "power_note": note,
            "power_history": [[round(ht - t, 4), None if p is None else round(float(p), 2)]
                              for ht, p in self.history],
            "rest": rest, "rest_frame": self.rest_frame, "aim": None,
            "release_frame": self.recent[-1][1] if self.recent else self.rest_frame,
            "radius": r, "cue_track": [], "object_track": [],
            "cue_lost_at": None, "object_lost_at": None, "frames": 0,
        }
        cue = self.cue
        self.shot["cue"] = cue
        if self.cue_follower is not None:
            self.cue_follower.start(self.recent[-1][0] if self.recent else t)
        self.shot["cue_track"].append([0.0, round(float(cue[0]), 2), round(float(cue[1]), 2)])
        # The aim the game showed, if it was read from this cue ball (with the
        # struck ball nearly touching the cue ball, the aiming frame's
        # detection can read a ray from another ball).
        aim = self.last_aim
        if aim is not None and np.hypot(aim.cue[0] - cue[0], aim.cue[1] - cue[1]) > cue[2] * .6:
            aim = None
            self.shot["power_note"] = (self.shot["power_note"] + "; " if self.shot["power_note"] else "") +                 "aim not read from the cue ball"
        self.shot["aim"] = aim
        # The struck ball is found from the motion (the first other ball to
        # leave its spot), then followed.
        self.object_follower = None
        self.struck = None
        # The fastest the cue ball has moved this shot (px/s): no ball it
        # hits can go faster.
        self.cue_top_speed = 0.0
        self.prev_t = self.recent[-1][0] if self.recent else t
        self.headings = {}
        if aim is not None and aim.aim_dir is not None:
            self.headings["cue"] = tuple(aim.aim_dir)
        self.last_thumb = None
        self.still_since = None
        self.state = "moving"
        return self._moving(frame, t)

    def _cue_step_speed(self, t: float) -> float:
        """The cue ball's speed over its last tracked step (px/s)."""
        track = self.shot["cue_track"]
        if len(track) < 2 or track[-1][0] <= track[-2][0]:
            return 0.0
        return float(np.hypot(track[-1][1] - track[-2][1], track[-1][2] - track[-2][2]) / (track[-1][0] - track[-2][0]))

    def _could_be_struck(self, bx: float, by: float, t: float) -> bool:
        """Whether the ball at (bx, by) that just left its spot can be the one
        the cue ball hit first: the cue ball is next to it now, its path came
        within reach of it, and it was not deflected before somewhere else
        than at a cushion (then it hit a ball the detection missed)."""
        shot = self.shot
        r = shot["radius"]
        cf = self.cue_follower
        cx, cy = cf.pos if cf is not None else shot["cue"][:2]
        speed = float(np.hypot(*cf.vel)) if cf is not None else 0.0
        if np.hypot(bx - cx, by - cy) > STRUCK_NEAR * r + 2 * speed * max(0.0, t - self.prev_t):
            return False
        path = np.array([p[1:] for p in shot["cue_track"]] + [[cx, cy]], np.float64)
        if len(path) >= 2:
            a, b = path[:-1], path[1:]
            ab = b - a
            length2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
            u = np.clip(((np.array([bx, by]) - a) * ab).sum(axis=1) / length2, 0, 1)
            closest = a + ab * u[:, None]
            if float(np.min(np.hypot(closest[:, 0] - bx, closest[:, 1] - by))) > STRUCK_PATH * r:
                return False
            # The first clear turn of the path (a collision).
            steps = np.diff(path, axis=0)
            moving = np.hypot(steps[:, 0], steps[:, 1]) > .5
            heading = None
            tx, ty, tw, th = shot["rest"].table
            for k in range(len(steps)):
                if not moving[k]:
                    continue
                v = steps[k] / np.hypot(*steps[k])
                if heading is not None and float(v @ heading) < np.cos(np.radians(15)):
                    px, py = path[k]
                    at_cushion = min(px - tx, tx + tw - px, py - ty, ty + th - py) < 1.6 * r
                    if not at_cushion and np.hypot(px - bx, py - by) > STRUCK_NEAR * r:
                        return False
                    break
                heading = v
        return True

    def _table_thumb(self, frame):
        x, y, w, h = self.shot["rest"].table
        return change_thumbnail(frame[max(0, y):y + h, max(0, x):x + w])

    def _moving(self, frame: np.ndarray, t: float):
        shot = self.shot
        rel = round(t - shot["t_start"], 4)
        shot["frames"] += 1
        for name, follower in (("cue", self.cue_follower), ("object", getattr(self, "object_follower", None))):
            if follower is None or follower.lost:
                continue
            limit = None if name == "cue" else self.cue_top_speed
            pos = follower.update(frame, t, heading=self.headings.get(name), max_speed=limit)
            if name == "cue" and pos is not None:
                self.cue_top_speed = max(self.cue_top_speed, float(np.hypot(*follower.vel)), self._cue_step_speed(t))
            if pos is not None:
                shot[f"{name}_track"].append([rel, round(pos[0], 2), round(pos[1], 2)])
            elif follower.lost and shot[f"{name}_lost_at"] is None:
                shot[f"{name}_lost_at"] = rel
        if self.struck is None:
            index = self._ball_left(frame)
            if index is not None:
                bx, by, br, _ = shot["rest"].balls[index]
                if not self._could_be_struck(bx, by, t):
                    # A ball moving far from the cue ball was not struck by it
                    # (the struck one was missed by the detection, e.g. one
                    # touching the cue ball). The struck ball stays unknown.
                    self.struck = (float("nan"), float("nan"))
                    index = None
            if index is not None:
                follower = BallFollower(self.rest_frame, bx, by, shot["radius"],
                                        exclude=self._lines_at(bx, by, shot["radius"]))
                follower.calibrate(self.rest_frame, bx, by)
                self.struck = (bx, by)
                if follower.template is not None:
                    follower.start(self.prev_t)
                    self.object_follower = follower
                    shot["object_track"].append([round(self.prev_t - shot["t_start"], 4), round(bx, 2), round(by, 2)])
                    # It leaves roughly along the line from the cue ball.
                    cx, cy = self.cue_follower.pos if self.cue_follower is not None else shot["cue"][:2]
                    n = float(np.hypot(bx - cx, by - cy))
                    if n > 1e-6:
                        self.headings["object"] = ((bx - cx) / n, (by - cy) / n)
                    found_at = follower.update(frame, t, heading=self.headings.get("object"),
                                               max_speed=self.cue_top_speed)
                    if found_at is not None:
                        shot["object_track"].append([rel, round(found_at[0], 2), round(found_at[1], 2)])
        self.prev_t = t
        thumb = self._table_thumb(frame)
        if self.last_thumb is not None and not screen_changed(thumb, self.last_thumb):
            if self.still_since is None:
                self.still_since = t
        else:
            self.still_since = None
        self.last_thumb = thumb
        done = self.still_since is not None and t - self.still_since >= STILL_SECONDS
        if done or t - shot["t_start"] >= MAX_SHOT_SECONDS:
            return self._finish(frame, t, timed_out=not done)
        return None

    def _finish(self, frame: np.ndarray, t: float, timed_out: bool):
        shot = self.shot
        rest, aim = shot["rest"], shot["aim"]
        after = self.full_detect(frame)
        if after.balls:
            after = refined(frame, after)
        r = shot["radius"]
        cue = shot["cue"]
        # A ball lost by its follower next to a pocket went in. (The full
        # detection of the table afterwards always names some ball the cue
        # ball, also when the real one was potted.) Where the cue ball
        # stopped is the end of its track (the detection's otherwise).
        follow = self.cue_follower
        cue_potted = bool(follow is not None and follow.lost and near_pocket(*follow.pos, rest.table, r))
        cue_after = None
        if follow is not None and not follow.lost:
            cue_after = [round(follow.pos[0], 2), round(follow.pos[1], 2)]
        elif after.cue is not None and not cue_potted:
            cue_after = [round(float(after.cue[0]), 2), round(float(after.cue[1]), 2)]
        before_balls = _balls_with_signatures(shot["rest_frame"], rest, cue[:2])
        after_balls = _balls_with_signatures(frame, after, cue_after) if after.balls else []
        matched = match_balls(before_balls, after_balls, r)
        for b, m in zip(before_balls, matched):
            b["after"] = m
        object_index = None
        if self.struck is not None:
            near = [(np.hypot(b["x"] - self.struck[0], b["y"] - self.struck[1]), i) for i, b in enumerate(before_balls)]
            if near and min(near)[0] < r:
                object_index = min(near)[1]
        aimed_at = None
        if aim is not None and aim.target is not None:
            tx, ty = aim.target[:2]
            near = [(np.hypot(b["x"] - tx, b["y"] - ty), i) for i, b in enumerate(before_balls)]
            if near and min(near)[0] < r:
                aimed_at = min(near)[1]
        if shot["power"] is None and len(shot["cue_track"]) <= 2 and not cue_potted:
            # No power, and the cue ball could not be followed at all: the
            # whole table changed at once (a new rack, another screen), not a
            # shot.
            self.notes.append(f"{self._when(shot['t_start'])}: the table changed at once (new rack or another screen?)")
            self.state = "aiming"
            self.history.clear()
            self.recent.clear()
            self.tracker.reset()
            self._set_still(frame, after)
            self.shot = None
            return None
        if shot["power"] is None and not any(
                m is None or np.hypot(after_balls[m]["x"] - b["x"], after_balls[m]["y"] - b["y"]) > r * .5
                for b, m in zip(before_balls, matched)):
            # Only the cue ball moved and no power was set: it was moved by
            # hand (ball in hand). Not a shot.
            self.notes.append(f"{self._when(shot['t_start'])}: cue ball moved without a shot (ball in hand?)")
            self.state = "aiming"
            self.history.clear()
            self.recent.clear()
            self.tracker.reset()
            self._set_still(frame, after, cue_hint=cue_after)
            self.shot = None
            return None
        objf = getattr(self, "object_follower", None)
        object_potted = None if objf is None else bool(objf.lost and near_pocket(*objf.pos, rest.table, r))
        self.shots += 1
        data = {
            "shot": self.shots,
            "time": datetime.now().isoformat(timespec="seconds"),
            "power": None if shot["power"] is None else round(shot["power"], 2),
            "power_note": shot["power_note"],
            "power_history": shot["power_history"],
            "table": list(rest.table),
            "ball_radius": round(r, 2),
            "before": {
                "cue": [round(float(cue[0]), 2), round(float(cue[1]), 2)],
                "balls": before_balls,
                "aim_degrees": None if aim is None else _angle(aim.aim_dir),
                "ghost": None if aim is None or aim.ghost is None else list(aim.ghost[:2]),
                "object_ball": object_index,
                "aimed_at_ball": aimed_at,
                "object_dir_degrees": None if aim is None else _angle(aim.object_dir),
                "cue_dir_degrees": None if aim is None else _angle(aim.cue_dir),
                "pot_call": None if aim is None else aim.pot,
            },
            "cue_track": shot["cue_track"],
            "object_track": shot["object_track"],
            "cue_lost_at": shot["cue_lost_at"],
            "object_lost_at": shot["object_lost_at"],
            "after": {"cue": cue_after, "balls": after_balls},
            "potted": [i for i, m in enumerate(matched) if m is None],
            "cue_potted": cue_potted,
            "object_potted": object_potted,
            "t_start": round(shot["t_start"], 4),
            "duration": round(t - shot["t_start"], 3),
            "frames_while_moving": shot["frames"],
            "timed_out": timed_out,
            "spin": "not read",
        }
        log = ShotLog(self.shots, data["power"], data)
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"shot_{self.shots:03d}"
            if self.save_frames:
                cv2.imwrite(str(self.out_dir / f"{stem}_release.png"), shot["release_frame"])
                cv2.imwrite(str(self.out_dir / f"{stem}_rest.png"), frame)
                data["frames"] = {"release": f"{stem}_release.png", "rest": f"{stem}_rest.png"}
            log.path = self.out_dir / f"{stem}.json"
            log.path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        # Aiming again from the new rest position.
        self.state = "aiming"
        self.history.clear()
        self.recent.clear()
        self.tracker.reset()
        # The table as it stopped is the next shot's still table.
        self._set_still(frame, after, cue_hint=cue_after)
        self.shot = None
        return log


# --- capturing the mirror ------------------------------------------------------

def find_mirror(title: str | None = None) -> int | None:
    """The mirror window: one whose title contains ``title``, else a known
    mirror title (``window_capture.MIRROR_TITLES``)."""
    import ctypes
    import window_capture as wc
    if not hasattr(wc, "_user32"):
        return None
    user32 = wc._user32
    hwnd = user32.GetTopWindow(None)
    fallback = None
    while hwnd:
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            name = wc._title(hwnd)
            if title and title.lower() in name.lower():
                return int(hwnd)
            if name in wc.MIRROR_TITLES or name.startswith(wc.MIRROR_TITLE_PREFIXES):
                fallback = fallback or int(hwnd)
        hwnd = user32.GetWindow(hwnd, 2)
    del ctypes
    return fallback


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(HERE / "outputs" / "shots"),
                        help="folder for the logs (a dated subfolder is made)")
    parser.add_argument("--title", default=None, help="part of the mirror window's title")
    parser.add_argument("--no-frames", action="store_true", help="do not save the release/rest PNGs")
    args = parser.parse_args(argv)
    import window_capture as wc
    hwnd = find_mirror(args.title)
    if hwnd is None:
        print("No mirror window found. Start the AirPlay receiver (or pass --title).")
        return 1
    print(f"Recording shots from \"{wc._title(hwnd)}\". Play normally; Ctrl+C to stop.")
    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    recorder = ShotRecorder(out, save_frames=not args.no_frames)
    capture = wc.WindowCapture(hwnd)
    frames = 0
    started = time.monotonic()
    try:
        while True:
            client = wc._client_rect(hwnd)
            if client is None:
                print("The mirror window is gone.")
                break
            region = {"left": client[0], "top": client[1], "width": client[2], "height": client[3]}
            frame = capture.grab(region)
            if frame is None:
                time.sleep(.05)
                continue
            frames += 1
            log = recorder.feed(frame, time.monotonic())
            for note in recorder.notes:
                print("  " + note)
            recorder.notes.clear()
            if log is not None:
                d = log.data
                power = "no power read" if d["power"] is None else f"power {d['power']:.1f}%"
                print(f"shot {log.index}: {power}, {len(d['cue_track'])} cue-ball positions, "
                      f"{len(d['potted'])} potted{', cue ball potted' if d['cue_potted'] else ''} -> {log.path}")
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()
    seconds = time.monotonic() - started
    print(f"{recorder.shots} shots logged in {out} ({frames / max(seconds, 1e-6):.0f} frames/s analysed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
