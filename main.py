"""Pool Practice Overlay: read-only screen analysis and a transparent visual guide."""
from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPoint, QPointF, QRect, QRectF, QSettings, QObject, Signal, QRunnable, QThreadPool
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QImage, QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QFileDialog
from pool_physics import Ball as PhysicsBall, simulate_shot


@dataclass
class Detection:
    table: tuple[int, int, int, int] | None = None
    balls: list[tuple[int, int, int, str]] | None = None  # x,y,r,kind
    cue: tuple[int, int, int] | None = None
    aim_segments: list[tuple[tuple[int, int], tuple[int, int]]] | None = None
    cue_segments: list[tuple[tuple[int, int], tuple[int, int]]] | None = None
    object_segments: list[tuple[tuple[int, int], tuple[int, int]]] | None = None
    target: tuple[int, int, int] | None = None
    # Miniclip's hollow ghost-ball reticle: where the cue ball sits at impact.
    ghost: tuple[int, int, int] | None = None
    # Balls that would stop a predicted path before it reaches a cushion/pocket.
    blockers: list[tuple[int, int, int]] | None = None
    # Unit aim direction (cue ball toward the ghost ball), not pixel-rounded.
    aim_dir: tuple[float, float] | None = None
    # Unit directions the object ball and cue ball leave the impact in.
    object_dir: tuple[float, float] | None = None
    cue_dir: tuple[float, float] | None = None
    note: str = "Guide is off"


class CueTracker:
    """Keep cue-ball identity stable across brief visual occlusions."""
    def __init__(self):
        self.last: tuple[float, float, int] | None = None
        self.last_seen = 0.0
        self.geometry: tuple[int, int, int, int] | None = None

    def reset(self):
        self.last = None
        self.last_seen = 0.0
        self.geometry = None

    def update(self, detection: Detection, geometry: QRect) -> Detection:
        rect = geometry.getRect()
        if self.geometry != rect:
            self.reset()
            self.geometry = rect
        now = time.monotonic()
        if detection.cue:
            current = detection.cue
            if self.last:
                distance = float(np.hypot(current[0] - self.last[0], current[1] - self.last[1]))
                if distance <= max(12, current[2] * 2.0):
                    current = (round(self.last[0] * .3 + current[0] * .7),
                               round(self.last[1] * .3 + current[1] * .7), current[2])
            self.last = current
            self.last_seen = now
            detection.cue = current
            detection.balls = [(current[0], current[1], current[2], "cue") if kind == "cue"
                               else (bx, by, br, kind)
                               for bx, by, br, kind in detection.balls or []]
        elif self.last and now - self.last_seen <= .30:
            detection.cue = self.last
            detection.balls = [(bx, by, br, kind) for bx, by, br, kind in detection.balls or []
                               if kind != "cue"]
            detection.balls.append((*self.last, "cue"))
            detection.note = "Cue briefly obscured · holding its last position"
        else:
            self.last = None
        return detection


class AnalysisSignals(QObject):
    completed = Signal(object, object)
    failed = Signal(str)


class AnalysisTask(QRunnable):
    """Run OpenCV away from the window thread to keep the controls smooth."""
    def __init__(self, frame: np.ndarray, geometry: QRect):
        super().__init__()
        self.frame = frame
        self.geometry = QRect(geometry)
        self.signals = AnalysisSignals()

    def run(self):
        try:
            self.signals.completed.emit(detect(self.frame), self.geometry)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


def white_guide_mask(roi: np.ndarray, balls: list[tuple[int, int, int, str]],
                     ball_scale: float = 1.18) -> np.ndarray:
    """Pixels of Miniclip's white guide graphics, with ball faces removed.

    Miniclip draws its aim ray, ghost-ball reticle, object-ball line and
    cue-deflection line in near-white. Ball faces (cue ball, stripes, number
    badges) are white too, so they are masked out.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Measured on both bundled frames: guide strokes have saturation <= ~60,
    # while even the brightly lit, washed-out felt centre stays >= ~95. The
    # older <= 115 limit let pale felt inside the hollow reticle through.
    mask = np.where((hsv[:, :, 2] >= 165) & (hsv[:, :, 1] <= 70), 255, 0).astype(np.uint8)
    for bx, by, br, _ in balls:
        cv2.circle(mask, (int(bx), int(by)), int(br * ball_scale), 0, thickness=-1)
    return mask


def _ring_kernel(radius: float) -> np.ndarray:
    """Correlation kernel that rewards a thin white ring with a hollow centre."""
    size = int(np.ceil(radius * 1.5)) + 2
    kernel = np.zeros((2 * size + 1, 2 * size + 1), np.float32)
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    for ring_radius, weight in ((radius, 1.0), (radius * .5, -1.0), (radius * 1.45, -1.0)):
        points = {(int(round(size + ring_radius * np.cos(a))), int(round(size + ring_radius * np.sin(a))))
                  for a in angles}
        for x, y in points:
            kernel[y, x] += weight / len(points)
    return kernel


def find_ghost_ball(white: np.ndarray, cue: tuple[int, int, int], direction: tuple[float, float] | None,
                    radius: float) -> tuple[float, float, float, float] | None:
    """Locate Miniclip's hollow ghost-ball reticle along the rough aim ray.

    The reticle centre is where the game places the cue ball at impact, so it
    pins both the true aim angle and the contact point. Reading it directly is
    far more precise than intersecting a slightly-off ray with a ball whose
    radius is only known to a pixel or two (3 px of error there turned the
    object-ball path by ~5°).
    """
    cx, cy, cr = cue
    h, w = white.shape[:2]
    pad = int(np.ceil(radius * 1.6)) + 4
    if direction is None:
        # No visible ray (target right next to the cue ball): look for the
        # ring all around the cue ball instead, within a few radii.
        reach = radius * 5 + pad
        x0, x1 = max(0, int(cx - reach)), min(w, int(cx + reach) + 1)
        y0, y1 = max(0, int(cy - reach)), min(h, int(cy + reach) + 1)
    else:
        ux, uy = direction
        # Restrict the search to a cone around the rough ray and to the
        # region in front of the cue ball.
        far = float(np.hypot(w, h))
        ends = [(cx + ux * t + s * uy * t * .07, cy + uy * t - s * ux * t * .07)
                for t in (0, far) for s in (-1, 1)]
        x0 = max(0, int(min(p[0] for p in ends)) - pad)
        x1 = min(w, int(max(p[0] for p in ends)) + pad)
        y0 = max(0, int(min(p[1] for p in ends)) - pad)
        y1 = min(h, int(max(p[1] for p in ends)) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = white[y0:y1, x0:x1].astype(np.float32) / 255.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    if direction is None:
        dist = np.hypot(xx - cx, yy - cy)
        corridor = (dist >= cr * 1.6) & (dist <= radius * 5)
        threshold = .35
    else:
        along = (xx - cx) * ux + (yy - cy) * uy
        across = np.abs((xx - cx) * uy - (yy - cy) * ux)
        # A ghost ball can sit closer than two radii when the target is
        # nearly touching the cue ball.
        corridor = (along > cr * 1.6) & (across <= np.maximum(4.0, along * .06))
        threshold = .25
    if not corridor.any():
        return None
    best = None
    for ring_radius in np.arange(radius * .78, radius * 1.16 + .01, 1.0):
        score = cv2.filter2D(crop, -1, _ring_kernel(float(ring_radius)), borderType=cv2.BORDER_CONSTANT)
        score[~corridor] = -1
        index = int(np.argmax(score))
        value = float(score.flat[index])
        if best is None or value > best[0]:
            iy, ix = divmod(index, score.shape[1])
            # Sub-pixel peak: a parabola through the neighbours in x and in y.
            # A whole-pixel centre was ~1 degree of aim on a short shot.
            fx = fy = 0.0
            if 0 < ix < score.shape[1] - 1:
                a, b, c = score[iy, ix - 1], score[iy, ix], score[iy, ix + 1]
                if a > -1 and c > -1 and (a - 2 * b + c) < 0:
                    fx = float(np.clip(.5 * (a - c) / (a - 2 * b + c), -.5, .5))
            if 0 < iy < score.shape[0] - 1:
                a, b, c = score[iy - 1, ix], score[iy, ix], score[iy + 1, ix]
                if a > -1 and c > -1 and (a - 2 * b + c) < 0:
                    fy = float(np.clip(.5 * (a - c) / (a - 2 * b + c), -.5, .5))
            best = (value, x0 + ix + fx, y0 + iy + fy, float(ring_radius))
    if best is None or best[0] < threshold:
        return None
    # (x, y, ring radius, ring score)
    return float(best[1]), float(best[2]), best[3], float(best[0])


def ray_direction_to_ghost(white: np.ndarray, cue: tuple[float, float, float], ghost: tuple[float, float],
                           radius: float) -> tuple[float, float] | None:
    """Aim direction fitted on the game's ray pixels.

    The ray is drawn exactly along the aim, from the cue-ball centre into the
    ghost ring's centre. Both centres carry some error (a pixel on the cue
    ball is a degree on a short shot), so the angle comes from a line fit on
    the ray's own pixels between the cue ball's edge and the ghost centre,
    skipping the ring stroke. Returns the unit direction from cue to ghost,
    or None if too little ray is visible.
    """
    cx, cy, cr = cue
    gx, gy = ghost
    dx, dy = gx - cx, gy - cy
    length = float(np.hypot(dx, dy))
    if length < cr * 2:
        return None
    ux, uy = dx / length, dy / length
    h, w = white.shape[:2]
    x0, x1 = max(0, int(min(cx, gx) - 4)), min(w, int(max(cx, gx) + 5))
    y0, y1 = max(0, int(min(cy, gy) - 4)), min(h, int(max(cy, gy) + 5))
    ys, xs = np.nonzero(white[y0:y1, x0:x1])
    if xs.size == 0:
        return None
    px, py = xs + x0 - cx, ys + y0 - cy           # relative to the cue centre
    along = px * ux + py * uy
    across = np.abs(px * uy - py * ux)
    to_ghost = np.hypot(px - dx, py - dy)
    ring_band = np.abs(to_ghost - radius) <= max(3.0, radius * .25)
    keep = (along >= cr * 1.3) & (along <= length - 1.5) & (across <= 3.0) & ~ring_band
    if int(keep.sum()) < 12:
        return None
    points = np.column_stack((px[keep], py[keep])).astype(np.float64)
    if float(np.ptp(points @ np.array([ux, uy]))) < radius * .5:
        return None
    # The ghost centre is located to a fraction of a pixel and lies on the
    # ray, so it is weighted in heavily: on a short shot only ~15 px of ray
    # are visible and a free line fit over that span is good to ~1 degree.
    anchor = np.repeat([[dx, dy]], max(10, len(points) // 2), axis=0)
    points = np.vstack([points, anchor])
    centre = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - centre).T))
    vx, vy = vectors[:, -1]
    if vx * ux + vy * uy < 0:
        vx, vy = -vx, -vy
    # Only a small correction is expected; anything larger is not the ray.
    if vx * ux + vy * uy < np.cos(np.radians(3)):
        return None
    return float(vx), float(vy)


def game_line_direction(white: np.ndarray, origin: tuple[float, float], expected: tuple[float, float],
                        radius: float, cone_degrees: float = 16.0
                        ) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Read a short white game line leaving ``origin``: (direction, anchor).

    Used for Miniclip's object-ball line (starting at the target centre) and
    its cue-deflection line (starting at the ghost-ball centre). Only pixels in
    a cone around the physics estimate are considered, so the incoming aim ray
    and the reticle ring are ignored. The line is fitted on its own pixels
    (principal axis), not forced through ``origin``: a ball centre from Hough
    wanders by a few pixels, and forcing the line through it swung the angle by
    up to 3°. ``anchor`` is ``origin`` projected onto the fitted line.
    Returns None when no clear line exists.
    """
    ox, oy = origin
    ex, ey = expected
    norm = float(np.hypot(ex, ey))
    if norm < 1e-6:
        return None
    ex, ey = ex / norm, ey / norm
    reach = radius * 9
    h, w = white.shape[:2]
    x0, x1 = max(0, int(ox - reach)), min(w, int(ox + reach) + 1)
    y0, y1 = max(0, int(oy - reach)), min(h, int(oy + reach) + 1)
    ys, xs = np.nonzero(white[y0:y1, x0:x1])
    if xs.size == 0:
        return None
    dx, dy = xs + x0 - ox, ys + y0 - oy
    dist = np.hypot(dx, dy)
    keep = (dist >= radius * 1.35) & (dist <= reach)
    keep &= (dx * ex + dy * ey) >= dist * np.cos(np.radians(cone_degrees))
    if int(keep.sum()) < max(12, int(radius)):
        return None
    if float(dist[keep].max() - dist[keep].min()) < radius * 1.5:
        return None
    def principal_axis(points):
        centre = points.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov((points - centre).T))
        vx, vy = vectors[:, -1]
        if vx * ex + vy * ey < 0:
            vx, vy = -vx, -vy
        return centre, float(vx), float(vy)

    every = np.column_stack((dx, dy)).astype(np.float64)
    centre, vx, vy = principal_axis(every[keep])
    # The cone edge can slice a thick stroke diagonally and tilt the fit, so
    # refit on a straight corridor around the first estimate.
    for _ in range(2):
        rel = every - centre
        across = np.abs(rel @ np.array([-vy, vx]))
        corridor = (across <= max(3.0, radius * .22)) & (dist >= radius * 1.35) & (dist <= reach)
        corridor &= (dx * vx + dy * vy) > 0
        if int(corridor.sum()) < max(12, int(radius)):
            return None
        centre, vx, vy = principal_axis(every[corridor])
    points = every[corridor]
    # A real stroke is long and thin; reject blobs or scattered pixels.
    spread = (points - centre) @ np.array([-vy, vx])
    if float(np.std(spread)) > max(2.5, radius * .2):
        return None
    along = float(-(centre[0] * vx + centre[1] * vy))
    anchor = (ox + centre[0] + vx * along, oy + centre[1] + vy * along)
    # Miniclip's lines start at a ball/ghost centre. A "line" that misses the
    # centre is stray white (number badges, stripes in a rack), not the guide.
    if float(np.hypot(anchor[0] - ox, anchor[1] - oy)) > radius * .3:
        return None
    return (float(vx), float(vy)), (float(anchor[0]), float(anchor[1]))


def aim_direction(roi: np.ndarray, cue: tuple[int, int, int], balls: list[tuple[int, int, int, str]]) -> tuple[float, float] | None:
    """Read Miniclip's thin white aim ray starting at the cue ball (rough angle).

    Scans outward from the cue ball's edge in every direction (0.5 degree
    steps) and takes the direction with the longest unbroken white run. A
    probabilistic line finder used here before split long rays into pieces
    and then found none starting at the ball, and it could not see short rays
    (a target close to the cue ball). The cue stick is ignored: it can point
    the opposite way in replays. The ghost ball later fixes the exact angle.
    """
    candidates = aim_candidates(roi, cue, balls, limit=1)
    return candidates[0] if candidates else None


def aim_candidates(roi: np.ndarray, cue: tuple[int, int, int],
                   balls: list[tuple[int, int, int, str]], limit: int = 4) -> list[tuple[float, float]]:
    """Up to ``limit`` possible ray directions from the cue ball, longest run first.

    The longest white run is not always the ray: on a short shot the ghost
    ring's edge plus the game's cue-deflection line can form a longer run
    in a wrong direction. The caller keeps the one that ends in a ghost ball.
    """
    cx, cy, cr = cue
    mask = cv2.dilate(white_guide_mask(roi, balls), np.ones((3, 3), np.uint8))
    h, w = mask.shape[:2]
    start = cr * 1.3
    steps = np.arange(start, start + max(6 * cr, 40), 1.0)
    angles = np.deg2rad(np.arange(0, 360, .5))
    xs = np.rint(cx + np.cos(angles)[:, None] * steps[None]).astype(int)
    ys = np.rint(cy + np.sin(angles)[:, None] * steps[None]).astype(int)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    white = np.zeros(xs.shape, bool)
    white[inside] = mask[ys[inside], xs[inside]] > 0
    runs = np.where(white.all(axis=1), white.shape[1], np.argmin(white, axis=1))
    return _run_peaks(angles, runs, max(1.2 * cr, 12))[:limit]


def _run_peaks(angles: np.ndarray, runs: np.ndarray, minimum: float) -> list[tuple[float, float]]:
    """Directions of separate white-run peaks, longest first.

    A stroke is a few pixels wide, so neighbouring angles tie; each peak is
    the average of the near-best angles within 3 degrees of it.
    """
    n = len(angles)
    order = np.argsort(-runs, kind="stable")
    taken = np.zeros(n, bool)
    peaks = []
    for top in order:
        if runs[top] < minimum:
            break
        if taken[top]:
            continue
        offset = np.abs((np.arange(n) - top + n // 2) % n - n // 2)
        near = offset <= 6
        chosen = near & (runs >= runs[top] * .9)
        taken |= offset <= 16          # suppress the rest of this peak (8 degrees)
        ux, uy = float(np.cos(angles[chosen]).sum()), float(np.sin(angles[chosen]).sum())
        norm = float(np.hypot(ux, uy))
        if norm > 1e-6:
            peaks.append((ux / norm, uy / norm))
    return peaks


# A circle on the stick line is a real ball when it is full ball size and at
# least this share of its rim is a clear ball-to-cloth edge. Measured on the
# real frame and synthetic tables: stick joints are 0.69-0.88 of the median
# ball radius, real balls 0.92-1.12; both can have clean round rims.
STICK_BALL_RIM = .5
STICK_BALL_SIZE = .9


def separate_cue_stick_detections(
    balls: list[tuple[int, int, int, str]],
    cue: tuple[int, int, int],
    aim: tuple[float, float],
    rim_open: dict[tuple[int, int], float] | None = None,
) -> list[tuple[int, int, int, str]]:
    """Drop circle-like cue-stick ornaments lying behind the tracked cue ball.

    The game aim ray points from the cue ball toward the target. The physical
    stick lies on the opposite side. Real balls offset from that narrow stick
    axis stay in the list; the cue itself is always retained. A real ball
    lying under the stick also stays when ``rim_open`` shows a clear
    ball-to-cloth edge around most of its rim (stick ornaments have none).
    """
    cx, cy, cr = cue
    ux, uy = aim
    typical_r = float(np.median([b[2] for b in balls])) if balls else float(cr)
    kept = []
    for bx, by, br, kind in balls:
        if kind == "cue" or (bx, by, br) == (cx, cy, cr):
            kept.append((bx, by, br, kind))
            continue
        dx, dy = bx - cx, by - cy
        behind = -(dx * ux + dy * uy)
        across = abs(dx * uy - dy * ux)
        shaft_half_width = max(9.0, min(cr * 1.05, br * 1.1))
        # The stick reaches ~30 ball radii behind the cue ball on real frames.
        if cr * 1.35 < behind < typical_r * 40 and across <= shaft_half_width:
            ball_like = (rim_open is not None and rim_open.get((bx, by), 0.0) >= STICK_BALL_RIM
                         and br >= typical_r * STICK_BALL_SIZE)
            if not ball_like:
                continue
        kept.append((bx, by, br, kind))
    return kept


def separate_cue_overlap_detections(
    balls: list[tuple[int, int, int, str]],
    cue: tuple[int, int, int],
) -> list[tuple[int, int, int, str]]:
    """Reject circles that overlap the cue-ball face, such as cue-tip graphics."""
    cx, cy, cr = cue
    kept = []
    for bx, by, br, kind in balls:
        if kind == "cue" or (bx, by, br) == cue:
            kept.append((bx, by, br, kind))
            continue
        # Touching balls are a full diameter apart. A circle much closer than
        # that is a UI cue-tip/hand marker on the cue face, not a ball (Hough
        # radii vary by a few pixels, hence 0.8 rather than 1.0).
        if np.hypot(bx - cx, by - cy) < (cr + br) * .80:
            continue
        kept.append((bx, by, br, kind))
    return kept


def _first_obstacle(x: float, y: float, ux: float, uy: float, limit: float, radius: float,
                    obstacles) -> tuple[float, tuple[int, int, int]] | None:
    """Distance along a ray at which a moving ball first touches an obstacle ball."""
    best = None
    for bx, by, br in obstacles or ():
        dx, dy = bx - x, by - y
        along = dx * ux + dy * uy
        across2 = dx * dx + dy * dy - along * along
        reach2 = (radius + br) ** 2
        if along <= 0 or across2 >= reach2:
            continue
        s = along - np.sqrt(reach2 - across2)
        if 1e-3 < s < limit and (best is None or s < best[0]):
            best = (float(s), (int(bx), int(by), int(br)))
    return best


def trace_bounces(start: tuple[float, float], direction: tuple[float, float],
                  bounds: tuple[int, int, int, int], radius: int,
                  max_bounces: int = 1, obstacles=None,
                  blocked: list | None = None) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Trace a ball to the cushions and reflect it, stopping at a pocket mouth.

    ``bounds`` are the limits for the ball's centre (cushion nose inset by the
    radius). When ``obstacles`` (x, y, r) are given, the path ends where it
    would first touch one of those balls, and that ball is appended to
    ``blocked`` so the overlay can mark it.
    """
    left, top, right, bottom = bounds
    # A ball frozen on a cushion can be detected a pixel past the centre limit;
    # start from inside so the next-cushion distance is never negative.
    x = float(min(max(start[0], left), right))
    y = float(min(max(start[1], top), bottom))
    ux, uy = direction
    norm = float(np.hypot(ux, uy))
    if norm < 1e-6 or not np.isfinite(norm):
        return []
    ux, uy = ux / norm, uy / norm
    # Already touching a cushion and heading into it: it rebounds at once.
    if (x <= left and ux < 0) or (x >= right and ux > 0):
        ux = -ux
    if (y <= top and uy < 0) or (y >= bottom and uy > 0):
        uy = -uy
    pockets = [(left, top), ((left + right)//2, top), (right, top),
               (left, bottom), ((left + right)//2, bottom), (right, bottom)]
    result = []
    for _ in range(max_bounces + 1):
        tx = ((right - x) / ux if ux > 1e-6 else
              (left - x) / ux if ux < -1e-6 else float("inf"))
        ty = ((bottom - y) / uy if uy > 1e-6 else
              (top - y) / uy if uy < -1e-6 else float("inf"))
        options = [t for t in (tx, ty) if 1e-4 < t < float("inf")]
        if not options:
            break
        distance = min(options)
        hit = _first_obstacle(x, y, ux, uy, distance, radius, obstacles)
        if hit is not None:
            s, ball = hit
            result.append(((int(round(x)), int(round(y))),
                           (int(round(x + ux * s)), int(round(y + uy * s)))))
            if blocked is not None:
                blocked.append(ball)
            break
        end_x, end_y = x + ux * distance, y + uy * distance
        end = (int(round(end_x)), int(round(end_y)))
        result.append(((int(round(x)), int(round(y))), end))
        # A rail intersection near a pocket is treated as a pot, not a bounce.
        if min(np.hypot(end_x-px, end_y-py) for px, py in pockets) <= radius * 2.0:
            break
        hit_x = abs(tx - distance) < 1e-3
        hit_y = abs(ty - distance) < 1e-3
        if hit_x:
            ux = -ux
        if hit_y:
            uy = -uy
        x, y = end_x, end_y
    return result


def collision_paths(cue: tuple[int, int, int], target: tuple[int, int, int],
                    direction: tuple[float, float], table: tuple[int, int, int, int],
                    contact: tuple[float, float] | None = None, radius: float | None = None,
                    obstacles=None, object_dir: tuple[float, float] | None = None,
                    cue_dir: tuple[float, float] | None = None, blocked: list | None = None):
    """Shared equal-mass, no-spin collision model used by live guide and simulator.

    ``radius`` is one shared ball radius (all pool balls are the same size; the
    per-circle Hough radii vary by a few pixels). ``contact`` is the cue-ball
    centre at impact, taken from the game's ghost-ball reticle when visible.
    ``object_dir``/``cue_dir`` override the physics directions when the game's
    own white object/deflection lines were read from the frame.
    """
    cx, cy, cr = cue
    bx, by, br = target
    left, top, width, height = table
    ux, uy = direction
    ball_r = float(radius) if radius else (cr + br) / 2.0
    combined = 2.0 * ball_r
    if contact is None:
        along = (bx-cx)*ux + (by-cy)*uy
        across = abs((bx-cx)*uy - (by-cy)*ux)
        if along <= cr or across >= combined:
            return [], [], [], None
        lead = along - np.sqrt(max(0.0, combined**2 - across**2))
        contact = (cx + ux*lead, cy + uy*lead)
    nx, ny = bx-contact[0], by-contact[1]
    nlen = float(np.hypot(nx, ny))
    if nlen < 1e-6:
        return [], [], [], None
    nx, ny = nx/nlen, ny/nlen
    incoming_normal = ux*nx + uy*ny
    if incoming_normal <= 0:
        return [], [], [], None
    physics_cue = (ux-incoming_normal*nx, uy-incoming_normal*ny)
    obj = object_dir or (nx, ny)
    cue_after = cue_dir or physics_cue
    bounds = (left+ball_r, top+ball_r, left+width-ball_r, top+height-ball_r)
    approach = [((int(round(cx)), int(round(cy))), (int(round(contact[0])), int(round(contact[1]))))]
    others = [(ox, oy, orr) for ox, oy, orr in obstacles or ()
              if (ox, oy) != (bx, by) and np.hypot(ox-cx, oy-cy) > ball_r * .5]
    object_path = trace_bounces((bx, by), obj, bounds, ball_r, max_bounces=1,
                                obstacles=others, blocked=blocked)
    cue_path = (trace_bounces(contact, cue_after, bounds, ball_r, max_bounces=1,
                              obstacles=others, blocked=blocked)
                if np.hypot(*physics_cue) > .12 else [])
    return approach, cue_path, object_path, (int(bx), int(by), int(br))


def predict_shot(roi: np.ndarray, table: tuple[int, int, int, int],
                 balls: list[tuple[int, int, int, str]], cue: tuple[int, int, int]) -> dict:
    """Follow the game's aim graphics and estimate cue/object paths.

    Returns a dict with aim/cue/object segments, target, ghost, blockers and a
    note. All coordinates are in frame pixels (the same frame ``table`` is in).
    """
    left, top, width, height = table
    empty = {"aim": [], "cue": [], "object": [], "target": None, "ghost": None, "blockers": [],
             "aim_dir": None, "object_dir": None, "cue_dir": None}
    cue_local = (cue[0] - left, cue[1] - top, cue[2])
    balls_local = [(bx-left, by-top, br, kind) for bx, by, br, kind in balls]
    cx, cy, cr = cue
    # All balls share one size; use the robust median rather than each circle's
    # slightly noisy Hough radius.
    ball_r = float(np.median([br for _, _, br, _ in balls])) if balls else float(cr)
    white = white_guide_mask(roi, balls_local)
    # The reticle ring touches the struck ball, so mask only the ball faces
    # here; the wider mask above would erase the ring's near side.
    ghost_mask = white_guide_mask(roi, balls_local, ball_scale=.9)
    # Of the candidate ray directions, the real one ends in the ghost ball.
    direction, ghost = None, None
    for candidate in aim_candidates(roi, cue_local, balls_local):
        found = find_ghost_ball(ghost_mask, cue_local, candidate, ball_r)
        if direction is None:
            direction = candidate
        if found is not None and (ghost is None or found[3] > ghost[3]):
            direction, ghost = candidate, found
    if ghost is None:
        ghost = find_ghost_ball(ghost_mask, cue_local, None, ball_r) if direction is None else None
    if direction is None and ghost is None:
        return {**empty, "note": "Cue found · Miniclip aim line not detected"}
    if direction is None:
        direction = (ghost[0] - cue_local[0], ghost[1] - cue_local[1])
        norm = float(np.hypot(*direction))
        direction = (direction[0] / norm, direction[1] / norm)
    ghost_frame = None
    if ghost is not None:
        gx, gy = ghost[0] + left, ghost[1] + top
        gdx, gdy = gx - cx, gy - cy
        glen = float(np.hypot(gdx, gdy))
        refined = ray_direction_to_ghost(white, cue_local, (ghost[0], ghost[1]), ball_r)
        if refined is not None:
            gdx, gdy = refined[0] * glen, refined[1] * glen
        if glen > cr * 1.5:
            # The reticle centre lies on the game's ray, and it is far from the
            # cue ball, so this gives the aim angle to a fraction of a degree.
            direction = (gdx / glen, gdy / glen)
            ghost_frame = (gx, gy, ghost[2])
    ux, uy = direction
    obstacles = [(bx, by, br) for bx, by, br, kind in balls if kind != "cue"]
    angle = float(np.degrees(np.arctan2(uy, ux)))
    selected = None
    if ghost_frame is not None:
        gx, gy, _ = ghost_frame
        # The struck ball touches the ghost ball: its centre is two radii from
        # the reticle centre, on the far side.
        touching = []
        # Distance from the ghost centre to the nearest cushion limit.
        on_cushion = min(gx - (left + ball_r), left + width - ball_r - gx,
                         gy - (top + ball_r), top + height - ball_r - gy) < ball_r * .35
        for bx, by, br, kind in balls:
            if kind != "ball":
                continue
            gap = abs(float(np.hypot(bx - gx, by - gy)) - 2 * ball_r)
            if gap > ball_r * .5 or (bx - gx) * ux + (by - gy) * uy <= 0:
                continue
            # A ghost resting on a cushion can have a ball lying just beside
            # it without touching (a near tie between rail and ball). Within
            # measurement error that cannot be told apart, so trust the game:
            # it draws an object-ball line only when a ball is really hit.
            if on_cushion and gap > ball_r * .2:
                hint = (bx - gx, by - gy)
                if game_line_direction(white, (bx - left, by - top), hint, ball_r) is None:
                    continue
            touching.append((gap, bx, by, br))
        if touching:
            _, bx, by, br = min(touching)
            selected = (bx, by, br)
        else:
            # Reticle sits against a cushion (or the struck ball was missed by
            # the detector): show the game ray up to the reticle only.
            ghost_int = (int(round(gx)), int(round(gy)), int(round(ghost_frame[2])))
            return {**empty, "aim": [((cx, cy), ghost_int[:2])], "ghost": ghost_int, "aim_dir": (ux, uy),
                    "note": f"Aim ray matched at {angle:.1f}° · ghost ball has no object ball beside it"}
    if selected is None:
        selected = _select_target_on_ray(roi, table, balls, cue, direction, ball_r)
        if selected is None:
            bounds = (left + ball_r, top + ball_r, left + width - ball_r, top + height - ball_r)
            blockers: list = []
            aim = trace_bounces((cx, cy), direction, bounds, ball_r, max_bounces=1,
                                obstacles=obstacles, blocked=blockers)
            return {**empty, "aim": aim, "blockers": blockers, "aim_dir": (ux, uy),
                    "note": f"Aim ray found at {angle:.1f}° · no ball directly in its path"}
    bx, by, br = selected
    contact = ghost_frame[:2] if ghost_frame else None
    # Prefer the game's own short white lines for the post-impact directions;
    # physics only fills in what the game does not draw.
    approach, _, _, target = collision_paths(cue, selected, direction, table,
                                             contact=contact, radius=ball_r)
    if target is None:
        return {**empty, "note": "Target found · impact angle is unclear"}
    contact_point = approach[0][1] if contact is None else contact
    physics_obj = (bx - contact_point[0], by - contact_point[1])
    obj_dir = None
    object_line = game_line_direction(white, (bx - left, by - top), physics_obj, ball_r)
    if object_line is not None:
        obj_dir, (ax, ay) = object_line
        # The game's line runs through the true ball centre; snap the noisy
        # Hough centre onto it so the yellow path starts on the game's line.
        selected = (int(round(ax + left)), int(round(ay + top)), br)
    elif ghost_frame is not None:
        # No visible object line (it runs across other balls): the struck
        # ball touches the ghost, so its centre is one diameter from the
        # precisely known ghost centre; fit only the direction.
        fitted = fit_target_on_ghost(roi, (ghost_frame[0] - left, ghost_frame[1] - top),
                                     (bx - left, by - top), ball_r)
        if fitted is not None:
            selected = (int(round(fitted[0] + left)), int(round(fitted[1] + top)), br)
            physics_obj = (fitted[0] + left - contact_point[0], fitted[1] + top - contact_point[1])
            n = float(np.hypot(*physics_obj))
            obj_dir = (physics_obj[0] / n, physics_obj[1] / n)
    n = float(np.hypot(*physics_obj))
    nx, ny = physics_obj[0] / n, physics_obj[1] / n
    dot = ux * nx + uy * ny
    physics_cue = (ux - dot * nx, uy - dot * ny)
    cue_dir = None
    if np.hypot(*physics_cue) > .12:
        cue_line = game_line_direction(white, (contact_point[0] - left, contact_point[1] - top),
                                       physics_cue, ball_r, cone_degrees=20.0)
        cue_dir = cue_line[0] if cue_line else None
    blockers = []
    approach, cue_path, object_path, target = collision_paths(
        cue, selected, direction, table, contact=contact, radius=ball_r, obstacles=obstacles,
        object_dir=obj_dir, cue_dir=cue_dir, blocked=blockers)
    sources = ["ghost ball" if ghost_frame else "ray only"]
    if object_line is not None:
        sources.append("game object line")
    if cue_dir:
        sources.append("game cue line")
    ghost_int = (tuple(int(round(v)) for v in ghost_frame) if ghost_frame else None)
    # The directions the balls leave the impact in (paths may rebound at once
    # when a ball is frozen on a cushion, so report these separately).
    if obj_dir is None:
        ox, oy = selected[0] - contact_point[0], selected[1] - contact_point[1]
        on = float(np.hypot(ox, oy))
        obj_dir = (ox / on, oy / on) if on > 1e-6 else None
    pn = float(np.hypot(*physics_cue))
    final_cue = cue_dir or ((physics_cue[0] / pn, physics_cue[1] / pn) if pn > .12 else None)
    return {"aim": approach, "cue": cue_path, "object": object_path, "target": target,
            "ghost": ghost_int, "blockers": blockers, "aim_dir": (ux, uy),
            "object_dir": obj_dir, "cue_dir": final_cue,
            "note": f"Aim ray matched · target ({bx},{by}) at {angle:.1f}° · "
                    f"{' + '.join(sources)} · paths are estimates"}


def fit_target_on_ghost(roi: np.ndarray, ghost: tuple[float, float], target: tuple[float, float],
                        radius: float) -> tuple[float, float] | None:
    """Centre of the struck ball, constrained to touch the ghost ball.

    The struck ball's centre lies one diameter from the ghost centre, so only
    its direction is unknown. The edge fit ignores the side facing the ghost
    (the white ring touches the ball there). Returns the fitted centre, or
    None when the fit is not clearly better than the detected centre.
    """
    gx, gy = ghost
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    distance = cv2.GaussianBlur(felt_distance_image(roi, hsv, as_float=True), (0, 0), 1.0)
    base = float(np.arctan2(target[1] - gy, target[0] - gx))
    thetas = base + np.deg2rad(np.arange(-10, 10.01, .2))
    best = None
    for spacing in np.arange(2 * radius - 1.5, 2 * radius + 1.51, .5):
        centres = np.column_stack([gx + spacing * np.cos(thetas), gy + spacing * np.sin(thetas)])
        # Rim samples, skipping +-40 degrees around the direction to the ghost.
        rim = _RIM_ANGLES[None, :]
        toward_ghost = (thetas + np.pi)[:, None]
        facing = np.abs((rim - toward_ghost + np.pi) % (2 * np.pi) - np.pi) < np.radians(40)
        xs_in = centres[:, :1] + (radius - 1.5) * np.cos(rim)
        ys_in = centres[:, 1:] + (radius - 1.5) * np.sin(rim)
        xs_out = centres[:, :1] + (radius + 1.5) * np.cos(rim)
        ys_out = centres[:, 1:] + (radius + 1.5) * np.sin(rim)
        inside = cv2.remap(distance, xs_in.astype(np.float32), ys_in.astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        outside = cv2.remap(distance, xs_out.astype(np.float32), ys_out.astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        contrast = np.where(facing, np.nan, inside - outside)
        scores = np.nanmean(contrast, axis=1)
        i = int(np.nanargmax(scores))
        if best is None or scores[i] > best[0]:
            best = (float(scores[i]), float(centres[i, 0]), float(centres[i, 1]))
    if best is None or best[0] < 5:
        return None
    return best[1], best[2]


def _select_target_on_ray(roi, table, balls, cue, direction, ball_r):
    """First object ball on the aim ray when the ghost-ball reticle is not visible."""
    left, top, width, height = table
    cx, cy, cr = cue
    ux, uy = direction
    # Select only an object ball intersected by the live aim ray; do not guess a
    # target based on distance or nearest pocket.
    hits = []
    for bx, by, br, kind in balls:
        if kind != "ball":
            continue
        dx, dy = bx-cx, by-cy
        along = dx*ux + dy*uy
        across = abs(dx*uy - dy*ux)
        collision_radius = 2 * ball_r
        if along > cr and across < collision_radius:
            lead = along - np.sqrt(max(0.0, collision_radius**2 - across**2))
            # Rank overlapping detections by color difference from felt so a
            # hollow aim reticle ahead of the real ball does not win the hit.
            lx, ly = bx-left, by-top
            yy, xx = np.ogrid[:height, :width]
            dist2 = (xx-lx)**2 + (yy-ly)**2
            inside = roi[dist2 <= (br*.5)**2]
            outside = roi[(dist2 >= (br*1.05)**2) & (dist2 <= (br*1.45)**2)]
            score = 0.0
            if inside.size and outside.size:
                ihsv = cv2.cvtColor(inside.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[:, 0]
                oh = cv2.cvtColor(outside.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[:, 0]
                im, om = np.median(ihsv, axis=0), np.median(oh, axis=0)
                hue_delta = np.abs((ihsv[:, 0].astype(float) - float(om[0]) + 90) % 180 - 90)
                # Striped balls can have a white center band, so assess the
                # colored pixels across the interior instead of its median.
                colored_fraction = float(np.mean((ihsv[:, 1] > 65) & (hue_delta > 12)))
                score = colored_fraction * 100 + max(0.0, float(im[1])-float(om[1])) * .15
            hits.append((lead, score, bx, by, br, along, across))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    selected = hits[0]
    # A reticle and a nearby ball can overlap along the shot ray. Prefer the
    # stronger-colored ball when it is within one combined ball radius behind it.
    for hit in hits[1:]:
        if hit[0] - selected[0] > 2 * ball_r * 1.8:
            break
        if hit[1] > max(selected[1] * 1.2 + 8, 22):
            selected = hit
    _, _, bx, by, br, _, _ = selected
    return bx, by, br


# The cushion nose (playing edge) sits this fraction of the corner-pocket span
# inside each corner-pocket centre. Measured on both bundled Miniclip frames.
POCKET_TO_CUSHION = .0095
# Pocket mouths are near-black; dark UI panels and navy side panels sit at
# brightness 32-60 and must not merge into the pocket blobs.
POCKET_DARKNESS = 45


def table_from_pockets(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Fit the playing area from the four dark corner pockets.

    The felt colour alone also catches the same-coloured cushions on one side,
    which shifted the old fit right by ~13 px (middle pockets landed off the
    real ones). The pockets are symmetric and unambiguous.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = (hsv[:, :, 2] < POCKET_DARKNESS).astype(np.uint8)
    felt_like = cv2.inRange(hsv, np.array([75, 45, 65]), np.array([125, 255, 255]))
    count, _, stats, centres = cv2.connectedComponentsWithStats(dark, connectivity=8)
    blobs = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        # Pocket mouths are drop-shadowed discs; their blob fills ~50% of its box.
        if area < 150 or not (.7 <= bw / max(1, bh) <= 1.43) or area < .4 * bw * bh:
            continue
        # A pocket opens onto the felt-coloured cushions. Requiring that keeps
        # dark wallpaper, letterbox bars and icons out of the candidates on a
        # full multi-monitor desktop capture.
        pad = int(bw * .6)
        around = felt_like[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]
        if around.size == 0 or float(np.count_nonzero(around)) / around.size < .08:
            continue
        blobs.append((float(centres[i][0]), float(centres[i][1]), float(bw), float(area)))
    blobs = sorted(blobs, key=lambda b: -b[3])[:40]
    best = None
    for a in blobs:
        for b in blobs:
            dx = b[0] - a[0]
            if dx < 200 or abs(b[1] - a[1]) > dx * .03:
                continue
            size = (a[2] + b[2]) / 2
            if not (.035 * dx <= size <= .1 * dx) or max(a[3], b[3]) > 2 * min(a[3], b[3]):
                continue
            for c in blobs:
                dy = c[1] - a[1]
                if abs(c[0] - a[0]) > dx * .03 or not (.44 * dx <= dy <= .58 * dx):
                    continue
                for d in blobs:
                    if abs(d[0] - b[0]) > dx * .03 or abs(d[1] - c[1]) > dx * .03:
                        continue
                    sizes = [p[3] for p in (a, b, c, d)]
                    if max(sizes) > 2 * min(sizes):
                        continue
                    error = (abs(b[1] - a[1]) + abs(d[1] - c[1]) + abs(c[0] - a[0]) + abs(d[0] - b[0])
                             + abs(dy / dx - .51) * dx * .2)
                    if best is None or error < best[0]:
                        best = (error, a, b, c, d)
    if best is None:
        return None
    _, a, b, c, d = best
    span = ((b[0] - a[0]) + (d[0] - c[0])) / 2
    inset = POCKET_TO_CUSHION * span
    x0, x1 = (a[0] + c[0]) / 2 + inset, (b[0] + d[0]) / 2 - inset
    y0, y1 = (a[1] + b[1]) / 2 + inset, (c[1] + d[1]) / 2 - inset
    inner = hsv[int(y0 + (y1 - y0) * .25):int(y1 - (y1 - y0) * .25),
                int(x0 + (x1 - x0) * .25):int(x1 - (x1 - x0) * .25)]
    felt = (inner[:, :, 0] >= 75) & (inner[:, :, 0] <= 125) & (inner[:, :, 1] >= 45)
    if inner.size == 0 or float(felt.mean()) < .5:
        return None
    return int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))


def refine_table_edges(frame: np.ndarray, table: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Snap each side of a rough table fit onto the actual cushion nose.

    Going from cushion to cloth, brightness drops sharply at the nose (a lit
    cushion edge, then the darker cloth). For each side, a median brightness
    profile across the rail (pocket mouths skipped, so balls resting on the
    cushion barely matter) locates that drop to a fraction of a pixel. A
    pocket fit can be pulled off by a dark ball near a pocket; this is not.
    """
    x, y, w, h = table
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fh, fw = gray.shape
    reach = max(6, int(round(w * .03)))

    def edge(profile_values: np.ndarray, start: int) -> float | None:
        # ``profile_values`` runs from outside (cushion) to inside (cloth).
        drops = profile_values[:-1] - profile_values[1:]
        i = int(np.argmax(drops))
        if drops[i] < 15:
            return None
        offset = 0.0
        if 0 < i < len(drops) - 1:
            a, b, c = drops[i - 1], drops[i], drops[i + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-6:
                offset = float(np.clip(.5 * (a - c) / denom, -.5, .5))
        return start + i + .5 + offset

    along_x = np.r_[np.arange(int(x + w * .08), int(x + w * .44)), np.arange(int(x + w * .56), int(x + w * .92))]
    along_y = np.arange(int(y + h * .12), int(y + h * .88))
    along_x = along_x[(along_x >= 0) & (along_x < fw)]
    along_y = along_y[(along_y >= 0) & (along_y < fh)]
    sides = {}
    if along_x.size > 20:
        top0, top1 = max(0, y - reach), min(fh, y + reach)
        if top1 - top0 > 3:
            sides["top"] = edge(np.median(gray[top0:top1][:, along_x], axis=1), top0)
        bot0, bot1 = max(0, y + h - reach), min(fh, y + h + reach)
        if bot1 - bot0 > 3:
            profile = np.median(gray[bot0:bot1][:, along_x], axis=1)[::-1]
            found = edge(profile, 0)
            # Reversed sample j is row bot1 - 1 - j.
            sides["bottom"] = None if found is None else bot1 - 1 - found
    if along_y.size > 20:
        left0, left1 = max(0, x - reach), min(fw, x + reach)
        if left1 - left0 > 3:
            sides["left"] = edge(np.median(gray[along_y][:, left0:left1], axis=0), left0)
        right0, right1 = max(0, x + w - reach), min(fw, x + w + reach)
        if right1 - right0 > 3:
            profile = np.median(gray[along_y][:, right0:right1], axis=0)[::-1]
            found = edge(profile, 0)
            sides["right"] = None if found is None else right1 - 1 - found
    left = sides.get("left") or x
    top = sides.get("top") or y
    right = sides.get("right") or x + w
    bottom = sides.get("bottom") or y + h
    # The playing area is 2:1; reject a refinement that breaks that badly.
    if right - left <= 0 or not (1.85 <= (right - left) / max(1.0, bottom - top) <= 2.15):
        return table
    return (int(round(left)), int(round(top)), int(round(right - left)), int(round(bottom - top)))


def find_table_bounds(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Playing area (cushion noses) as (x, y, width, height) in frame pixels."""
    pocket_fit = table_from_pockets(frame)
    if pocket_fit is not None:
        return refine_table_edges(frame, pocket_fit)
    # Fallback when the pockets are cropped out (an overlay drawn tightly
    # around the felt): fit the felt colour instead.
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Miniclip's blue felt sits around hue 95–110; exclude the much darker
    # navy side panels and black letterbox bars when fitting the full desktop.
    # The felt is less saturated than the cyan cushions. A broad threshold
    # merges both and makes the fitted guide too tall and the cushion paths wrong.
    felt = cv2.inRange(hsv, np.array([75, 45, 65]), np.array([125, 190, 255]))
    contours, _ = cv2.findContours(felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, tw, th = cv2.boundingRect(c)
        aspect = tw / max(1, th)
        if tw * th > w * h * 0.01 and 1.7 <= aspect <= 2.3:
            # The playable cloth has a near 2:1 aspect ratio. Blue trim below
            # the cloth often extends the color contour, so trim excess from
            # the bottom while retaining the accurately detected top edge.
            th = min(th, max(1, int(round(tw / 2.0))))
            pad = max(1, int(min(tw, th) * .003))
            table = (max(0, x + pad), max(0, y + pad), min(w, tw - 2 * pad), min(h, th - 2 * pad))
            table = refine_table_edges(frame, table)
    return table


def felt_reference(roi_hsv: np.ndarray) -> tuple[float, float]:
    """Median hue and saturation of the cloth (balls and guide strokes excluded)."""
    hue, sat = roi_hsv[:, :, 0], roi_hsv[:, :, 1]
    cloth = (hue >= 80) & (hue <= 115) & (sat >= 50) & (sat <= 235)
    if np.count_nonzero(cloth) < roi_hsv.shape[0] * roi_hsv.shape[1] * .2:
        cloth = np.ones_like(hue, bool)
    return float(np.median(hue[cloth])), float(np.median(sat[cloth]))


def felt_distance_image(roi: np.ndarray, roi_hsv: np.ndarray, as_float: bool = False) -> np.ndarray:
    """How far each pixel's colour is from the cloth (bright = not cloth).

    A blue ball on blue cloth has almost no brightness edge, so grey-level
    circle finding misses it; its colour still differs. Lightness is weighted
    down so the cloth's own vignetting stays dark here.
    """
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    hue, sat = roi_hsv[:, :, 0], roi_hsv[:, :, 1]
    cloth = (hue >= 80) & (hue <= 115) & (sat >= 50) & (sat <= 235)
    reference = np.median(lab[cloth] if np.count_nonzero(cloth) > 100 else lab.reshape(-1, 3), axis=0)
    diff = lab - reference
    distance = np.sqrt((diff[:, :, 0] * .5) ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
    if as_float:
        return distance.astype(np.float32)
    return np.clip(distance * 3, 0, 255).astype(np.uint8)


_RIM_ANGLES = np.linspace(0, 2 * np.pi, 72, endpoint=False)


def _rim_scores(distance: np.ndarray, centres: np.ndarray, radius: float) -> np.ndarray:
    """Mean of (inside - outside) cloth distance just across a circle's rim.

    ``centres`` is an (N, 2) array; returns N scores. High where the circle
    sits exactly on a ball's edge. Where two balls touch, inside and outside
    are both "ball", so that stretch adds nothing and cannot pull the fit.
    """
    cos, sin = np.cos(_RIM_ANGLES), np.sin(_RIM_ANGLES)
    inner, outer = radius - 1.5, radius + 1.5
    xs = np.concatenate([centres[:, :1] + inner * cos, centres[:, :1] + outer * cos], axis=1)
    ys = np.concatenate([centres[:, 1:] + inner * sin, centres[:, 1:] + outer * sin], axis=1)
    values = cv2.remap(distance, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    n = len(_RIM_ANGLES)
    return (values[:, :n] - values[:, n:]).mean(axis=1)


def rim_open_fraction(distance: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Share of the rim where a ball clearly meets cloth (inside differs, outside is cloth).

    A real ball scores high except where neighbours or the stick touch it; a
    circle fitted onto the cue stick or a guide line scores low.
    """
    cos, sin = np.cos(_RIM_ANGLES), np.sin(_RIM_ANGLES)
    outside = cv2.remap(distance, (cx + (r + 3) * cos)[None].astype(np.float32),
                        (cy + (r + 3) * sin)[None].astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)[0]
    inside = cv2.remap(distance, (cx + (r - 3) * cos)[None].astype(np.float32),
                       (cy + (r - 3) * sin)[None].astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)[0]
    return float(np.mean(inside - outside > 8))


def refine_circle(distance: np.ndarray, cx: float, cy: float, r: float,
                  search: float = 4.0) -> tuple[float, float, float]:
    """Sub-pixel ball centre and radius on the cloth-distance image.

    Hough centres are only good to a few pixels; 4 px on the cue ball turned
    the aim by ~1 degree on synthetic tests. Coarse centre grid, then radius,
    then a fine centre grid.
    """
    def best_centre(x0, y0, radius, reach, step):
        offsets = np.arange(-reach, reach + 1e-6, step)
        gx, gy = np.meshgrid(x0 + offsets, y0 + offsets)
        centres = np.column_stack([gx.ravel(), gy.ravel()])
        scores = _rim_scores(distance, centres, radius)
        i = int(np.argmax(scores))
        return float(centres[i, 0]), float(centres[i, 1])

    x0, y0, r0 = cx, cy, r
    # A ball boxed in by touching neighbours (inside a rack) shows almost no
    # cloth around its rim, so the score is flat there and a fit would drift
    # on noise. Refine only balls with enough open rim.
    if rim_open_fraction(distance, cx, cy, r) < .35:
        return x0, y0, r0
    cx, cy = best_centre(cx, cy, r, search, 1.0)
    radii = np.arange(r * .8, r * 1.2 + 1e-6, .5)
    scores = [_rim_scores(distance, np.array([[cx, cy]]), rr)[0] for rr in radii]
    r = float(radii[int(np.argmax(scores))])
    cx, cy = best_centre(cx, cy, r, 1.5, .25)
    return cx, cy, r


def find_balls(roi: np.ndarray, gray: np.ndarray, min_r: int, max_r: int):
    """Ball circles in the table ROI: (x, y, r, sat, val, white_fraction, red_spot).

    Candidates come from circle finding on grey levels and on colour distance
    from the cloth (the second one is what finds blue balls on blue cloth).
    A candidate is a ball when its interior is mostly *not* cloth-coloured:
    hollow shapes such as the game's ghost-ball ring, watermark rings and
    lettering, ball shadows and guide-line crossings all have cloth inside.
    Balls cannot overlap, so of two detections closer than 1.4 radii the one
    with less cloth inside is kept.
    """
    th, tw = roi.shape[:2]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    felt_hue, felt_sat = felt_reference(roi_hsv)
    distance = felt_distance_image(roi, roi_hsv, as_float=True)
    # Colour-distance edges of a blue ball on blue cloth are weaker than grey
    # edges of other balls, hence the lower Canny threshold for that image.
    sources = ((cv2.medianBlur(np.clip(distance * 3, 0, 255).astype(np.uint8), 5), 60), (gray, 100))
    smooth = cv2.GaussianBlur(distance, (0, 0), 1.0)
    proposals: list[tuple[int, int, int]] = []
    for image, edge_threshold in sources:
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.25,
                                   minDist=max(12, min_r * 1.8), param1=edge_threshold,
                                   param2=20, minRadius=min_r, maxRadius=max_r)
        if circles is None:
            continue
        for hx, hy, hr in circles[0]:
            # Snap the proposal onto the ball's edge before judging it: Hough
            # centres can be a few pixels off, enough to fail the cloth test.
            fx, fy, fr = refine_circle(smooth, float(hx), float(hy), float(hr))
            if np.hypot(fx - hx, fy - hy) > hr * .5:
                fx, fy, fr = float(hx), float(hy), float(hr)
            cx, cy, r = int(round(fx)), int(round(fy)), int(round(fr))
            # The colour-distance proposal comes first and wins duplicates.
            if all((cx - px) ** 2 + (cy - py) ** 2 > (min_r * .6) ** 2 for px, py, _ in proposals):
                proposals.append((cx, cy, r))
    # Pocket centres lie just outside the cushion noses (corners ~1 inset,
    # middles ~2 insets beyond the rail line).
    inset = POCKET_TO_CUSHION * tw
    pockets_local = [(-inset, -inset), (tw / 2, -2 * inset), (tw + inset, -inset),
                     (-inset, th + inset), (tw / 2, th + 2 * inset), (tw + inset, th + inset)]
    accepted = []
    for cx, cy, r in proposals:
        if not (0 <= cx < tw and 0 <= cy < th):
            continue
        # Ignore partial circles cut by the rail line. A ball frozen to a
        # cushion has its centre exactly one radius from the edge and stays.
        if min(cx, cy, tw - cx, th - cy) < r * .75:
            continue
        # A ball sitting in the jaws of a pocket is about to drop.
        if min((cx - px) ** 2 + (cy - py) ** 2 for px, py in pockets_local) < (r * 1.6) ** 2:
            continue
        extent = int(np.ceil(r * 1.8))
        x0, x1 = max(0, cx - extent), min(tw, cx + extent + 1)
        y0, y1 = max(0, cy - extent), min(th, cy + extent + 1)
        patch = roi_hsv[y0:y1, x0:x1].astype(np.int16)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inner = patch[dist2 <= (r * .62) ** 2]
        if inner.size == 0:
            continue
        # Local cloth colour from a ring just outside the ball, so the
        # cloth's own gradient (hue 92 at the rails to 98 mid-table) and
        # lighting do not matter.
        ring = patch[(dist2 >= (r * 1.35) ** 2) & (dist2 <= (r * 1.8) ** 2)]
        ring_cloth = ring[(np.abs(ring[:, 0] - felt_hue) <= 12) & (np.abs(ring[:, 1] - felt_sat) <= 60)]
        local_hue, local_sat = ((float(np.median(ring_cloth[:, 0])), float(np.median(ring_cloth[:, 1])))
                                if len(ring_cloth) >= 12 else (felt_hue, felt_sat))
        cloth_inside = ((np.abs(inner[:, 0] - local_hue) <= 7) & (np.abs(inner[:, 1] - local_sat) <= 35)
                        & (inner[:, 2] >= 50))
        cloth_fraction = float(np.mean(cloth_inside))
        if cloth_fraction > .35:
            continue
        sat = float(np.median(inner[:, 1]))
        val = float(np.median(inner[:, 2]))
        white_fraction = float(np.mean((inner[:, 1] < 90) & (inner[:, 2] > 150)))
        centre = patch[dist2 <= (r * .4) ** 2]
        red_spot = float(np.mean((centre[:, 0] < 30) & (centre[:, 1] > 70) & (centre[:, 2] < 90)))
        accepted.append((cloth_fraction, cx, cy, r, sat, val, white_fraction, red_spot))
    # Balls never overlap (touching balls are one diameter apart). Two
    # proposals on the same ball are ~0.3 diameters apart; touching rack
    # balls can look as close as ~0.75 diameters when the radius estimate is
    # high and boxed-in centres are a little off, so detections closer than
    # 0.6 diameters are merged, keeping the one with less cloth inside. Two
    # mostly white circles are both kept; the cue-ball check decides which is
    # the cue ball and drops the other (a cue-tip/hand marker).
    if not accepted:
        return []
    diameter = 2 * float(np.median([item[3] for item in accepted]))
    accepted.sort(key=lambda item: item[0])
    kept = []
    for item in accepted:
        _, cx, cy, r = item[:4]
        clash = [k for k in kept if (cx - k[1]) ** 2 + (cy - k[2]) ** 2 < (diameter * .6) ** 2]
        if all(item[6] >= .5 and k[6] >= .5 for k in clash):
            kept.append(item)
    return [(cx, cy, r, *rest, rim_open_fraction(smooth, cx, cy, r)) for _, cx, cy, r, *rest in kept]


def detect(frame: np.ndarray) -> Detection:
    """Find the Miniclip felt, ball centers, and current game aim ray."""
    h, w = frame.shape[:2]
    table = find_table_bounds(frame)
    if table is None:
        return Detection(note="Table not found — adjust the overlay to include the full felt.")

    x, y, tw, th = table
    roi = frame[y:y + th, x:x + tw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    # At Miniclip's usual aspect ratio, ball radii are about 2.5–3% of felt height.
    # Keeping this range tight avoids treating pockets and logo strokes as balls.
    min_r = max(8, int(min(tw, th) * .021))
    # Miniclip balls are close to 3% of the felt's short side. A tighter upper
    # bound prevents the larger hollow aim reticle from merging with a nearby ball.
    max_r = max(min_r + 2, int(min(tw, th) * .039))
    balls: list[tuple[int, int, int, str]] = []
    cue_candidates: list[tuple[int, int, int, float, float, float, float]] = []
    rim_open: dict[tuple[int, int], float] = {}
    for cx, cy, r, sat, val, white_fraction, red_spot, open_rim in find_balls(roi, gray, min_r, max_r):
        ball = (x + cx, y + cy, r)
        rim_open[(x + cx, y + cy)] = open_rim
        # Keep every detection eligible as an object ball until the actual
        # cue is identified from its shaft (white stripes and number badges
        # are common on object balls too).
        cue_candidates.append((*ball, sat, val, white_fraction, red_spot))
        balls.append((*ball, "ball"))
    cue = None
    if cue_candidates:
        # The cue ball is the white candidate at the end of the long cue stick.
        # This distinguishes it from white stripes and bright aiming markers.
        edges = cv2.Canny(gray, 50, 140)
        # Leave the shaft intact: masking all balls breaks it into fragments
        # and lets a nearby aim marker be mistaken for the cue-ball endpoint.
        stick_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=16,
                                      minLineLength=max(45, min_r * 6), maxLineGap=max(12, int(min_r * 1.2)))
        # The cue ball is bright white over most of its face; striped object
        # balls and Miniclip's reticle are much less white across the full disc.
        white_candidates = [c for c in cue_candidates if c[5] >= .68]
        shaft_candidates = white_candidates or [c for c in cue_candidates if c[5] >= .52 and c[3] < 80]
        stick_matches = []
        if stick_lines is not None:
            for candidate in shaft_candidates:
                cx, cy, cr, sat, val = candidate[:5]
                lx, ly = cx-x, cy-y
                for line in stick_lines[:, 0, :]:
                    x1, y1, x2, y2 = map(float, line)
                    dx, dy = x2-x1, y2-y1
                    length = float(np.hypot(dx, dy))
                    if length < 1:
                        continue
                    gap = abs((lx-x1)*dy - (ly-y1)*dx) / length
                    endpoint_gap = min(np.hypot(lx-x1, ly-y1), np.hypot(lx-x2, ly-y2))
                    if gap <= cr*1.4 and endpoint_gap <= cr*2.2:
                        # Prefer a long shaft that actually ends at this ball;
                        # a long aim ray often passes through markers or targets.
                        shaft_score = (-length + 5.0 * (gap + endpoint_gap)
                                       - 100.0 * min(candidate[6], .2))
                        stick_matches.append((shaft_score, -length, gap, endpoint_gap, sat, -val, candidate))
        if stick_matches:
            cue = min(stick_matches, key=lambda item: item[:6])[6]
        else:
            if white_candidates:
                cue = max(white_candidates, key=lambda item: (item[6], item[5], item[4]))
            elif shaft_candidates:
                cue = max(shaft_candidates, key=lambda item: (item[6], item[5], -item[3]))
        # The decisive sign is the game's own guide: its white aim ray starts at
        # the cue ball and ends in the hollow ghost ball. The stick can be
        # mostly off the table (cue ball near a rail) and then points at a
        # white-striped ball instead, so a candidate whose ray ends in a ghost
        # ball wins; the stick choice only breaks ties.
        ray_candidates = [c for c in cue_candidates if c[5] >= .52]
        if ray_candidates:
            local_all = [(bx - x, by - y, br, kind) for bx, by, br, kind in balls]
            radius = float(np.median([b[2] for b in balls]))
            ghost_mask = white_guide_mask(roi, local_all, ball_scale=.9)
            verified = []
            for candidate in ray_candidates:
                local = (candidate[0] - x, candidate[1] - y, candidate[2])
                directions = aim_candidates(roi, local, local_all)
                if not directions:
                    continue
                ghost = None
                for direction in directions:
                    ghost = find_ghost_ball(ghost_mask, local, direction, radius)
                    if ghost is not None:
                        break
                verified.append((ghost is not None, candidate is cue, candidate[5], candidate))
            if verified:
                best = max(verified, key=lambda item: item[:3])
                if best[0] or cue is None:
                    cue = best[3]
        if cue:
            cue_position = cue[:3]
            cue_index = min(range(len(balls)), key=lambda i: np.hypot(balls[i][0]-cue[0], balls[i][1]-cue[1]))
            bx, by, br, _ = balls[cue_index]
            balls[cue_index] = (bx, by, br, "cue")
            balls = separate_cue_overlap_detections(balls, cue_position)
            local_balls = [(bx-x, by-y, br, kind) for bx, by, br, kind in balls]
            aim = aim_direction(roi, (cue_position[0]-x, cue_position[1]-y, cue_position[2]), local_balls)
            if aim:
                balls = separate_cue_stick_detections(balls, cue_position, aim, rim_open)
            # Exactly one ball is the cue ball; other white/striped detections
            # remain valid object-ball targets under 8-ball rules.
            cue_index = min(range(len(balls)), key=lambda i: np.hypot(balls[i][0]-cue[0], balls[i][1]-cue[1]))
            bx, by, br, _ = balls[cue_index]
            balls[cue_index] = (bx, by, br, "cue")
    if len(balls) > 24:
        return Detection(table=table, balls=[], note=f"Table found · {len(balls)} circle candidates · too many to track confidently")
    if cue:
        shot = predict_shot(roi, table, balls, cue[:3])
        if shot["ghost"]:
            # Nothing can occupy the spot where the cue ball will be at impact:
            # a "ball" centred on the ghost ring is the ring itself (the ray and
            # deflection line inside it can make it look filled). Drop it and
            # predict again so it is not masked or used as an obstacle.
            gx, gy, gr = shot["ghost"]
            cleaned = [b for b in balls if b[3] == "cue" or np.hypot(b[0] - gx, b[1] - gy) > gr * .6]
            if len(cleaned) != len(balls):
                balls = cleaned
                shot = predict_shot(roi, table, balls, cue[:3])
        return Detection(table=table, balls=balls, cue=cue[:3], aim_segments=shot["aim"],
                         cue_segments=shot["cue"], object_segments=shot["object"],
                         target=shot["target"], ghost=shot["ghost"], blockers=shot["blockers"],
                         aim_dir=shot["aim_dir"], object_dir=shot["object_dir"], cue_dir=shot["cue_dir"],
                         note=f"{shot['note']} · {len(balls)} ball candidates")
    return Detection(table=table, balls=balls, note="Table found · cue ball not confidently identified")


# Every guide stroke stays at or below this opacity. The overlay is visible in
# screenshots and screen shares, so its own drawing is in every analysis frame.
# Partial opacity lets ``remove_overlay`` recover the game pixels underneath
# exactly instead of having to guess them.
GUIDE_ALPHA = 150
# Seconds to wait after an overlay repaint before reading the screen, so the
# desktop compositor is showing that repaint (the layer we subtract).
PAINT_SETTLE = .07


def paint_guide(p: QPainter, d: Detection, width: int, height: int, adjusting: bool,
                active: bool, ui_scale: float = 1.0):
    """Draw the guide in detection (capture-pixel) coordinates.

    Shared by the live overlay, the offscreen layer used to undo the overlay's
    own pixels before analysis, and the screenshot tester, so all three match.
    """
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    a = GUIDE_ALPHA

    def pen(color, w, style=Qt.PenStyle.SolidLine):
        return QPen(color, w * ui_scale, style)

    if adjusting:
        p.fillRect(QRect(0, 0, width, height), QColor(20, 170, 170, 13))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(pen(QColor(45, 220, 205, a), 2, Qt.PenStyle.DashLine))
        p.drawRect(QRectF(2, 2, width - 5, height - 5))
        p.setPen(pen(QColor(255, 255, 255, a), 1))
        font = QFont("Arial")
        font.setPixelSize(max(10, round(13 * ui_scale)))
        p.setFont(font)
        p.drawText(QPoint(round(14 * ui_scale), round(24 * ui_scale)),
                   "Drag to position · drag lower-right corner to resize")
        handle = 12 * ui_scale
        p.drawRect(QRectF(width - handle - 7 * ui_scale, height - handle - 7 * ui_scale, handle, handle))
    if not active:
        return
    if d.table:
        x, y, w, h = d.table
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(pen(QColor(60, 255, 180, 110), 2))
        p.drawRect(x, y, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 220, 70, 110))
        for px, py in ((x, y), (x+w//2, y), (x+w, y), (x, y+h), (x+w//2, y+h), (x+w, y+h)):
            p.drawEllipse(QPointF(px, py), 6 * ui_scale, 6 * ui_scale)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for bx, by, br, kind in d.balls or []:
        color = QColor(80, 245, 255, a) if kind == "cue" else QColor(255, 255, 255, 110)
        p.setPen(pen(color, 2))
        # Ring just outside the ball so the ball itself stays clear.
        p.drawEllipse(QPointF(bx, by), br + 3, br + 3)
    for segments, color, style, width_px in (
        (d.aim_segments, QColor(45, 225, 255, a), Qt.PenStyle.SolidLine, 3),
        (d.object_segments, QColor(255, 210, 55, a), Qt.PenStyle.SolidLine, 3),
        (d.cue_segments, QColor(255, 70, 180, a), Qt.PenStyle.DashLine, 2.5),
    ):
        for index, (start, end) in enumerate(segments or []):
            segment_color = QColor(color)
            if index:
                segment_color.setAlpha(round(a * .75))
            p.setPen(pen(segment_color, width_px, Qt.PenStyle.DashLine if index else style))
            p.drawLine(QPoint(*start), QPoint(*end))
    for bx, by, br in d.blockers or []:
        p.setPen(pen(QColor(255, 140, 40, a), 2, Qt.PenStyle.DotLine))
        p.drawEllipse(QPointF(bx, by), br + 5, br + 5)
    if d.target:
        tx, ty, tr = d.target
        p.setPen(pen(QColor(255, 210, 55, a), 3))
        p.drawEllipse(QPointF(tx, ty), tr + 5, tr + 5)


def render_guide_layer(d: Detection, width: int, height: int, adjusting: bool, active: bool,
                       ui_scale: float = 1.0) -> np.ndarray:
    """The overlay's drawing as premultiplied BGRA pixels (what the compositor blends)."""
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    p = QPainter(image)
    paint_guide(p, d, width, height, adjusting, active, ui_scale)
    p.end()
    stride = image.bytesPerLine() // 4
    pixels = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes())
    return pixels.reshape(height, stride, 4)[:, :width].copy()


def composite_overlay(frame: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Blend a premultiplied layer over a BGR frame like the desktop compositor."""
    alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
    blended = layer[:, :, :3].astype(np.float32) + frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def remove_overlay(frame: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Recover the game pixels under the overlay's own (known) drawing.

    The capture shows ``C = P + B * (1 - a)``, where P/a are the overlay's
    premultiplied colour/alpha and B the game. Solving for B undoes the guide,
    so the analysis never mistakes its own lines for Miniclip's. Pixels the
    guide covers almost completely (overlapping strokes) are inpainted.
    """
    frame = np.ascontiguousarray(frame)
    alpha = layer[:, :, 3]
    covered = alpha > 0
    if not covered.any():
        return frame
    out = frame.copy()
    a = alpha[covered].astype(np.float32)[:, None] / 255.0
    c = frame[covered].astype(np.float32)
    premul = layer[:, :, :3][covered].astype(np.float32)
    background = (c - premul) / np.maximum(1.0 - a, .06)
    out[covered] = np.clip(np.rint(background), 0, 255).astype(np.uint8)
    solid = (alpha >= 240).astype(np.uint8)
    if solid.any():
        out = cv2.inpaint(out, solid, 3, cv2.INPAINT_TELEA)
    return out


def _debug_dump(folder: str, name: str, image: np.ndarray):
    """Write a diagnostics image atomically (readers never see half a file)."""
    try:
        temporary = os.path.join(folder, "~" + name)
        if cv2.imwrite(temporary, image):
            os.replace(temporary, os.path.join(folder, name))
    except Exception:
        pass


def _screen_scale(screen) -> tuple[QRect, float]:
    return screen.geometry(), float(screen.devicePixelRatio())


def logical_to_physical(rect: QRect, screen) -> dict:
    """Qt logical window rect → physical-pixel capture rect (mss coordinates).

    Qt keeps each screen's top-left at its native position and scales within
    the screen. At 100% scaling this is the identity.
    """
    origin, scale = _screen_scale(screen)
    left = origin.x() + round((rect.x() - origin.x()) * scale)
    top = origin.y() + round((rect.y() - origin.y()) * scale)
    return {"left": left, "top": top, "width": max(1, round(rect.width() * scale)),
            "height": max(1, round(rect.height() * scale))}


def physical_to_logical(x: int, y: int, width: int, height: int) -> QRect:
    """Physical desktop rect (from a full-desktop capture) → Qt logical rect."""
    screens = QGuiApplication.screens()
    chosen = QGuiApplication.primaryScreen()
    cx, cy = x + width / 2, y + height / 2
    for screen in screens:
        origin, scale = _screen_scale(screen)
        if (origin.x() <= cx < origin.x() + origin.width() * scale
                and origin.y() <= cy < origin.y() + origin.height() * scale):
            chosen = screen
            break
    origin, scale = _screen_scale(chosen)
    return QRect(origin.x() + round((x - origin.x()) / scale), origin.y() + round((y - origin.y()) / scale),
                 max(1, round(width / scale)), max(1, round(height / scale)))


class Overlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.settings = QSettings("Codex", "PoolPracticeOverlay")
        # The earlier geometry was saved against a smaller AirPlay window. Use
        # a fresh key so the current full-screen mirror starts over the whole
        # table once, then remember the user's later adjustments.
        saved_geometry = self.settings.value("overlay/geometry_v2")
        if not saved_geometry or not self.restoreGeometry(saved_geometry):
            # Match the complete table visible in the current 1808x1080 mirror.
            self.setGeometry(308, 280, 1190, 660)
        self.active = False
        self.adjusting = True
        self.detection = Detection()
        self._drag = None
        self._resize = False
        # What the last paintEvent drew, so analysis can subtract exactly that.
        self._painted = None
        self.painted_at = 0.0
        self.setWindowTitle("Pool Practice Guide")
        self.show()
        self._allow_capture()

    def _allow_capture(self):
        """Keep the guide in screenshots, recordings and screen shares.

        WDA_NONE explicitly; the analysis removes the guide's own pixels with
        ``remove_overlay`` instead of hiding the window from capture.
        """
        if sys.platform == "win32":
            try:
                ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x00000000)
            except Exception:
                pass

    def scale(self) -> float:
        screen = self.screen()
        return float(screen.devicePixelRatio()) if screen else 1.0

    def capture_rect(self) -> dict:
        """The overlay's area in physical screen pixels (what mss captures)."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        return logical_to_physical(self.geometry(), screen)

    def render_layer(self) -> np.ndarray | None:
        """Pixels the overlay currently shows on screen, or None if it is hidden."""
        if not self.isVisible() or self._painted is None:
            return None
        detection, adjusting, active, width, height, ui_scale = self._painted
        return render_guide_layer(detection, width, height, adjusting, active, ui_scale)

    def set_active(self, active: bool):
        self.active = active
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self.adjusting)
        self.update()

    def set_adjusting(self, adjusting: bool):
        self.adjusting = adjusting
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not adjusting)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, not adjusting)
        self.show()
        if not adjusting:
            self.settings.setValue("overlay/geometry_v2", self.saveGeometry())
        self.update()

    def closeEvent(self, event):
        self.settings.setValue("overlay/geometry_v2", self.saveGeometry())
        super().closeEvent(event)

    def mousePressEvent(self, e):
        if not self.adjusting or e.button() != Qt.MouseButton.LeftButton:
            return
        self._resize = e.position().x() > self.width()-26 and e.position().y() > self.height()-26
        self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if not self.adjusting or self._drag is None:
            return
        p = e.globalPosition().toPoint()
        if self._resize:
            self.resize(max(300, p.x()-self.x()), max(200, p.y()-self.y()))
        else:
            self.move(p-self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None
        self._resize = False

    def paintEvent(self, _):
        # Detection coordinates are physical capture pixels; draw in that space
        # so the guide lines up at any Windows display scaling.
        scale = self.scale()
        width, height = round(self.width() * scale), round(self.height() * scale)
        p = QPainter(self)
        p.scale(1 / scale, 1 / scale)
        paint_guide(p, self.detection, width, height, self.adjusting, self.active, scale)
        p.end()
        self._painted = (self.detection, self.adjusting, self.active, width, height, scale)
        self.painted_at = time.monotonic()


class SimCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(860, 500)
        self.cue = [0.0, 0.0, 12, QColor(248, 248, 242), 0]
        self.balls = []
        self.angle = 0
        self.power = 0.68
        self.dragging = None
        self.result = None
        self.frame_index = 0
        self.pocketed = set()
        self._last_table = None
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.advance)
        self.reset_rack()

    def table(self):
        margin = 38
        available_width = max(200, self.width() - margin * 2)
        available_height = max(100, self.height() - margin * 2)
        if available_width / available_height > 2.0:
            height = available_height
            width = height * 2.0
        else:
            width = available_width
            height = width / 2.0
        left = (self.width() - width) / 2
        top = (self.height() - height) / 2
        return (int(round(left)), int(round(top)), int(round(width)), int(round(height)))

    def reset_rack(self):
        left, top, width, height = self.table()
        radius = max(9, min(14, int(height * .031)))
        center_y = top + height / 2
        self.cue = [left + width * .22, center_y, radius, QColor(248, 248, 242), 0]
        palette = {
            1: (246, 190, 31), 2: (43, 108, 196), 3: (204, 54, 55),
            4: (122, 70, 167), 5: (230, 123, 36), 6: (42, 142, 68),
            7: (133, 55, 45), 8: (28, 30, 36), 9: (246, 190, 31),
            10: (43, 108, 196), 11: (204, 54, 55), 12: (122, 70, 167),
            13: (230, 123, 36), 14: (42, 142, 68), 15: (133, 55, 45),
        }
        # Standard five-row triangle with the 8 ball in the center.
        rack_order = (1, 9, 2, 10, 8, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)
        rack_tip_x = left + width * .72
        index = 0
        self.balls = []
        for row in range(5):
            x = rack_tip_x + row * radius * 1.78
            for slot in range(row + 1):
                number = rack_order[index]
                index += 1
                y = center_y + (slot - row / 2) * radius * 2.04
                self.balls.append([x, y, radius, QColor(*palette[number]), number])
        self.result = None
        self.frame_index = 0
        self.pocketed.clear()
        self.timer.stop()
        self._last_table = (left, top, width, height)
        self.update()

    def resizeEvent(self, event):
        new_table = self.table()
        old_table = self._last_table
        if old_table and old_table[2] > 0 and old_table[3] > 0:
            old_left, old_top, old_width, old_height = old_table
            new_left, new_top, new_width, new_height = new_table
            sx, sy = new_width / old_width, new_height / old_height
            for ball in [self.cue] + self.balls:
                ball[0] = new_left + (ball[0] - old_left) * sx
                ball[1] = new_top + (ball[1] - old_top) * sy
                ball[2] = max(8, round(ball[2] * min(sx, sy)))
            self.result = None
            self.pocketed.clear()
            self.timer.stop()
        self._last_table = new_table
        super().resizeEvent(event)

    def first_hit(self):
        theta = np.radians(self.angle)
        direction = (float(np.cos(theta)), float(np.sin(theta)))
        cx, cy, cr = self.cue[:3]
        hits = []
        for i, ball in enumerate(self.balls):
            if ball[4] in self.pocketed:
                continue
            bx, by, br = ball[:3]
            dx, dy = bx - cx, by - cy
            along = dx * direction[0] + dy * direction[1]
            across = abs(dx * direction[1] - dy * direction[0])
            if along > cr and across < cr + br:
                lead = along - np.sqrt(max(0, (cr + br) ** 2 - across ** 2))
                if lead > 0:
                    hits.append((lead, i))
        if not hits:
            return direction, None, None
        _, index = min(hits)
        left, top, width, height = self.table()
        paths = collision_paths(tuple(map(int, self.cue[:3])),
                                tuple(map(int, self.balls[index][:3])), direction,
                                (left, top, width, height))
        return direction, index, paths

    def shoot(self):
        left, top, width, height = self.table()
        physics_balls = [PhysicsBall(0, self.cue[0], self.cue[1], self.cue[2])]
        physics_balls.extend(PhysicsBall(ball[4], ball[0], ball[1], ball[2])
                             for ball in self.balls if ball[4] not in self.pocketed)
        self._physics_ball_order = physics_balls[1:]
        self.pocketed_at_shot = set(self.pocketed)
        self.result = simulate_shot(physics_balls, (left, top, left + width, top + height),
                                    self.angle, self.power)
        self.frame_index = 0
        if self.result.frames:
            self.timer.start()
        self.update()

    def advance(self):
        if self.result is None or not self.result.frames:
            self.timer.stop()
            return
        if self.frame_index >= len(self.result.frames):
            self.timer.stop()
            self.update()
            return
        frame = self.result.frames[self.frame_index]
        self.cue[0], self.cue[1] = frame[0].x, frame[0].y
        self.pocketed = set(getattr(self, "pocketed_at_shot", set()))
        by_number = {ball[4]: ball for ball in self.balls}
        for i, physics_ball in enumerate(self._physics_ball_order, start=1):
            state = frame[i]
            target = by_number.get(physics_ball.number)
            if target is not None:
                target[0], target[1] = state.x, state.y
                if state.pocketed:
                    self.pocketed.add(physics_ball.number)
        self.frame_index += 1
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, top, width, height = self.table(); right, bottom = left + width, top + height
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(112, 61, 36))
        p.drawRoundedRect(left - 14, top - 14, width + 28, height + 28, 20, 20)
        p.setBrush(QColor(36, 138, 148)); p.drawRoundedRect(left, top, width, height, 12, 12)
        p.setBrush(QColor(8, 14, 20, 245))
        pocket_centers = [(left, top), ((left + right)//2, top), (right, top),
                          (left, bottom), ((left + right)//2, bottom), (right, bottom)]
        for px, py in pocket_centers:
            p.drawEllipse(QPoint(px, py), 16, 16)

        direction, target_index, paths = self.first_hit()
        ux, uy = direction; cx, cy, cue_radius = self.cue[:3]
        if self.result is not None:
            for i, trail in enumerate(self.result.trails):
                color = QColor(255, 255, 255, 95) if i == 0 else QColor(255, 213, 70, 100)
                if len(trail) > 1:
                    p.setPen(QPen(color, 1.5, Qt.PenStyle.DotLine))
                    for a, b in zip(trail, trail[1:]):
                        p.drawLine(QPoint(round(a[0]), round(a[1])), QPoint(round(b[0]), round(b[1])))
        elif paths:
            approach, cue_path, object_path, _ = paths
            for segments, color, style, stroke in (
                (approach, QColor(45, 225, 255, 225), Qt.PenStyle.SolidLine, 3),
                (object_path, QColor(255, 210, 55, 225), Qt.PenStyle.SolidLine, 3),
                (cue_path, QColor(255, 70, 180, 220), Qt.PenStyle.DashLine, 2),
            ):
                for index, (a, b) in enumerate(segments):
                    segment_color = QColor(color)
                    if index: segment_color.setAlpha(145)
                    p.setPen(QPen(segment_color, stroke, Qt.PenStyle.DashLine if index else style))
                    p.drawLine(QPoint(*a), QPoint(*b))
        else:
            bounds = (left + cue_radius, top + cue_radius, right - cue_radius, bottom - cue_radius)
            ray = trace_bounces((cx, cy), direction, bounds, cue_radius, max_bounces=1)
            p.setPen(QPen(QColor(255, 255, 255, 220), 2, Qt.PenStyle.DashLine))
            for a, b in ray: p.drawLine(QPoint(*a), QPoint(*b))

        by_number = {ball[4]: ball for ball in self.balls}
        if self.result is not None and self.result.frames:
            for i, physics_ball in enumerate(self._physics_ball_order, start=1):
                state = self.result.frames[min(self.frame_index, len(self.result.frames)-1)][i]
                if state.pocketed: self.pocketed.add(physics_ball.number)
        for ball in self.balls:
            bx, by, radius, color, number = ball
            if number in self.pocketed: continue
            p.setPen(QPen(QColor(245, 245, 245, 230), 1.5)); p.setBrush(color)
            p.drawEllipse(QPoint(round(bx), round(by)), radius, radius)
            if 9 <= number <= 15:
                p.setBrush(QColor(245, 245, 238)); p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(round(bx-radius), round(by-radius*.42), radius*2, round(radius*.84), radius*.3, radius*.3)
            # Miniclip-style number badges keep each ball identifiable even in a
            # compact window; the 8-ball gets the same white badge as the rest.
            p.setBrush(QColor(248, 248, 242)); p.setPen(QPen(QColor(25, 28, 32, 210), 0.8))
            badge_radius = max(5, round(radius * .48))
            p.drawEllipse(QPoint(round(bx), round(by)), badge_radius, badge_radius)
            p.setPen(QPen(QColor(12, 14, 17), 1))
            p.setFont(QFont("Arial", max(8, round(radius * .82)), QFont.Weight.Bold))
            p.drawText(QRectF(bx - badge_radius, by - badge_radius, badge_radius * 2, badge_radius * 2),
                       Qt.AlignmentFlag.AlignCenter, str(number))
            if target_index is not None and ball is self.balls[target_index] and self.result is None:
                p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor(255, 230, 85), 2.5))
                p.drawEllipse(QPoint(round(bx), round(by)), radius + 4, radius + 4)

        p.setPen(QPen(QColor(65, 245, 255), 2.5)); p.setBrush(QColor(248, 248, 242))
        p.drawEllipse(QPoint(round(cx), round(cy)), cue_radius, cue_radius)
        p.setPen(QPen(QColor(120, 30, 30), 2)); p.drawPoint(QPoint(round(cx), round(cy)))
        if self.result is not None and self.result.frames:
            moving = self.frame_index < len(self.result.frames)
            if not moving:
                p.setPen(QColor(235, 240, 245)); p.setFont(QFont("Arial", 10))
                p.drawText(left + 10, bottom - 10,
                           f"{self.result.ball_collisions} ball hits · {self.result.cushion_collisions} cushion hits · potted {len(self.result.pocketed)}")
        p.end()

    def mousePressEvent(self,e):
        x,y=e.position().x(),e.position().y()
        if np.hypot(x-self.cue[0],y-self.cue[1])<25:self.dragging='cue';return
        for i,b in enumerate(self.balls):
            if np.hypot(x-b[0],y-b[1])<25:self.dragging=i;return

    def mouseMoveEvent(self,e):
        if self.dragging is None:return
        left,top,w,h=self.table();x=np.clip(e.position().x(),left+24,left+w-24);y=np.clip(e.position().y(),top+24,top+h-24)
        if self.dragging=='cue':self.cue[0],self.cue[1]=x,y
        else:self.balls[self.dragging][0],self.balls[self.dragging][1]=x,y
        self.result=None;self.pocketed.clear();self.timer.stop()
        self.update()

    def mouseReleaseEvent(self,e):self.dragging=None


class ShotSimulator(QMainWindow):
    closed = Signal()
    def __init__(self):
        super().__init__();self.setWindowTitle('Pool Shot Simulator');self.resize(1000,700)
        self.canvas=SimCanvas();self.angle_text=QLabel('Aim: 0°');self.power_text=QLabel('Power: 68%')
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setRange(-180,180);self.slider.setValue(0)
        self.slider.valueChanged.connect(self.set_angle)
        self.power_slider=QSlider(Qt.Orientation.Horizontal);self.power_slider.setRange(10,100);self.power_slider.setValue(68)
        self.power_slider.valueChanged.connect(self.set_power)
        self.shoot_button=QPushButton('Shoot');self.shoot_button.clicked.connect(self.shoot)
        self.reset=QPushButton('Reset rack');self.reset.clicked.connect(self.reset_scene)
        controls=QHBoxLayout();controls.addWidget(self.angle_text);controls.addWidget(self.slider);controls.addWidget(self.power_text);controls.addWidget(self.power_slider);controls.addWidget(self.shoot_button);controls.addWidget(self.reset)
        intro=QLabel('A full 15-ball practice table. Drag balls, set aim and power, then shoot. '
                     'Ball collisions, cushion rebounds, cloth drag and pockets run without a phone or game feed.')
        intro.setWordWrap(True)
        limits=QLabel('Approximate 2D physics: no spin, english, swerve or jump, and no 8-ball rules '
                      '(fouls, solids/stripes, turns). Use it to practise angles, not to predict exact results.')
        limits.setWordWrap(True);limits.setStyleSheet('color:#667085')
        layout=QVBoxLayout();layout.addWidget(intro);layout.addWidget(limits);layout.addWidget(self.canvas);layout.addLayout(controls)
        root=QWidget();root.setLayout(layout);self.setCentralWidget(root)
    def set_angle(self,value):self.canvas.angle=value;self.angle_text.setText(f'Aim: {value}°');self.canvas.update()
    def set_power(self,value):self.canvas.power=value/100;self.power_text.setText(f'Power: {value}%')
    def shoot(self):
        self.canvas.shoot()
        self.shoot_button.setEnabled(False)
        QTimer.singleShot(max(500, len(self.canvas.result.frames)*16 + 80),lambda:self.shoot_button.setEnabled(True))
    def reset_scene(self):
        self.canvas.reset_rack();self.slider.setValue(0);self.power_slider.setValue(68);self.shoot_button.setEnabled(True)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class ScreenshotCanvas(QWidget):
    """Show detector output on a saved frame without needing a live mirror."""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 420)
        self.image = None
        self.detection = Detection()

    def load_image(self, path: str) -> Detection | None:
        frame = cv2.imread(path)
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.image = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        self.detection = detect(frame)
        self.update()
        return self.detection

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 28, 34))
        if self.image is None:
            p.setPen(QColor(225, 230, 235))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a saved pool screenshot to inspect the detected lines")
            p.end()
            return
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        ox, oy = (self.width() - iw * scale) / 2, (self.height() - ih * scale) / 2
        p.translate(ox, oy)
        p.scale(scale, scale)
        p.drawImage(0, 0, self.image)
        # Same drawing as the live overlay, with stroke widths kept constant on screen.
        paint_guide(p, self.detection, iw, ih, adjusting=False, active=True, ui_scale=1 / scale)
        p.end()


class ScreenshotTester(QMainWindow):
    closed = Signal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test a Pool Screenshot")
        self.resize(1100, 760)
        self.canvas = ScreenshotCanvas()
        self.status = QLabel("Cyan: cue approach to the game's ghost ball · yellow: object ball · "
                             "magenta dashes: cue ball after impact · orange dots: a ball in the way.")
        self.status.setWordWrap(True)
        self.official_button = QPushButton("Miniclip aim sample")
        self.official_button.clicked.connect(lambda: self.load_sample("miniclip_example_aiming.png", "Miniclip aiming-guide sample"))
        self.earlier_button = QPushButton("Earlier table sample")
        self.earlier_button.clicked.connect(lambda: self.load_sample("example_pool_screenshot.png", "Earlier table screenshot"))
        self.open_button = QPushButton("Open screenshot…")
        self.open_button.clicked.connect(self.open_image)
        bar = QHBoxLayout(); bar.addWidget(self.official_button); bar.addWidget(self.earlier_button); bar.addWidget(self.open_button); bar.addWidget(self.status, 1)
        layout = QVBoxLayout(); layout.addLayout(bar); layout.addWidget(self.canvas, 1)
        root = QWidget(); root.setLayout(layout); self.setCentralWidget(root)
        self.load_sample("miniclip_example_aiming.png", "Miniclip aiming-guide sample")

    def load_sample(self, filename: str, label: str):
        sample = Path(__file__).resolve().parent / "outputs" / filename
        detection = self.canvas.load_image(str(sample))
        if detection:
            self.status.setText(f"{label} · {detection.note}")
        else:
            self.status.setText("The built-in screenshot sample is missing. Open a saved screenshot instead.")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open a pool screenshot", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        detection = self.canvas.load_image(path)
        if detection is None:
            self.status.setText("That image could not be opened. Choose a PNG, JPG, or BMP screenshot.")
        else:
            self.status.setText(detection.note)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class Control(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowTitle("Pool Practice Overlay")
        self.setMinimumWidth(350)
        self.overlay = Overlay()
        self.capture = mss.MSS()
        self.last_thumbnail = None
        self.analysis_pending = False
        self.analysis_task = None
        self.thread_pool = QThreadPool.globalInstance()
        self.cue_tracker = CueTracker()
        self.label = QLabel("Position the guide over the complete pool table, then enable it.")
        self.label.setWordWrap(True)
        self.toggle = QPushButton("Enable guide")
        self.adjust = QPushButton("Finish positioning")
        self.fit = QPushButton("Fit table")
        self.simulator = QPushButton("Shot simulator")
        self.simulator.clicked.connect(self.open_simulator)
        self.simulator_window = None
        self.screenshot_test = QPushButton("Test a screenshot")
        self.screenshot_test.clicked.connect(self.open_screenshot_tester)
        self.screenshot_window = None
        self.synthetic = QPushButton("Synthetic test table")
        self.synthetic.setToolTip("A generated pool table with a white aim guide, for checking the overlay "
                                  "without AirPlay. Space or → shows the next shot.")
        self.synthetic.clicked.connect(self.open_synthetic_table)
        self.synthetic_window = None
        self.redetect = QPushButton("Re-detect")
        self.redetect.setEnabled(False)
        self.toggle.clicked.connect(self.toggle_guide)
        self.adjust.clicked.connect(self.toggle_adjust)
        self.fit.clicked.connect(self.fit_table)
        self.redetect.clicked.connect(self.analyze)
        row = QHBoxLayout(); row.addWidget(self.toggle); row.addWidget(self.adjust); row.addWidget(self.fit)
        layout = QVBoxLayout(); layout.addWidget(QLabel("Miniclip 8 Ball Pool · practice overlay")); layout.addLayout(row)
        layout.addWidget(self.simulator)
        layout.addWidget(self.screenshot_test)
        layout.addWidget(self.synthetic)
        layout.addWidget(self.redetect); layout.addWidget(self.label)
        note = QLabel("Visual estimates only. No gameplay inputs are sent.")
        note.setStyleSheet("color:#667085")
        layout.addWidget(note)
        root = QWidget(); root.setLayout(layout); self.setCentralWidget(root)
        self.timer = QTimer(self); self.timer.setInterval(125); self.timer.timeout.connect(self.analyze)
        self.resize(420, 190)
        # Keep the controls in the black letterbox area above the table so
        # they cannot hide cue/object balls from the screen analysis.
        self.move(12, 18)
        QTimer.singleShot(700, self.fit_table)

    def open_simulator(self):
        if self.simulator_window is None:
            self.simulator_window=ShotSimulator()
            self.simulator_window.closed.connect(self.restore_overlay)
        self.suspend_overlay()
        self.simulator_window.show();self.simulator_window.raise_();self.simulator_window.activateWindow()

    def open_synthetic_table(self):
        """Open a generated table to try the live overlay without a phone.

        The overlay stays visible (that is the point): press Fit table, then
        Enable guide, and compare the guide with the white game-style lines.
        """
        import random
        import fake_game
        if self.synthetic_window is None:
            screen = QGuiApplication.primaryScreen().availableGeometry()
            width = int(min(1836, screen.width() * .85))
            self.synthetic_window = fake_game.make_table_window(seed=random.randint(0, 10 ** 6),
                                                                interval_ms=4000, width=width)
            self.synthetic_window.move(screen.x() + (screen.width() - width) // 2,
                                       screen.y() + max(0, (screen.height() - int(width * .6)) // 2))
        self.synthetic_window.show()
        self.synthetic_window.raise_()
        self.label.setText("Synthetic test table open · press Fit table, then Enable guide")

    def open_screenshot_tester(self):
        if self.screenshot_window is None:
            self.screenshot_window = ScreenshotTester()
            self.screenshot_window.closed.connect(self.restore_overlay)
        self.suspend_overlay()
        self.screenshot_window.show(); self.screenshot_window.raise_(); self.screenshot_window.activateWindow()

    def suspend_overlay(self):
        self.overlay.hide()
        if self.overlay.active:
            self.timer.stop()

    def restore_overlay(self):
        self.overlay.show()
        self.overlay.raise_()
        if self.overlay.active:
            self.timer.start()
            self.analyze()

    def _grab_without_overlay(self, region: dict) -> np.ndarray | None:
        """Capture ``region`` (physical pixels) and undo the guide's own drawing.

        Returns None when the overlay changed after its last on-screen paint,
        so the subtraction would not match what the capture shows.
        """
        frame = np.ascontiguousarray(np.asarray(self.capture.grab(region))[:, :, :3])
        # This control panel is always on top and dark; where it overlaps the
        # capture (often right above the table) it can merge with a pocket.
        # Neutral grey is neither dark, felt nor white, so analysis ignores it.
        if self.isVisible() and not self.isMinimized():
            panel = logical_to_physical(self.frameGeometry(), self.screen() or QGuiApplication.primaryScreen())
            shadow = 8  # Windows 11 draws a soft drop shadow outside the frame.
            px0 = max(0, panel["left"] - shadow - region["left"])
            py0 = max(0, panel["top"] - shadow - region["top"])
            px1 = min(frame.shape[1], panel["left"] + panel["width"] + shadow - region["left"])
            py1 = min(frame.shape[0], panel["top"] + panel["height"] + shadow - region["top"])
            if px1 > px0 and py1 > py0:
                frame[py0:py1, px0:px1] = 128
        layer = self.overlay.render_layer()
        if layer is None:
            return frame
        debug_dir = os.environ.get("POOL_OVERLAY_DEBUG_DIR")
        if debug_dir:
            # Opt-in diagnostics: what the screen showed, the guide layer that
            # is subtracted, and the frame the detector actually analyses.
            _debug_dump(debug_dir, "capture.png", frame)
            _debug_dump(debug_dir, "layer.png", layer)
        overlay = self.overlay.capture_rect()
        if (layer.shape[1], layer.shape[0]) != (overlay["width"], overlay["height"]):
            return None
        # Place the layer where the overlay sits inside the captured region.
        x0 = overlay["left"] - region["left"]
        y0 = overlay["top"] - region["top"]
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1 = min(frame.shape[1], x0 + layer.shape[1])
        fy1 = min(frame.shape[0], y0 + layer.shape[0])
        if fx1 <= fx0 or fy1 <= fy0:
            return frame
        part = layer[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        frame[fy0:fy1, fx0:fx1] = remove_overlay(frame[fy0:fy1, fx0:fx1], part)
        if debug_dir:
            _debug_dump(debug_dir, "cleaned.png", frame)
        return frame

    def _place_overlay_on_table(self, x: int, y: int, width: int, height: int):
        """Cover a table found at a physical desktop rect, plus a pocket margin.

        The margin keeps the six pockets inside each analysis frame, so every
        frame refits the table from its pockets instead of the felt colour.
        """
        margin = round(width * .06)
        desktop = self.capture.monitors[0]
        left = max(desktop["left"], x - margin)
        top = max(desktop["top"], y - margin)
        right = min(desktop["left"] + desktop["width"], x + width + margin)
        bottom = min(desktop["top"] + desktop["height"], y + height + margin)
        self.overlay.setGeometry(physical_to_logical(left, top, right - left, bottom - top))
        self.overlay.settings.setValue("overlay/geometry_v2", self.overlay.saveGeometry())
        self.overlay.update()

    def fit_table(self):
        """Find the table on the full desktop and resize the guide around it."""
        if time.monotonic() - self.overlay.painted_at < PAINT_SETTLE:
            QTimer.singleShot(100, self.fit_table)
            return
        try:
            desktop = self.capture.monitors[0]
            frame = self._grab_without_overlay(desktop)
            if frame is None:
                QTimer.singleShot(100, self.fit_table)
                return
            table = find_table_bounds(frame)
            if table is None:
                if self._fit_airplay_layout():
                    self.label.setText("Table fitted to AirPlay view · check alignment, then enable guide")
                else:
                    self.label.setText("Table not found on screen · move the mirrored pool table into view")
                return
            x, y, width, height = table
            self._place_overlay_on_table(desktop["left"] + x, desktop["top"] + y, width, height)
            self.label.setText("Table fitted · finish positioning, then enable the guide")
            if self.overlay.active:
                self.last_thumbnail = None
                QTimer.singleShot(150, self.analyze)
        except Exception as exc:
            if self._fit_airplay_layout():
                self.label.setText("Table fitted to AirPlay view · check alignment, then enable guide")
                if self.overlay.active:
                    QTimer.singleShot(150, self.analyze)
            else:
                self.label.setText(f"Could not fit table: {exc}")

    def _fit_airplay_layout(self) -> bool:
        """Fallback fit for the known UxPlay window when desktop capture fails."""
        if sys.platform != "win32":
            return False
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
            user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
            user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
            user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
            found = []
            enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def visit(hwnd, _):
                size = user32.GetWindowTextLengthW(hwnd)
                if size:
                    title = ctypes.create_unicode_buffer(size + 1)
                    user32.GetWindowTextW(hwnd, title, size + 1)
                    if title.value in ("AirPlay Video Stream", "uxplay-windows") and not user32.IsIconic(hwnd):
                        found.append(hwnd)
                return True
            user32.EnumWindows(enum_proc(visit), 0)
            if not found:
                return False
            valid=[]
            for hwnd in found:
                client=RECT()
                if user32.GetClientRect(hwnd,ctypes.byref(client)):
                    width,height=client.right-client.left,client.bottom-client.top
                    if width>=900 and height>=500:valid.append((width*height,hwnd,width,height))
            if not valid:return False
            _,hwnd,width,height=max(valid,key=lambda item:item[0])
            origin = POINT(0, 0)
            if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
                return False
            # UxPlay's Miniclip view places the felt inside fixed dark side
            # panels. These proportions follow the visible 2:1 playing area.
            x, y = origin.x + int(width*.205), origin.y + int(height*.225)
            table_w, table_h = int(width*.594), int(height*.642)
            # Win32 client rects are physical pixels, like the mss capture.
            self._place_overlay_on_table(x, y, table_w, table_h)
            return True
        except Exception:
            return False

    def toggle_guide(self):
        active = not self.overlay.active
        if active and self.overlay.adjusting:
            self.overlay.set_adjusting(False)
            self.adjust.setText("Adjust overlay")
        self.overlay.set_active(active)
        self.toggle.setText("Disable guide" if active else "Enable guide")
        self.redetect.setEnabled(active)
        if active:
            self.last_thumbnail = None
            self.analyze(); self.timer.start()
        else:
            self.timer.stop(); self.last_thumbnail = None; self.overlay.detection = Detection(note="Guide is off")
            self.cue_tracker.reset()
            self.overlay.update(); self.label.setText("Guide is off")

    def toggle_adjust(self):
        adjusting = not self.overlay.adjusting
        self.overlay.set_adjusting(adjusting)
        self.adjust.setText("Finish positioning" if adjusting else "Adjust overlay")

    def analyze(self):
        if not self.overlay.active:
            return
        if self.analysis_pending:
            return
        if self.overlay.adjusting:
            self.label.setText("Positioning mode · analysis paused")
            return
        # Read the screen only once the compositor shows the latest guide
        # drawing; that drawing is subtracted below, so it must be the one on
        # screen. The 125 ms timer simply retries on its next tick.
        if time.monotonic() - self.overlay.painted_at < PAINT_SETTLE:
            return
        g = self.overlay.geometry()
        try:
            # Capture in physical pixels; detection coordinates stay in that
            # space and the overlay paints in it too (see Overlay.paintEvent).
            shot = self._grab_without_overlay(self.overlay.capture_rect())
            if shot is None:
                return
            thumbnail = cv2.resize(cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY), (128, 64), interpolation=cv2.INTER_AREA)
            if self.last_thumbnail is not None and float(cv2.absdiff(thumbnail, self.last_thumbnail).mean()) < .35:
                return
            self.last_thumbnail = thumbnail
            self.analysis_pending = True
            self.analysis_task = AnalysisTask(shot, g)
            self.analysis_task.signals.completed.connect(self._analysis_completed)
            self.analysis_task.signals.failed.connect(self._analysis_failed)
            self.thread_pool.start(self.analysis_task)
        except Exception as exc:
            self.label.setText(f"Screen capture failed: {exc}")

    def _analysis_completed(self, detection: Detection, geometry: QRect):
        self.analysis_pending = False
        self.analysis_task = None
        if not self.overlay.active or self.overlay.adjusting or self.overlay.geometry() != geometry:
            return
        detection = self.cue_tracker.update(detection, geometry)
        self.overlay.detection = detection
        self.label.setText(detection.note)
        self.overlay.update()

    def _analysis_failed(self, message: str):
        self.analysis_pending = False
        self.analysis_task = None
        if self.overlay.active:
            self.label.setText(f"Image analysis failed: {message}")

    def closeEvent(self, e):
        self.timer.stop()
        self.overlay.close()
        if self.synthetic_window is not None:
            self.synthetic_window.close()
        self.capture.close()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    control = Control(); control.show(); control.raise_(); control.activateWindow()
    sys.exit(app.exec())
