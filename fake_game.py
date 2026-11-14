"""Synthetic pool-table frames with exactly known geometry, for testing the overlay.

This is a test fixture, not a game. It draws a generic pool table in the
visual language the detector relies on: blue cloth with lighter cushions,
dark pockets, numbered solid/striped balls with shading, a decorated cue
stick, and a white aim guide (ray, hollow ghost ball, short object-ball and
cue-ball lines). Because every frame is drawn from known numbers, a test can
check the detector's table fit, ball list, aim angle, contact point and
predicted directions against the truth.

Run ``python fake_game.py`` for a window that shows such frames for a live
overlay test (see ``FakeTableWindow``); it never reads or controls any game.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

# Cushion nose sits this fraction of the corner-pocket span inside each pocket
# centre (same constant the detector uses; measured on real frames).
POCKET_TO_CUSHION = .0095

# Ball colours (RGB) by number; stripes 9-15 reuse 1-7.
PALETTE = {1: (246, 190, 31), 2: (43, 108, 196), 3: (204, 54, 55), 4: (122, 70, 167),
           5: (230, 123, 36), 6: (42, 142, 68), 7: (133, 55, 45), 8: (28, 30, 36)}


def _bgr(rgb):
    return np.array(rgb[::-1], np.float32)


def _hsv_bgr(h, s, v):
    pixel = np.uint8([[[h, s, v]]])
    return cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


@dataclass
class Scene:
    width: int
    height: int
    table: tuple[float, float, float, float]          # cushion-nose rect x, y, w, h
    radius: float
    balls: dict[int, tuple[float, float]]             # number -> centre
    cue: tuple[float, float]
    aim_degrees: float
    show_guide: bool = True
    stick_pull: float = 0.0                           # extra stick gap (in radii)
    jpeg_quality: int | None = None
    blur: float = 0.0
    letterbox: float = .07
    seed: int = 0


@dataclass
class Truth:
    table: tuple[float, float, float, float]
    radius: float
    cue: tuple[float, float]
    balls: dict[int, tuple[float, float]]
    aim_degrees: float
    ghost: tuple[float, float] | None
    target: int | None
    object_dir: tuple[float, float] | None
    cue_dir: tuple[float, float] | None
    pockets: list[tuple[float, float]] = field(default_factory=list)


def pocket_centres(table):
    x, y, w, h = table
    inset = POCKET_TO_CUSHION * w / (1 - 2 * POCKET_TO_CUSHION)
    return [(x - inset, y - inset), (x + w / 2, y - 2 * inset), (x + w + inset, y - inset),
            (x - inset, y + h + inset), (x + w / 2, y + h + 2 * inset), (x + w + inset, y + h + inset)]


def truth_path(start, direction, table, radius, bounces=3, balls=(), pockets=None):
    """Exact path of a ball's centre: cushion bounces as mirrors, ending in a
    pocket mouth (rail contact within two radii of where the cushion nose
    meets the pocket) or where it first touches one of ``balls``.

    Returns (points, stop) with stop "bounces", "pocket" or "ball". Solved by
    unfolding the table (the path is a straight line in mirrored copies),
    independently of the detector's step-by-step tracer.
    """
    x0, y0, w, h = table
    lo_x, lo_y, span_x, span_y = x0 + radius, y0 + radius, w - 2 * radius, h - 2 * radius
    ux, uy = direction
    n = float(np.hypot(ux, uy))
    ux, uy = ux / n, uy / n
    sx = min(max(start[0] - lo_x, 0.0), span_x)
    sy = min(max(start[1] - lo_y, 0.0), span_y)

    def fold(v, span):
        v = v % (2 * span)
        return v if v <= span else 2 * span - v

    def crossings(s0, u, span):
        # Distances along the line at which it crosses a cushion line.
        if abs(u) < 1e-12:
            return []
        k = np.floor(s0 / span) + (1 if u > 0 else 0)
        if u < 0 and s0 <= 0:
            k = -1.0
        out = []
        for _ in range(bounces + 2):
            t = (k * span - s0) / u
            if t > .5:  # a ball already on that cushion rebounds at once
                out.append(t)
            k += 1 if u > 0 else -1
        return out

    events = sorted(set(round(t, 6) for t in crossings(sx, ux, span_x) + crossings(sy, uy, span_y)))
    corners = [(lo_x, lo_y), (lo_x + span_x / 2, lo_y), (lo_x + span_x, lo_y),
               (lo_x, lo_y + span_y), (lo_x + span_x / 2, lo_y + span_y), (lo_x + span_x, lo_y + span_y)]
    points = [(lo_x + sx, lo_y + sy)]
    travelled = 0.0
    for t in events[:bounces + 1]:
        px, py = points[-1]
        # Direction on this leg, from the fold of a point just past its start.
        e = travelled + 1e-3
        qx, qy = lo_x + fold(sx + ux * e, span_x), lo_y + fold(sy + uy * e, span_y)
        lx, ly = qx - px, qy - py
        ln = float(np.hypot(lx, ly)) or 1.0
        lx, ly = lx / ln, ly / ln
        leg = t - travelled
        hit = None
        for bx, by in balls:
            dx, dy = bx - px, by - py
            along = dx * lx + dy * ly
            across2 = dx * dx + dy * dy - along * along
            if along > 0 and across2 < (2 * radius) ** 2:
                d = along - np.sqrt((2 * radius) ** 2 - across2)
                if 1e-3 < d < leg and (hit is None or d < hit):
                    hit = d
        if hit is not None:
            points.append((px + lx * hit, py + ly * hit))
            return points, "ball"
        end = (lo_x + fold(sx + ux * t, span_x), lo_y + fold(sy + uy * t, span_y))
        points.append(end)
        if min(np.hypot(end[0] - cx, end[1] - cy) for cx, cy in corners) <= radius * 2.0:
            return points, "pocket"
        travelled = t
    return points, "bounces"


def solve_truth(scene: Scene) -> Truth:
    """Where the cue ball first touches a ball or cushion along the aim."""
    x, y, w, h = scene.table
    r = scene.radius
    cx, cy = scene.cue
    a = np.radians(scene.aim_degrees)
    ux, uy = float(np.cos(a)), float(np.sin(a))
    # Distance to the first cushion contact (ball centre one radius from nose).
    limits = []
    if ux > 1e-9:
        limits.append((x + w - r - cx) / ux)
    if ux < -1e-9:
        limits.append((x + r - cx) / ux)
    if uy > 1e-9:
        limits.append((y + h - r - cy) / uy)
    if uy < -1e-9:
        limits.append((y + r - cy) / uy)
    best_s, target = min(limits), None
    for number, (bx, by) in scene.balls.items():
        dx, dy = bx - cx, by - cy
        along = dx * ux + dy * uy
        across2 = dx * dx + dy * dy - along * along
        if along <= 0 or across2 >= (2 * r) ** 2:
            continue
        s = along - np.sqrt((2 * r) ** 2 - across2)
        if 0 < s < best_s:
            best_s, target = float(s), number
    ghost = (cx + ux * best_s, cy + uy * best_s)
    object_dir = cue_dir = None
    if target is not None:
        bx, by = scene.balls[target]
        nx, ny = bx - ghost[0], by - ghost[1]
        n = float(np.hypot(nx, ny))
        nx, ny = nx / n, ny / n
        object_dir = (nx, ny)
        dot = ux * nx + uy * ny
        tx, ty = ux - dot * nx, uy - dot * ny
        t = float(np.hypot(tx, ty))
        cue_dir = (tx / t, ty / t) if t > 1e-6 else None
    return Truth(scene.table, r, scene.cue, dict(scene.balls), scene.aim_degrees,
                 ghost if scene.show_guide else None, target, object_dir, cue_dir,
                 pocket_centres(scene.table))


# --- antialiased primitives on a float32 BGR canvas -------------------------

def _patch(canvas, cx, cy, reach):
    h, w = canvas.shape[:2]
    x0, x1 = max(0, int(np.floor(cx - reach))), min(w, int(np.ceil(cx + reach)) + 1)
    y0, y1 = max(0, int(np.floor(cy - reach))), min(h, int(np.ceil(cy + reach)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    return (slice(y0, y1), slice(x0, x1)), xx, yy


def _blend(canvas, region, coverage, color):
    view = canvas[region]
    cov = coverage[..., None]
    if np.ndim(color) == 3:
        view[:] = view * (1 - cov) + color * cov
    else:
        view[:] = view * (1 - cov) + np.asarray(color, np.float32) * cov


def fill_disc(canvas, cx, cy, radius, color, alpha=1.0):
    got = _patch(canvas, cx, cy, radius + 2)
    if got is None:
        return
    region, xx, yy = got
    d = np.hypot(xx - cx, yy - cy)
    _blend(canvas, region, np.clip(radius + .5 - d, 0, 1) * alpha, color)


def stroke_ring(canvas, cx, cy, radius, width, color):
    got = _patch(canvas, cx, cy, radius + width + 2)
    if got is None:
        return
    region, xx, yy = got
    d = np.abs(np.hypot(xx - cx, yy - cy) - radius)
    _blend(canvas, region, np.clip(width / 2 + .5 - d, 0, 1), color)


def stroke_line(canvas, p0, p1, width, color, alpha=1.0):
    (x0, y0), (x1, y1) = p0, p1
    reach = width + 2
    h, w = canvas.shape[:2]
    bx0, bx1 = max(0, int(min(x0, x1) - reach)), min(w, int(max(x0, x1) + reach) + 1)
    by0, by1 = max(0, int(min(y0, y1) - reach)), min(h, int(max(y0, y1) + reach) + 1)
    if bx1 <= bx0 or by1 <= by0:
        return
    yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / max(length2, 1e-9), 0, 1)
    d = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
    _blend(canvas, (slice(by0, by1), slice(bx0, bx1)), np.clip(width / 2 + .5 - d, 0, 1) * alpha, color)


def draw_ball(canvas, cx, cy, r, number, rng):
    """Shaded ball: solid, striped, 8 ball or cue ball (number 0)."""
    # Soft drop shadow first.
    got = _patch(canvas, cx + r * .2, cy + r * .25, r * 1.4)
    if got is not None:
        region, xx, yy = got
        d = np.hypot(xx - cx - r * .2, yy - cy - r * .25)
        _blend(canvas, region, np.clip((r * 1.12 - d) / (r * .35), 0, 1) * .35, (8, 12, 20))
    got = _patch(canvas, cx, cy, r + 2)
    if got is None:
        return
    region, xx, yy = got
    dx, dy = xx - cx, yy - cy
    d = np.hypot(dx, dy)
    white = np.array([236, 240, 240], np.float32)
    if number == 0:
        base = np.broadcast_to(white, dx.shape + (3,)).copy()
    else:
        colour = _bgr(PALETTE[number if number <= 8 else number - 8])
        base = np.broadcast_to(colour, dx.shape + (3,)).copy()
        if number >= 9:
            # Stripe: coloured band across a white ball at a random tilt.
            tilt = rng.uniform(-.6, .6)
            band = np.abs(-dx * np.sin(tilt) + dy * np.cos(tilt)) < r * .52
            base[~band] = white
    # Shading towards the rim and a specular highlight.
    shade = 1 - .38 * np.clip(d / r, 0, 1) ** 2.5
    base *= shade[..., None]
    spot = np.exp(-((dx + r * .35) ** 2 + (dy + r * .38) ** 2) / (2 * (r * .18) ** 2)) * .55
    base += (255 - base) * spot[..., None]
    _blend(canvas, region, np.clip(r + .5 - d, 0, 1), base)
    if number == 0:
        ox, oy = rng.uniform(-.35, .35) * r, rng.uniform(-.35, .35) * r
        fill_disc(canvas, cx + ox, cy + oy, r * .13, (40, 40, 200))
        return
    # Number badge with the number on it.
    bx, by = cx + rng.uniform(-.12, .12) * r, cy + rng.uniform(-.12, .12) * r
    fill_disc(canvas, bx, by, r * .45, (240, 242, 242))
    text = str(number)
    scale = r / 30
    thickness = max(1, int(round(r / 14)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(canvas, text, (int(round(bx - tw / 2)), int(round(by + th / 2))),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), thickness, cv2.LINE_AA)


def draw_pocket(canvas, px, py, pr):
    """Pocket mouth as on the real frames: a black core (brightness ~0) on the
    pocket centre and a dark-red rim (brightness ~55-80) on its upper-left."""
    fill_disc(canvas, px - pr * .18, py - pr * .18, pr * 1.12, (6, 10, 72))
    fill_disc(canvas, px, py, pr * .86, (2, 2, 4))


def render(scene: Scene) -> tuple[np.ndarray, Truth]:
    """Draw the scene; return (BGR uint8 frame, truth)."""
    rng = np.random.default_rng(scene.seed)
    W, H = scene.width, scene.height
    x, y, w, h = scene.table
    r = scene.radius
    span = w / (1 - 2 * POCKET_TO_CUSHION)
    canvas = np.empty((H, W, 3), np.float32)
    canvas[:] = _hsv_bgr(112, 150, 62)                    # navy surround
    bar = int(H * scene.letterbox)
    canvas[:bar] = 0                                      # letterbox bars
    canvas[H - bar:] = 0
    # Wooden rail.
    rail = span * .062
    rail_rect = (int(x - rail), int(y - rail), int(x + w + rail), int(y + h + rail))
    cv2.rectangle(canvas, rail_rect[:2], rail_rect[2:], tuple(float(v) for v in _bgr((150, 32, 26))), -1)
    cv2.rectangle(canvas, rail_rect[:2], rail_rect[2:], (40, 25, 60), 2)
    pockets = pocket_centres(scene.table)
    corner_r, middle_r = span * .032, span * .024
    for i, (px, py) in enumerate(pockets):
        pr = middle_r if i in (1, 4) else corner_r
        fill_disc(canvas, px, py, pr * 1.22, (178, 172, 176))   # metal rim
    # Cushions: a lighter blue band inside the rail, open at the pockets.
    cushion = span * .02
    cv2.rectangle(canvas, (int(round(x - cushion)), int(round(y - cushion))),
                  (int(round(x + w + cushion)), int(round(y + h + cushion))),
                  tuple(float(v) for v in _hsv_bgr(99, 175, 205)), -1)
    for i, (px, py) in enumerate(pockets):
        pr = middle_r if i in (1, 4) else corner_r
        draw_pocket(canvas, px, py, pr)
    # Cloth: lighter in the middle, faint noise. HSV measured on the real
    # frame: hue ~98, saturation ~135-150, value ~185 mid to ~137 at the rails.
    yy, xx = np.mgrid[0:int(np.ceil(h)) + 2, 0:int(np.ceil(w)) + 2].astype(np.float32)
    radial = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    value = (188 - 46 * np.clip(radial, 0, 1.4) + rng.normal(0, 2.2, radial.shape)).clip(0, 255)
    saturation = (134 + 14 * np.clip(radial, 0, 1.4)).clip(0, 255)
    cloth = np.dstack([np.full_like(value, 98), saturation, value]).astype(np.uint8)
    cloth = cv2.cvtColor(cloth, cv2.COLOR_HSV2BGR).astype(np.float32)
    x0i, y0i = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0i, y - y0i
    ch, cw = cloth.shape[:2]
    gy, gx = np.mgrid[y0i:y0i + ch, x0i:x0i + cw].astype(np.float32)
    # Antialiased cloth edge exactly on the nose rect.
    cov = (np.clip(gx - x + .5, 0, 1) * np.clip(x + w - gx + .5, 0, 1)
           * np.clip(gy - y + .5, 0, 1) * np.clip(y + h - gy + .5, 0, 1))
    region = (slice(y0i, y0i + ch), slice(x0i, x0i + cw))
    _blend(canvas, region, cov, cloth)
    # Bright cushion-nose highlight just outside the cloth.
    for p0, p1 in (((x, y - 1.2), (x + w, y - 1.2)), ((x, y + h + 1.2), (x + w, y + h + 1.2)),
                   ((x - 1.2, y), (x - 1.2, y + h)), ((x + w + 1.2, y), (x + w + 1.2, y + h))):
        stroke_line(canvas, p0, p1, 1.6, tuple(float(v) for v in _hsv_bgr(98, 150, 245)))
    # Re-open the pocket mouths over cushion and highlight.
    for i, (px, py) in enumerate(pockets):
        pr = middle_r if i in (1, 4) else corner_r
        draw_pocket(canvas, px, py, pr)
    # Diamonds on the rail.
    for k in range(1, 8):
        if k == 4:
            continue
        fill_disc(canvas, x + w * k / 8, y - rail * .55, r * .18, (235, 235, 235))
        fill_disc(canvas, x + w * k / 8, y + h + rail * .55, r * .18, (235, 235, 235))
    for k in (1, 2, 3):
        fill_disc(canvas, x - rail * .55, y + h * k / 4, r * .18, (235, 235, 235))
        fill_disc(canvas, x + w + rail * .55, y + h * k / 4, r * .18, (235, 235, 235))
    # Faint table decorations that must not become balls or aim lines:
    # a head-string line and a large watermark ring with lettering.
    stroke_line(canvas, (x + w / 4, y + 2), (x + w / 4, y + h - 2), 1.4,
                tuple(float(v) for v in _hsv_bgr(98, 120, 215)), alpha=.7)
    darker = tuple(float(v) for v in _hsv_bgr(101, 165, 140))
    stroke_ring(canvas, x + w * .55, y + h * .52, r * 3.2, r * .45, darker)
    cv2.putText(canvas, "PRACTICE", (int(x + w * .36), int(y + h * .6)), cv2.FONT_HERSHEY_DUPLEX,
                r / 9, darker, max(2, int(r / 3)), cv2.LINE_AA)
    # Balls.
    for number, (bx, by) in scene.balls.items():
        draw_ball(canvas, bx, by, r, number, rng)
    cx, cy = scene.cue
    draw_ball(canvas, cx, cy, r, 0, rng)
    truth = solve_truth(scene)
    a = np.radians(scene.aim_degrees)
    ux, uy = float(np.cos(a)), float(np.sin(a))
    # Cue stick behind the cue ball, with decorative rings and joints.
    start = r * (1.25 + scene.stick_pull)
    length = span * .52
    steps = 48
    for k in range(steps):
        t0 = start + length * k / steps
        t1 = start + length * (k + 1) / steps
        width = r * (.32 + .45 * k / steps)
        colour = (235, 235, 230) if k == 0 else ((70, 80, 40) if k % 9 < 6 else (160, 90, 30))
        stroke_line(canvas, (cx - ux * t0, cy - uy * t0), (cx - ux * t1, cy - uy * t1), width, colour)
    for frac in (.18, .31, .62, .78):
        t = start + length * frac
        stroke_ring(canvas, cx - ux * t, cy - uy * t, r * .42, r * .12, (200, 205, 200))
    # Round metal joints on the shaft, about 0.8 ball radii, like the real
    # stick's decorations (the real frame has one the detector saw as a ball).
    for frac in (.1, .45):
        t = start + length * frac
        jx, jy = cx - ux * t, cy - uy * t
        fill_disc(canvas, jx, jy, r * .8, (95, 100, 95))
        stroke_ring(canvas, jx, jy, r * .62, r * .14, (40, 45, 40))
        fill_disc(canvas, jx, jy, r * .3, (190, 195, 190))
    if scene.show_guide:
        stroke = max(2.0, r * .17)
        gx, gy = truth.ghost
        lines = [((cx, cy), (gx, gy))]
        if truth.target is not None:
            bx, by = scene.balls[truth.target]
            cos_cut = float(np.dot(truth.object_dir, (ux, uy)))
            lines.append(((bx, by), (bx + truth.object_dir[0] * r * (1.2 + 4.2 * cos_cut),
                                     by + truth.object_dir[1] * r * (1.2 + 4.2 * cos_cut))))
            if truth.cue_dir is not None:
                sin_cut = float(np.sqrt(max(0.0, 1 - cos_cut ** 2)))
                if sin_cut > .12:
                    lines.append(((gx, gy), (gx + truth.cue_dir[0] * r * (1.2 + 4.2 * sin_cut),
                                             gy + truth.cue_dir[1] * r * (1.2 + 4.2 * sin_cut))))
        ring_r = r - stroke * .5
        # Dark outlines first, then white strokes on top.
        for p0, p1 in lines:
            stroke_line(canvas, p0, p1, stroke + 2.4, (20, 22, 26))
        stroke_ring(canvas, gx, gy, ring_r, stroke + 2.4, (20, 22, 26))
        for p0, p1 in lines:
            stroke_line(canvas, p0, p1, stroke, (250, 250, 250))
        stroke_ring(canvas, gx, gy, ring_r, stroke, (250, 250, 250))
    frame = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
    if scene.blur > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), scene.blur)
    if scene.jpeg_quality:
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, scene.jpeg_quality])
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return frame, truth


# --- random scenes ---------------------------------------------------------

def _free(point, others, gap):
    return all(np.hypot(point[0] - ox, point[1] - oy) >= gap for ox, oy in others)


def random_scene(seed: int, kind: str | None = None) -> Scene:
    """A random but plausible table state with a guided aim.

    kinds: "scatter" (balls anywhere), "rack" (tight cluster), "cushion"
    (several balls frozen on cushions), "miss" (aim at an empty cushion).
    """
    rng = np.random.default_rng(seed)
    kind = kind or rng.choice(["scatter", "scatter", "rack", "cushion", "miss"])
    width = int(rng.choice([1280, 1500, 1836, 1920, 2200]))
    height = int(round(width * rng.uniform(.56, .62)))
    tw = width * rng.uniform(.52, .6)
    th = tw / 2
    x0 = (width - tw) / 2 + rng.uniform(-.02, .02) * width
    y0 = (height - th) / 2 + rng.uniform(-.015, .015) * height
    table = (float(x0), float(y0), float(tw), float(th))
    r = tw * .01545
    pockets = pocket_centres(table)

    def inside(px, py, margin=0.0):
        return (x0 + r + margin <= px <= x0 + tw - r - margin
                and y0 + r + margin <= py <= y0 + th - r - margin)

    def clear_of_pockets(px, py):
        return min(np.hypot(px - qx, py - qy) for qx, qy in pockets) > r * 3.2

    placed: dict[int, tuple[float, float]] = {}
    numbers = list(range(1, 16))
    rng.shuffle(numbers)
    if kind == "rack":
        apex = (x0 + tw * rng.uniform(.62, .75), y0 + th * rng.uniform(.4, .6))
        spots = []
        gap = 2.06 * r      # touching-but-not-overlapping rack spacing
        for row in range(5):
            for slot in range(row + 1):
                spots.append((apex[0] + row * gap * np.sqrt(3) / 2, apex[1] + (slot - row / 2) * gap))
        for number, spot in zip(numbers, spots):
            jitter = rng.uniform(-.02, .02, 2) * r
            placed[number] = (spot[0] + jitter[0], spot[1] + jitter[1])
        # Spread a few out so the rack is partly broken.
        for number in numbers[:int(rng.integers(0, 6))]:
            del placed[number]
    if kind == "cushion":
        for number in numbers[:5]:
            for _ in range(200):
                side = rng.integers(4)
                if side < 2:
                    px = rng.uniform(x0 + r * 4, x0 + tw - r * 4)
                    py = y0 + r if side == 0 else y0 + th - r
                else:
                    px = x0 + r if side == 2 else x0 + tw - r
                    py = rng.uniform(y0 + r * 4, y0 + th - r * 4)
                if clear_of_pockets(px, py) and _free((px, py), placed.values(), 2.15 * r):
                    placed[number] = (float(px), float(py))
                    break
    for number in numbers:
        if number in placed:
            continue
        for _ in range(400):
            px, py = rng.uniform(x0 + r, x0 + tw - r), rng.uniform(y0 + r, y0 + th - r)
            if inside(px, py, r * .1) and clear_of_pockets(px, py) and _free((px, py), placed.values(), 2.2 * r):
                placed[number] = (float(px), float(py))
                break
    cue = None
    for _ in range(400):
        px, py = rng.uniform(x0 + r * 2, x0 + tw - r * 2), rng.uniform(y0 + r * 2, y0 + th - r * 2)
        if clear_of_pockets(px, py) and _free((px, py), placed.values(), 2.6 * r):
            cue = (float(px), float(py))
            break
    if cue is None:
        cue = (x0 + tw * .25, y0 + th * .5)
    scene = Scene(width, height, table, float(r), placed, cue, 0.0, seed=seed,
                  stick_pull=float(rng.uniform(0, 2.5)),
                  jpeg_quality=int(rng.choice([0, 72, 85, 92])) or None,
                  blur=float(rng.choice([0, 0, .5, .8])),
                  letterbox=float(rng.uniform(.04, .1)))
    # Choose an aim: at a reachable target with a random cut, or at an empty cushion.
    order = list(placed)
    rng.shuffle(order)
    for number in order if kind != "miss" else []:
        bx, by = placed[number]
        dist = float(np.hypot(bx - cue[0], by - cue[1]))
        if dist < r * 5:
            continue
        cut = rng.uniform(-.9, .9)
        offset = cut * 2 * r
        base = np.arctan2(by - cue[1], bx - cue[0])
        scene.aim_degrees = float(np.degrees(base - np.arcsin(offset / dist)))
        truth = solve_truth(scene)
        if truth.target == number:
            gx, gy = truth.ghost
            # Keep the ghost ring off the cue ball: when they overlap the game's
            # guide is hidden under the cue ball (a known limit of reading it).
            if np.hypot(gx - cue[0], gy - cue[1]) < 2.2 * r:
                continue
            # Ghost ring must not overlap a third ball (the game would hit it).
            if all(np.hypot(gx - ox, gy - oy) >= 2 * r * .999 for n, (ox, oy) in placed.items() if n != number):
                return scene
    for _ in range(200):
        scene.aim_degrees = float(rng.uniform(-180, 180))
        truth = solve_truth(scene)
        if np.hypot(truth.ghost[0] - cue[0], truth.ghost[1] - cue[1]) < 2.2 * r:
            continue
        if truth.target is None and kind == "miss":
            return scene
        if truth.target is not None and kind != "miss":
            return scene
    return scene


def fit_scene_to(scene: Scene, table: tuple[float, float, float, float], width: int, height: int) -> Scene:
    """Map a scene onto a fixed frame size and table rectangle (same layout,
    scaled), so a sequence of scenes keeps the table where the overlay is."""
    x0, y0, w0, _ = scene.table
    x1, y1, w1, _ = table
    s = w1 / w0

    def move(point):
        return (x1 + (point[0] - x0) * s, y1 + (point[1] - y0) * s)

    scene.balls = {n: move(p) for n, p in scene.balls.items()}
    scene.cue = move(scene.cue)
    scene.radius = scene.radius * s
    scene.table = table
    scene.width, scene.height = width, height
    return scene


def live_scene(seed: int, width: int = 1836) -> Scene:
    """Scene for the live window: same table rectangle every time, like a real
    mirror, so the overlay is fitted once and must stay aligned."""
    height = int(round(width * .6))
    table_w = width * .58
    table = ((width - table_w) / 2, (height - table_w / 2) / 2, table_w, table_w / 2)
    return fit_scene_to(random_scene(seed), table, width, height)


def truth_json(truth: Truth) -> str:
    data = asdict(truth)
    data["balls"] = {str(k): v for k, v in truth.balls.items()}
    return json.dumps(data)


# --- live window for overlay tests ------------------------------------------

def make_table_window(seed: int = 1, interval_ms: int = 2500, truth_path: str | None = None,
                      width: int = 1836, frameless: bool = False):
    """A window that shows synthetic shots for testing the overlay live.

    A new random shot appears every ``interval_ms`` (0 = only on demand);
    Space or the right arrow shows the next one. The table stays in the same
    place, like a real mirror. With ``truth_path`` the current truth is
    written there as JSON (with the window's screen position) for a harness.
    """
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QLabel

    class TableWindow(QLabel):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Synthetic test table (not a game, no inputs)")
            if frameless:
                self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.seed = seed
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.show_next)
            if interval_ms > 0:
                self.timer.start(interval_ms)
            self.show_next()

        def show_next(self):
            scene = live_scene(self.seed, width)
            self.seed += 1
            frame, truth = render(scene)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], int(rgb.strides[0]),
                           QImage.Format.Format_RGB888).copy()
            self.setPixmap(QPixmap.fromImage(image))
            self.setFixedSize(image.size())
            if truth_path:
                data = json.loads(truth_json(truth))
                origin = self.mapToGlobal(QPoint(0, 0))
                data["window"] = [origin.x(), origin.y()]
                data["seed"] = scene.seed
                Path(truth_path).write_text(json.dumps(data))

        def keyPressEvent(self, event):
            if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Right):
                self.show_next()
            else:
                super().keyPressEvent(event)

        def closeEvent(self, event):
            self.timer.stop()
            super().closeEvent(event)

    return TableWindow()


def run_window(seed: int = 1, interval_ms: int = 2500, truth_path: str | None = None,
               position: tuple[int, int] = (0, 120), width: int = 1836):
    """Run the synthetic table window on its own (used by live_check.py)."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    window = make_table_window(seed, interval_ms, truth_path, width, frameless=True)
    window.move(*position)
    window.show()
    # Redraw the first scene now that the window is placed, so the truth file
    # records the real screen position.
    window.seed -= 1
    window.show_next()
    app.exec()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--interval", type=int, default=2500, help="ms between scenes")
    parser.add_argument("--truth", help="write the current truth JSON here")
    args = parser.parse_args()
    run_window(args.seed, args.interval, args.truth)
