"""Detection of the table, balls, cue ball, Miniclip's aim graphics, and the predicted shot paths.

Pure image analysis (OpenCV/numpy): no windows, no screen access."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np




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
    # Does the struck ball drop: "pot", "jaw" (it hits the pocket's jaw) or
    # None (no pocket on its path, or a ball in the way), and that pocket's
    # centre in frame pixels.
    pot: str | None = None
    pot_pocket: tuple[int, int] | None = None
    # How far the cue stick is pulled back: the gap from the cue ball's rim
    # to the stick's tip, in ball radii (None when no stick lies behind the
    # cue ball). It grows as the player drags the power bar; turning it into
    # a power percentage needs measurements from the real game.
    stick_pull: float | None = None
    # One level through the balls in the way: where a ball that a predicted
    # path runs into goes (``knock_paths``), and where the moving ball goes
    # after that hit (``deflect_paths``). Two entries each when a target was
    # hit: [after the object path, after the cue path], an empty list where
    # that path is not blocked. ``blockers`` stays as it was.
    knock_paths: list | None = None
    deflect_paths: list | None = None
    # Miniclip's power slider left of the table, when the capture includes
    # it: its outer rectangle (x, y, w, h) in frame pixels, and the power set
    # on it in percent: 0 with the cue at rest, 100 with the cue's top at the
    # slot's bottom. That 100 is Miniclip's full power, and that power grows
    # evenly along the slot, still has to be checked on real frames.
    power_slider: tuple[int, int, int, int] | None = None
    power_percent: float | None = None
    note: str = "Guide is off"


def white_guide_mask(roi: np.ndarray, balls: list[tuple[int, int, int, str]],
                     ball_scale: float = 1.18, hsv: np.ndarray | None = None) -> np.ndarray:
    """Pixels of Miniclip's white guide graphics, with ball faces removed.

    Miniclip draws its aim ray, ghost-ball reticle, object-ball line and
    cue-deflection line in near-white. Ball faces (cue ball, stripes, number
    badges) are white too, so they are masked out.
    """
    if hsv is None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Measured on both bundled frames: guide strokes have saturation <= ~60,
    # while even the brightly lit, washed-out felt centre stays >= ~95. The
    # older <= 115 limit let pale felt inside the hollow reticle through.
    mask = cv2.inRange(hsv, (0, 0, 165), (255, 70, 255))
    if _light_cloth(hsv):
        # A white or pale cloth passes that test itself: a guide stroke must
        # also be clearly brighter than the cloth around it, measured as the
        # cloth's lighter values over a wide window (strokes are thin, and
        # compression ringing beside their dark outlines is not brighter).
        v = hsv[:, :, 2]
        around = local_lightness(v)
        mask &= ((v.astype(np.float32) - around) >= LIGHT_CLOTH_STROKE).astype(np.uint8) * 255
        # Compression speckles next to dark edges pass that too; the guide's
        # strokes are long connected lines and rings.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            keep = stats[:, cv2.CC_STAT_AREA] >= 10
            keep[0] = False
            mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    for bx, by, br, _ in balls:
        cv2.circle(mask, (int(bx), int(by)), int(br * ball_scale), 0, thickness=-1)
    return mask


# Lab lightness (8-bit) from which a cloth counts as light in the ball check.
LIGHT_CLOTH_L = 150.0
# On a light, pale cloth, how much a darker shade of it (lettering, logo,
# shadows) counts in the cloth-distance image; a lighter one counts in full.
DARKER_SHADE_WEIGHT = .25
# On a light cloth a guide stroke is at least this much brighter than the
# cloth around it.
LIGHT_CLOTH_STROKE = 10


def _light_cloth(hsv: np.ndarray) -> bool:
    """True when the cloth itself would pass the white-stroke test (white,
    light grey, beige, pale tables): bright and barely saturated."""
    sample = hsv[::8, ::8].reshape(-1, 3)
    return float(np.median(sample[:, 2])) >= 150 and float(np.median(sample[:, 1])) <= 90


def _ring_kernel(radius: float) -> np.ndarray:
    """Correlation kernel that rewards a thin white ring with a hollow centre."""
    # Built once per radius: the same few radii are asked for on every frame.
    return _ring_kernel_cached(round(float(radius), 4))


@lru_cache(maxsize=256)
def _ring_kernel_cached(radius: float) -> np.ndarray:
    size = int(np.ceil(radius * 1.5)) + 2
    kernel = np.zeros((2 * size + 1, 2 * size + 1), np.float32)
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    for ring_radius, weight in ((radius, 1.0), (radius * .5, -1.0), (radius * 1.45, -1.0)):
        points = {(int(round(size + ring_radius * np.cos(a))), int(round(size + ring_radius * np.sin(a))))
                  for a in angles}
        for x, y in points:
            kernel[y, x] += weight / len(points)
    kernel.setflags(write=False)  # shared between calls
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
    yy, xx = np.ogrid[y0:y1, x0:x1]
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
    # Coarse pass: one ring size over the whole corridor finds where rings
    # could be. Only small windows around the strongest few spots then get
    # every ring size (that full sweep over the corridor was most of the time).
    # Concentric rings of the other sizes peak at the same centres, so the
    # result matches a full sweep.
    coarse = cv2.filter2D(crop, -1, _ring_kernel(float(radius)), borderType=cv2.BORDER_CONSTANT)
    coarse[~corridor] = -1
    windows = []
    half = int(np.ceil(radius * .5)) + 3
    margin = _ring_kernel(radius * 1.16 + .01).shape[0] // 2 + 1
    for _ in range(3):
        index = int(np.argmax(coarse))
        if coarse.flat[index] <= -1:
            break
        py, px = divmod(index, coarse.shape[1])
        windows.append((max(0, py - half), min(crop.shape[0], py + half + 1),
                        max(0, px - half), min(crop.shape[1], px + half + 1)))
        coarse[max(0, py - int(radius)):py + int(radius) + 1, max(0, px - int(radius)):px + int(radius) + 1] = -1
    best = None
    for wy0, wy1, wx0, wx1 in windows:
        # Filter a slightly larger patch so values inside the window are the
        # same as when filtering the whole crop.
        py0, py1 = max(0, wy0 - margin), min(crop.shape[0], wy1 + margin)
        px0, px1 = max(0, wx0 - margin), min(crop.shape[1], wx1 + margin)
        patch = crop[py0:py1, px0:px1]
        inside = corridor[wy0:wy1, wx0:wx1]
        for ring_radius in np.arange(radius * .78, radius * 1.16 + .01, 1.0):
            score = cv2.filter2D(patch, -1, _ring_kernel(float(ring_radius)), borderType=cv2.BORDER_CONSTANT)
            score = score[wy0 - py0:wy1 - py0, wx0 - px0:wx1 - px0]
            score[~inside] = -1
            best = _ring_peak(score, best, wx0 + x0, wy0 + y0, float(ring_radius))
    if best is None or best[0] < threshold:
        return None
    # (x, y, ring radius, ring score)
    return float(best[1]), float(best[2]), best[3], float(best[0])


def _ring_peak(score: np.ndarray, best, ox: int, oy: int, ring_radius: float):
    """Keep ``best`` or replace it with this score map's sub-pixel peak.

    ``(ox, oy)`` is where the map's top-left pixel sits in the white mask.
    """
    index = int(np.argmax(score))
    value = float(score.flat[index])
    if best is not None and value <= best[0]:
        return best
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
    return value, ox + ix + fx, oy + iy + fy, ring_radius


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
    # Miniclip's lines are solid strokes at least 2 px wide. One-pixel
    # threads are not: the anti-aliased edge of our own guide line (yellow
    # blended into blue felt turns pale grey) leaves exactly such a thread from
    # the target centre, and reading it as the game's line locked the guide
    # onto its own, wrong, angle (real frame: -123 deg instead of -110 deg).
    ys, xs = np.nonzero(cv2.morphologyEx(white[y0:y1, x0:x1], cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)))
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
# A circle whose centre lies within this share of a ball radius of the
# stick's axis is a decoration on the stick.
STICK_AXIS_SHARE = .2
# Towards the butt Miniclip's stick is nearly a ball wide, and its end cap and
# rings are round, full ball size and have clean rims (real frame: 1.11 r,
# rim 0.79, 1-3 px off the stick axis, ~30 r behind the cue ball). Such a
# circle is no wider than the stick around it, while a ball lying under a
# narrower part of the stick sticks out on both sides. A circle whose stick
# half-width (beside it, along the axis) is at least this share of the
# typical ball radius is taken as part of the stick (real butt ring: 0.75;
# synthetic balls under the stick, rings included: 0.40-0.60).
STICK_WIDTH_SHARE = .68


def stick_half_width(distance: np.ndarray, cx: float, cy: float, ux: float, uy: float,
                     radius: float) -> float | None:
    """Half-width of the stick just beside a circle on its axis, in pixels.

    Measured across the axis (``ux``, ``uy``) at 1.6 radii before and after
    the circle, where it does not cover the stick. Only places where the axis
    itself is not cloth count (the butt end may have no stick beyond it). A
    reading wider than the ball is a neighbouring ball, not the stick, and is
    skipped; of the rest the narrower is used.
    ``distance`` is the float cloth-distance image. None when no stick is
    seen beside the circle.
    """
    steps = np.arange(0.0, radius * 1.5, .5)
    widths = []
    for t in (-1.6 * radius, 1.6 * radius):
        px, py = cx + ux * t, cy + uy * t
        sides = []
        for sign in (-1, 1):
            xs = (px - sign * uy * steps).astype(np.float32)[None]
            ys = (py + sign * ux * steps).astype(np.float32)[None]
            profile = cv2.remap(distance, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
            if profile[0] <= 8:
                break
            cloth = np.nonzero(profile <= 8)[0]
            sides.append(float(steps[cloth[0]]) if cloth.size else float(steps[-1]))
        if len(sides) == 2 and max(sides) < radius * 1.1:
            widths.append(float(np.mean(sides)))
    return min(widths) if widths else None


def _wider_than_stick(distance, cx: float, cy: float, ux: float, uy: float, radius: float) -> bool:
    """True when non-cloth shows ~0.95 radii to both sides of the stick axis
    at (cx, cy), i.e. the circle there is as wide as a ball."""
    if distance is None:
        return True
    h, w = distance.shape[:2]
    hits = 0
    for side in (-1, 1):
        values = []
        for reach in (.93, .97):
            px = cx - uy * side * radius * reach
            py = cy + ux * side * radius * reach
            if 0 <= px < w and 0 <= py < h:
                values.append(float(distance[int(py), int(px)]))
        hits += bool(values) and max(values) > 8
    return hits == 2


def separate_cue_stick_detections(
    balls: list[tuple[int, int, int, str]],
    cue: tuple[int, int, int],
    aim: tuple[float, float],
    rim_open: dict[tuple[int, int], float] | None = None,
    distance: np.ndarray | None = None,
    offset: tuple[int, int] = (0, 0),
) -> list[tuple[int, int, int, str]]:
    """Drop circle-like cue-stick ornaments lying behind the tracked cue ball.

    The game aim ray points from the cue ball toward the target. The physical
    stick lies on the opposite side. Real balls offset from that narrow stick
    axis stay in the list; the cue itself is always retained. A real ball
    lying under the stick also stays when ``rim_open`` shows a clear
    ball-to-cloth edge around most of its rim (stick ornaments have none),
    unless the stick beside it is about as wide as the circle (the butt's
    round end cap and rings; needs ``distance``, the float cloth-distance
    image of the region starting at ``offset`` in frame pixels).
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
            # A ring centred on the stick's own axis is one of its
            # decorations (bands, joints, the glowing collar on real cues)
            # unless something non-cloth shows on both sides of the stick at
            # most of a ball radius out: a ball under the stick does, a
            # decoration is narrower.
            if across <= typical_r * STICK_AXIS_SHARE and not _wider_than_stick(
                    distance, bx - offset[0], by - offset[1], ux, uy, typical_r):
                continue
            ball_like = (rim_open is not None and rim_open.get((bx, by), 0.0) >= STICK_BALL_RIM
                         and br >= typical_r * STICK_BALL_SIZE)
            if ball_like and distance is not None:
                width = stick_half_width(distance, bx - offset[0], by - offset[1], ux, uy, br)
                ball_like = width is None or width < typical_r * STICK_WIDTH_SHARE
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


# Cushion bounces the live guide follows each predicted path through. Every
# bounce adds a little angle error (the game's cushions are not perfect
# mirrors), so the path still stops at a pocket or the first ball in the way.
PATH_BOUNCES = 3


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


# Pocket holes as a share of the pocket-to-pocket span (as drawn on Miniclip's
# table; the same numbers as fake_game).
CORNER_POCKET_SPAN = .032
MIDDLE_POCKET_SPAN = .024
# A ball drops when its centre passes within the hole radius minus
# POT_DROP_RADII ball radii of the pocket centre; within the hole radius plus
# POT_JAW_RADII it hits the jaw. An approximation: no pocket-angle or speed
# effects (a fast ball can rattle out of a pocket in the game).
POT_DROP_RADII = .4
POT_JAW_RADII = .5
# How far past the end of a leg (where it meets the cushion) a pocket counts.
POT_REACH_RADII = 6.0


def pocket_holes(table: tuple[int, int, int, int]) -> list[tuple[float, float, float]]:
    """(x, y, hole radius) of the six pockets, from the cushion-nose rectangle."""
    left, top, width, height = table
    span = width / (1 - 2 * POCKET_TO_CUSHION)
    inset = POCKET_TO_CUSHION * span
    right, bottom = left + width, top + height
    corner, middle = span * CORNER_POCKET_SPAN, span * MIDDLE_POCKET_SPAN
    return [(left - inset, top - inset, corner), (left + width / 2, top - 2 * inset, middle),
            (right + inset, top - inset, corner), (left - inset, bottom + inset, corner),
            (left + width / 2, bottom + 2 * inset, middle), (right + inset, bottom + inset, corner)]


def pocket_outcome(segments, table: tuple[int, int, int, int], radius: float):
    """("pot" | "jaw" | None, pocket centre) for a ball following ``segments``.

    On each leg, a pocket ahead of the ball near where the leg meets the
    cushion decides it by how close the ball's centre line passes the pocket
    centre. A leg that ends short of the cushion was stopped by a ball, and a
    pocket beyond that ball is not reached.
    """
    left, top, width, height = table
    bounds = (left + radius, top + radius, left + width - radius, top + height - radius)
    holes = pocket_holes(table)
    for (sx, sy), (ex, ey) in segments or []:
        length = float(np.hypot(ex - sx, ey - sy))
        if length < 1e-6:
            continue
        ux, uy = (ex - sx) / length, (ey - sy) / length
        at_cushion = min(abs(ex - bounds[0]), abs(ey - bounds[1]),
                         abs(ex - bounds[2]), abs(ey - bounds[3])) <= 1.5
        best = None
        for px, py, hole in holes:
            t = (px - sx) * ux + (py - sy) * uy
            if t <= 0 or t > length + POT_REACH_RADII * radius or (not at_cushion and t > length):
                continue
            d = abs((px - sx) * uy - (py - sy) * ux)
            if d <= hole - POT_DROP_RADII * radius:
                outcome = "pot"
            elif d <= hole + POT_JAW_RADII * radius:
                outcome = "jaw"
            else:
                continue
            if best is None or t < best[0]:
                best = (t, outcome, (int(round(px)), int(round(py))))
        if best is not None:
            return best[1], best[2]
        if not at_cushion:
            break
    return None, None


# The knock-on depends on exactly where two balls touch, so a small angle
# error after a long path (through cushions) swings it a lot: only predict it
# for a ball reached within this many ball radii of travel. And the moving
# ball's deflection is only drawn for a hit this far from full (sine of the
# cut), where its direction is stable.
KNOCK_REACH_RADII = 25.0
KNOCK_MIN_CUT = .25


def knock_on(path, blockers, obstacles, bounds, radius: float, max_bounces: int):
    """(hit ball's path, moving ball's path after the hit) when ``path``
    ends against one of ``blockers``; ([], []) otherwise.

    Same equal-mass, no-spin rule as the main shot: the hit ball leaves
    along the line of centres, the moving ball along the tangent (nothing
    when the hit is nearly full, as the moving ball then stops).
    """
    if not path or not blockers:
        return [], []
    travel = sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in path)
    if travel > KNOCK_REACH_RADII * radius:
        return [], []
    (sx, sy), (ex, ey) = path[-1]
    hit = min(blockers, key=lambda b: abs(np.hypot(b[0] - ex, b[1] - ey) - 2 * radius))
    gap = float(np.hypot(hit[0] - ex, hit[1] - ey))
    if abs(gap - 2 * radius) > max(2.5, radius * .2):
        return [], []
    length = float(np.hypot(ex - sx, ey - sy))
    if length < 1e-6 or gap < 1e-6:
        return [], []
    ux, uy = (ex - sx) / length, (ey - sy) / length
    nx, ny = (hit[0] - ex) / gap, (hit[1] - ey) / gap
    if ux * nx + uy * ny <= 0:
        return [], []
    # The hit ball and the moving ball's old spot are not obstacles.
    others = [o for o in obstacles if (o[0], o[1]) != (hit[0], hit[1])
              and np.hypot(o[0] - ex, o[1] - ey) > radius * .5]
    knocked = trace_bounces((hit[0], hit[1]), (nx, ny), bounds, radius, max_bounces=max_bounces,
                            obstacles=others)
    dot = ux * nx + uy * ny
    tx, ty = ux - dot * nx, uy - dot * ny
    deflected = (trace_bounces((ex, ey), (tx, ty), bounds, radius, max_bounces=max_bounces, obstacles=others)
                 if np.hypot(tx, ty) > KNOCK_MIN_CUT else [])
    return knocked, deflected


def collision_paths(cue: tuple[int, int, int], target: tuple[int, int, int],
                    direction: tuple[float, float], table: tuple[int, int, int, int],
                    contact: tuple[float, float] | None = None, radius: float | None = None,
                    obstacles=None, object_dir: tuple[float, float] | None = None,
                    cue_dir: tuple[float, float] | None = None, blocked: list | None = None,
                    max_bounces: int = 1):
    """Shared equal-mass, no-spin collision model used by live guide and simulator.

    ``radius`` is one shared ball radius (all pool balls are the same size; the
    per-circle Hough radii vary by a few pixels). ``contact`` is the cue-ball
    centre at impact, taken from the game's ghost-ball reticle when visible.
    ``object_dir``/``cue_dir`` override the physics directions when the game's
    own white object/deflection lines were read from the frame.
    ``max_bounces`` is how many cushion bounces each path follows (the live
    guide uses ``PATH_BOUNCES``; the simulator keeps one).
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
    object_path = trace_bounces((bx, by), obj, bounds, ball_r, max_bounces=max_bounces,
                                obstacles=others, blocked=blocked)
    cue_path = (trace_bounces(contact, cue_after, bounds, ball_r, max_bounces=max_bounces,
                              obstacles=others, blocked=blocked)
                if np.hypot(*physics_cue) > .12 else [])
    return approach, cue_path, object_path, (int(bx), int(by), int(br))


# Weighted Lab distance (lightness halved, as in ``felt_distance_image``) at
# which a pixel on the stick's line no longer looks like the cloth beside it.
STICK_CONTRAST = 10.0


def stick_pull(roi: np.ndarray, cue_local: tuple[float, float], aim: tuple[float, float],
               radius: float) -> float | None:
    """Gap from the cue ball's rim to the cue stick's tip, in ball radii.

    Samples only the line behind the cue ball (opposite the aim) and two
    lines 1.4 radii to either side of it, so each spot is compared with the
    cloth right beside it (lighting and cloth colour do not matter). The tip
    is where the line stops looking like cloth and keeps differing for four
    radii (a stick is long; a ball or shadow is not). None when there is no
    stick there, e.g. on the opponent's turn.
    """
    if aim is None or radius <= 0:
        return None
    cx, cy = cue_local
    ux, uy = aim
    norm = float(np.hypot(ux, uy))
    if norm < 1e-6:
        return None
    ux, uy = ux / norm, uy / norm
    height, width = roi.shape[:2]
    # Start just outside the cue ball's shadow-darkened rim.
    steps = np.arange(radius * 1.12, radius * 16, .5, dtype=np.float32)
    side = radius * 1.4
    xs = cx - ux * steps
    ys = cy - uy * steps
    points = [(xs, ys), (xs - uy * side, ys + ux * side), (xs + uy * side, ys - ux * side)]
    valid = [(px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1) for px, py in points]
    # The line can run off the table (the stick crosses the cushion); only
    # the part on the cloth is used.
    count = int(np.argmin(valid[0])) if not valid[0].all() else len(steps)
    if count < 8:
        return None
    lab = []
    for px, py in points:
        mapx = np.clip(px[:count], 0, width - 1).reshape(1, -1).astype(np.float32)
        mapy = np.clip(py[:count], 0, height - 1).reshape(1, -1).astype(np.float32)
        sampled = cv2.remap(roi, mapx, mapy, cv2.INTER_LINEAR)
        lab.append(cv2.cvtColor(sampled, cv2.COLOR_BGR2LAB)[0].astype(np.float32))
    weights = np.array([.5, 1.0, 1.0], np.float32)

    def apart(a, b):
        return np.sqrt((((a - b) * weights) ** 2).sum(axis=1))

    # A ball beside the line spoils one side; the other side is cloth. A
    # side off the table does not count.
    far = np.float32(1e6)
    left = np.where(valid[1][:count], apart(lab[0], lab[1]), far)
    right = np.where(valid[2][:count], apart(lab[0], lab[2]), far)
    contrast = np.minimum(left, right)
    usable = contrast < far
    # Two limits: a firm one where the tip starts, a lower one to follow a
    # dark stick whose colour comes close to the cloth's.
    on = (contrast > STICK_CONTRAST) & usable
    along = (contrast > STICK_CONTRAST * .6) & usable
    full = int(round(radius * 4 / .5))
    least = int(round(radius * 1.5 / .5))
    solid = max(2, int(round(radius / .5)))
    tip_index = None
    for i in np.nonzero(on)[0]:
        # The stick keeps going for four radii, or until the line leaves the
        # table (at least 1.5 radii of it on the cloth).
        stretch = along[i:i + full]
        if len(stretch) < least or stretch.mean() < .85:
            continue
        # The tip starts a solid stretch: the cue ball's drop shadow is a
        # short dark patch with cloth after it.
        if on[i:i + solid].mean() < .9 and not along[i:i + least].all():
            continue
        if len(stretch) < full and i + full <= count:
            continue
        tip_index = int(i)
        break
    if tip_index is None:
        return None
    i = tip_index
    if i == 0:
        # Already stick at the first sample: the tip touches the cue ball.
        return float(steps[0] / radius - 1)
    # Sub-sample tip: where the contrast crosses halfway between the cloth
    # just before it and the stick just after.
    level = (float(contrast[i - 1]) + float(np.median(contrast[i:i + full]))) / 2
    a, b = float(contrast[i - 1]), float(contrast[i])
    frac = 0.0 if b <= a else float(np.clip((level - a) / (b - a), 0, 1))
    tip = float(steps[i - 1] + .5 * frac)
    return tip / radius - 1


def predict_shot(roi: np.ndarray, table: tuple[int, int, int, int],
                 balls: list[tuple[int, int, int, str]], cue: tuple[int, int, int]) -> dict:
    """Follow the game's aim graphics and estimate cue/object paths.

    Returns a dict with aim/cue/object segments, target, ghost, blockers,
    pot call, stick pull and a note. All coordinates are in frame pixels (the
    same frame ``table`` is in).
    """
    shot = _predict_paths(roi, table, balls, cue)
    pull = None
    if shot.get("aim_dir") is not None:
        ball_r = float(np.median([br for _, _, br, _ in balls])) if balls else float(cue[2])
        pull = stick_pull(roi, (cue[0] - table[0], cue[1] - table[1]), shot["aim_dir"], ball_r)
    return {**shot, "stick_pull": pull}


def ghost_from_object_line(white: np.ndarray, balls_local, cue_local, radius: float):
    """(ghost x, y, struck ball) in ROI pixels from the game's object-ball
    line, or None.

    When the struck ball lies right next to the cue ball, Miniclip's ghost
    ring sits on top of the cue ball and cannot be read. The game still draws
    its object line from the struck ball's centre, and the ghost touches that
    ball on the far side: one diameter back along the line.
    """
    cx, cy = cue_local[:2]
    best = None
    for bx, by, br, kind in balls_local:
        if kind == "cue":
            continue
        dist = float(np.hypot(bx - cx, by - cy))
        if not (1.8 * radius <= dist <= 4.4 * radius):
            continue
        hint = ((bx - cx) / dist, (by - cy) / dist)
        found = game_line_direction(white, (bx, by), hint, radius, cone_degrees=70.0)
        if found is None:
            continue
        (dx, dy), (ax, ay) = found
        gx, gy = ax - 2 * radius * dx, ay - 2 * radius * dy
        gap = float(np.hypot(gx - cx, gy - cy))
        # The ghost overlaps (or nearly touches) the cue ball, and the cue
        # ball travels through it into the struck ball.
        if not (.1 * radius <= gap <= 2.4 * radius) or (gx - cx) * dx + (gy - cy) * dy <= 0:
            continue
        if best is None or dist < best[0]:
            best = (dist, gx, gy, (bx, by, br))
    return None if best is None else (best[1], best[2], best[3])


def _predict_paths(roi: np.ndarray, table: tuple[int, int, int, int],
                   balls: list[tuple[int, int, int, str]], cue: tuple[int, int, int]) -> dict:
    """Follow the game's aim graphics and estimate cue/object paths.

    Returns a dict with aim/cue/object segments, target, ghost, blockers and a
    note. All coordinates are in frame pixels (the same frame ``table`` is in).
    """
    left, top, width, height = table
    empty = {"aim": [], "cue": [], "object": [], "target": None, "ghost": None, "blockers": [],
             "aim_dir": None, "object_dir": None, "cue_dir": None, "pot": None, "pot_pocket": None,
             "knock": [], "deflect": []}
    cue_local = (cue[0] - left, cue[1] - top, cue[2])
    balls_local = [(bx-left, by-top, br, kind) for bx, by, br, kind in balls]
    cx, cy, cr = cue
    # All balls share one size; use the robust median rather than each circle's
    # slightly noisy Hough radius.
    ball_r = float(np.median([br for _, _, br, _ in balls])) if balls else float(cr)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = white_guide_mask(roi, balls_local, hsv=roi_hsv)
    # The reticle ring touches the struck ball, so mask only the ball faces
    # here; the wider mask above would erase the ring's near side.
    ghost_mask = white_guide_mask(roi, balls_local, ball_scale=.9, hsv=roi_hsv)
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
    near_ghost = None

    def ball_touches(gxl, gyl):
        return any(abs(float(np.hypot(bx - gxl, by - gyl)) - 2 * ball_r) <= ball_r * .5
                   for bx, by, _, kind in balls_local if kind == "ball")

    def against_cushion(gxl, gyl):
        return min(gxl - ball_r, width - ball_r - gxl, gyl - ball_r, height - ball_r - gyl) < ball_r * .35

    # A ghost against a cushion with no ball beside it is a plain rebound
    # shot, not a hidden one; leave it alone.
    if (ghost is None or np.hypot(ghost[0] - cue_local[0], ghost[1] - cue_local[1]) <= cr * 1.5
            or not (ball_touches(ghost[0], ghost[1]) or against_cushion(ghost[0], ghost[1]))):
        near_ghost = ghost_from_object_line(white, balls_local, cue_local, ball_r)
        if near_ghost is not None:
            ghost = (near_ghost[0], near_ghost[1], ball_r, 1.0)
            dx, dy = near_ghost[0] - cue_local[0], near_ghost[1] - cue_local[1]
            n = float(np.hypot(dx, dy))
            direction = (dx / n, dy / n)
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
        refined = None if near_ghost is not None else ray_direction_to_ghost(white, cue_local, (ghost[0], ghost[1]), ball_r)
        if refined is not None:
            gdx, gdy = refined[0] * glen, refined[1] * glen
        if near_ghost is not None:
            # Ghost found from the object line, on top of the cue ball.
            direction = (gdx / glen, gdy / glen)
            ghost_frame = (gx, gy, ball_r)
        elif glen > cr * 1.5:
            # The reticle centre lies on the game's ray, and it is far from the
            # cue ball, so this gives the aim angle to a fraction of a degree.
            direction = (gdx / glen, gdy / glen)
            ghost_frame = (gx, gy, ghost[2])
    ux, uy = direction
    obstacles = [(bx, by, br) for bx, by, br, kind in balls if kind != "cue"]
    angle = float(np.degrees(np.arctan2(uy, ux)))
    selected = None
    if near_ghost is not None:
        # The ball whose object line placed the ghost is the struck ball.
        bx, by, br = near_ghost[2]
        selected = (int(round(bx + left)), int(round(by + top)), int(br))
    if ghost_frame is not None and selected is None:
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
            ghost_int = (int(round(gx)), int(round(gy)), int(round(ghost_frame[2])))
            if not on_cushion:
                # No ball and no cushion beside the reticle: the struck ball
                # was probably missed, so show the game ray only.
                return {**empty, "aim": [((cx, cy), ghost_int[:2])], "ghost": ghost_int, "aim_dir": (ux, uy),
                        "note": f"Aim ray matched at {angle:.1f}° · ghost ball has no object ball beside it"}
            # Reticle in a pocket mouth: the cue ball goes in (a scratch), it
            # does not rebound off the cushion beside the pocket.
            if pocket_outcome([((cx, cy), (gx, gy))], table, ball_r)[0] == "pot":
                return {**empty, "aim": [((cx, cy), ghost_int[:2])], "ghost": ghost_int, "aim_dir": (ux, uy),
                        "note": f"Aim ray matched at {angle:.1f}° · cue ball goes into the pocket"}
            # Reticle against a cushion: the cue ball rebounds there. Mirror
            # the aim off each cushion the reticle touches (both in a corner)
            # and trace on from the reticle, like the after-impact paths.
            near = ball_r * .35
            rx, ry = ux, uy
            if (gx - (left + ball_r) < near and ux < 0) or (left + width - ball_r - gx < near and ux > 0):
                rx = -ux
            if (gy - (top + ball_r) < near and uy < 0) or (top + height - ball_r - gy < near and uy > 0):
                ry = -uy
            bounds = (left + ball_r, top + ball_r, left + width - ball_r, top + height - ball_r)
            blockers: list = []
            # The bounce at the reticle is the first of PATH_BOUNCES.
            rebound = trace_bounces((gx, gy), (rx, ry), bounds, ball_r, max_bounces=PATH_BOUNCES - 1,
                                    obstacles=obstacles, blocked=blockers) if (rx, ry) != (ux, uy) else []
            return {**empty, "aim": [((cx, cy), ghost_int[:2])], "cue": rebound, "ghost": ghost_int,
                    "blockers": blockers, "aim_dir": (ux, uy),
                    "cue_dir": (rx, ry) if rebound else None,
                    "note": f"Aim ray matched at {angle:.1f}° · cue ball rebounds off the cushion"}
    if selected is None:
        selected = _select_target_on_ray(roi, table, balls, cue, direction, ball_r)
        if selected is None:
            bounds = (left + ball_r, top + ball_r, left + width - ball_r, top + height - ball_r)
            blockers: list = []
            aim = trace_bounces((cx, cy), direction, bounds, ball_r, max_bounces=PATH_BOUNCES,
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
        if ghost_frame is not None:
            # The struck ball touches the ghost ball, so its centre is one
            # diameter from the precise ghost centre along that line. This
            # also fixes the position along the line, which snapping cannot
            # (a red ball on a red table: 6 px off).
            tx, ty = ghost_frame[0] + 2 * ball_r * obj_dir[0], ghost_frame[1] + 2 * ball_r * obj_dir[1]
            if np.hypot(tx - selected[0], ty - selected[1]) < ball_r * .6:
                selected = (int(round(tx)), int(round(ty)), br)
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
    if object_line is not None:
        # The game's own object line is the impact normal to a fraction of a
        # degree; the cue ball leaves at 90 deg to it. The centre-to-centre
        # estimate can be several degrees off (synthetic seed 253: 9 deg),
        # which turned the fallback cue path just as far.
        nx, ny = obj_dir
    else:
        n = float(np.hypot(*physics_obj))
        nx, ny = physics_obj[0] / n, physics_obj[1] / n
    dot = ux * nx + uy * ny
    physics_cue = (ux - dot * nx, uy - dot * ny)
    cue_dir = None
    if np.hypot(*physics_cue) > .12:
        cue_line = game_line_direction(white, (contact_point[0] - left, contact_point[1] - top),
                                       physics_cue, ball_r, cone_degrees=20.0)
        cue_dir = cue_line[0] if cue_line else None
    pn = float(np.hypot(*physics_cue))
    final_cue = cue_dir or ((physics_cue[0] / pn, physics_cue[1] / pn) if pn > .12 else None)
    blockers = []
    # Draw the cue path along final_cue too: collision_paths' own physics
    # uses the raw Hough centre, which is several degrees off the game's
    # object line (synthetic seed 11: 7 deg).
    approach, cue_path, object_path, target = collision_paths(
        cue, selected, direction, table, contact=contact, radius=ball_r, obstacles=obstacles,
        object_dir=obj_dir, cue_dir=final_cue, blocked=blockers, max_bounces=PATH_BOUNCES)
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
    pot, pot_pocket = pocket_outcome(object_path, table, ball_r)
    # One level through a ball in the way of either path. The struck ball
    # has left its spot, so it is no obstacle any more.
    bounds = (left + ball_r, top + ball_r, left + width - ball_r, top + height - ball_r)
    moving_free = [o for o in obstacles if np.hypot(o[0] - selected[0], o[1] - selected[1]) > ball_r * .5]
    knock, deflect = [], []
    for path in (object_path, cue_path):
        hit_path, after = knock_on(path, blockers, moving_free, bounds, ball_r, PATH_BOUNCES - 1)
        knock.append(hit_path)
        deflect.append(after)
    return {"aim": approach, "cue": cue_path, "object": object_path, "target": target,
            "ghost": ghost_int, "blockers": blockers, "aim_dir": (ux, uy),
            "object_dir": obj_dir, "cue_dir": final_cue, "pot": pot, "pot_pocket": pot_pocket,
            "knock": knock, "deflect": deflect,
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
    # Only a patch around the ghost and struck ball is read (the rim samples
    # reach at most ~3 radii from the ghost), so only that patch is converted.
    # The cloth colour is still measured on the whole table, as before.
    reach = int(np.ceil(radius * 3 + 12))
    ox, oy = max(0, int(ghost[0]) - reach), max(0, int(ghost[1]) - reach)
    patch = roi[oy:int(ghost[1]) + reach + 1, ox:int(ghost[0]) + reach + 1]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # The cloth colour from every other pixel of the table: the same median
    # to within a level, a quarter of the work (this runs on every frame).
    distance = cv2.GaussianBlur(felt_distance_image(patch, hsv, as_float=True, reference=_felt_lab(roi[::2, ::2])),
                                (0, 0), 1.0)
    gx, gy = ghost[0] - ox, ghost[1] - oy
    target = (target[0] - ox, target[1] - oy)
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
    return best[1] + ox, best[2] + oy


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
# On a black cloth (brightness ~12 at the rails) the pocket's black core
# (0-5) is still darker; this limit separates them.
GREY_POCKET_DARKNESS = 8


# Miniclip's default cloth is blue (OpenCV hue ~98), but tables come in green,
# red, purple, brown... Every cloth test is a hue band around the measured
# cloth hue; the offsets below are the band the blue table was tuned with.
DEFAULT_CLOTH_HUE = 98.0


def cloth_hue(hsv: np.ndarray, step: int = 1) -> float:
    """Dominant hue of the saturated, not-dark pixels: the cloth and its cushions.

    ``step`` subsamples a large frame. Falls back to blue when too few pixels
    are coloured (e.g. a frame without a table).
    """
    sample = hsv[::step, ::step] if step > 1 else hsv
    mask = cv2.inRange(sample, (0, 45, 65), (180, 255, 255))
    if cv2.countNonZero(mask) < mask.size * .05:
        return DEFAULT_CLOTH_HUE
    counts = cv2.calcHist([sample], [0], mask, [180], [0, 180]).ravel()
    # Hue is circular (red sits on both ends); smooth across the wrap.
    smooth = sum(np.roll(counts, k) for k in range(-4, 5))
    peak = int(np.argmax(smooth))
    window = np.arange(peak - 6, peak + 7)
    weights = counts[window % 180]
    return float((peak + (weights @ (window - peak)) / max(1.0, weights.sum())) % 180)


# How far (weighted Lab distance, lightness at 0.35 so the cloth's own
# shading counts as cloth) a pixel may be from the measured cloth colour.
CLOTH_LAB_REACH = 24.0
LAB_WEIGHTS = np.array([.35, 1.0, 1.0], np.float32)


def cloth_model(frame: np.ndarray, hsv: np.ndarray | None = None, step: int = 4) -> dict:
    """The cloth, measured instead of assumed: the most common colour in the
    middle of the image (where the table is), so white, grey, pale, dark and
    any hue work the same way.

    Returns ``lab`` (reference colour), ``v_low`` (brightness near the rails:
    5th percentile of cloth pixels), ``v`` and ``s`` (median brightness and
    saturation of the cloth).
    """
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    middle = (slice(h // 5, h * 4 // 5, step), slice(w // 5, w * 4 // 5, step))
    lab = cv2.cvtColor(frame[middle], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) == 0:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # Peak of a coarse colour histogram (chroma fine, lightness coarse, so the
    # cloth's light-to-dark shading stays one peak).
    bins = np.stack([lab[:, 0] // 48, lab[:, 1] // 6, lab[:, 2] // 6], axis=1).astype(np.int32)
    keys = bins[:, 0] * 10000 + bins[:, 1] * 100 + bins[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    peak = values[np.argmax(counts)]
    pb = np.array([peak // 10000, (peak // 100) % 100, peak % 100])
    near = np.all(np.abs(bins - pb) <= np.array([1, 1, 1]), axis=1)
    reference = np.median(lab[near], axis=0)
    cloth = np.sqrt((((lab - reference) * LAB_WEIGHTS) ** 2).sum(axis=1)) < CLOTH_LAB_REACH
    sample = hsv[middle].reshape(-1, 3)[cloth] if cloth.any() else hsv[middle].reshape(-1, 3)
    return {"lab": reference.astype(np.float32),
            "v_low": float(np.percentile(sample[:, 2], 5)),
            "v": float(np.median(sample[:, 2])),
            "s": float(np.median(sample[:, 1]))}


def cloth_like(frame: np.ndarray, model: dict) -> np.ndarray:
    """Boolean mask of pixels close to the measured cloth colour."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = (lab - model["lab"]) * LAB_WEIGHTS
    return (diff ** 2).sum(axis=2) < CLOTH_LAB_REACH ** 2


def cloth_is_grey(hsv: np.ndarray, step: int = 4) -> bool:
    """True for a cloth with no colour (black or grey tables, e.g. Lucky
    Shot): few saturated pixels in the middle of the image."""
    h, w = hsv.shape[:2]
    middle = hsv[h // 4:h * 3 // 4:step, w // 4:w * 3 // 4:step]
    return middle.size > 0 and float(np.mean(middle[:, :, 1] >= 45)) < .25


def grey_cloth_mask(hsv: np.ndarray) -> np.ndarray:
    """Pixels that can be a grey cloth: unsaturated, not black, not bright."""
    return (hsv[:, :, 1] < 50) & (hsv[:, :, 2] >= 8) & (hsv[:, :, 2] <= 140)


def hue_near(hue: np.ndarray, centre: float, below: float, above: float) -> np.ndarray:
    """Boolean mask: ``hue`` within [centre - below, centre + above], circularly."""
    if hue.dtype == np.uint8:
        # Same test through a 256-entry table (an 8-bit hue has few values);
        # converting whole frames to float was the slowest step of detect.
        values = (np.arange(256, dtype=np.float32) - centre + 90) % 180 - 90
        table = ((values >= -below) & (values <= above)).astype(np.uint8)
        return cv2.LUT(hue, table).astype(bool)
    delta = (hue.astype(np.float32) - centre + 90) % 180 - 90
    return (delta >= -below) & (delta <= above)


def hue_delta(hue, centre: float):
    """Signed circular hue difference in OpenCV units (-90..90)."""
    return (np.asarray(hue, np.float32) - centre + 90) % 180 - 90


def _three_corner_fit(blobs, dark: np.ndarray):
    """(error, tl, tr, bl, br) from three corner pockets, the fourth inferred.

    ``blobs`` are pocket candidates (x, y, width, area). Two corners on one
    long side, a third straight above or below one of them, and a middle
    pocket on that long side (checked in the ``dark`` mask directly: the
    cushion cuts it into a shape the blob filter drops); the hidden corner
    completes the rectangle.
    """
    h, w = dark.shape[:2]
    best = None
    for a in blobs:
        for b in blobs:
            dx = b[0] - a[0]
            if dx < 200 or abs(b[1] - a[1]) > dx * .03:
                continue
            size = (a[2] + b[2]) / 2
            if not (.035 * dx <= size <= .1 * dx) or max(a[3], b[3]) > 2 * min(a[3], b[3]):
                continue
            row_y = (a[1] + b[1]) / 2
            # The middle pocket of this side: halfway along, a little outside
            # the corner row (middle pockets sit further out), smaller.
            mid_x = (a[0] + b[0]) / 2
            half = max(3, int(round(dx * .012)))
            x0m, x1m = int(mid_x - half), int(mid_x + half) + 1
            y0m, y1m = int(row_y - dx * .03), int(row_y + dx * .03) + 1
            box = dark[max(0, y0m):min(h, y1m), max(0, x0m):min(w, x1m)]
            if box.size == 0 or float(box.mean()) < .25:
                continue
            for c in blobs:
                if c is a or c is b:
                    continue
                for corner, other in ((a, b), (b, a)):
                    dy = c[1] - corner[1]
                    if abs(c[0] - corner[0]) > dx * .03 or not (.44 * dx <= abs(dy) <= .58 * dx):
                        continue
                    if max(c[3], corner[3]) > 2 * min(c[3], corner[3]):
                        continue
                    # The hidden corner: under/over the other one, level with c.
                    hidden = (other[0], c[1], other[2], other[3])
                    if dy > 0:
                        tl, tr = (a, b)
                        bl, br = (c, hidden) if corner is a else (hidden, c)
                    else:
                        bl, br = (a, b)
                        tl, tr = (c, hidden) if corner is a else (hidden, c)
                    height = (bl[1] + br[1]) / 2 - (tl[1] + tr[1]) / 2
                    error = (abs(c[0] - corner[0]) + abs(b[1] - a[1]) + abs(height / dx - .51) * dx * .2
                             + dx * .01)  # a little worse than any four-corner fit
                    if best is None or error < best[0]:
                        best = (error, tl, tr, bl, br)
    return best


def table_from_pockets(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Fit the playing area from the four dark corner pockets.

    The felt colour alone also catches the same-coloured cushions on one side,
    which shifted the old fit right by ~13 px (middle pockets landed off the
    real ones). The pockets are symmetric and unambiguous.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = cloth_is_grey(hsv)
    model = cloth_model(frame, hsv)
    # Pockets are darker than the cloth's darkest part (its rails): on a dark
    # cloth (navy, dark green, black) a fixed limit would swallow the cloth.
    limit = min(POCKET_DARKNESS, max(GREY_POCKET_DARKNESS, .6 * model["v_low"]))
    dark = (hsv[:, :, 2] < limit).astype(np.uint8)
    # Cut thin dark lines (the neon table's rail bands, outlines) away from
    # the pockets, which they would otherwise join into long blobs. Pockets
    # are far wider than this kernel.
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    hue = cloth_hue(hsv, step=4)
    # Next to a pocket: the measured cloth colour, or (on a coloured cloth)
    # its hue band, which also covers cushions of the same hue (neon table).
    felt_like = cloth_like(frame, model)
    if not grey:
        felt_like |= hue_near(hsv[:, :, 0], hue, 23, 27) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 65)
    felt_like = felt_like.astype(np.uint8)
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
        # A corner pocket can be hidden (the cue ball or the stick on it).
        # Three corners still fix the table when the long side's middle
        # pocket confirms the row that has both corners.
        best = _three_corner_fit(blobs, dark)
    if best is None:
        return None
    _, a, b, c, d = best
    span = ((b[0] - a[0]) + (d[0] - c[0])) / 2
    inset = POCKET_TO_CUSHION * span
    x0, x1 = (a[0] + c[0]) / 2 + inset, (b[0] + d[0]) / 2 - inset
    y0, y1 = (a[1] + b[1]) / 2 + inset, (c[1] + d[1]) / 2 - inset
    inner = hsv[int(y0 + (y1 - y0) * .25):int(y1 - (y1 - y0) * .25),
                int(x0 + (x1 - x0) * .25):int(x1 - (x1 - x0) * .25)]
    inner_bgr = frame[int(y0 + (y1 - y0) * .25):int(y1 - (y1 - y0) * .25),
                      int(x0 + (x1 - x0) * .25):int(x1 - (x1 - x0) * .25)]
    felt = cloth_like(inner_bgr, model) if inner_bgr.size else np.zeros(1, bool)
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
        # The nose is the innermost strong drop: a chrome rail outside a
        # dark cushion bevel (black Lucky Shot table) drops first, and more.
        # (On a dark cloth the nose step can be small: navy cushion 32 -> cloth 21.)
        strong = np.nonzero(drops >= max(8.0, drops[i] * .5))[0]
        last = int(strong[-1])
        # Step to that drop's own peak (the edge can span two samples).
        while last + 1 < len(drops) and drops[last + 1] > drops[last]:
            last += 1
        i = last
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
    # The felt sits in a hue band around the measured cloth hue (blue: 95–110);
    # exclude the much darker side panels and black letterbox bars when fitting
    # the full desktop. The felt is less saturated than the cushions. A broad
    # threshold merges both and makes the fitted guide too tall.
    hue = cloth_hue(hsv, step=4)
    if cloth_is_grey(hsv):
        band = grey_cloth_mask(hsv)
    else:
        band = (hue_near(hsv[:, :, 0], hue, 23, 27) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 1] <= 190)
                & (hsv[:, :, 2] >= 65))
    # On a red or brown table the wooden rail falls in the same hue band;
    # keep only pixels close to the cloth's own colour.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    # The cloth colour is measured in the middle of the frame, where the
    # table is; the rail can outnumber the cloth over the whole frame.
    fh, fw = band.shape
    middle = (slice(fh // 4, fh * 3 // 4, 4), slice(fw // 4, fw * 3 // 4, 4))
    sample = band[middle]
    if np.count_nonzero(sample) > 100:
        reference = np.median(lab[middle][sample], axis=0)
        diff = lab - reference
        near = (diff[:, :, 0] * .5) ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2 < 20.0 ** 2
        band &= near
    felt = band.astype(np.uint8) * 255
    # Cut thin outlines (rail edges, guide lines) that would join other areas
    # of a similar colour onto the cloth.
    felt = cv2.morphologyEx(felt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
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
    centre = cloth_hue(roi_hsv, step=2)
    cloth = hue_near(hue, centre, 18, 17) & (sat >= 50) & (sat <= 235)
    if np.count_nonzero(cloth) < roi_hsv.shape[0] * roi_hsv.shape[1] * .2:
        cloth = np.ones_like(hue, bool)
    # Median of the circular offset, so a red cloth (hue 175..5) is not torn apart.
    return (float((centre + np.median(hue_delta(hue[cloth], centre))) % 180),
            float(np.median(sat[cloth])))


def _felt_lab(roi: np.ndarray, roi_hsv: np.ndarray | None = None) -> np.ndarray:
    """Median Lab colour of the cloth (all pixels if too little cloth shows)."""
    if roi_hsv is None:
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    if cloth_is_grey(roi_hsv, step=2):
        # Colourless cloth of any brightness (black to white): the measured
        # cloth colour, not a fixed brightness range.
        model = cloth_model(roi, roi_hsv, step=2)
        cloth = cloth_like(roi, model).astype(np.uint8)
        if cv2.countNonZero(cloth) <= 100:
            return model["lab"]
    else:
        centre = cloth_hue(roi_hsv, step=2)
        cloth = (hue_near(roi_hsv[:, :, 0], centre, 18, 17) & (roi_hsv[:, :, 1] >= 50)
                 & (roi_hsv[:, :, 1] <= 235)).astype(np.uint8)
    if cv2.countNonZero(cloth) <= 100:
        cloth = None
    # Exact median of 8-bit values from each channel's histogram (much faster
    # than sorting a million pixels; same result as np.median).
    reference = []
    for channel in range(3):
        counts = cv2.calcHist([lab], [channel], cloth, [256], [0, 256]).ravel()
        cumulative = np.cumsum(counts)
        n = cumulative[-1]
        low = int(np.searchsorted(cumulative, (n - 1) // 2 + 1))
        high = int(np.searchsorted(cumulative, n // 2 + 1))
        reference.append((low + high) / 2)
    return np.array(reference, np.float32)


def local_lightness(lightness: np.ndarray) -> np.ndarray:
    """The cloth's own lightness around each pixel, over ~90 px.

    The lightest part of each 8 px block (lettering strokes are thinner, so
    this is the cloth beside them), then a median over 11x11 blocks so a
    white ball (a few blocks) barely moves it."""
    h, w = lightness.shape[:2]
    if h < 16 or w < 16:
        return np.full((h, w), float(np.median(lightness)), np.float32)
    clean = cv2.medianBlur(np.clip(lightness, 0, 255).astype(np.uint8), 3)
    bh, bw = h // 8, w // 8
    small = clean[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).max(axis=(1, 3))
    small = cv2.medianBlur(np.ascontiguousarray(small), 11 if min(bh, bw) > 11 else 3)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def lettering_shade(lab: np.ndarray, cloth_lightness: np.ndarray, felt: np.ndarray) -> float | None:
    """How much darker than the cloth its watermark lettering/logo is (Lab L,
    negative), or None when there is none: the commonest darker shade among
    cloth-coloured pixels across the table."""
    step = 2
    l = lab[::step, ::step, 0]
    rel = l - cloth_lightness[::step, ::step]
    dab = np.hypot(lab[::step, ::step, 1] - felt[1], lab[::step, ::step, 2] - felt[2])
    shade = rel[(dab < 10) & (rel < -20) & (rel > -.45 * cloth_lightness[::step, ::step])]
    if shade.size < .003 * rel.size:
        return None
    counts, edges = np.histogram(shade, bins=np.arange(-120, -19, 3))
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2)


def felt_distance_image(roi: np.ndarray, roi_hsv: np.ndarray, as_float: bool = False,
                        reference: np.ndarray | None = None) -> np.ndarray:
    """How far each pixel's colour is from the cloth (bright = not cloth).

    A blue ball on blue cloth has almost no brightness edge, so grey-level
    circle finding misses it; its colour still differs. Lightness is weighted
    down so the cloth's own vignetting stays dark here. ``reference`` is the
    cloth's Lab colour when it was measured elsewhere (e.g. on the whole table
    while ``roi`` is only a patch of it).
    """
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    if reference is None:
        reference = _felt_lab(roi, roi_hsv)
    diff = lab - reference
    if reference[0] >= LIGHT_CLOTH_L and np.hypot(reference[1] - 128, reference[2] - 128) < 24:
        # Light, pale cloth: the cue ball and white stripes are only a little
        # brighter than it, so lightness counts in full that way; darker
        # shades of the cloth (lettering, logo, shadows) count little. The
        # lighting falls off across the table, so lightness is measured
        # against the cloth around each spot, not one reference.
        dl = lab[:, :, 0] - local_lightness(lab[:, :, 0])
        diff[:, :, 0] = np.where(dl > 0, dl, dl * DARKER_SHADE_WEIGHT)
    else:
        diff[:, :, 0] *= .5
    distance = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
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


def refit_centre(distance: np.ndarray, cx: float, cy: float, r: float,
                 reach: float = 3.0) -> tuple[float, float]:
    """Best centre for a circle of a fixed radius ``r`` near (cx, cy)."""
    best = (cx, cy)
    for step, span in ((1.0, reach), (.25, 1.5)):
        offsets = np.arange(-span, span + 1e-6, step)
        gx, gy = np.meshgrid(best[0] + offsets, best[1] + offsets)
        centres = np.column_stack([gx.ravel(), gy.ravel()])
        scores = _rim_scores(distance, centres, r)
        i = int(np.argmax(scores))
        best = (float(centres[i, 0]), float(centres[i, 1]))
    return best


# All pool balls are the same size. A fitted radius further than this share
# from the typical one has grown onto the ball's drop shadow (or shrunk
# against a neighbour); its centre is fitted again at the typical radius.
BALL_SIZE_SPREAD = .06


def has_number_badge(patch_hsv: np.ndarray, xx, yy, cx: float, cy: float, r: float,
                     min_v: float = 150) -> bool:
    """True when a compact white disc (a ball's number badge, ~0.45 r) sits
    near the centre: white fills the middle around its own centroid, with
    almost no white just outside it. The game's guide lines crossing at the
    ghost ball's centre are white there too, but they run on outwards.
    ``min_v`` is raised on a light cloth, which is itself "white" by the
    default rule (a badge is still brighter than it).
    """
    white = (patch_hsv[:, :, 1] < 90) & (patch_hsv[:, :, 2] > min_v)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    near = white & (dist2 <= (r * .7) ** 2)
    ys, xs = np.nonzero(near)
    if len(xs) < 6:
        return False
    my, mx = ys.mean() + yy.ravel()[0], xs.mean() + xx.ravel()[0]
    around = (xx - mx) ** 2 + (yy - my) ** 2
    middle = white[around <= (r * .36) ** 2]
    outside = white[(around >= (r * .6) ** 2) & (around <= (r * .78) ** 2)]
    if middle.size == 0 or outside.size == 0:
        return False
    return float(middle.mean()) >= .5 and float(outside.mean()) <= .08


def find_balls(roi: np.ndarray, gray: np.ndarray, min_r: int, max_r: int, shared: dict | None = None):
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
    # A grey, black or pale cloth has no reliable hue to compare; judge it by
    # Lab colour.
    grey = cloth_is_grey(roi_hsv, step=2) or felt_sat < 75
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32) if grey else None
    felt_lab = _felt_lab(roi, roi_hsv) if grey else None
    lab_weights = np.array([.5, 1.0, 1.0], np.float32)
    roi_light = None
    letter_shade = False   # not measured yet
    distance = felt_distance_image(roi, roi_hsv, as_float=True)
    if shared is not None:
        # detect() needs the same image again for the cue-stick filter.
        shared["distance"] = distance
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
    # A ball the same colour as the cloth (the black 8 on a black table) has
    # no edge for circle finding, but its white number badge (a disc ~0.45
    # radii) stands out. Propose a ball on each such disc; the checks below
    # (and has_number_badge) decide whether it is one.
    typical = float(np.median([p[2] for p in proposals])) if proposals else (min_r + max_r) / 2
    from_badge: set[tuple[int, int]] = set()
    white = ((roi_hsv[:, :, 1] < 90) & (roi_hsv[:, :, 2] > 150)).astype(np.uint8)
    count, _, stats, centres = cv2.connectedComponentsWithStats(white, connectivity=8)
    badge_area = np.pi * (typical * .45) ** 2
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if not (.45 * badge_area <= area <= 1.6 * badge_area):
            continue
        if not (.7 <= bw / max(1, bh) <= 1.43) or area < .55 * bw * bh:
            continue
        cx, cy = int(round(centres[i][0])), int(round(centres[i][1]))
        if all((cx - px) ** 2 + (cy - py) ** 2 > (min_r * .6) ** 2 for px, py, _ in proposals):
            proposals.append((cx, cy, int(round(typical))))
            from_badge.add((cx, cy))
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
        # Hue differences are circular so a red cloth (hue 175..5) works too.
        ring_cloth = ring[(np.abs(hue_delta(ring[:, 0], felt_hue)) <= 12) & (np.abs(ring[:, 1] - felt_sat) <= 60)]
        local_hue, local_sat = (((felt_hue + float(np.median(hue_delta(ring_cloth[:, 0], felt_hue)))) % 180,
                                 float(np.median(ring_cloth[:, 1])))
                                if len(ring_cloth) >= 12 else (felt_hue, felt_sat))
        cloth_inside = ((np.abs(hue_delta(inner[:, 0], local_hue)) <= 7) & (np.abs(inner[:, 1] - local_sat) <= 35)
                        & (inner[:, 2] >= 50))
        if grey:
            lab_patch = roi_lab[y0:y1, x0:x1]
            ring_lab = lab_patch[(dist2 >= (r * 1.35) ** 2) & (dist2 <= (r * 1.8) ** 2)]
            near_cloth = ring_lab[np.sqrt(((ring_lab - felt_lab) * lab_weights) ** 2).sum(axis=1) < 25]                 if len(ring_lab) else ring_lab
            local = np.median(near_cloth, axis=0) if len(near_cloth) >= 12 else felt_lab
            inner_lab = lab_patch[dist2 <= (r * .62) ** 2]
            cloth_inside = np.sqrt((((inner_lab - local) * lab_weights) ** 2).sum(axis=1)) < 10
            if roi_light is None and felt_lab[0] >= LIGHT_CLOTH_L - 30:
                roi_light = local_lightness(roi_lab[:, :, 0])
            # Lettering or a logo ring in the ring pulls its median down, so
            # whether the cloth is light is judged over a wider area too.
            if max(local[0], roi_light[cy, cx] if roi_light is not None else 0) >= LIGHT_CLOTH_L:
                if roi_light is None:
                    roi_light = local_lightness(roi_lab[:, :, 0])
                # The cloth is the lighter part of the ring...
                if len(near_cloth) >= 12:
                    local = local.copy()
                    # ...but not above the cloth's lightness over a wider
                    # area: in a rack the ring holds the neighbours' white.
                    local[0] = max(local[0], min(float(np.percentile(near_cloth[:, 0], 80)),
                                                 float(roi_light[cy, cx])))
                # Light cloth: the table's watermark lettering and logo are
                # the cloth's colour in one darker shade, still cloth;
                # lighter (the cue ball, white stripes), another colour, or
                # other shades (a ball's shading and badge edge) are not.
                if letter_shade is False:
                    letter_shade = lettering_shade(roi_lab, roi_light, felt_lab)
                dl = inner_lab[:, 0] - local[0]
                dab = np.hypot(inner_lab[:, 1] - local[1], inner_lab[:, 2] - local[2])
                cloth_inside |= (dab < 10) & (np.abs(dl) < 6)
                if letter_shade is not None:
                    cloth_inside |= (dab < 10) & (np.abs(dl - letter_shade) < 10)
                # A white ball on near-white cloth is only a little brighter
                # than it on its lit side, and its shaded side looks like a
                # darker shade of the cloth. Lettering has nothing brighter.
                if float(np.mean((dl > 8) & (dab < 12))) >= .3:
                    cloth_inside = (np.abs(dl) < 4) & (dab < 10)
        cloth_fraction = float(np.mean(cloth_inside))
        centre = patch[dist2 <= (r * .4) ** 2]
        # A badge-only proposal stands for a ball that looks like cloth; a
        # light-centred disc that is not cloth-coloured (a metal joint on the
        # cue stick) would have been found by circle finding if it were a ball.
        if (cx, cy) in from_badge:
            # Judge the ball's own colour around the badge (white excluded).
            body = ~((inner[:, 1] < 90) & (inner[:, 2] > 150))
            if not body.any() or float(np.mean(cloth_inside[body])) <= .25:
                continue
        if cloth_fraction > .35:
            # A solid ball the same colour as the cloth (purple 4 on a purple
            # table, red 3 on red) looks like cloth inside, but its white
            # number badge fills the centre. Hollow rings, lettering and
            # shadows have cloth there.
            badge_v = 150.0
            if grey and len(ring):
                badge_v = max(badge_v, float(np.percentile(ring[:, 2], 80)) + 10)
            if not has_number_badge(patch, xx, yy, cx, cy, r, badge_v):
                continue
        sat = float(np.median(inner[:, 1]))
        val = float(np.median(inner[:, 2]))
        white_fraction = float(np.mean((inner[:, 1] < 90) & (inner[:, 2] > 150)))
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
    # Same size for every ball: a shadow must not make one circle bigger or
    # pull its centre towards the shadow.
    typical = float(np.median([item[3] for item in kept]))
    result = []
    for _, cx, cy, r, *rest in kept:
        # Smaller circles keep their size: the cue-stick filter tells stick
        # rings from balls by it. Every ball is drawn at one size in the end
        # (``uniform_ball_size``).
        # Only a white ball (the cue ball, whose centre sets the aim) has an
        # edge strong enough to beat its shadow in this fit; on a ball close
        # to the cloth colour the shadow would win and pull the centre.
        white_ball = rest[2] >= .5  # (sat, val, white_fraction, red_spot)
        if white_ball and r > typical * (1 + BALL_SIZE_SPREAD) and rim_open_fraction(smooth, cx, cy, typical) >= .65:
            fx, fy = refit_centre(smooth, float(cx), float(cy), typical)
            cx, cy = int(round(fx)), int(round(fy))
        result.append((cx, cy, r, *rest, rim_open_fraction(smooth, cx, cy, r)))
    return result


def uniform_ball_size(balls):
    """All pool balls are one size: report every ball at the median radius."""
    if not balls:
        return balls
    typical = int(round(float(np.median([b[2] for b in balls]))))
    return [(x, y, typical, kind) for x, y, _, kind in balls]


# Power slider, measured on a real frame (table 1136x568 px): a grey-framed
# slot 0.052 table widths wide and 0.79 table heights tall, centred 0.183
# table widths left of the table and level with it. At rest the cue's shaft
# (below the white tip and dark ferrule) starts SLIDER_REST of the slot
# height from its top; the slot's inside ends at SLIDER_END.
SLIDER_REST = .098
SLIDER_END = .965


def find_power_slider(frame: np.ndarray, table: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Outer rectangle of Miniclip's power slider left of the table, or None."""
    left, top, width, height = table
    if left < width * .1:
        return None
    region = frame[:, :left]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # The slot's grey frame: unsaturated, mid-bright.
    mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] >= 90) & (hsv[:, :, 2] <= 190)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not (.55 * height <= h <= 1.05 * height and .025 * width <= w <= .09 * width and h >= 5 * w):
            continue
        # A framed slot, not a thin edge line (the real frame fills ~17% of
        # its box, a rail edge ~5%).
        if area < .1 * w * h:
            continue
        if abs(y + h / 2 - (top + height / 2)) > height * .2:
            continue
        candidates.append((int(x), int(y), int(w), int(h)))
    # Closest to the table first (other tall icons sit further left); it must
    # hold the cue, or be an empty dark slot (cue dragged all the way down).
    for rect in sorted(candidates, key=lambda r: -r[0]):
        if _slider_shaft_top(frame, rect) is not None or _slider_empty(frame, rect):
            return rect
    return None


def _slider_empty(frame: np.ndarray, slider: tuple[int, int, int, int]) -> bool:
    """True when the slot's middle is dark from top to bottom (no cue showing)."""
    x, y, w, h = slider
    centre = x + w / 2
    half = max(2, int(round(w * .06)))
    strip = frame[int(y + h * .06):int(y + h * .94), int(centre - half):int(centre + half) + 1]
    if strip.size == 0:
        return False
    value = np.median(cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)[:, :, 2], axis=1)
    return bool(np.mean(value < 110) > .95)


def _slider_shaft_top(frame: np.ndarray, slider: tuple[int, int, int, int]) -> float | None:
    """Row (fraction of the slot height from its top) where the cue's shaft starts."""
    x, y, w, h = slider
    centre = x + w / 2
    half = max(2, int(round(w * .06)))
    strip = frame[y:y + h, int(centre - half):int(centre + half) + 1]
    if strip.size == 0:
        return None
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    row = np.median(hsv, axis=1)
    shaft = (row[:, 0] >= 5) & (row[:, 0] <= 25) & (row[:, 1] >= 70) & (row[:, 1] <= 190) & (row[:, 2] >= 170)
    # The shaft runs down to the slot's bottom (clipped there); take the run
    # that reaches lowest, allowing a few odd rows (segment marks).
    ys = np.nonzero(shaft)[0]
    if ys.size < h * .03:
        return None
    bottom = ys[-1]
    if bottom < h * .8:
        return None
    filled = shaft.copy()
    gap_ok = max(2, int(round(h * .01)))
    top_row = bottom
    misses = 0
    for r in range(bottom, -1, -1):
        if filled[r]:
            top_row, misses = r, 0
        else:
            misses += 1
            if misses > gap_ok:
                break
    return top_row / h


def read_power_slider(frame: np.ndarray, slider: tuple[int, int, int, int] | None) -> float | None:
    """Power set on the slider in percent: 0 at rest, 100 at the bottom."""
    if slider is None:
        return None
    shaft = _slider_shaft_top(frame, slider)
    if shaft is None:
        return 100.0 if _slider_empty(frame, slider) else None
    return float(np.clip((shaft - SLIDER_REST) / (SLIDER_END - SLIDER_REST), 0.0, 1.0)) * 100


def detect(frame: np.ndarray) -> Detection:
    """Find the Miniclip felt, ball centers, and current game aim ray."""
    h, w = frame.shape[:2]
    table = find_table_bounds(frame)
    if table is None:
        return Detection(note="Table not found — adjust the overlay to include the full felt.")

    slider = find_power_slider(frame, table)
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
    shared: dict = {}
    for cx, cy, r, sat, val, white_fraction, red_spot, open_rim in find_balls(roi, gray, min_r, max_r, shared):
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
                distance = shared.get("distance")
                if distance is None:
                    distance = felt_distance_image(roi, cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), as_float=True)
                balls = separate_cue_stick_detections(balls, cue_position, aim, rim_open,
                                                      distance=distance, offset=(x, y))
            # Exactly one ball is the cue ball; other white/striped detections
            # remain valid object-ball targets under 8-ball rules.
            cue_index = min(range(len(balls)), key=lambda i: np.hypot(balls[i][0]-cue[0], balls[i][1]-cue[1]))
            bx, by, br, _ = balls[cue_index]
            balls[cue_index] = (bx, by, br, "cue")
    if len(balls) > 24:
        return Detection(table=table, balls=[], power_slider=slider, power_percent=read_power_slider(frame, slider),
                         note=f"Table found · {len(balls)} circle candidates · too many to track confidently")
    if cue:
        shot = predict_shot(roi, table, balls, cue[:3])
        aim_now = shot.get("aim_dir")
        if aim_now is not None and (aim is None or aim_now[0] * aim[0] + aim_now[1] * aim[1] < .985):
            # The aim used for the cue-stick filter above was wrong (the ghost
            # ring on top of the cue ball hid the ray); filter the stick's
            # rings along the real aim and predict again.
            filtered = separate_cue_stick_detections(
                balls, cue[:3], aim_now, rim_open,
                distance=shared.get("distance") if shared.get("distance") is not None
                else felt_distance_image(roi, cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), as_float=True),
                offset=(x, y))
            if len(filtered) != len(balls):
                balls = filtered
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
        balls = uniform_ball_size(balls)
        cue = (*cue[:2], balls[0][2]) if balls else cue
        return Detection(table=table, balls=balls, cue=cue[:3], aim_segments=shot["aim"],
                         power_slider=slider, power_percent=read_power_slider(frame, slider),
                         cue_segments=shot["cue"], object_segments=shot["object"],
                         target=shot["target"], ghost=shot["ghost"], blockers=shot["blockers"],
                         aim_dir=shot["aim_dir"], object_dir=shot["object_dir"], cue_dir=shot["cue_dir"],
                         pot=shot["pot"], pot_pocket=shot["pot_pocket"], stick_pull=shot["stick_pull"],
                         knock_paths=shot["knock"], deflect_paths=shot["deflect"],
                         note=f"{shot['note']} · {len(balls)} ball candidates")
    return Detection(table=table, balls=uniform_ball_size(balls), power_slider=slider,
                     power_percent=read_power_slider(frame, slider),
                     note="Table found · cue ball not confidently identified")
