"""Detection of the table, balls, cue ball, Miniclip's aim graphics, and the predicted shot paths.

Pure image analysis (OpenCV/numpy): no windows, no screen access."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

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
                     ball_scale: float = 1.18, hsv: np.ndarray | None = None, ridges: bool = False) -> np.ndarray:
    """Pixels of Miniclip's white guide graphics, with ball faces removed.

    Miniclip draws its aim ray, ghost-ball reticle, object-ball line and
    cue-deflection line in near-white. Ball faces (cue ball, stripes, number
    badges) are white too, so they are masked out.
    """
    mask = white_strokes(roi, hsv, ridges=ridges)
    for bx, by, br, _ in balls:
        cv2.circle(mask, (int(bx), int(by)), int(br * ball_scale), 0, thickness=-1)
    return mask


def white_strokes(roi: np.ndarray, hsv: np.ndarray | None = None, ridges: bool = False) -> np.ndarray:
    """``white_guide_mask`` before the ball faces are removed (a new array).

    ``ridges``: on a white cloth also thin bright ridges between dark
    outlines (see ``white_stroke_masks``)."""
    plain, with_ridges = white_stroke_masks(roi, hsv, ridges)
    return with_ridges if ridges else plain


def white_stroke_masks(roi: np.ndarray, hsv: np.ndarray | None = None, ridges: bool = True,
                       ridge_windows=None):
    """(plain, with ridges) masks of the guide's white strokes, computed
    together (the cloth's local lightness is measured once).

    On a white cloth the strokes' white core is no brighter than the cloth
    itself (both ~190 where the lighting falls off): it shows as a thin
    ridge between its dark outlines (~125). The mask with ridges is for
    reading the lines (the aim ray, the object and cue lines); the plain one
    is for the ghost ring: where two strokes meet inside it, the cloth
    between their outlines reads as a ridge too and fills the hollow ring.
    Elsewhere (or with ``ridges`` False) both are the same array.

    ``ridge_windows`` (x0, y0, x1, y1 rectangles) limits the ridge search to
    where lines are read (the live tracker's hinted read): it is the slow
    part on a white cloth."""
    if hsv is None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Measured on both bundled frames: guide strokes have saturation <= ~60,
    # while even the brightly lit, washed-out felt centre stays >= ~95. The
    # older <= 115 limit let pale felt inside the hollow reticle through.
    base = cv2.inRange(hsv, (0, 0, 165), (255, 70, 255))
    cloth_s = _cloth_saturation(hsv) if _light_cloth(hsv) else None
    if cloth_s is None and float(np.mean(base[::8, ::8] > 0)) >= LIT_CLOTH_SHARE:
        # Not a light cloth, but its brightly lit middle passes the white
        # test itself (light grey: 7% of it, the guide and balls ~1%): the
        # strokes must stand out from it there too, or the lines are read in
        # a flood of cloth (lost or 3-12 degrees off, the ghost missed).
        cloth_s = _cloth_saturation(hsv)
    if cloth_s is not None and cloth_s >= 40:
        # A pale but coloured cloth (beige, sky): a thin stroke blurred with
        # its dark outline can be no brighter than the cloth, but it is far
        # less saturated.
        mask = cv2.inRange(hsv, (0, 0, 165), (255, int(min(70, cloth_s * .55)), 255))
        return mask, mask
    if cloth_s is None:
        return base, base
    # A white or pale cloth passes that test itself: a guide stroke must also
    # be clearly brighter than the cloth around it, measured as the cloth's
    # lighter values over a wide window (strokes are thin, and compression
    # ringing beside their dark outlines is not brighter).
    v = hsv[:, :, 2]
    around = local_lightness(v)
    brighter = (v.astype(np.float32) - around) >= LIGHT_CLOTH_STROKE
    plain = _long_strokes(base & (brighter.astype(np.uint8) * 255))
    if not ridges:
        return plain, plain
    if ridge_windows is None:
        core = _ridge_pieces(v, base)
    else:
        core = np.zeros_like(base)
        h, w = base.shape[:2]
        for x0, y0, x1, y1 in ridge_windows:
            x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                core[y0:y1, x0:x1] |= _ridge_pieces(v[y0:y1, x0:x1], base[y0:y1, x0:x1])
    # (Both parts are long already: ridge pieces by length, strokes by area.)
    return plain, plain | core


def _ridge_pieces(v: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Thin-ridge pixels of the guide's strokes, in long pieces only: bright
    cloth between two close dark marks (the table's lettering) is a ridge
    too, in short specks, and they tilted the object line's fit by 6
    degrees (seed 40007)."""
    core = base & (thin_ridges(v, LIGHT_CLOTH_RIDGE).astype(np.uint8) * 255)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8, ltype=cv2.CV_16U)
    if count <= 1:
        return core
    keep = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= RIDGE_MIN_LENGTH
    keep[0] = False
    return np.take((keep * 255).astype(np.uint8), labels)


def _long_strokes(mask: np.ndarray) -> np.ndarray:
    """Compression speckles next to dark edges pass the light-cloth stroke
    tests too; the guide's strokes are long connected lines and rings."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_16U)
    if count <= 1:
        return mask
    keep = stats[:, cv2.CC_STAT_AREA] >= 10
    keep[0] = False
    return np.take((keep * 255).astype(np.uint8), labels)


# Lab lightness (8-bit) from which a cloth counts as light in the ball check.
LIGHT_CLOTH_L = 150.0
# A light cloth with less colour (Lab chroma) than this is pale (white,
# light grey, beige, sky): striped balls are fitted on their band there.
PALE_CLOTH_CHROMA = 25.0
# A colourless cloth at least this light (grey, not the near-black Lucky
# Shot cloth) has watermark lettering in a darker shade of it.
GREY_CLOTH_L = 60.0
# On a light, pale cloth, how much a darker shade of it (lettering, logo,
# shadows) counts in the cloth-distance image; a lighter one counts in full.
DARKER_SHADE_WEIGHT = .25
# On a light cloth a guide stroke is at least this much brighter than the
# cloth around it,
LIGHT_CLOTH_STROKE = 10
# ...a test also used where this share of the cloth passes the white test.
LIT_CLOTH_SHARE = .03
# or a thin ridge at least this much brighter than both sides, in a piece at
# least this long (px).
LIGHT_CLOTH_RIDGE = 40
RIDGE_MIN_LENGTH = 20


def thin_ridges(v: np.ndarray, rise: float) -> np.ndarray:
    """Pixels at least ``rise`` brighter than the darkest pixel 1-3 px away
    on *both* sides, in one of four directions: the core of a thin bright
    line between dark outlines. (Cloth beside a single dark line is darker
    on one side only; a wide bright area has no dark sides.) The darkest
    pixel on a side is an erosion with a one-sided line kernel."""
    v8 = v if v.dtype == np.uint8 else np.clip(v, 0, 255).astype(np.uint8)
    out = np.zeros(v8.shape, bool)
    for kernel_a, kernel_b in _RIDGE_SIDES:
        side_a = cv2.erode(v8, kernel_a, borderType=cv2.BORDER_REPLICATE)
        side_b = cv2.erode(v8, kernel_b, borderType=cv2.BORDER_REPLICATE)
        out |= cv2.subtract(v8, cv2.max(side_a, side_b)) >= rise
    return out


def _ridge_sides():
    """For each of four directions, the 7x7 kernels picking the pixels 1-3
    px away on one side and on the other (centre anchored)."""
    pairs = []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        pair = []
        for sign in (1, -1):
            kernel = np.zeros((7, 7), np.uint8)
            for step in (1, 2, 3):
                kernel[3 + sign * step * dy, 3 + sign * step * dx] = 1
            pair.append(kernel)
        pairs.append(tuple(pair))
    return tuple(pairs)


_RIDGE_SIDES = _ridge_sides()


def _cloth_saturation(hsv: np.ndarray) -> float:
    return float(np.median(hsv[::8, ::8, 1]))


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
                    radius: float, roi: np.ndarray | None = None,
                    along_range: tuple[float, float] | None = None,
                    spots: int = 3) -> tuple[float, float, float, float] | None:
    """Locate Miniclip's hollow ghost-ball reticle along the rough aim ray.

    The reticle centre is where the game places the cue ball at impact, so it
    pins both the true aim angle and the contact point. Reading it directly is
    far more precise than intersecting a slightly-off ray with a ball whose
    radius is only known to a pixel or two (3 px of error there turned the
    object-ball path by ~5°).

    ``along_range`` (with a direction) limits the search to that stretch of
    the ray, as distances from the cue-ball centre: the live tracker looks
    near last frame's ghost first, where there is one ring, so ``spots``
    (how many of the strongest coarse spots get every ring size) can be 1.
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
        near, far = (0.0, float(np.hypot(w, h))) if along_range is None else along_range
        spread = .07 if along_range is None else .09
        ends = [(cx + ux * t + s * uy * max(t * spread, radius * .7), cy + uy * t - s * ux * max(t * spread, radius * .7))
                for t in (near, far) for s in (-1, 1)]
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
        if along_range is not None:
            # Last frame's ray can be a few degrees off this frame's: near the
            # cue ball the cone is only a few pixels wide and would clip the
            # ring's peak (its centre then moved a pixel, half a degree).
            corridor = ((along > cr * 1.6) & (along >= along_range[0]) & (along <= along_range[1])
                        & (across <= np.maximum(radius * .6, along * .08)))
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
    for _ in range(spots):
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
    # The reticle is hollow: cloth inside (besides its own lines, or the cue
    # ball when they overlap). White bits of packed balls can add up to a
    # ring, but a ball's colour fills it.
    if roi is not None and not ring_interior_is_cloth(roi, float(best[1]), float(best[2]), best[3]):
        return None
    # (x, y, ring radius, ring score)
    return float(best[1]), float(best[2]), best[3], float(best[0])


def ring_interior_is_cloth(roi: np.ndarray, gx: float, gy: float, r: float) -> bool:
    """False when the inside of a ring at (gx, gy) is largely another colour
    than the cloth around it (a ball's). Lighter or darker shades of the
    cloth (the guide lines' dark outlines, shading) and colourless pixels
    (white lines, a cue ball under the ring) do not count against it."""
    h, w = roi.shape[:2]
    reach = int(np.ceil(r * 1.7))
    x0, x1 = max(0, int(gx) - reach), min(w, int(gx) + reach + 1)
    y0, y1 = max(0, int(gy) - reach), min(h, int(gy) + reach + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return True
    lab = cv2.cvtColor(roi[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128, lab[:, :, 2] - 128
    chroma = np.hypot(a, b)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.hypot(xx - gx, yy - gy)
    around = (d >= r * 1.25) & (d <= r * 1.6)
    inside = d <= r * .55
    cloth_a, cloth_b = float(np.median(a[around])), float(np.median(b[around]))
    coloured = inside & (chroma >= 15)
    if int(coloured.sum()) < max(10, .15 * int(inside.sum())):
        # Mostly colourless inside (white lines, outlines, a cue ball under
        # the ring, or a colourless cloth): nothing to judge by.
        return True
    if np.hypot(cloth_a, cloth_b) < 8:
        # Colourless cloth: plenty of colour inside is a ball's.
        return False
    cloth_hue = np.arctan2(cloth_b, cloth_a)
    turn = np.abs((np.arctan2(b[coloured], a[coloured]) - cloth_hue + np.pi) % (2 * np.pi) - np.pi)
    return float(np.mean(turn > np.radians(RING_INTERIOR_HUE))) < .5


# Hue turn (degrees, Lab) beyond which a pixel inside a ghost ring is
# another colour than the cloth.
RING_INTERIOR_HUE = 35.0


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


# --- the ghost ring found anywhere on the table ------------------------------

# Rays (angles) and offsets across the circle (px) along which a ring's
# stroke is read.
_STROKE_ANGLES = np.linspace(0, 2 * np.pi, 96, endpoint=False)
_STROKE_OFFSETS = np.arange(-5.0, 5.01, .5)
_STROKE_NEAR = np.nonzero(np.abs(_STROKE_OFFSETS) <= 2.5)[0]
_PEAK_OFFSETS = np.arange(-7.0, 7.01, .5)
_PEAK_AT = np.nonzero(np.abs(_PEAK_OFFSETS) <= 4.0)[0]
_CORE_OFFSETS = np.arange(-1.5, 1.51, .5)
# A stroke's core is at least this much brighter (Lab L, 8-bit) than the
# darkest pixel 1-3 px away on each side.
RING_STROKE_RISE = 15.0


def _ray_profiles(image: np.ndarray, circles, offsets: np.ndarray) -> np.ndarray:
    """``image`` sampled along each circle's rays at ``offsets`` px across
    it: (circles, rays, offsets)."""
    circles = np.asarray(circles, np.float64).reshape(-1, 3)
    n = len(circles)
    radii = circles[:, 2, None, None] + offsets[None, None, :]
    xs = circles[:, 0, None, None] + np.cos(_STROKE_ANGLES)[None, :, None] * radii
    ys = circles[:, 1, None, None] + np.sin(_STROKE_ANGLES)[None, :, None] * radii
    # One row per circle: remap takes fewer than 32767 rows.
    profiles = cv2.remap(image, xs.reshape(n, -1).astype(np.float32), ys.reshape(n, -1).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return profiles.reshape(n, len(_STROKE_ANGLES), len(offsets))


def _stroke_contrast(profiles: np.ndarray, at) -> np.ndarray:
    """For each offset index in ``at``: how much brighter the profile is
    there than the darkest sample 1-3 px (six samples) away on its darker
    side."""
    at = np.asarray(at)
    # Darkest of each run of six samples (past the ends: none), then per
    # offset the run just before it and the one just after.
    padded = np.pad(profiles, [(0, 0)] * (profiles.ndim - 1) + [(6, 6)], constant_values=np.inf)
    darkest = np.lib.stride_tricks.sliding_window_view(padded, 6, axis=-1).min(axis=-1)
    centre = profiles[..., at]
    return np.minimum(centre - darkest[..., at], centre - darkest[..., at + 7]).astype(np.float32)


def _stroke_shares(lightness: np.ndarray, circles) -> np.ndarray:
    """For each circle (x, y, r): the share of rays that cross a thin bright
    stroke within 2.5 px of it (see ``_thin_stroke_share``)."""
    return (_ray_stroke_contrast(lightness, circles) >= RING_STROKE_RISE).mean(axis=-1)


def _ray_stroke_contrast(lightness: np.ndarray, circles) -> np.ndarray:
    """For each circle and ray, the strongest stroke contrast within 2.5 px
    of the circle: (circles, rays)."""
    return _stroke_contrast(_ray_profiles(lightness, circles, _STROKE_OFFSETS), _STROKE_NEAR).max(axis=-1)


def _stroke_points(lightness: np.ndarray, cx: float, cy: float, rho: float):
    """Per ray, where the thin bright stroke is strongest within 2.5 px of
    the circle: (x, y) arrays, rays without one left out."""
    contrast = _stroke_contrast(_ray_profiles(lightness, (cx, cy, rho), _STROKE_OFFSETS)[0], _STROKE_NEAR)
    best = np.argmax(contrast, axis=1)
    ok = contrast[np.arange(len(best)), best] >= RING_STROKE_RISE
    radii = rho + _STROKE_OFFSETS[_STROKE_NEAR][best[ok]]
    return cx + np.cos(_STROKE_ANGLES[ok]) * radii, cy + np.sin(_STROKE_ANGLES[ok]) * radii


def _fit_circle(xs, ys, keep: float, rounds: int = 3) -> tuple[float, float, float] | None:
    """Least-squares circle through the points, fitted again on those within
    ``keep`` px of it: (x, y, r), or None when fewer than 12 are left."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    use = np.ones(len(xs), bool)
    fit = None
    for _ in range(rounds):
        if use.sum() < 12:
            return None
        # Normal equations of u^2 + v^2 = a u + b v + c, (u, v) taken from
        # the points' mean for precision: centre (a/2, b/2), r^2 = c + a^2/4 + b^2/4.
        mx, my = float(xs[use].mean()), float(ys[use].mean())
        u, v = xs[use] - mx, ys[use] - my
        a = np.column_stack((u, v, np.ones(len(u))))
        try:
            sol = np.linalg.solve(a.T @ a, a.T @ (u * u + v * v))
        except np.linalg.LinAlgError:
            return None
        cu, cv = sol[0] / 2, sol[1] / 2
        cx, cy = cu + mx, cv + my
        fit = (float(cx), float(cy), float(np.sqrt(max(sol[2] + cu * cu + cv * cv, 1e-6))))
        use = np.abs(np.hypot(xs - cx, ys - cy) - fit[2]) <= keep
    return fit


def _ring_consensus(lightness: np.ndarray, cx: float, cy: float, rho: float,
                    radius: float) -> tuple[float, float, float] | None:
    """The circle near (cx, cy, rho) that the most rays have a stroke peak on.

    Where the ring lies on the cue ball, a thin strip of the ball's face
    between the ring's outline and the ball's own shaded rim is a bright
    ridge as well, right beside the stroke; a least-squares fit took both
    and landed 3-6 px off (synthetic near shots). Every local stroke peak
    within 4 px of the circle is kept per ray; circles through peaks on three
    rays 120 degrees apart are scored by how many rays have a peak within
    0.75 px of them, and the best is fitted on those peaks."""
    contrast = _stroke_contrast(_ray_profiles(lightness, (cx, cy, rho), _PEAK_OFFSETS)[0], _PEAK_AT)
    padded = np.pad(contrast, ((0, 0), (1, 1)), constant_values=-np.inf)
    peak = ((contrast >= RING_STROKE_RISE) & (contrast >= padded[:, :-2]) & (contrast >= padded[:, 2:]))
    rays, cols = np.nonzero(peak)
    if len(rays) < 20:
        return None
    radii = rho + _PEAK_OFFSETS[_PEAK_AT][cols]
    px = cx + np.cos(_STROKE_ANGLES[rays]) * radii
    py = cy + np.sin(_STROKE_ANGLES[rays]) * radii
    n = len(_STROKE_ANGLES)
    # (np.nonzero lists the peaks ray by ray.)
    bounds = np.searchsorted(rays, np.arange(n + 1))
    on_ray = [np.arange(bounds[a], bounds[a + 1]) for a in range(n)]
    triples = []
    for a in range(0, n // 3, 2):
        groups = (on_ray[a], on_ray[a + n // 3], on_ray[a + 2 * n // 3])
        if all(len(g) for g in groups):
            triples.extend(product(*groups))
    if not triples:
        return None
    t = np.array(triples)
    ax, ay, bx, by, qx, qy = px[t[:, 0]], py[t[:, 0]], px[t[:, 1]], py[t[:, 1]], px[t[:, 2]], py[t[:, 2]]
    d = 2 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by))
    ok = np.abs(d) > 1e-9
    d = np.where(ok, d, 1.0)
    a2, b2, q2 = ax * ax + ay * ay, bx * bx + by * by, qx * qx + qy * qy
    ux = (a2 * (by - qy) + b2 * (qy - ay) + q2 * (ay - by)) / d
    uy = (a2 * (qx - bx) + b2 * (ax - qx) + q2 * (bx - ax)) / d
    rr = np.hypot(ax - ux, ay - uy)
    ok &= (rr >= .8 * radius) & (rr <= 1.15 * radius)
    if not ok.any():
        return None
    ux, uy, rr = ux[ok], uy[ok], rr[ok]
    inlier = np.abs(np.hypot(px[None] - ux[:, None], py[None] - uy[:, None]) - rr[:, None]) <= .75
    hit = np.zeros((len(ux), n), bool)
    ci, pi = np.nonzero(inlier)
    hit[ci, rays[pi]] = True
    best = int(np.argmax(hit.sum(axis=1)))
    return _fit_circle(px[inlier[best]], py[inlier[best]], keep=1.0)


def _fit_ghost_ring(lightness: np.ndarray, ridge: np.ndarray, px: float, py: float,
                    radius: float) -> tuple[float, float, float] | None:
    """A ring fitted around the proposed centre (px, py): a circle through
    the thin-ridge pixels about a ring's size from it, then on the strongest
    stroke point along each ray (twice), then the rays' consensus."""
    h, w = ridge.shape
    reach = int(radius * 1.5) + 2
    x0, x1 = max(0, int(px) - reach), min(w, int(px) + reach + 1)
    y0, y1 = max(0, int(py) - reach), min(h, int(py) + reach + 1)
    ys, xs = np.nonzero(ridge[y0:y1, x0:x1])
    xs, ys = xs + x0, ys + y0
    band = np.abs(np.hypot(xs - px, ys - py) - radius * .92) <= radius * .45
    if band.sum() < 12:
        return None
    fit = _fit_circle(xs[band], ys[band], keep=2.0)
    if fit is None:
        return None
    for _ in range(2):
        if not (.6 * radius <= fit[2] <= 1.3 * radius):
            break
        sx, sy = _stroke_points(lightness, *fit)
        if len(sx) < 30:
            break
        refit = _fit_circle(sx, sy, keep=1.0)
        if refit is None:
            break
        fit = refit
    return _ring_consensus(lightness, *fit, radius) or fit


def _ring_core_white(hsv: np.ndarray, cx: float, cy: float, rho: float) -> float:
    """Share of rays whose brightest pixel within 1.5 px of the circle is
    guide white (V >= 165, S <= 70 as in ``white_strokes``)."""
    h, w = hsv.shape[:2]
    reach = int(rho) + 6
    x0, y0 = max(0, int(cx) - reach), max(0, int(cy) - reach)
    patch = hsv[y0:min(h, int(cy) + reach + 1), x0:min(w, int(cx) + reach + 1)]
    circle = (cx - x0, cy - y0, rho)
    v = _ray_profiles(patch[:, :, 2].astype(np.float32), circle, _CORE_OFFSETS)[0]
    s = _ray_profiles(patch[:, :, 1].astype(np.float32), circle, _CORE_OFFSETS)[0]
    k = np.argmax(v, axis=1)
    rows = np.arange(len(k))
    return float(np.mean((v[rows, k] >= 165) & (s[rows, k] <= 70)))


def _circle_kernel(rho: float) -> np.ndarray:
    """Correlation kernel: a thin circle of radius ``rho``, summing to one."""
    return _circle_kernel_cached(round(float(rho), 4))


@lru_cache(maxsize=64)
def _circle_kernel_cached(rho: float) -> np.ndarray:
    size = int(np.ceil(rho)) + 2
    kernel = np.zeros((2 * size + 1, 2 * size + 1), np.float32)
    points = {(int(round(size + rho * np.cos(a))), int(round(size + rho * np.sin(a))))
              for a in np.linspace(0, 2 * np.pi, 120, endpoint=False)}
    for x, y in points:
        kernel[y, x] = 1.0 / len(points)
    kernel.setflags(write=False)  # shared between calls
    return kernel


def find_ghost_ring(roi: np.ndarray, radius: float, hsv: np.ndarray | None = None,
                    lightness: np.ndarray | None = None,
                    around: tuple[float, float, float] | None = None) -> tuple[float, float, float, float] | None:
    """Miniclip's ghost ring found anywhere on the table, without knowing the
    cue ball: (x, y, ring radius, stroke share) in ROI pixels, or None.

    The ring is a thin bright stroke between dark outlines all the way round.
    Spots are proposed by correlating a circle with the thin-ridge map at
    half size; each is fitted on its own stroke (``_fit_ghost_ring``) and
    kept when nearly every ray crosses the stroke on the circle, the circle
    is a ghost ring's size and the stroke is guide white. A striped ball's
    white caps and the stick's metal joints make ring-like ridges too, but
    smaller (0.73-0.83 radii against 0.87-0.94), less complete or not white.
    Synthetic frames, every cloth and kind of shot: the ring found in 2821
    of 2838, 0.05 px typical, no other ring taken. The game shows one guide:
    the most complete ring is returned.

    ``around`` (x, y, reach) takes only a ring centred within ``reach`` of
    (x, y), and looks only there (a few milliseconds instead of ~25)."""
    ox = oy = 0
    if around is not None:
        ax, ay, reach = around
        h, w = roi.shape[:2]
        margin = reach + radius * 1.5 + 10
        x0, x1 = max(0, int(ax - margin)), min(w, int(ax + margin) + 1)
        y0, y1 = max(0, int(ay - margin)), min(h, int(ay + margin) + 1)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        roi = roi[y0:y1, x0:x1]
        hsv = None if hsv is None else hsv[y0:y1, x0:x1]
        lightness = None if lightness is None else lightness[y0:y1, x0:x1]
        ox, oy, ax, ay = x0, y0, ax - x0, ay - y0
    if lightness is None:
        lightness = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    if hsv is None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ridge = thin_ridges(lightness, RING_STROKE_RISE).astype(np.uint8)
    # (Dilated a pixel: the stroke's width and a ring size known to a pixel.)
    half = cv2.resize(cv2.dilate(ridge, np.ones((3, 3), np.uint8)).astype(np.float32), None,
                      fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    score = cv2.filter2D(half, -1, _circle_kernel(radius * .46), borderType=cv2.BORDER_CONSTANT)
    if around is not None:
        yy, xx = np.ogrid[:score.shape[0], :score.shape[1]]
        score[np.hypot(xx * 2 + .5 - ax, yy * 2 + .5 - ay) > reach + radius * .3] = 0
    gap = max(2, int(radius * .4))
    spots = []
    for _ in range(RING_PROPOSALS):
        index = int(np.argmax(score))
        if score.flat[index] < RING_PROPOSAL_SCORE:
            break
        sy, sx = divmod(index, score.shape[1])
        score[max(0, sy - gap):sy + gap + 1, max(0, sx - gap):sx + gap + 1] = 0
        spots.append((sx * 2 + .5, sy * 2 + .5))
    if not spots:
        return None
    # Cheap first look: the best stroke share of three ring sizes at each
    # spot (a real ring 0.90 or more there; 96% of other spots below 0.7,
    # and the full fit is skipped for them).
    sizes = (.86, .93, 1.0)
    shares = _stroke_shares(lightness, [(px, py, radius * f) for px, py in spots for f in sizes])
    best = None
    for (px, py), first in zip(spots, shares.reshape(len(spots), len(sizes)).max(axis=1)):
        if first < RING_FIRST_LOOK:
            continue
        fit = _fit_ghost_ring(lightness, ridge, px, py, radius)
        if fit is None or not (GHOST_RING_SIZE[0] * radius <= fit[2] <= GHOST_RING_SIZE[1] * radius):
            continue
        if around is not None and np.hypot(fit[0] - ax, fit[1] - ay) > reach:
            continue
        contrast = _ray_stroke_contrast(lightness, fit)[0]
        share = float(np.mean(contrast >= RING_STROKE_RISE))
        if share < GHOST_RING_SHARE or (best is not None and share <= best[3]):
            continue
        # The game's ring has dark outlines on both sides: its stroke stands
        # far above both (synthetic: median 73 or more, real frames 117-220).
        # A light ring drawn round a ball has the ball's shadow inside but
        # bare cloth outside (real frame: 26).
        if float(np.median(contrast)) < GHOST_RING_CONTRAST:
            continue
        if _ring_core_white(hsv, *fit) < GHOST_RING_WHITE:
            continue
        best = (*fit, share)
    return None if best is None else (best[0] + ox, best[1] + oy, best[2], best[3])


# Ring search: at most this many proposed spots, each with at least this
# share of a circle on (dilated) thin ridges.
RING_PROPOSALS = 24
RING_PROPOSAL_SCORE = .45
# A spot is fitted only when a circle there has this share of rays on a
# stroke (synthetic: real rings 0.90 or more, 96% of other spots below).
RING_FIRST_LOOK = .7
# A ghost ring: this share of rays crossing its stroke, a radius in this
# range (ball radii; the game's is ~0.91), and this share of its stroke
# guide white.
GHOST_RING_SHARE = .88
GHOST_RING_SIZE = (.85, 1.2)
GHOST_RING_WHITE = .9
GHOST_RING_CONTRAST = 50.0
# Only the cue ball can overlap the ghost ring: a ball centre within this
# many radii of the ring's centre is under it. The struck ball's is two true
# radii away; the radius is known to whole pixels (12 for 12.3: a cue ball
# 1.7 true radii away measured 1.74).
RING_OVERLAP = 1.85
# A ghost found along the ray within this many radii of the cue ball is
# checked to be the game's ring when a ring lies on the cue ball.
SHORT_RAY = 3.5
# A ball at least this white (share of its face) is taken for the cue ball
# by the ring, whatever lines leave it.
CUE_WHITE = .7
# A circle centred within this many radii of the ghost ring's centre is the
# ring itself (circle finding locks onto it), not a ball beside it.
RING_ITSELF = .3
# Once the ghost is found, no ball other than the cue ball lies within this
# many ghost radii of its centre (the struck ball touches it, two radii out).
GHOST_OVERLAP = 1.3
# A ghost this many radii or more from a ball, at the end of its ray, is a
# normal shot's (a ring right beside a ball can be the struck ball's own).
RAY_GHOST_REACH = 2.4


def ball_under_ring(candidates, ring_xy: tuple[float, float], radius: float,
                    white: np.ndarray | None = None, offset: tuple[int, int] = (0, 0)):
    """The cue-ball candidate under the ghost ring at ``ring_xy``, or None:
    the whitest beside the ring's centre, else the ring's own circle (circle
    finding locks onto the ring, a crisp white circle, and can miss the ball
    under it; it is fitted again beside the ring later).

    The struck ball touches the ghost ball two radii out, the cue ball under
    the ring is closer, but with the radius known to a pixel and centres
    beside a ring a pixel or two off, a ball 1.5-1.85 radii out can be
    either (synthetic: struck balls measured at 1.66-1.8, cue balls at up to
    1.84). There the game's object line tells: it leaves the struck ball's
    centre, away from the ring. ``white`` is the guide's line mask (ball
    faces removed) of the region starting at ``offset``."""
    gx, gy = ring_xy
    near = [(float(np.hypot(c[0] - gx, c[1] - gy)), c) for c in candidates]
    near = [(d, c) for d, c in near if d <= radius * RING_OVERLAP]
    if white is not None:
        # (Not for a ball as white as a cue ball: its stick also points away
        # from the ring, and where the stick's light tip and rings show as a
        # line, the cue ball read as the struck ball.)
        near = [(d, c) for d, c in near
                if d <= radius * 1.5 or c[5] >= CUE_WHITE or game_line_direction(
                    white, (c[0] - offset[0], c[1] - offset[1]), (c[0] - gx, c[1] - gy), radius,
                    cone_degrees=20.0) is None]
    if not near:
        return None
    beside = [c for d, c in near if d >= radius * RING_ITSELF]
    if beside:
        return max(beside, key=lambda c: c[5])
    return min(near, key=lambda item: item[0])[1]


def paint_out_ring(roi: np.ndarray, ring, radius: float) -> np.ndarray:
    """A copy of ``roi`` with the ghost ring (x, y, ring radius; ROI px)
    painted out: its stroke and outlines filled in from beside them. Lying
    on the cue ball, the ring is a crisp circle that circle finding and rim
    fits lock onto or are pulled towards."""
    gx, gy, rho = ring[:3]
    half = radius * .1 + 2.5     # the stroke's half width with its outline
    h, w = roi.shape[:2]
    reach = int(rho + half) + 8
    x0, x1 = max(0, int(gx) - reach), min(w, int(gx) + reach + 1)
    y0, y1 = max(0, int(gy) - reach), min(h, int(gy) + reach + 1)
    out = roi.copy()
    if x1 - x0 < 4 or y1 - y0 < 4:
        return out
    yy, xx = np.ogrid[y0:y1, x0:x1]
    band = (np.abs(np.hypot(xx - gx, yy - gy) - rho) <= half).astype(np.uint8) * 255
    out[y0:y1, x0:x1] = cv2.inpaint(roi[y0:y1, x0:x1], band, 3, cv2.INPAINT_TELEA)
    return out


def ball_beside_ring(roi: np.ndarray, ring, radius: float, min_r: int, max_r: int):
    """The whitest ball under the ghost ring (ROI px) found once the ring is
    painted out: (x, y, r, sat, val, white_fraction, red_spot, open_rim) in
    ROI px, or None. Circle finding can miss the cue ball under the ring
    altogether or fit the ring instead (synthetic near shots, white cloth:
    no cue ball found in 18 of 100)."""
    gx, gy = ring[:2]
    h, w = roi.shape[:2]
    reach = int(radius * (RING_OVERLAP + 2.5)) + 4
    x0, x1 = max(0, int(gx) - reach), min(w, int(gx) + reach + 1)
    y0, y1 = max(0, int(gy) - reach), min(h, int(gy) + reach + 1)
    if x1 - x0 < radius * 4 or y1 - y0 < radius * 4:
        return None
    window = paint_out_ring(roi[y0:y1, x0:x1], (gx - x0, gy - y0, ring[2]), radius)
    gray = cv2.medianBlur(cv2.cvtColor(window, cv2.COLOR_BGR2GRAY), 5)
    best = None
    # (The cue ball is white: no ball the cloth's colour is looked for.)
    for found in find_balls(window, gray, min_r, max_r, cloth_coloured=False):
        bx, by = found[0] + x0, found[1] + y0
        if np.hypot(bx - gx, by - gy) <= radius * RING_OVERLAP and (best is None or found[5] > best[5]):
            best = (bx, by, *found[2:])
    return best


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


# Half-width (px) of the straight band a game line's first estimate is
# read from (``game_line_direction``).
LINE_BAND = 2.0


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
    # First estimate: the narrow straight band holding the most stroke
    # pixels, among bands passing within 0.3 radii of the origin (as the
    # line must, see the anchor test below), then the principal axis of
    # those. One axis through all the white in the cone was tilted by
    # whatever else lies there: the cushion's lit edge along the table's
    # border, the white cap of a striped ball read a few pixels off its
    # centre, the table's faint head string (synthetic light grey: 4-6
    # degrees).
    pts = every[keep]
    angles = np.arctan2(ey, ex) + np.radians(np.arange(-cone_degrees, cone_degrees + 1e-6, .5))
    across = np.outer(pts[:, 1], np.cos(angles)) - np.outer(pts[:, 0], np.sin(angles))
    ahead = np.outer(pts[:, 0], np.cos(angles)) + np.outer(pts[:, 1], np.sin(angles)) > 0
    best = None
    for shift in np.linspace(-radius * .3, radius * .3, 7):
        band = (np.abs(across - shift) <= LINE_BAND) & ahead
        counts = band.sum(axis=0)
        i = int(np.argmax(counts))
        if best is None or counts[i] > best[0]:
            best = (int(counts[i]), band[:, i])
    if best[0] < max(12, int(radius)):
        return None
    centre, vx, vy = principal_axis(pts[best[1]])
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


def _clear_border_runs(white: np.ndarray, radius: float) -> None:
    """Clear long white runs along the table's border (the outermost
    ``LINE_BORDER`` rows and columns of the ROI) from a line mask, in place.
    The border is the cushion nose, and its lit edge (white on a light grey
    cloth) can lie in the outermost row: in a line's cone it tilted the fit
    (synthetic light grey: 6 degrees). A game line only crosses the border,
    in short runs, and a line running into a cushion near a ball needs
    every pixel it has there (clearing the whole border broke 3 of 1900
    synthetic near shots)."""
    run = max(8, int(radius))
    b = LINE_BORDER
    for rows in (slice(0, b), slice(white.shape[0] - b, white.shape[0])):
        strip = np.ascontiguousarray(white[rows])
        white[rows][cv2.morphologyEx(strip, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)) > 0] = 0
    for cols in (slice(0, b), slice(white.shape[1] - b, white.shape[1])):
        strip = np.ascontiguousarray(white[:, cols])
        white[:, cols][cv2.morphologyEx(strip, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8)) > 0] = 0


# Pixels along the table's border looked at by ``_clear_border_runs``.
LINE_BORDER = 3


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
                   balls: list[tuple[int, int, int, str]], limit: int = 4,
                   guide: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Up to ``limit`` possible ray directions from the cue ball, longest run first.

    ``guide`` is the guide mask (``white_guide_mask`` with ridges) when the
    caller has made it already.

    The longest white run is not always the ray: on a short shot the ghost
    ring's edge plus the game's cue-deflection line can form a longer run
    in a wrong direction. The caller keeps the one that ends in a ghost ball.
    """
    cx, cy, cr = cue
    # ``guide``: ``white_guide_mask(roi, balls, ridges=True)`` made already.
    if guide is None:
        guide = white_guide_mask(roi, balls, ridges=True)
    mask = cv2.dilate(guide, np.ones((3, 3), np.uint8))
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
    body: dict[tuple[int, int], tuple[float, float, float]] | None = None,
    min_behind: float = 1.35,
    axis_rule: bool = True,
    confirmed: set | None = None,
    white_cloth: bool = False,
    rings_only: bool = False,
) -> list[tuple[int, int, int, str]]:
    """Drop circle-like cue-stick ornaments lying behind the tracked cue ball.

    The game aim ray points from the cue ball toward the target. The physical
    stick lies on the opposite side. Real balls offset from that narrow stick
    axis stay in the list; the cue itself is always retained. A real ball
    lying under the stick also stays when ``rim_open`` shows a clear
    ball-to-cloth edge around most of its rim (stick ornaments have none),
    unless the stick beside it is about as wide as the circle (the butt's
    round end cap and rings; needs ``distance``, the float cloth-distance
    image of the region starting at ``offset`` in frame pixels). Balls in
    ``confirmed`` (a ball-sized rim fitted around them) always stay: on a
    white cloth a striped ball's white caps look like cloth beside the
    stick, so one lying on the stick's line read as one of its rings. On a
    ``white_cloth`` for the same reason a full-size circle with a clear rim
    on the stick's axis is not dropped as a ring at once: the ball-under-
    the-stick test below decides (size, rim, the stick's width beside it).
    ``rings_only``: drop only circles smaller than a ball centred on the axis
    (rings and joints), keeping every other circle near the stick (the
    ball-under-the-stick test is for a fresh look, not for a small
    correction of the axis).
    """
    cx, cy, cr = cue
    ux, uy = aim
    typical_r = float(np.median([b[2] for b in balls])) if balls else float(cr)
    kept = []
    for bx, by, br, kind in balls:
        if kind == "cue" or (bx, by, br) == (cx, cy, cr) or (confirmed and (bx, by) in confirmed):
            kept.append((bx, by, br, kind))
            continue
        dx, dy = bx - cx, by - cy
        behind = -(dx * ux + dy * uy)
        across = abs(dx * uy - dy * ux)
        shaft_half_width = max(9.0, min(cr * 1.05, br * 1.1))
        # The stick reaches ~30 ball radii behind the cue ball on real frames.
        if cr * min_behind < behind < typical_r * 40 and across <= shaft_half_width:
            # A ring centred on the stick's own axis is one of its
            # decorations (bands, joints, the glowing collar on real cues)
            # unless something non-cloth shows on both sides of the stick at
            # most of a ball radius out: a ball under the stick does, a
            # decoration is narrower.
            full_and_rimmed = (white_cloth and br >= typical_r * STICK_BALL_SIZE and rim_open is not None
                               and rim_open.get((bx, by), 0.0) >= STICK_BALL_RIM)
            # On a small correction of the axis a full-size circle on it can
            # be a ball under the stick, looking narrow and grey under one of
            # its joints (green 6 on green cloth): only the first look, with
            # the ball-under-the-stick test, drops one. (Rings and joints
            # are smaller: the joints ~0.8-0.9 of a ball.)
            if rings_only and br >= typical_r * STICK_FULL_SIZE:
                kept.append((bx, by, br, kind))
                continue
            if (axis_rule and across <= typical_r * STICK_AXIS_SHARE and not full_and_rimmed
                    and not _wider_than_stick(distance, bx - offset[0], by - offset[1], ux, uy, typical_r)):
                continue
            # A grey metal body (barely saturated, mid brightness, not white)
            # on the stick's axis is one of its joints: no ball looks like
            # that (the 8 is dark, the cue ball and stripes mostly white).
            sat, val, white = (body or {}).get((bx, by), (255.0, 0.0, 0.0))
            if across <= typical_r * STICK_AXIS_SHARE and sat < 60 and 50 < val < 175 and white < .3:
                continue
            if rings_only:
                kept.append((bx, by, br, kind))
                continue
            # (Without the aim's axis rule a ball half under the stick, its
            # circle shrunk by the stick, must not be lost on size alone.)
            ball_like = (rim_open is not None and rim_open.get((bx, by), 0.0) >= STICK_BALL_RIM
                         and (br >= typical_r * STICK_BALL_SIZE or not axis_rule))
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
                 balls: list[tuple[int, int, int, str]], cue: tuple[int, int, int],
                 hint: dict | None = None) -> dict:
    """Follow the game's aim graphics and estimate cue/object paths.

    Returns a dict with aim/cue/object segments, target, ghost, blockers,
    pot call, stick pull and a note. All coordinates are in frame pixels (the
    same frame ``table`` is in).

    ``hint`` is last frame's ``shot["hint"]`` (the live tracker passes it):
    while aiming the angle changes only a little per frame, so the ghost
    ring is first looked for near where it was, along last frame's aim,
    instead of scanning every ray direction (most of the time).
    ``shot["fast"]`` says whether that worked.
    """
    info: dict = {}
    shot = _predict_paths(roi, table, balls, cue, hint, info)
    pull = None
    if shot.get("aim_dir") is not None:
        ball_r = float(np.median([br for _, _, br, _ in balls])) if balls else float(cue[2])
        pull = stick_pull(roi, (cue[0] - table[0], cue[1] - table[1]), shot["aim_dir"], ball_r)
    # Only a ring found on the ray is a hint (not one placed from the object
    # line on top of the cue ball).
    next_hint = None
    if shot.get("ghost") is not None and shot.get("aim_dir") is not None and info.get("score"):
        score = info["score"]
        if info.get("fast") and hint is not None:
            score = hint["score"] + (score - hint["score"]) * HINT_EASE
        next_hint = {"aim_dir": shot["aim_dir"], "ghost": shot["ghost"][:2], "score": score}
    return {**shot, "stick_pull": pull, "hint": next_hint, "fast": info.get("fast", False)}


def ghost_from_object_line(white: np.ndarray, balls_local, cue_local, radius: float,
                           ring_mask: np.ndarray | None = None):
    """(ghost x, y, struck ball) in ROI pixels from the game's object-ball
    line, or None.

    When the struck ball lies right next to the cue ball, Miniclip's ghost
    ring sits on top of the cue ball and cannot be read. The game still draws
    its object line from the struck ball's centre, and the ghost touches that
    ball on the far side: one diameter back along the line.

    With ``ring_mask`` (the guide's strokes, ball faces removed) the ghost is
    then placed on its ring along that line (``ring_on_line``): one
    diameter back from the struck ball's centre was 1.4 px off (the ball
    radius is known to whole pixels, ~0.5 px small), and on a ghost this
    close to the cue ball that turned the aim by degrees.
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
            best = (dist, gx, gy, (bx, by, br), (ax, ay), (dx, dy))
    if best is None:
        return None
    _, gx, gy, struck, anchor, direction = best
    if ring_mask is not None:
        mask = ring_mask.copy()
        # The cue ball's own rim is a ring of the same size right beside it.
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(radius * 1.05), 0, thickness=-1)
        on_ring = ring_on_line(mask, anchor, direction, radius, (cx, cy), struck[:2])
        if on_ring is not None and np.hypot(on_ring[0] - gx, on_ring[1] - gy) <= 3.0:
            gx, gy = on_ring[0], on_ring[1]
    return gx, gy, struck


_RING_ANGLES = np.linspace(0, 2 * np.pi, 120, endpoint=False)


def ring_on_line(mask: np.ndarray, anchor, direction, radius: float, cue_xy, struck_xy,
                 reach: float = 3.0):
    """The ghost ring's centre on the object line ``anchor - s * direction``,
    searched within ``reach`` px of s = one diameter, or None.

    For each spot a circle of 0.8-1.05 radii is scored by how much of it
    lies on the strokes (blurred), leaving out its parts on the cue ball and
    the struck ball; the best spot is taken to a fraction of a pixel.
    Synthetic near shots: 0.14 px typical against 1.46 for one diameter
    back (the line itself is exact across, so only the spot along it is
    searched)."""
    blurred = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.0)
    ax, ay = anchor
    ux, uy = direction
    cx, cy = cue_xy
    tx, ty = struck_xy
    spots = np.arange(2 * radius - reach, 2 * radius + reach + 1e-6, .25)
    sizes = np.arange(.8 * radius, 1.05 * radius + 1e-6, .25)
    scores = np.full((len(spots), len(sizes)), -1.0)
    for i, s in enumerate(spots):
        gx, gy = ax - s * ux, ay - s * uy
        for j, rho in enumerate(sizes):
            px, py = gx + rho * np.cos(_RING_ANGLES), gy + rho * np.sin(_RING_ANGLES)
            use = (np.hypot(px - cx, py - cy) > 1.1 * radius) & (np.hypot(px - tx, py - ty) > 1.1 * radius)
            if use.sum() < 30:
                continue
            values = cv2.remap(blurred, px[use].astype(np.float32)[None], py[use].astype(np.float32)[None],
                               cv2.INTER_LINEAR)[0]
            scores[i, j] = float(values.mean())
    i, j = np.unravel_index(int(np.argmax(scores)), scores.shape)
    if scores[i, j] < RING_ON_LINE_SCORE:
        return None
    s = float(spots[i])
    if 0 < i < len(spots) - 1:
        a, b, c = scores[i - 1, j], scores[i, j], scores[i + 1, j]
        if min(a, c) > -1 and a - 2 * b + c < 0:
            s += float(np.clip(.5 * (a - c) / (a - 2 * b + c), -.5, .5)) * .25
    return ax - s * ux, ay - s * uy, float(sizes[j]), float(scores[i, j])


def _is_ghost_ring(roi: np.ndarray, x: float, y: float, radius: float) -> bool:
    """True when a thin bright stroke with dark outlines runs along nearly
    the whole circle of about ``radius`` at (x, y): the ghost ring, not a
    ball's rim (synthetic near shots on blue: a cue ball detected on the
    ring scored 0.98-1.00, real cue balls at most 0.97, typically 0.31)."""
    lightness = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    return max(_thin_stroke_share(lightness, x, y, radius * f) for f in (.9, 1.0)) >= GHOST_RING_STROKE


# The share of thin bright stroke along a circle from which it is the ring.
GHOST_RING_STROKE = .98
# A circle counts as the ghost ring when this share of it (blurred) lies on
# the guide's strokes.
RING_ON_LINE_SCORE = .35


# Live tracking: the ghost is first looked for within this many ball radii
# (along the ray) of last frame's, and taken only with at least this share of
# the reference ring score (the last full reading's, eased towards later
# fast ones so a ring that slowly weakens is kept, a sudden drop is not).
HINT_REACH = 2.5
HINT_SCORE = .5
HINT_EASE = .2
# A ring found from the hint counts only if the game's ray leads to it: no
# stretch without ray pixels longer than this many ball radii. (The real
# ray is thin and broken into short pieces in the mask, ~70% covered.)
RAY_GAP = 1.5


def ray_reaches(white: np.ndarray, cue: tuple[float, float, float], ghost: tuple[float, float],
                radius: float) -> bool:
    """True when the white ray runs from the cue ball's edge to the ring.

    Miniclip's ray stops at the ghost ball. A ring found near last frame's
    ghost after the aim moved onto a nearer ball has no ray leading to it
    (the ray now ends at the nearer ghost), so a long stretch is empty."""
    cx, cy, cr = cue
    gx, gy = ghost
    length = float(np.hypot(gx - cx, gy - cy))
    start, end = cr * 1.3, length - radius * 1.25
    if end - start < 4:
        return True
    ux, uy = (gx - cx) / length, (gy - cy) / length
    t = np.arange(start, end, 1.0)
    h, w = white.shape[:2]
    hit = np.zeros(t.shape, bool)
    for s in (-2, -1, 0, 1, 2):
        xs = np.clip(np.round(cx + ux * t - uy * s).astype(int), 0, w - 1)
        ys = np.clip(np.round(cy + uy * t + ux * s).astype(int), 0, h - 1)
        hit |= white[ys, xs] > 0
    gap = longest = 0
    for present in hit:
        gap = 0 if present else gap + 1
        longest = max(longest, gap)
    return longest <= max(6.0, radius * RAY_GAP)


def _predict_paths(roi: np.ndarray, table: tuple[int, int, int, int],
                   balls: list[tuple[int, int, int, str]], cue: tuple[int, int, int],
                   hint: dict | None = None, info: dict | None = None) -> dict:
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
    windows = None
    if hint is not None:
        # The hinted read only reads lines near last frame's aim: along the
        # ray (cue ball to ghost) and within ~11 radii of the ghost (the
        # object line from the struck ball, the cue line from the ghost).
        hgx, hgy = hint["ghost"][0] - left, hint["ghost"][1] - top
        pad, around = 12.0, 12 * ball_r
        windows = [(min(cue_local[0], hgx) - pad, min(cue_local[1], hgy) - pad,
                    max(cue_local[0], hgx) + pad + 1, max(cue_local[1], hgy) + pad + 1),
                   (hgx - around, hgy - around, hgx + around + 1, hgy + around + 1)]
    strokes, line_strokes = white_stroke_masks(roi, roi_hsv, ridge_windows=windows)

    def line_mask(line_strokes):
        # The game's lines are read on the mask with thin ridges (on a white
        # cloth their white core is no brighter than the cloth; the object
        # line was never read there and the struck ball's fit fell back on
        # the ghost, 4-5 px off).
        white = line_strokes.copy()
        for bx, by, br, _ in balls_local:
            cv2.circle(white, (int(bx), int(by)), int(br * 1.18), 0, thickness=-1)
        _clear_border_runs(white, ball_r)
        return white

    white = line_mask(line_strokes)
    # The reticle ring touches the struck ball, so mask only the ball faces
    # there; the wider mask (1.18 radii) would erase the ring's near side.
    ghost_mask = strokes.copy()
    for bx, by, br, _ in balls_local:
        cv2.circle(ghost_mask, (int(bx), int(by)), int(br * .9), 0, thickness=-1)
    direction, ghost = None, None
    fast = False
    if hint is not None:
        # Last frame's aim and ghost: look near the old ghost only. When the
        # ghost jumped (the aim moved onto another ball), the full scan
        # below finds it; a search along the whole old ray could lock onto
        # something ring-like further on.
        hux, huy = hint["aim_dir"]
        hgx, hgy = hint["ghost"][0] - left, hint["ghost"][1] - top
        g_along = (hgx - cue_local[0]) * hux + (hgy - cue_local[1]) * huy
        reach = HINT_REACH * ball_r
        found = find_ghost_ball(ghost_mask, cue_local, (hux, huy), ball_r, roi,
                                along_range=(g_along - reach, g_along + reach), spots=1)
        if (found is not None and found[3] >= hint["score"] * HINT_SCORE
                and ray_reaches(white, cue_local, found[:2], ball_r)):
            direction, ghost, fast = (hux, huy), found, True
    if not fast:
        if windows is not None and line_strokes is not strokes:
            # The full scan reads lines anywhere: ridges everywhere.
            strokes, line_strokes = white_stroke_masks(roi, roi_hsv)
            white = line_mask(line_strokes)
        # Of the candidate ray directions, the real one ends in the ghost ball.
        candidates = aim_candidates(roi, cue_local, balls_local, guide=white)
        for candidate in candidates:
            found = find_ghost_ball(ghost_mask, cue_local, candidate, ball_r, roi)
            if direction is None:
                direction = candidate
            if found is not None and (ghost is None or found[3] > ghost[3]):
                direction, ghost = candidate, found
        if ghost is None and candidates and line_strokes is not strokes:
            # On a white cloth the ring's white core can be no brighter than
            # the cloth, like the ray's (only its outlines show): then look
            # again on the strokes with ridges. Only then: where strokes meet
            # inside the ring, the ridges fill it and can hide a ring that
            # the plain strokes show.
            ridged = line_strokes.copy()
            for bx, by, br, _ in balls_local:
                cv2.circle(ridged, (int(bx), int(by)), int(br * .9), 0, thickness=-1)
            for candidate in candidates:
                found = find_ghost_ball(ridged, cue_local, candidate, ball_r, roi)
                if found is not None and (ghost is None or found[3] > ghost[3]):
                    direction, ghost = candidate, found
    if ghost is None:
        ghost = find_ghost_ball(ghost_mask, cue_local, None, ball_r, roi) if direction is None else None
    if ghost is not None and np.hypot(ghost[0] - cue_local[0], ghost[1] - cue_local[1]) <= ball_r * RING_OVERLAP:
        # A ring overlapping the cue ball: the ring score is pulled by the
        # ball's face (3 px on synthetic near shots); fit it on its stroke.
        exact = find_ghost_ring(roi, ball_r, hsv=roi_hsv, around=(ghost[0], ghost[1], ball_r * .5))
        if exact is not None:
            ghost = (exact[0], exact[1], exact[2], ghost[3])
    elif (ghost is not None and np.hypot(ghost[0] - cue_local[0], ghost[1] - cue_local[1]) <= ball_r * SHORT_RAY
          and find_ghost_ring(roi, ball_r, hsv=roi_hsv, around=(ghost[0], ghost[1], ball_r * .5)) is None
          and find_ghost_ring(roi, ball_r, hsv=roi_hsv,
                              around=(cue_local[0], cue_local[1], ball_r * RING_OVERLAP)) is not None):
        # A "ghost" a short ray away that is not the game's ring (white bits
        # of the struck ball and the ring's arc beside it can pass the ring
        # score), while the game's ring lies on the cue ball: a near shot.
        ghost = None
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
        near_ghost = ghost_from_object_line(white, balls_local, cue_local, ball_r, ring_mask=ghost_mask)
        # The ghost ring itself, found on its own stroke on top of the cue
        # ball, places the ghost exactly (0.05 px typical against 0.14 from
        # the object line, which can also not be read at all).
        on_cue = find_ghost_ring(roi, ball_r, hsv=roi_hsv,
                                 around=(cue_local[0], cue_local[1], ball_r * RING_OVERLAP))
        if on_cue is not None and np.hypot(on_cue[0] - cue_local[0], on_cue[1] - cue_local[1]) >= ball_r * .2:
            agrees = near_ghost is not None and np.hypot(near_ghost[0] - on_cue[0],
                                                         near_ghost[1] - on_cue[1]) <= ball_r * .3
            # (Without the object line the struck ball is the one touching
            # the ring, found further on.)
            near_ghost = (on_cue[0], on_cue[1], near_ghost[2] if agrees else None)
        if near_ghost is not None:
            ghost = (near_ghost[0], near_ghost[1], ball_r, 1.0)
            dx, dy = near_ghost[0] - cue_local[0], near_ghost[1] - cue_local[1]
            n = float(np.hypot(dx, dy))
            direction = (dx / n, dy / n)
    if info is not None:
        info["fast"] = fast
        info["score"] = ghost[3] if ghost is not None and near_ghost is None else None
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
        if ghost_frame is not None and glen <= ball_r * RING_OVERLAP:
            # The ghost on (or right by) the cue ball: from the cue ball's
            # centre to it the aim turns by degrees per pixel, but the stick
            # lies along it (synthetic near shots: aim more than 1 px off at
            # the ghost in 127 of 988, 33 with the stick).
            along_stick = stick_direction_near(roi, cue_local[:2], (ghost[0], ghost[1]), ball_r)
            if along_stick is not None:
                direction = along_stick
    ux, uy = direction
    obstacles = [(bx, by, br) for bx, by, br, kind in balls if kind != "cue"]
    angle = float(np.degrees(np.arctan2(uy, ux)))
    selected = None
    if near_ghost is not None and near_ghost[2] is not None:
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
    count, labels, stats, centres = cv2.connectedComponentsWithStats(dark, connectivity=8)
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
        # Centre on the hole's darkest core: a darker display (lower
        # brightness, higher gamma) pulls the pocket's shaded rim under the
        # limit too, and the whole blob's centre slides towards it.
        values = hsv[y:y + bh, x:x + bw, 2]
        member = labels[y:y + bh, x:x + bw] == i
        darkest = float(values[member].min())
        core = member & (values <= darkest + .5 * (limit - darkest))
        cx_, cy_ = float(centres[i][0]), float(centres[i][1])
        if int(core.sum()) >= .25 * area:
            ys, xs = np.nonzero(core)
            kx, ky = x + float(xs.mean()), y + float(ys.mean())
            # Only a real slide moves it (a sub-pixel change would shift the
            # whole table fit for nothing).
            if np.hypot(kx - cx_, ky - cy_) > 1.5:
                cx_, cy_ = kx, ky
        blobs.append((cx_, cy_, float(bw), float(area)))
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


# White-mask ring score at the cue ball's centre above which the "cue ball"
# is the ghost ring lying on it (measured: 0.17 locked, at most 0.03 on real
# cue balls).
RING_ON_CUE_SCORE = .12


def refit_beside_ring(distance: np.ndarray, cx: float, cy: float, r: float,
                      gx: float, gy: float, ring_r: float) -> tuple[float, float]:
    """Ball centre near (cx, cy) fitted on the cloth-distance image, leaving
    out rim samples that fall on the ghost ring drawn at (gx, gy). The fit
    is kept only when it is clearly better than (cx, cy) itself."""
    cos, sin = np.cos(_RIM_ANGLES), np.sin(_RIM_ANGLES)
    band = max(3.0, ring_r * .3)

    def score(px: float, py: float) -> float | None:
        ix, iy = px + (r - 1.5) * cos, py + (r - 1.5) * sin
        ox, oy = px + (r + 1.5) * cos, py + (r + 1.5) * sin
        clear = ((np.abs(np.hypot(ix - gx, iy - gy) - ring_r) > band)
                 & (np.abs(np.hypot(ox - gx, oy - gy) - ring_r) > band))
        if clear.mean() < .4:
            return None
        inside = cv2.remap(distance, ix[None].astype(np.float32), iy[None].astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        outside = cv2.remap(distance, ox[None].astype(np.float32), oy[None].astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        return float((inside - outside)[clear].mean())

    best, best_score = (cx, cy), None
    for step, span in ((1.0, r * 1.2), (.25, 1.5)):
        offsets = np.arange(-span, span + 1e-6, step)
        found = None
        for px in best[0] + offsets:
            for py in best[1] + offsets:
                value = score(float(px), float(py))
                if value is not None and (found is None or value > found[0]):
                    found = (value, (float(px), float(py)))
        if found is None:
            return cx, cy
        best_score, best = found
    start = score(cx, cy)
    # Measured: real fixes score >= ~1.9x the start (or the start sits on
    # the ring), wrong moves 1.1-1.4x.
    if start is not None and best_score < start * RING_REFIT_GAIN:
        return cx, cy
    return best


RING_REFIT_GAIN = 1.6


def stick_matches_cue(stick_matches, cue) -> bool:
    """True when the best stick match ends at the chosen cue ball."""
    best = min(stick_matches, key=lambda item: item[:6])[6]
    return best is cue


def stick_direction(stick_lines, cue_xy: tuple[float, float], radius: float,
                    roi_hsv: np.ndarray | None = None) -> tuple[float, float] | None:
    """Shot direction implied by the longest line ending at the cue ball.

    The stick and the game's aim ray lie on one axis, on opposite sides of
    the ball; the ray is the very bright side (near-white, light cyan on a
    blue table). The direction points from the stick towards the ray."""
    if stick_lines is None:
        return None
    cx, cy = cue_xy
    best = None
    for x1, y1, x2, y2 in stick_lines[:, 0, :].astype(float):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < radius * 4:
            continue
        gap = abs((cx - x1) * (y2 - y1) - (cy - y1) * (x2 - x1)) / length
        near, far = ((x1, y1), (x2, y2)) if np.hypot(cx - x1, cy - y1) < np.hypot(cx - x2, cy - y2)             else ((x2, y2), (x1, y1))
        if gap > radius * 1.4 or np.hypot(cx - near[0], cy - near[1]) > radius * 2.2:
            continue
        if best is None or length > best[0]:
            dx, dy = cx - far[0], cy - far[1]
            n = float(np.hypot(dx, dy))
            best = (length, (dx / n, dy / n))
    if best is None:
        return None
    ux, uy = best[1]
    if roi_hsv is not None:
        def bright_share(sign: float) -> float:
            t = np.arange(3, 12.01, .5) * radius
            hits = []
            for k in np.linspace(-.3, .3, 5) * radius:
                px = np.clip((cx + sign * ux * t - uy * k).astype(int), 0, roi_hsv.shape[1] - 1)
                py = np.clip((cy + sign * uy * t + ux * k).astype(int), 0, roi_hsv.shape[0] - 1)
                samples = roi_hsv[py, px]
                hits.append((samples[:, 1] <= 120) & (samples[:, 2] >= 225))
            return float(np.any(hits, axis=0).mean())
        ahead, behind = bright_share(1.0), bright_share(-1.0)
        if behind > ahead + .2:
            ux, uy = -ux, -uy
        elif ahead <= behind + .2:
            return None        # no clear ray side: do not guess
    return ux, uy


def stick_direction_near(roi: np.ndarray, cue_xy: tuple[float, float], ghost_xy: tuple[float, float],
                         radius: float) -> tuple[float, float] | None:
    """The aim read from the cue stick (ROI px; pointing from the cue ball
    towards the ghost), or None.

    With the ghost on top of the cue ball the aim from the cue ball's centre
    to it turns by degrees per pixel: the centre is known to about a pixel
    (half its rim lies under the ring), the ghost only a few pixels from it.
    The stick lies along the aim behind the cue ball. Its band of off-cloth
    shade gives the direction best (``_stick_band_direction``); where that
    is not found, its straight edges (``_stick_edge_direction``). The
    ghost's direction from the cue ball says which side the stick is on and
    roughly where (within 45 degrees). Synthetic near shots, 10 cloths: the
    stick read in 77%, 0.16 degrees at the 95th percentile, never 1.2 off."""
    cx, cy = cue_xy
    rx, ry = ghost_xy[0] - cx, ghost_xy[1] - cy
    norm = float(np.hypot(rx, ry))
    if norm < 1e-6:
        return None
    rough = (rx / norm, ry / norm)
    h, w = roi.shape[:2]
    reach = int(radius * 16)
    x0, x1 = max(0, int(cx) - reach), min(w, int(cx) + reach + 1)
    y0, y1 = max(0, int(cy) - reach), min(h, int(cy) + reach + 1)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    gray = cv2.cvtColor(roi[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    cue = (cx - x0, cy - y0)
    return (_stick_band_direction(gray, cue, rough, radius)
            or _stick_edge_direction(gray, cue, rough, radius))


def _stick_band_direction(gray: np.ndarray, cue_xy, rough, radius: float) -> tuple[float, float] | None:
    """The stick's axis from its band of off-cloth shade behind the cue ball.

    Off cloth: at least ``STICK_SHADE`` from the cloth's own shade around
    (the stick is far darker or lighter than the cloth; the table's
    lettering is only a little darker). Of the directions within 45 degrees
    of ``rough``, the one whose backward ray stays off cloth longest from
    within 5 radii of the cue ball (the tip) is taken; short cloth-like gaps
    (the stick's light rings and joints) are bridged, and the run must be at
    least 7 radii long (the game's own lines are at most ~5.4). Then the
    band's middle is measured across it every 1.5 px over 9 radii, and a
    line fitted through those middles turns the axis onto the band (three
    times). The axis must pass by the cue ball's centre: the stick points
    at it."""
    cx, cy = cue_xy
    off = (np.abs(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
                  - cv2.medianBlur(gray, 31).astype(np.float32)) >= STICK_SHADE).astype(np.float32)

    def sample(xs, ys):
        return cv2.remap(off, np.asarray(xs, np.float32), np.asarray(ys, np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) > .5

    a0 = float(np.arctan2(rough[1], rough[0]))
    ts = np.arange(radius * 1.1, radius * 14, 1.0)
    angles = a0 + np.radians(np.arange(-STICK_AIM_GATE, STICK_AIM_GATE + .01, .5))
    on = sample(cx - np.outer(np.cos(angles), ts), cy - np.outer(np.sin(angles), ts))
    gap = int(radius * .7)
    latest_start = int(radius * (5 - 1.1))
    runs = np.zeros(len(angles))
    starts = np.zeros(len(angles), int)
    for i, row in enumerate(on):
        hits = np.nonzero(row)[0]
        if not len(hits):
            continue
        # Runs of hits with gaps of at most ``gap`` samples between them.
        breaks = np.nonzero(np.diff(hits) > gap + 1)[0]
        firsts = np.concatenate(([hits[0]], hits[breaks + 1]))
        lasts = np.concatenate((hits[breaks], [hits[-1]]))
        ok = firsts <= latest_start
        if ok.any():
            lengths = (lasts - firsts + 1)[ok]
            k = int(np.argmax(lengths))
            runs[i], starts[i] = lengths[k], firsts[ok][k]
    top = float(runs.max())
    if top < radius * STICK_RUN:
        return None
    k = int(np.argmax(runs))
    lo = hi = k
    while lo > 0 and runs[lo - 1] >= top * .9:
        lo -= 1
    while hi < len(runs) - 1 and runs[hi + 1] >= top * .9:
        hi += 1
    a = float((angles[lo] + angles[hi]) / 2)
    tip = float(ts[starts[k]])
    across = np.arange(-radius * 1.3, radius * 1.3 + 1e-6, .5)
    centre = len(across) // 2
    tt = np.arange(tip + radius * .3, tip + radius * 9, 1.5)
    icpt = 0.0
    for _ in range(3):
        ux, uy = np.cos(a), np.sin(a)
        rows = sample(cx - np.outer(tt, np.full(len(across), ux)) - np.outer(np.ones(len(tt)), across) * uy,
                      cy - np.outer(tt, np.full(len(across), uy)) + np.outer(np.ones(len(tt)), across) * ux)
        mids, at = [], []
        for t, row in zip(tt, rows):
            hits = np.nonzero(row)[0]
            if not len(hits):
                continue
            c = int(hits[np.argmin(np.abs(hits - centre))])
            if abs(c - centre) > 4:          # 2 px
                continue
            left = right = c
            while left > 0 and row[left - 1]:
                left -= 1
            while right < len(row) - 1 and row[right + 1]:
                right += 1
            if left == 0 or right == len(row) - 1:   # wider than the window: a ball under it
                continue
            mids.append((across[left] + across[right]) / 2)
            at.append(t)
        if len(mids) < 8:
            return None
        at, mids = np.array(at), np.array(mids)
        use = np.ones(len(mids), bool)
        for _ in range(3):
            slope, icpt = np.polyfit(at[use], mids[use], 1)
            res = np.abs(mids - (slope * at + icpt))
            use = res <= max(1.0, 2.5 * float(np.median(res)))
            if use.sum() < 8:
                return None
        # The middles drift sideways by ``slope`` per pixel backwards.
        a -= float(np.arctan(slope))
    if abs(icpt) > max(3.0, radius * .3):
        return None
    return float(np.cos(a)), float(np.sin(a))


def _stick_edge_direction(gray: np.ndarray, cue_xy, rough, radius: float) -> tuple[float, float] | None:
    """The stick's direction from its long straight edges: lines (the same
    line search as for the stick in ``detect``) passing within 0.8 radii of
    the cue ball's centre, starting behind its face (the tip is 1.25-3.75
    radii out, an edge piece after a joint further) and pointing within 45
    degrees of ``rough``. Edges parallel to the longest one are averaged
    (the stick tapers; its two edges lean opposite ways)."""
    cx, cy = cue_xy
    rx, ry = rough
    min_r = max(8, int(radius * .75))
    lines = cv2.HoughLinesP(cv2.Canny(cv2.medianBlur(gray, 5), 50, 140), 1, np.pi / 180, threshold=16,
                            minLineLength=max(45, min_r * 6), maxLineGap=max(12, int(min_r * 1.2)))
    if lines is None:
        return None
    found = []
    for ax, ay, bx, by in lines[:, 0, :].astype(float):
        length = float(np.hypot(bx - ax, by - ay))
        if length < radius * 6:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        if ux * rx + uy * ry < 0:
            ux, uy = -ux, -uy
        if ux * rx + uy * ry < np.cos(np.radians(STICK_AIM_GATE)):
            continue
        if abs((cx - ax) * uy - (cy - ay) * ux) > radius * .8:
            continue
        near_end = max((ax - cx) * ux + (ay - cy) * uy, (bx - cx) * ux + (by - cy) * uy)
        if not -radius * 9 <= near_end <= -radius * .5:
            continue
        found.append((length, ux, uy))
    if not found:
        return None
    _, vx, vy = max(found)
    parallel = [(l, ux, uy) for l, ux, uy in found if ux * vx + uy * vy >= np.cos(np.radians(2.5))]
    sx, sy = sum(l * ux for l, ux, _ in parallel), sum(l * uy for l, _, uy in parallel)
    norm = float(np.hypot(sx, sy))
    return float(sx / norm), float(sy / norm)


# How far (degrees) the stick may point from the ghost's direction when that
# is read from a ghost on top of the cue ball.
STICK_AIM_GATE = 45.0
# Off-cloth shade (grey levels) of the stick against the cloth around it,
# and the shortest run of it (ball radii) behind the cue ball.
STICK_SHADE = 20.0
STICK_RUN = 7.0
# The stick's rings are filtered again along the aim read from the ghost
# when it turned more than this from the first rough reading (0.5 degree
# steps): 17 radii back a 0.75 degree turn is 2.4 px, beyond the 0.2
# radii that tell a ring on the axis.
STICK_REFILTER_COS = float(np.cos(np.radians(.25)))
# ...and then a circle at least this share of a ball's size is kept.
STICK_FULL_SIZE = .95


def stick_axis(stick_lines, cue_xy: tuple[float, float], radius: float,
               aim: tuple[float, float]) -> tuple[float, float] | None:
    """The stick's own direction (pointing like ``aim``), from the longest
    image line through the cue ball's centre that lies behind it within 15
    degrees of ``aim``, or None.

    The aim read from a shot whose struck ball nearly touches the cue ball
    is degrees off (the ghost lies on the cue ball), and at the stick's far
    rings, ~17 radii back, that put them 1.8 radii beside the assumed axis:
    they stayed as balls. A long straight edge of the stick gives its
    direction to ~0.5 degrees. (A line ahead of the cue ball, the game's
    own lines, is not the stick; ``aim`` tells the sides apart.)"""
    if stick_lines is None or aim is None:
        return None
    cx, cy = cue_xy
    ax, ay = aim
    best = None
    for x1, y1, x2, y2 in stick_lines[:, 0, :].astype(float):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < radius * 6:
            continue
        if abs((cx - x1) * (y2 - y1) - (cy - y1) * (x2 - x1)) / length > radius * .8:
            continue
        d1, d2 = float(np.hypot(x1 - cx, y1 - cy)), float(np.hypot(x2 - cx, y2 - cy))
        if min(d1, d2) > radius * 5:
            continue
        fx, fy = (x1, y1) if d1 > d2 else (x2, y2)
        if (fx - cx) * ax + (fy - cy) * ay >= 0:
            continue
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        if ux * ax + uy * ay < 0:
            ux, uy = -ux, -uy
        if ux * ax + uy * ay < np.cos(np.radians(15)):
            continue
        if best is None or length > best[0]:
            best = (length, (float(ux), float(uy)))
    return None if best is None else best[1]


def shadow_direction(lightness: np.ndarray, cloth: np.ndarray, balls, radius: float) -> tuple[float, float] | None:
    """Unit direction in which the balls' drop shadows fall, measured as the
    darkest side of a ring just outside each object ball (None when too few
    balls show one). All shadows fall the same way.

    Directions towards a neighbouring ball are left out of its ring (in a
    rack every ring touched others, so no ball counted), and the darkness is
    taken against the free arc's own average (a partial arc has a direction
    of its own). Only clearly one-sided rings count, and the direction most
    of them agree on wins: a false ball candidate (a ring of the stick or
    the guide, with its dark outlines) reads some other side, and a plain
    average fell apart (white cloth, synthetic seed 40001: 11 balls at ~51
    deg, 5 candidates at about -125 deg). Synthetic white racks: found on
    28/30 within 13 deg (was 26/30, up to 42 deg off)."""
    angles, weights = [], []
    points = [(b[0], b[1]) for b in balls]
    for i, (bx, by, _, kind) in enumerate(balls):
        if kind == "cue":
            continue
        free = np.ones(len(_RIM_ANGLES), bool)
        for j, (ox, oy) in enumerate(points):
            dist = float(np.hypot(ox - bx, oy - by))
            if j == i or dist < 1e-6 or dist > 2.8 * radius:
                continue
            free &= np.cos(_RIM_ANGLES - np.arctan2(oy - by, ox - bx)) < np.cos(np.radians(40))
        if free.mean() < .4:
            continue
        xs = (bx + radius * 1.2 * np.cos(_RIM_ANGLES))[None].astype(np.float32)
        ys = (by + radius * 1.2 * np.sin(_RIM_ANGLES))[None].astype(np.float32)
        value = cv2.remap(lightness, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        around = cv2.remap(cloth, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        dark = np.clip(around - value, 0, 60)[free]
        if dark.sum() < 1:
            continue
        arc = _RIM_ANGLES[free]
        excess = dark - dark.mean()
        v = np.array([np.sum(excess * np.cos(arc)), np.sum(excess * np.sin(arc))])
        one_sided = float(np.linalg.norm(v) / dark.sum())
        if one_sided < SHADOW_ONE_SIDED * free.mean():
            continue
        angles.append(float(np.arctan2(v[1], v[0])))
        weights.append(one_sided)
    if len(angles) < 2:
        return None
    angles, weights = np.array(angles), np.array(weights)
    agree = np.cos(angles[:, None] - angles[None, :])
    support = (np.clip(agree, 0, None) * weights[None, :]).sum(axis=1)
    inliers = agree[int(np.argmax(support))] >= np.cos(np.radians(40))
    if inliers.sum() < 2 or weights[inliers].sum() < .5 * weights.sum():
        return None
    mean = np.array([np.sum(weights[inliers] * np.cos(angles[inliers])),
                     np.sum(weights[inliers] * np.sin(angles[inliers]))])
    norm = float(np.linalg.norm(mean))
    return (float(mean[0] / norm), float(mean[1] / norm))


# A ball's ring counts for the shadow direction when its darkness is at
# least this one-sided (|sum of dark * direction| / sum of dark; a real
# shadow reads ~0.7-0.85, a ring among strokes or packed balls ~0.1-0.5).
SHADOW_ONE_SIDED = .45


class _WhiteCloth:
    """What the white-cloth rim fits need, measured once per frame: the
    table's lightness, the cloth's local lightness, the ball radius and the
    shadow direction. ``of(roi, balls, offset)`` is None unless the cloth is
    near-white and a shadow direction is found."""

    def __init__(self, lab, lightness, cloth, local, radius, sdir):
        self.lab, self.lightness, self.cloth = lab, lightness, cloth
        self.local, self.radius, self.sdir = local, radius, sdir

    @classmethod
    def of(cls, roi: np.ndarray, balls, offset: tuple[int, int]):
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0].astype(np.float32)
        cloth = local_lightness(lightness)
        # Only a near-white cloth: on a light grey one the balls are far
        # brighter than the cloth and the usual fit finds their whole rim.
        # (Judged on the table's brightest cloth: the lighting falls off
        # towards the rails, where white cloth is as dark as light grey.)
        if float(np.percentile(cloth[::8, ::8], 95)) < WHITE_CLOTH_L or not balls:
            return None
        # A colourless cloth only (a bright yellow or pink one is lit as
        # strongly but its colour finds every rim).
        sample = lab[::8, ::8].reshape(-1, 3).astype(np.float32)
        if float(np.hypot(np.median(sample[:, 1]) - 128, np.median(sample[:, 2]) - 128)) >= WHITE_CLOTH_CHROMA:
            return None
        local = [(bx - offset[0], by - offset[1], br, kind) for bx, by, br, kind in balls]
        radius = float(np.median([b[2] for b in local]))
        sdir = shadow_direction(lightness, cloth, local, radius)
        return None if sdir is None else cls(lab, lightness, cloth, local, radius, sdir)

    def white_here(self, x: float, y: float) -> bool:
        """The cloth around (x, y) is light and colourless (on a coloured
        cloth the colour difference finds a ball's whole rim)."""
        h, w = self.lightness.shape
        if not (0 <= int(y) < h and 0 <= int(x) < w) or self.cloth[int(y), int(x)] < LIGHT_CLOTH_L:
            return False
        reach = 60
        y0, y1, x0, x1 = max(0, int(y) - reach), min(h, int(y) + reach + 1), max(0, int(x) - reach), min(w, int(x) + reach + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        ring = np.hypot(xx - int(x), yy - int(y)) < reach
        a = float(np.median(self.lab[y0:y1, x0:x1, 1][ring])) - 128
        b = float(np.median(self.lab[y0:y1, x0:x1, 2][ring])) - 128
        return float(np.hypot(a, b)) < WHITE_CLOTH_CHROMA

    def rim_fit(self, x: float, y: float) -> tuple[float, float] | None:
        """Centre of the ball near (x, y) from its lit-side rim dip, or None."""
        radius = self.radius
        x0, y0 = float(x), float(y)
        others = [(bx, by) for bx, by, _, _ in self.local if np.hypot(bx - x0, by - y0) > radius * .6]
        cx, cy = x0, y0
        fit = None
        for inner, outer in ((.4, 1.9), (.55, 1.35)):
            points = _lit_rim_dips(self.lightness, cx, cy, radius, self.sdir, inner, outer, others)
            found = _consensus_circle(points, cx, cy, radius) if len(points) >= WHITE_RIM_MIN_POINTS else None
            if found is None or np.hypot(found[0] - x0, found[1] - y0) > .8 * radius:
                # The narrow pass from a good first fit can be short of
                # points (stroke crossings): keep the first fit then.
                if fit is None:
                    return None
                break
            fit = found
            cx, cy = fit[0], fit[1]
        if _thin_stroke_share(self.lightness, *fit) > WHITE_RING_STROKE:
            return None
        return fit[0], fit[1]

    def ring_white(self, x: float, y: float) -> float:
        """Share of pure white (colourless, bright) between 0.5 and 0.85
        radii from (x, y): a striped ball's white caps (0.28-0.46 on
        synthetic white tables) or the cue ball (~0.9); a solid ball's only
        white is its number badge inside 0.5 radii (at most 0.15)."""
        return _ring_white(self.lab, x, y, self.radius)


def _ring_white(lab: np.ndarray, x: float, y: float, r: float) -> float:
    """Share of pure white (colourless, Lab L 170 or more) between 0.5 and
    0.85 radii ``r`` from (x, y) on the Lab image ``lab``: a striped ball's
    white caps (0.28-0.46 on synthetic white tables) or the cue ball (~0.9);
    a solid ball's only white is its number badge inside 0.5 radii (at most
    0.15)."""
    h, w = lab.shape[:2]
    reach = int(np.ceil(r)) + 1
    x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 1)
    y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    ring = (d >= .5 * r) & (d <= .85 * r)
    if not ring.any():
        return 0.0
    px = lab[y0:y1, x0:x1][ring].astype(np.float32)
    chroma = np.hypot(px[:, 1] - 128, px[:, 2] - 128)
    return float(np.mean((chroma < 12) & (px[:, 0] >= 170)))


# The cue ball is white right out to its rim, but for its small red dot. A
# striped ball turned with its band towards the rim (seen end-on: a white
# face ringed by its colour) is as white inside; its rim is coloured (Lab
# chroma CUE_RIM_CHROMA or more) along STRIPE_RIM of it or more.
CUE_RIM_CHROMA = 25.0
STRIPE_RIM = .25
_RIM_COLOUR_ANGLES = np.linspace(0, 2 * np.pi, 48, endpoint=False)


def coloured_rim_share(lab: np.ndarray, x: float, y: float, r: float) -> float:
    """Share of the rim of the ball at (x, y) (0.8 and 0.88 radii ``r``, the
    more coloured of the two on each ray) that is clearly coloured, on the
    Lab image ``lab``: 0 on synthetic cue balls (real frames: at most 0.15),
    0.3 or more on 95% of synthetic striped balls turned end-on or mostly
    white."""
    radii = np.array([.8, .88])[:, None] * r
    xs = (x + np.cos(_RIM_COLOUR_ANGLES)[None] * radii).astype(np.float32)
    ys = (y + np.sin(_RIM_COLOUR_ANGLES)[None] * radii).astype(np.float32)
    px = cv2.remap(lab, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    chroma = np.hypot(px[..., 1] - 128, px[..., 2] - 128).max(axis=0)
    return float(np.mean(chroma >= CUE_RIM_CHROMA))


def white_cloth_cue_centre(roi: np.ndarray, cue_xy: tuple[float, float], balls,
                           offset: tuple[int, int], context: "_WhiteCloth | None" = None
                           ) -> tuple[float, float] | None:
    """Cue-ball centre on a white cloth, or None elsewhere (or when unsure).

    There the ball's rim is shaded darker than the cloth, and on the shadow
    side its drop shadow continues that darkness, so circle finding slides
    towards the shadow (3-8 px). The steepest brightness step is inside the
    ball (its lit middle falling off into the shaded rim), so fitting the
    step slid the same way. What marks the rim on the lit side is the dip:
    darker than the ball inside and the cloth outside. On each ray of the
    lit half the outermost clear dip is taken, and a circle is fitted to
    those points by consensus (a stroke crossing the rim, the cue ball's
    dots or the aim line inside it fall out), first from the rough centre
    over a wide band, then again from that fit. Synthetic white tables:
    0.22 px typical, against 1.1 px (7 px at the 90th percentile) before.

    A ghost ring lying on top of the cue ball (the struck ball right beside
    it) is a crisp circle too: a fit with a thin bright stroke along most of
    it is that ring, and None is returned.
    """
    context = context or _WhiteCloth.of(roi, balls, offset)
    if context is None or not context.white_here(*cue_xy):
        return None
    return context.rim_fit(*cue_xy)


def white_cloth_missed_balls(roi: np.ndarray, balls, offset: tuple[int, int], rejected) -> list[tuple[float, float]]:
    """Balls on a white cloth that circle finding took for cloth: centres
    (ROI px) of the ``rejected`` circles (x, y, r, cloth share inside; see
    ``find_balls``) that are balls after all.

    The cue ball on a white cloth is white on white: most of its face passes
    for cloth and it has no number badge, so it was dropped, and with it
    the aim. A circle counts when its lit-side rim dip fits a ball
    (``_WhiteCloth.rim_fit``), it casts a drop shadow (the cloth just
    outside its rim on the shadow side at least ``RESCUE_SHADOW`` darker
    than the cloth) and its lit face is brighter than the cloth, its inside
    is not all cloth, and no ball found lies within ``RESCUE_CLEAR`` radii
    (a circle on the cloth beside a ball fits that ball's rim and shadow:
    5 of ~1060 such circles passed the rest). On 250 synthetic white frames
    (150 shots of five kinds, 100 near shots) 9 circles were taken back,
    all real balls, 5 of them the cue ball.
    """
    context = _WhiteCloth.of(roi, balls, offset)
    if context is None or not rejected:
        return []
    radius = context.radius
    found = [(bx - offset[0], by - offset[1]) for bx, by, _, _ in balls]
    sx, sy = context.sdir
    shadow_rays = np.arctan2(sy, sx) + np.radians(np.linspace(-40, 40, 9))
    h, w = context.lightness.shape
    rescued: list[tuple[float, float]] = []
    for cx, cy, _, cloth_share in rejected:
        if cloth_share >= RESCUE_CLOTH or not context.white_here(cx, cy):
            continue
        if any(np.hypot(fx - cx, fy - cy) < radius * RESCUE_CLEAR for fx, fy in found + rescued):
            continue
        fit = context.rim_fit(cx, cy)
        if fit is None or any(np.hypot(fx - fit[0], fy - fit[1]) < radius * RESCUE_CLEAR
                              for fx, fy in found + rescued):
            continue
        fx, fy = fit
        cloth = float(context.cloth[int(cy), int(cx)])
        rim = cv2.remap(context.lightness, (fx + np.cos(shadow_rays) * radius * 1.15)[None].astype(np.float32),
                        (fy + np.sin(shadow_rays) * radius * 1.15)[None].astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)[0]
        lx, ly = int(fx - sx * radius * .35), int(fy - sy * radius * .35)
        if not (0 <= lx < w and 0 <= ly < h):
            continue
        if float(np.median(rim)) - cloth > -RESCUE_SHADOW or float(context.lightness[ly, lx]) - cloth < RESCUE_FACE:
            continue
        rescued.append((fx, fy))
    return rescued


# A rejected circle on a white cloth is a ball (``white_cloth_missed_balls``)
# with at most this share of cloth inside, a drop shadow at least
# RESCUE_SHADOW darker than the cloth, a lit face RESCUE_FACE brighter, and
# no other ball within RESCUE_CLEAR radii.
RESCUE_CLOTH = .9
RESCUE_SHADOW = 25.0
RESCUE_FACE = 5.0
RESCUE_CLEAR = 3.0


def pale_cloth_cue_centre(roi: np.ndarray, cue_xy: tuple[float, float], balls,
                          offset: tuple[int, int], ring=None) -> tuple[float, float] | None:
    """Cue-ball centre on a light, tinted cloth (beige, sky, yellow, lime) or a
    light grey one, or None elsewhere (or when unsure).

    There the ball's shaded rim is about as light as the cloth, and its drop
    shadow, a darker shade of the cloth, has lost some of its colour like
    the white ball: the usual fit slid towards the shadow (synthetic sky:
    3.5 px at the 90th percentile, beige 2.1). The white ball stays
    colourless right up to its rim, where the cloth's tint begins, so on
    each ray of its lit half the rim is where the colour difference from the
    cloth crosses halfway from the ball's to the cloth's (``_lit_rim_edge``);
    a circle is fitted to those points by consensus, from the rough centre
    and again from that fit, as on a white cloth. Sky 1.1 px, beige 1.3 at
    the 90th percentile, never 3.

    A light grey cloth has no tint, and the rim there is no darker than the
    cloth either (no dip as on a white cloth): the ball's shading falls to
    the cloth's level right at the rim and the profile goes flat. There the
    rim is the outermost place still ``KNEE_RISE`` above the cloth (synthetic
    light grey: 3-5 px towards the shadow in 5 of 150 cue balls before, none
    after). A white cloth keeps its dip fit (``white_cloth_cue_centre``).

    ``ring`` (x, y, radius; ROI px): a ghost ring on the ball, painted out of
    ``roi`` in blended colours, so rim points on its band are left out (they
    were 3-7 px off). On a tinted cloth None is returned then (near shots
    got worse)."""
    h, w = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    felt = _felt_lab(roi, hsv)
    tinted = np.hypot(felt[1] - 128, felt[2] - 128) >= WHITE_CLOTH_CHROMA
    if (felt[0] < LIGHT_CLOTH_L or (not tinted and felt[0] >= LIGHT_GREY_MAX_L) or not balls
            or (tinted and ring is not None)):
        return None
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    local = [(bx - offset[0], by - offset[1], br, kind) for bx, by, br, kind in balls]
    radius = float(np.median([b[2] for b in local]))
    sdir = shadow_direction(lightness, local_lightness(lightness), local, radius)
    if sdir is None:
        return None
    x0, y0 = cue_xy
    reach = int(radius * 3)
    wx0, wx1 = max(0, int(x0) - reach), min(w, int(x0) + reach + 1)
    wy0, wy1 = max(0, int(y0) - reach), min(h, int(y0) + reach + 1)
    patch = lab[wy0:wy1, wx0:wx1].astype(np.float32)
    if tinted:
        image = cv2.GaussianBlur(np.hypot(patch[:, :, 1] - felt[1], patch[:, :, 2] - felt[2]).astype(np.float32),
                                 (0, 0), .7)
    else:
        image = patch[:, :, 0].copy()
    others = [(bx - wx0, by - wy0) for bx, by, _, _ in local if np.hypot(bx - x0, by - y0) > radius * .6]
    cx, cy = x0 - wx0, y0 - wy0
    fit = None
    for inner, outer in ((.4, 1.7), (.6, 1.35)):
        points = _lit_rim_edge(image, cx, cy, radius, sdir, inner, outer, others,
                               knee=None if tinted else KNEE_RISE)
        if ring is not None and len(points):
            off_band = np.abs(np.hypot(points[:, 0] + wx0 - ring[0], points[:, 1] + wy0 - ring[1]) - ring[2])
            points = points[off_band > radius * .1 + 4.5]
        found = _consensus_circle(points, cx, cy, radius) if len(points) >= WHITE_RIM_MIN_POINTS else None
        if found is None or np.hypot(found[0] - (x0 - wx0), found[1] - (y0 - wy0)) > .8 * radius:
            break
        fit = found
        cx, cy = fit[0], fit[1]
    return None if fit is None else (fit[0] + wx0, fit[1] + wy0)


def _lit_rim_edge(image: np.ndarray, cx: float, cy: float, radius: float, sdir, inner: float, outer: float,
                  others=(), knee: float | None = None) -> np.ndarray:
    """Rim points on the lit side (rays more than ~100 degrees from the
    shadow) of a ball: on each ray from (cx, cy) between ``inner`` and
    ``outer`` radii, the outermost place where ``image`` falls from the
    ball's level (0.5-0.8 radii out) through a threshold to the cloth's (the
    ray's outer fifth), to a fraction of a pixel. The threshold is halfway
    between the two (``image`` the colour difference from a tinted cloth),
    or ``knee`` above the cloth's level (``image`` the lightness on a light
    grey cloth, where the ball's shading meets the cloth at its rim). Rays
    where something else runs on beyond it are left out, and so are rays
    towards ``others`` (balls) within 3.2 radii, as in ``_lit_rim_dips``."""
    shadow = np.arctan2(sdir[1], sdir[0])
    keep = np.cos(_WHITE_RIM_ANGLES - shadow) < -.2
    for ox, oy in others:
        dist = float(np.hypot(ox - cx, oy - cy))
        if radius * .6 < dist <= 3.2 * radius:
            half = np.arcsin(min(1.0, 1.2 * radius / dist)) + np.radians(5)
            keep &= np.cos(_WHITE_RIM_ANGLES - np.arctan2(oy - cy, ox - cx)) < np.cos(half)
    angles = _WHITE_RIM_ANGLES[keep]
    if not len(angles):
        return np.zeros((0, 2))
    step = .25
    steps = np.arange(inner * radius, outer * radius, step)
    xs = (cx + np.outer(np.cos(angles), steps)).astype(np.float32)
    ys = (cy + np.outer(np.sin(angles), steps)).astype(np.float32)
    profiles = cv2.remap(image, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ball_part = (steps >= .5 * radius) & (steps <= .8 * radius)
    points = []
    for angle, p in zip(angles, profiles):
        level = float(np.median(p[-max(4, len(p) // 5):]))
        inside = float(np.median(p[ball_part])) if ball_part.any() else float(p[:len(p) // 3].max())
        if inside - level < (TINT_RIM_STEP if knee is None else KNEE_BALL_RISE):
            continue
        cut = (inside + level) / 2 if knee is None else level + knee
        above = p >= cut
        crossings = np.nonzero(above[:-1] & ~above[1:])[0]
        if not len(crossings):
            continue
        j = int(crossings[-1])
        beyond = p[j + 4:]
        if len(beyond) and float(np.mean(beyond > cut if knee is None else np.abs(beyond - level) > 10)) > .15:
            continue
        t = steps[j] + (p[j] - cut) / max(float(p[j] - p[j + 1]), 1e-6) * step
        points.append((cx + t * np.cos(angle), cy + t * np.sin(angle)))
    return np.array(points) if points else np.zeros((0, 2))


# On a tinted cloth, a ball's colour difference from the cloth is at least
# this much above the cloth's own (Lab a/b) for its rim to be read by it.
TINT_RIM_STEP = 6.0
# On a light grey cloth (colourless, Lab L at most LIGHT_GREY_MAX_L; a white
# one has its dip fit) the rim is the outermost place KNEE_RISE above the
# cloth's level, on rays where the ball is KNEE_BALL_RISE above it inside.
LIGHT_GREY_MAX_L = 175.0
KNEE_RISE = 5.0
KNEE_BALL_RISE = 20.0


def white_cloth_stripe_centres(context: "_WhiteCloth", balls, offset: tuple[int, int], cue_xy,
                               confirmed: set | None = None) -> list:
    """``balls`` with every striped ball on a white cloth moved to its
    rim-fitted centre (added to ``confirmed``: a ball-sized rim was found
    around it, so it is a ball, not one of the stick's rings). Their white caps' shaded rims run into the drop
    shadow like the cue ball's, so circle finding put them 3-6 px towards
    the shadow (synthetic white tables, loose stripes: 2.9 px typical, 0.65
    after). Solid balls have a clear colour edge all round and keep theirs
    (the fit would move some of them by a pixel the wrong way)."""
    out = []
    for bx, by, br, kind in balls:
        lx, ly = bx - offset[0], by - offset[1]
        if (kind != "cue" and np.hypot(bx - cue_xy[0], by - cue_xy[1]) > br * .6
                and context.ring_white(lx, ly) >= STRIPE_RING_WHITE and context.white_here(lx, ly)):
            fit = context.rim_fit(lx, ly)
            if fit is not None and np.hypot(fit[0] - lx, fit[1] - ly) <= br * .5:
                bx, by = int(round(fit[0] + offset[0])), int(round(fit[1] + offset[1]))
                if confirmed is not None:
                    confirmed.add((bx, by))
        out.append((bx, by, br, kind))
    return out


def merge_duplicate_balls(balls, confirmed: set) -> list:
    """One detection per ball where it is clear which: an object-ball
    circle closer than 1.2 radii to one with a fitted rim (``confirmed``) is
    a second circle on that ball (real balls are at least ~1.75 radii apart,
    even packed). ``find_balls`` keeps two mostly white circles for the
    cue-ball check to sort out; on a white cloth a striped ball is mostly
    white too, and a second circle on it, 0.5-0.6 radii off, stayed as a
    false ball. Without a fitted rim on either, both stay (which one is the
    ball is not known: on a ball lying on the table's lettering the better
    rim score belonged to the wrong circle)."""
    if not confirmed:
        return balls
    radius = float(np.median([b[2] for b in balls])) if balls else 0.0
    out = []
    for bx, by, br, kind in balls:
        if (kind != "cue" and (bx, by) not in confirmed
                and any(np.hypot(cx - bx, cy - by) < 1.2 * radius for cx, cy in confirmed)):
            continue
        out.append((bx, by, br, kind))
    return out


def stripe_centres(roi: np.ndarray, balls, offset: tuple[int, int], cue_xy, stick=None,
                   confirmed: set | None = None, ghost=None) -> list:
    """``balls`` with every striped ball moved onto the circle through the
    two arcs where its coloured band meets the cloth.

    A striped ball's white caps are close to a light cloth (white, light
    grey, beige, sky), so circle finding slid along the band or towards the
    drop shadow (synthetic loose stripes: 0.11-0.20 radii off at the 90th
    percentile on those cloths, solid balls 0.05). The band is coloured
    right up to the rim at both its ends, and those two arcs lie opposite
    each other (also for a band seen at a slant on a 3D-drawn ball: two
    great circles cross at opposite points). On each ray from the rough
    centre whose inside is band-coloured, the rim is where the colour
    difference from the cloth falls halfway to the cloth's
    (``_band_rim_points``); a circle of the balls' radius (within 10%) is
    fitted to those points by consensus, from the rough centre and again
    from that fit, when its points lie on both sides of the ball (both
    arcs: one alone, beside a ball in a rack, moved a ball half a radius).
    Only on a pale cloth (Lab L at least ``LIGHT_CLOTH_L``, chroma below
    ``PALE_CLOTH_CHROMA``: white, light grey, beige, sky) is the ball moved.
    On a darker one circle finding places stripes well already (0.05 radii
    at the 90th percentile) and the fit, its halfway level pulled by the
    drop shadow's colour, a little worse (0.06-0.08); on a light coloured
    one (yellow, lime) it gained little, and on a real light blue table it
    moved Miniclip's 3D-drawn stripes (curved bands, outlined) by 2-3 px
    with nothing to show it was right.
    ``stick`` ((x, y) of the cue ball, unit direction of the aim): a ball
    the cue stick lies across behind the cue ball keeps its centre (the
    stick's colour hides the band). A fitted ball is added to ``confirmed``:
    a ball-sized circle through its band's two arcs makes it a ball, not
    one of the stick's rings (moved onto the stick's axis, a stripe under
    the stick was dropped as one in near shots, where the stick is read
    later). ``ghost`` ((x, y), frame px): the ghost ring when known; rays
    towards it are left out like rays towards balls. The struck ball
    touches it, one diameter from its centre: the game's object line is
    drawn from that ball's centre across its band (fitted, it was 3-6 px
    off in 4 of 2850 synthetic frames), so its fit is used only when it
    puts the ball a diameter from the ghost (within 0.1 radii) and the
    circle found did not (0.25 radii or more off); the ghost and the
    object line place it later anyway."""
    if not balls:
        return balls
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    felt = _felt_lab(roi)
    move = felt[0] >= LIGHT_CLOTH_L and np.hypot(felt[1] - 128, felt[2] - 128) < PALE_CLOTH_CHROMA
    colour = cv2.GaussianBlur(np.hypot(lab[:, :, 1] - felt[1], lab[:, :, 2] - felt[2]), (0, 0), .7)
    light = cv2.GaussianBlur(lab[:, :, 0], (0, 0), .7)
    radius = float(np.median([b[2] for b in balls]))
    local = [(bx - offset[0], by - offset[1]) for bx, by, _, _ in balls]
    blocking = local + ([(ghost[0] - offset[0], ghost[1] - offset[1])] if ghost is not None else [])
    out = []
    for (bx, by, br, kind), (lx, ly) in zip(balls, local):
        struck = ghost is not None and np.hypot(bx - ghost[0], by - ghost[1]) < radius * BAND_STRUCK
        # A ball the stick lies across keeps its centre.
        keep = False
        if stick is not None and not struck:
            (sx, sy), (ux, uy) = stick
            behind = -((bx - sx) * ux + (by - sy) * uy)
            keep = 0 < behind < radius * 40 and abs((bx - sx) * uy - (by - sy) * ux) <= radius * 2
        if (kind == "cue" or keep or np.hypot(bx - cue_xy[0], by - cue_xy[1]) <= br * .6
                or _ring_white(lab, lx, ly, radius) < STRIPE_RING_WHITE):
            out.append((bx, by, br, kind))
            continue
        others = [(ox, oy) for ox, oy in (local if struck else blocking)
                  if np.hypot(ox - lx, oy - ly) > radius * .6]
        cx, cy, fit = lx, ly, None
        for inner, outer in ((.3, 1.7), (.55, 1.35)):
            points = _band_rim_points(colour, light, cx, cy, radius, inner, outer, others, rough=fit is None)
            found = (_consensus_circle(points, cx, cy, radius, radii=BAND_RADII, min_points=BAND_MIN_POINTS,
                                       min_arc=BAND_MIN_ARC) if len(points) >= BAND_MIN_POINTS else None)
            if found is None or np.hypot(found[0] - lx, found[1] - ly) > radius * .6:
                break
            fit = found
            cx, cy = fit[0], fit[1]
        if fit is not None and struck:
            touch = 2 * fit[2]
            gap_before = abs(np.hypot(bx - ghost[0], by - ghost[1]) - touch)
            gap_after = abs(np.hypot(fit[0] + offset[0] - ghost[0], fit[1] + offset[1] - ghost[1]) - touch)
            if gap_before < fit[2] * .25 or gap_after > fit[2] * .1:
                fit = None
        if fit is not None:
            if move:
                bx, by = int(round(fit[0] + offset[0])), int(round(fit[1] + offset[1]))
            if confirmed is not None:
                confirmed.add((bx, by))
        out.append((bx, by, br, kind))
    return out


_BAND_ANGLES = np.radians(np.arange(0, 360, 3.0))


def _band_rim_points(colour: np.ndarray, light: np.ndarray, cx: float, cy: float, radius: float,
                     inner: float, outer: float, others=(), rough: bool = False) -> np.ndarray:
    """Rim points of a ball's coloured parts: on each ray from (cx, cy)
    between ``inner`` and ``outer`` radii whose inside (0.5-0.8 radii) is at
    least ``BAND_STEP`` above the cloth's level in ``colour`` (the colour
    difference from the cloth), the outermost place where it falls through
    halfway to that level, to a fraction of a pixel. Left out: rays where
    colour runs on beyond it; rays that only cross the band's long edge into
    a white cap (the cap, ``BAND_CAP`` lighter than the cloth in ``light``,
    lies just beyond: its rim is shaded darker); rays towards ``others``
    (balls within 3.2 radii). ``rough``: (cx, cy) may be half a radius
    off, so a ray's inside is its band colour anywhere from 0.2 radii out
    to the rim (the 80th percentile there): 0.5-0.8 radii from a centre
    that far off can be the white number badge, and the far end of the
    band was never found."""
    keep = np.ones(len(_BAND_ANGLES), bool)
    for ox, oy in others:
        dist = float(np.hypot(ox - cx, oy - cy))
        if dist < radius * 3.2:
            half = np.arcsin(min(1.0, radius * 1.15 / max(dist, 1e-6))) + np.radians(8)
            keep &= np.abs((_BAND_ANGLES - np.arctan2(oy - cy, ox - cx) + np.pi) % (2 * np.pi) - np.pi) > half
    angles = _BAND_ANGLES[keep]
    if not len(angles):
        return np.zeros((0, 2))
    step = .25
    steps = np.arange(inner * radius, outer * radius, step)
    xs = (cx + np.outer(np.cos(angles), steps)).astype(np.float32)
    ys = (cy + np.outer(np.sin(angles), steps)).astype(np.float32)
    profiles = cv2.remap(colour, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lights = cv2.remap(light, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ball_part = ((steps >= .2 * radius) & (steps <= 1.0 * radius) if rough
                 else (steps >= .5 * radius) & (steps <= .8 * radius))
    tail = max(4, len(steps) // 5)
    ahead = int(radius * .3 / step)
    points = []
    for angle, p, lp in zip(angles, profiles, lights):
        level = float(np.median(p[-tail:]))
        inside = float(np.percentile(p[ball_part], 80) if rough else np.median(p[ball_part]))
        if inside - level < BAND_STEP:
            continue
        cut = (inside + level) / 2
        above = p >= cut
        crossings = np.nonzero(above[:-1] & ~above[1:])[0]
        if not len(crossings):
            continue
        j = int(crossings[-1])
        beyond = p[j + 4:]
        if len(beyond) and float(np.mean(beyond > cut)) > .15:
            continue
        near = lp[j + 2:j + 2 + ahead]
        if len(near) and float(near.max()) > float(np.median(lp[-tail:])) + BAND_CAP:
            continue
        t = steps[j] + (p[j] - cut) / max(float(p[j] - p[j + 1]), 1e-6) * step
        points.append((cx + t * np.cos(angle), cy + t * np.sin(angle)))
    return np.array(points) if points else np.zeros((0, 2))


# A ray's inside is a striped ball's band when its colour difference from
# the cloth is at least BAND_STEP above the cloth's own; a white cap lies
# beyond a band edge when the lightness there is BAND_CAP above the cloth's.
BAND_STEP = 15.0
BAND_CAP = 15.0
# The band fit: radius within BAND_RADII of the balls', at least
# BAND_MIN_POINTS rim points agreeing over BAND_MIN_ARC degrees.
BAND_RADII = (.9, 1.1)
BAND_MIN_POINTS = 8
BAND_MIN_ARC = 150.0
# A ball closer than this many radii to the ghost is the struck one (it
# touches the ghost: 2 radii).
BAND_STRUCK = 2.6


def _carry_ball_keys(old, new, rim_open: dict, cue_candidates: list, confirmed: set) -> list:
    """Balls moved (``old`` to ``new``, in the same order): what the stick
    filter looks up by a ball's position (its rim share, its colour among
    ``cue_candidates``, a fitted rim in ``confirmed``) goes along, or a
    stripe under the stick's line was dropped as one of its rings.
    Returns the updated ``cue_candidates``."""
    for (ox, oy, _, _), (nx, ny, _, _) in zip(old, new):
        if (ox, oy) == (nx, ny):
            continue
        if (ox, oy) in rim_open:
            rim_open[(nx, ny)] = rim_open[(ox, oy)]
        if (ox, oy) in confirmed:
            confirmed.add((nx, ny))
        cue_candidates = [(nx, ny, *c[2:]) if (c[0], c[1]) == (ox, oy) else c for c in cue_candidates]
    return cue_candidates


def drop_second_circles(balls, distance: np.ndarray | None, offset: tuple[int, int]) -> list:
    """One circle per object ball: of two closer than ``SECOND_CIRCLE`` radii
    (real balls are at least ~1.75 radii apart, even packed) the one whose
    rim lies better on the ball's edge stays (``_rim_scores`` on the
    cloth-distance image ``distance``, ROI px from ``offset``).

    ``find_balls`` keeps two mostly white circles for the cue-ball check to
    sort out, and on a light cloth a striped ball's white caps make it
    mostly white: a second circle on it, 0.4-0.8 radii off, stayed as a
    false ball (45 frames of 4750 synthetic) or was picked as the struck
    ball and the game's object line was looked for from the wrong centre.
    The better rim score was the circle nearer the true centre in 44 of
    those 45 pairs. The cue ball is not touched (its second circles are
    gone by now)."""
    others = [b for b in balls if b[3] != "cue"]
    if distance is None or len(others) < 2:
        return balls
    radius = float(np.median([b[2] for b in balls]))
    centres = np.array([(b[0] - offset[0], b[1] - offset[1]) for b in others], np.float64)
    scores = _rim_scores(distance, centres, radius)
    kept: list = []
    for i in np.argsort(-scores, kind="stable"):
        bx, by = others[i][:2]
        if all(np.hypot(bx - k[0], by - k[1]) >= radius * SECOND_CIRCLE for k in kept):
            kept.append(others[i])
    keep = {id(b) for b in kept}
    return [b for b in balls if b[3] == "cue" or id(b) in keep]


# Two object-ball circles closer than this many radii are on one ball.
SECOND_CIRCLE = .9


# A ball with at least this share of pure white between 0.5 and 0.85 radii
# is striped (or the cue ball); see ``_WhiteCloth.ring_white``.
STRIPE_RING_WHITE = .2


def _lit_rim_dips(lightness: np.ndarray, cx: float, cy: float, radius: float, sdir, inner: float,
                  outer: float, others=()) -> np.ndarray:
    """Rim points on the lit side (rays more than ~100 deg from the shadow):
    on each ray from (cx, cy) between ``inner`` and ``outer`` radii, the
    outermost dip at least ``WHITE_RIM_DIP`` below the cloth's level (the
    ray's outer quarter), with that level coming back right after it. (Not
    the deepest dip: the cue ball's dots and the aim line inside the ball
    are darker than its rim.) Rays where
    a stroke or the stick runs on beyond the dip are left out, and so are
    rays towards ``others`` (balls) within 3.2 radii: their near rim is a
    dip too (a rack: a second circle at 1.3-1.5 radii). A ball lying on the
    table's watermark lettering can show the letters' edges as a second
    circle; the consensus fit then finds no circle and the caller keeps
    its own centre."""
    shadow = np.arctan2(sdir[1], sdir[0])
    keep = np.cos(_WHITE_RIM_ANGLES - shadow) < -.2
    for ox, oy in others:
        dist = float(np.hypot(ox - cx, oy - cy))
        # (Closer than 0.6 radii is this ball itself, detected twice.)
        if radius * .6 < dist <= 3.2 * radius:
            half = np.arcsin(min(1.0, 1.2 * radius / dist)) + np.radians(5)
            keep &= np.cos(_WHITE_RIM_ANGLES - np.arctan2(oy - cy, ox - cx)) < np.cos(half)
    angles = _WHITE_RIM_ANGLES[keep]
    if not len(angles):
        # Every lit direction leads into a neighbour (a boxed-in ball).
        return np.zeros((0, 2))
    steps = np.arange(inner * radius, outer * radius, .25)
    xs = (cx + np.outer(np.cos(angles), steps)).astype(np.float32)
    ys = (cy + np.outer(np.sin(angles), steps)).astype(np.float32)
    profiles = cv2.remap(lightness, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    back = int(round(.35 * radius / .25))
    points = []
    for angle, p in zip(angles, profiles):
        level = float(np.median(p[-max(4, len(p) // 4):]))
        minima = np.nonzero((p[1:-1] <= p[:-2]) & (p[1:-1] <= p[2:]))[0] + 1
        if not len(minima):
            continue
        depth = level - p[minima]
        chosen = None
        for j, dj in zip(minima[::-1], depth[::-1]):
            if dj < WHITE_RIM_DIP or p[:j].max() - p[j] < 6:
                continue
            after = p[j + 1:j + 1 + back]
            if len(after) == 0 or after.max() < level - 6:
                continue
            chosen = int(j)
            break
        if chosen is None:
            continue
        j = chosen
        beyond = p[j + back:]
        if len(beyond) and ((beyond < level - 15).mean() > .3 or (beyond > level + 25).any()):
            continue
        a, b, c = p[j - 1], p[j], p[j + 1]
        den = a - 2 * b + c
        shift = float(np.clip(.5 * (a - c) / den, -.5, .5)) if den > 1e-6 else 0.0
        t = steps[j] + shift * .25
        points.append((cx + t * np.cos(angle), cy + t * np.sin(angle)))
    return np.array(points) if points else np.zeros((0, 2))


def _circle_through(points: np.ndarray) -> tuple[float, float, float]:
    a = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    sol, *_ = np.linalg.lstsq(a, points[:, 0] ** 2 + points[:, 1] ** 2, rcond=None)
    fx, fy = sol[0] / 2, sol[1] / 2
    return float(fx), float(fy), float(np.sqrt(max(1e-9, sol[2] + fx ** 2 + fy ** 2)))


def _consensus_circle(points: np.ndarray, cx: float, cy: float, radius: float, radii=(.8, 1.05),
                      min_points: int | None = None, min_arc: float = 90.0):
    """The circle most rim points agree on (within 0.75 px): (x, y, r), or
    None when fewer than ``min_points`` (default ``WHITE_RIM_MIN_POINTS``)
    agree or they cover less than ``min_arc`` degrees of the rim. Circles
    through three points at a time (a fixed random sequence, so the result
    is repeatable), radius ``radii`` (0.8-1.05) of the ball's."""
    low, high = radii[0] * radius, radii[1] * radius
    min_points = WHITE_RIM_MIN_POINTS if min_points is None else min_points
    rng = np.random.default_rng(0)
    n = len(points)
    best = best_circle = None
    for _ in range(200):
        fx, fy, fr = _circle_through(points[rng.choice(n, 3, replace=False)])
        if not (low <= fr <= high) or np.hypot(fx - cx, fy - cy) > .8 * radius:
            continue
        inliers = np.abs(np.hypot(points[:, 0] - fx, points[:, 1] - fy) - fr) < .75
        if best is None or inliers.sum() > best.sum():
            best, best_circle = inliers, (fx, fy, fr)
    if best is None or best.sum() < min_points or best.mean() < .5:
        return None
    chosen = points[best]
    angles = np.sort(np.arctan2(chosen[:, 1] - cy, chosen[:, 0] - cx))
    gaps = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
    if 2 * np.pi - gaps.max() < np.radians(min_arc):
        return None
    fx, fy, fr = _circle_through(chosen)
    # A short arc can make the free fit wander; then keep the consensus circle.
    if not (low <= fr <= high) or np.hypot(fx - best_circle[0], fy - best_circle[1]) > 1.5:
        fx, fy, fr = best_circle
    return fx, fy, fr


def _thin_stroke_share(lightness: np.ndarray, fx: float, fy: float, fr: float) -> float:
    """Share of angles with a thin bright stroke within 2.5 px of the circle
    (a peak at least 15 brighter than pixels within 3 px on both sides): a
    ghost ring's white line between its dark outlines. A ball's rim has its
    bright middle further in and no such peak (at most 0.47 on synthetic
    white tables; a ring 0.97 or more)."""
    return float(_stroke_shares(lightness, (fx, fy, fr))[0])


_WHITE_RIM_ANGLES = np.linspace(0, 2 * np.pi, 96, endpoint=False)
# Cloth lightness (Lab L, 8-bit) from which the cue ball is fitted on its
# lit side, as the table's brightest cloth (white ~225-231; light grey ~182).
WHITE_CLOTH_L = 205.0
# ...and its colour (Lab chroma) below this: a colourless cloth.
WHITE_CLOTH_CHROMA = 8.0
# A rim dip is at least this much darker than the cloth on its ray.
WHITE_RIM_DIP = 8.0
# A white-cloth rim fit needs at least this many agreeing rim points.
WHITE_RIM_MIN_POINTS = 12
# Above this share of thin bright strokes along it, a fitted circle is a
# ghost ring, not the cue ball.
WHITE_RING_STROKE = .7


def grid_spot(points, index: int, radius: float, need: int = 2) -> tuple[float, float] | None:
    """Where touching neighbours on a hexagonal grid put ball ``index``: the
    mean of the spots each touching pair predicts within half a radius of
    it, when at least ``need`` pairs do."""
    px, py = points[index]
    pts = np.array(points, np.float32)
    near = [j for j in range(len(points)) if j != index
            and 1.7 * radius <= float(np.hypot(pts[j, 0] - px, pts[j, 1] - py)) <= 2.4 * radius]
    guesses = []
    for a in near:
        for b in range(len(points)):
            if b in (a, index):
                continue
            d = float(np.hypot(pts[b, 0] - pts[a, 0], pts[b, 1] - pts[a, 1]))
            if not (1.85 * radius <= d <= 2.3 * radius):
                continue
            base = np.arctan2(pts[b, 1] - pts[a, 1], pts[b, 0] - pts[a, 0])
            for k in (1, 5):
                gx = pts[a, 0] + d * np.cos(base + k * np.pi / 3)
                gy = pts[a, 1] + d * np.sin(base + k * np.pi / 3)
                if np.hypot(gx - px, gy - py) <= radius * .5:
                    guesses.append((gx, gy))
    if len(guesses) < need:
        return None
    return float(np.mean([g[0] for g in guesses])), float(np.mean([g[1] for g in guesses]))


def rack_gaps(points, radius: float, width: float, height: float, pockets) -> list[tuple[float, float, float]]:
    """Grid spots next to touching balls with no ball found there: each one
    a diameter from two or more balls (in a rack every ball has neighbours
    all round), inside the cushions and clear of the pockets."""
    if len(points) < 3:
        return []
    pts = np.array(points, np.float32)
    spots = []
    for i, (ax, ay) in enumerate(pts):
        for j, (bx, by) in enumerate(pts):
            if i == j:
                continue
            d = float(np.hypot(bx - ax, by - ay))
            if not (1.85 * radius <= d <= 2.3 * radius):
                continue
            base = np.arctan2(by - ay, bx - ax)
            for k in range(1, 6):
                angle = base + k * np.pi / 3
                sx, sy = ax + d * np.cos(angle), ay + d * np.sin(angle)
                if min(sx, sy, width - sx, height - sy) < radius * .9:
                    continue
                if min(np.hypot(sx - px, sy - py) for px, py in pockets) < radius * 1.6:
                    continue
                gaps = np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)
                if gaps.min() < radius * 1.6 or int(np.sum(gaps <= radius * 2.35)) < 2:
                    continue
                spots.append((float(sx), float(sy)))
    # Every touching pair beside a spot predicts it: average them.
    merged: list[list[tuple[float, float]]] = []
    for sx, sy in spots:
        for group in merged:
            if np.hypot(sx - group[0][0], sy - group[0][1]) <= radius * .6:
                group.append((sx, sy))
                break
        else:
            merged.append([(sx, sy)])
    return [(float(np.mean([g[0] for g in group])), float(np.mean([g[1] for g in group])), float(radius))
            for group in merged][:40]


# All pool balls are the same size. A fitted radius further than this share
# from the typical one has grown onto the ball's drop shadow (or shrunk
# against a neighbour); its centre is fitted again at the typical radius.
BALL_SIZE_SPREAD = .06


def has_number_badge(patch_hsv: np.ndarray, xx, yy, cx: float, cy: float, r: float,
                     min_v: float = 150, max_s: float = 90) -> bool:
    """True when a compact white disc (a ball's number badge, ~0.45 r) sits
    near the centre: white fills the middle around its own centroid, with
    almost no white just outside it. The game's guide lines crossing at the
    ghost ball's centre are white there too, but they run on outwards.
    ``min_v`` is raised on a light cloth and ``max_s`` lowered on a pale
    coloured one, which are themselves "white" by the default rule (a badge
    is still brighter, and less saturated, than they are).
    """
    white = (patch_hsv[:, :, 1] < max_s) & (patch_hsv[:, :, 2] > min_v)
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


# Where the balls' highlight sits (radii right of and below the centre) when
# a frame has too few balls to measure it (``highlight_offset``): real frames
# show it at (-0.3..-0.4, -0.4..-0.6), the synthetic table at (-0.35, -0.38).
HIGHLIGHT = (-.38, -.45)
HIGHLIGHT_BALLS = 4
HIGHLIGHT_PEAK = 1.0
_HL_GRID = np.linspace(-.9, .9, 19)
_HL_GX, _HL_GY = np.meshgrid(_HL_GRID, _HL_GRID)
_HL_DISC = np.hypot(_HL_GX, _HL_GY) <= .85
_HL_SEARCH = np.hypot(_HL_GX, _HL_GY) <= .75
_HL_RING = np.linspace(0, 2 * np.pi, 48, endpoint=False)


def highlight_offset(smooth: np.ndarray, centres, radius: float) -> tuple[float, float]:
    """Where the balls' highlight sits, in radii from their centre: the peak
    of their average brightness pattern on ``smooth`` (lightness, blurred a
    little), each ball's own level and spread taken out so its colour does
    not count; number badges and stripes, turned every way, average out.
    HIGHLIGHT with fewer than HIGHLIGHT_BALLS balls or no clear peak (in the
    balls' own spread). Real frames: (-0.3..-0.4, -0.4..-0.6), peak 1.5-2.1."""
    if len(centres) < HIGHLIGHT_BALLS:
        return HIGHLIGHT
    total = np.zeros_like(_HL_GX)
    for x, y in centres:
        p = cv2.remap(smooth, (x + _HL_GX * radius).astype(np.float32), (y + _HL_GY * radius).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        v = p[_HL_DISC]
        total += (p - v.mean()) / max(float(v.std()), 1.0)
    score = np.where(_HL_SEARCH, total / len(centres), -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[iy, ix] < HIGHLIGHT_PEAK or np.hypot(_HL_GX[iy, ix], _HL_GY[iy, ix]) < .15:
        return HIGHLIGHT
    return float(_HL_GX[iy, ix]), float(_HL_GY[iy, ix])


# A circle that looks like cloth inside is a ball the cloth's colour
# (``BallLook.cloth_coloured_ball``) when it lies CLOTH_COLOURED_GAP radii or
# more from every ball found, its highlight is HIGHLIGHT_RISE (Lab L)
# brighter than the rest of the circle through it, HIGHLIGHT_OVER_CLOTH
# brighter than the cloth around and HIGHLIGHT_OVER_LEVEL brighter than the
# cloth's own level there, its inside differs from that cloth by
# CLOTH_COLOURED_DL in lightness or CLOTH_COLOURED_DAB in colour, and it is
# not the ghost ring: no circle near its rim with a thin bright stroke along
# CLOTH_COLOURED_RING of it whose core is guide white along
# CLOTH_COLOURED_RING_WHITE of it.
CLOTH_COLOURED_GAP = 1.75
HIGHLIGHT_RISE = 12.0
HIGHLIGHT_OVER_CLOTH = 15.0
HIGHLIGHT_OVER_LEVEL = 8.0
# ...and its highlight is not on a line: going out from it HIGHLIGHT_RUN
# radii in any direction, no more than HIGHLIGHT_LINE of its rise is left
# all the way (a spot fades: real balls 0.32 typical at 0.35 radii, 0.69 at
# the 90th percentile; along a line ~1), unless HIGHLIGHT_ACROSS or more is
# left HIGHLIGHT_SIDE radii to one side, at the spot or 0.3 radii along (a
# stripe's white cap is wide; the guide's strokes and the stick are not).
HIGHLIGHT_RUN = (.15, .3, .45)
HIGHLIGHT_SIDE = .25
HIGHLIGHT_LINE = .75
HIGHLIGHT_ACROSS = .5
_HL_DIRS = np.radians(np.arange(0, 360, 11.25))
CLOTH_COLOURED_RING = .8
CLOTH_COLOURED_RING_WHITE = .7
CLOTH_COLOURED_DL = 6.0
CLOTH_COLOURED_DAB = 4.0


class BallLook:
    """What the test for a ball the colour of the cloth needs, measured once
    per picture (``roi``, BGR): Lab, lightness (blurred a little too), HSV,
    where the highlight sits (``highlight_offset`` on the ``confident``
    balls' centres, or ``offset``) and the cloth's own lightness around each
    spot with the balls found (``found``) painted out (``local_lightness``:
    the table's lettering does not lower it; packed balls' white would
    raise it)."""

    def __init__(self, roi: np.ndarray, found, confident, radius: float, offset=None, hsv=None):
        self.lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        self.lightness = self.lab[:, :, 0].astype(np.float32)
        self.smooth = cv2.GaussianBlur(self.lightness, (0, 0), max(1.0, radius * .1))
        self.hsv = hsv if hsv is not None else cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.radius = radius
        self.offset = offset if offset is not None else highlight_offset(self.smooth, confident, radius)
        masked = self.lightness.copy()
        fill = float(np.median(self.lightness[::4, ::4]))
        for x, y in found:
            cv2.circle(masked, (int(round(x)), int(round(y))), int(round(radius * 1.25)), fill, -1)
        self.cloth = local_lightness(masked)

    def cloth_coloured_ball(self, cx: float, cy: float) -> bool:
        """True when the circle at (cx, cy) shows a ball's highlight and
        its own shade (see the constants above; the gap to other balls is
        checked by the caller).

        A ball the colour of the cloth (purple 4 on a purple table, red 3 on
        red, the 8 on black) looks like cloth inside, and with its number
        turned away (most of the time: the balls roll) it shows no badge
        either. What it still has is the glossy highlight every ball shows
        at the same spot (the light is the same for all), paler and brighter
        than the cloth, and its own shade. Rings, lettering and shadows have
        cloth there."""
        lab, radius = self.lab, self.radius
        h, w = lab.shape[:2]
        reach = int(np.ceil(radius * 1.85)) + 1
        x0, x1 = max(0, int(cx) - reach), min(w, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(h, int(cy) + reach + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return False
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        px = lab[y0:y1, x0:x1].astype(np.float32)
        ring = px[(d >= radius * 1.35) & (d <= radius * 1.8)]
        inner = px[d <= radius * .6]
        ox, oy = self.offset
        at_spot = np.hypot(xx - cx - ox * radius, yy - cy - oy * radius) <= radius * .15
        spot = px[at_spot]
        if len(ring) < 12 or len(inner) < 6 or len(spot) == 0:
            return False
        cloth = np.median(ring, axis=0)
        # Brighter than the cloth around, and than the cloth's own level
        # there (bare cloth between the dark strokes of the table's
        # lettering is brighter than a ring over the letters).
        spot_l = float(np.median(spot[:, 0]))
        level = float(np.median(self.cloth[y0:y1, x0:x1][at_spot]))
        if spot_l - float(cloth[0]) < HIGHLIGHT_OVER_CLOTH or spot_l - level < HIGHLIGHT_OVER_LEVEL:
            return False
        body = np.median(inner, axis=0)
        if (abs(float(body[0] - cloth[0])) < CLOTH_COLOURED_DL
                and float(np.hypot(body[1] - cloth[1], body[2] - cloth[2])) < CLOTH_COLOURED_DAB):
            return False
        rho = float(np.hypot(ox, oy))
        ang = _HL_RING[np.cos(_HL_RING - np.arctan2(oy, ox)) < np.cos(np.radians(45))]
        sx, sy = cx + ox * radius, cy + oy * radius
        ux, uy = np.cos(_HL_DIRS), np.sin(_HL_DIRS)
        run = np.asarray(HIGHLIGHT_RUN)[:, None] * radius
        side = HIGHLIGHT_SIDE * radius
        mid = HIGHLIGHT_RUN[1] * radius
        xs = np.concatenate(([sx], cx + np.cos(ang) * rho * radius, (sx + ux * run).ravel(),
                             sx - uy * side, sx + uy * side, sx + ux * mid - uy * side, sx + ux * mid + uy * side))
        ys = np.concatenate(([sy], cy + np.sin(ang) * rho * radius, (sy + uy * run).ravel(),
                             sy + ux * side, sy - ux * side, sy + uy * mid + ux * side, sy + uy * mid - ux * side))
        xs, ys = xs.astype(np.float32)[None], ys.astype(np.float32)[None]
        v = cv2.remap(self.smooth, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        base = float(np.median(v[1:1 + len(ang)]))
        rise = float(v[0]) - base
        if rise < HIGHLIGHT_RISE:
            return False
        # A highlight is a spot, fading every way. A white line (the game's
        # guide) or the cue stick through that point, or ending there, stays
        # bright along itself and is dark on both sides across it, at the
        # spot and further along (a stripe's white cap there is wide).
        # Read on the unblurred lightness (blurred, a thin line sinks to the
        # cloth's level) and from the cloth's own level where the circle's
        # is lower (lettering under it: the bare cloth beside a line is no
        # wider bright stroke).
        v = cv2.remap(self.lightness, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        ref = max(float(np.median(v[1:1 + len(ang)])), level)
        rest = (v[1 + len(ang):] - ref) / max(float(v[0]) - ref, 1.0)
        k = len(_HL_DIRS)
        along = rest[:len(HIGHLIGHT_RUN) * k].reshape(len(HIGHLIGHT_RUN), k).min(axis=0)
        across = rest[len(HIGHLIGHT_RUN) * k:].reshape(4, k)
        i = int(np.argmax(along))
        if along[i] > HIGHLIGHT_LINE and float(across[:, i].max()) < HIGHLIGHT_ACROSS:
            return False
        # Not the ghost ring (a circle found on it can sit a few px off its
        # centre; its stroke is guide white, a real ball's outline is tinted).
        return not any(_ring_core_white(self.hsv, *circle) >= CLOTH_COLOURED_RING_WHITE
                       for circle in _rings_nearby(self.lightness, cx, cy, radius))


_RING_NEAR_SHIFTS = np.arange(-.25, .26, .125)
_RING_NEAR_SIZES = (.8, .85, .9, .95, 1.0, 1.05)


def _rings_nearby(lightness: np.ndarray, cx: float, cy: float, radius: float, most: int = 8):
    """Circles within 0.25 radii of (cx, cy), 0.8-1.05 radii across, with a
    thin bright stroke (``_thin_stroke_share``) along CLOTH_COLOURED_RING or
    more of them, the best ``most`` first. Real frames: the game's ghost ring
    0.84-1.0 (cut by the cushion: 0.84); a ball 0.97 typical too (Miniclip
    outlines its balls), but tinted: guide white along 0.08 of it (0.49 at
    the 90th percentile), the ghost ring 0.76-0.97."""
    circles = [(cx + dx * radius, cy + dy * radius, radius * f)
               for dx in _RING_NEAR_SHIFTS for dy in _RING_NEAR_SHIFTS for f in _RING_NEAR_SIZES]
    shares = _stroke_shares(lightness, circles)
    order = [i for i in np.argsort(-shares)[:most] if shares[i] >= CLOTH_COLOURED_RING]
    return [circles[i] for i in order]


def find_balls(roi: np.ndarray, gray: np.ndarray, min_r: int, max_r: int, shared: dict | None = None,
               sites: list[tuple[float, float, float]] | None = None, cloth_coloured: bool = True):
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
    roi_bright = None
    if sites is not None and shared is not None and shared.get("distance") is not None:
        distance = shared["distance"]
    else:
        distance = felt_distance_image(roi, roi_hsv, as_float=True)
    if shared is not None:
        # detect() needs the same image again for the cue-stick filter.
        shared["distance"] = distance
    # Colour-distance edges of a blue ball on blue cloth are weaker than grey
    # edges of other balls, hence the lower Canny threshold for that image.
    sources = ((cv2.medianBlur(np.clip(distance * 3, 0, 255).astype(np.uint8), 5), 60), (gray, 100))
    smooth = cv2.GaussianBlur(distance, (0, 0), 1.0)
    proposals: list[tuple[int, int, int]] = []
    from_badge: set[tuple[int, int]] = set()
    if sites is not None:
        # Second pass: spots where a packed ball must be (see the end), each
        # snapped onto a rim nearby and then judged like any proposal.
        for sx, sy, sr in sites:
            fx, fy = refit_centre(smooth, float(sx), float(sy), float(sr), reach=float(sr) * .2)
            proposals.append((int(round(fx)), int(round(fy)), int(round(sr))))
        typical = float(np.median([site[2] for site in sites])) if sites else (min_r + max_r) / 2
    else:
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
    # Circles that look like cloth inside and show no number badge.
    cloth_like = []
    refit: set[tuple[int, int]] = set()
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
                    # Lettering borders only cloth. The same shade right next
                    # to something brighter than the cloth is the blurred
                    # edge of a ball's white badge or stripe.
                    if roi_bright is None:
                        roi_bright = cv2.dilate(((roi_lab[:, :, 0] > roi_light + 12) * 255).astype(np.uint8),
                                                np.ones((5, 5), np.uint8)) > 0
                    near_bright = roi_bright[y0:y1, x0:x1][dist2 <= (r * .62) ** 2]
                    cloth_inside |= (dab < 10) & (np.abs(dl - letter_shade) < 10) & ~near_bright
                # A white ball on near-white cloth is only a little brighter
                # than it on its lit side, and its shaded side looks like a
                # darker shade of the cloth. Lettering has nothing brighter.
                if float(np.mean((dl > 8) & (dab < 12))) >= .3:
                    cloth_inside = (np.abs(dl) < 4) & (dab < 10)
            elif np.hypot(felt_lab[1] - 128, felt_lab[2] - 128) < 8 and felt_lab[0] >= GREY_CLOTH_L:
                # A darker colourless cloth (grey): its watermark lettering
                # is the cloth in one darker shade as well, and the bowls of
                # its letters read as balls (synthetic grey: 66 of 150
                # frames had one).
                if roi_light is None:
                    roi_light = local_lightness(roi_lab[:, :, 0])
                if letter_shade is False:
                    letter_shade = lettering_shade(roi_lab, roi_light, felt_lab)
                if letter_shade is not None:
                    if len(near_cloth) >= 12:
                        # The ring round a letter's bowl lies on its stroke:
                        # the cloth is the lighter part of it, as on a light
                        # cloth, and the inside is judged against that.
                        local = local.copy()
                        local[0] = max(local[0], min(float(np.percentile(near_cloth[:, 0], 80)),
                                                     float(roi_light[cy, cx])))
                        cloth_inside |= np.sqrt((((inner_lab - local) * lab_weights) ** 2).sum(axis=1)) < 10
                    dl = inner_lab[:, 0] - local[0]
                    dab = np.hypot(inner_lab[:, 1] - local[1], inner_lab[:, 2] - local[2])
                    if roi_bright is None:
                        roi_bright = cv2.dilate(((roi_lab[:, :, 0] > roi_light + 12) * 255).astype(np.uint8),
                                                np.ones((5, 5), np.uint8)) > 0
                    near_bright = roi_bright[y0:y1, x0:x1][dist2 <= (r * .62) ** 2]
                    cloth_inside |= (dab < 10) & (np.abs(dl - letter_shade) < 10) & ~near_bright
        cloth_fraction = float(np.mean(cloth_inside))
        centre = patch[dist2 <= (r * .4) ** 2]
        # A badge-only proposal stands for a ball that looks like cloth; a
        # light-centred disc that is not cloth-coloured (a metal joint on the
        # cue stick) would have been found by circle finding if it were a ball.
        if (cx, cy) in from_badge:
            # Judge the ball's own colour around the badge (white excluded).
            body = ~((inner[:, 1] < 90) & (inner[:, 2] > 150))
            if not body.any():
                continue
            # A coloured body is a ball circle finding missed (packed in a
            # rack, crossed by a guide line); a grey one is the stick's joint.
            coloured = float(np.median(inner[body][:, 1])) >= 90 and not grey
            if float(np.mean(cloth_inside[body])) <= .25:
                if not coloured:
                    continue
                # Another hue than the cloth, so it has a rim to fit: the
                # badge is printed off the ball's centre.
                body_hue = float(np.median(hue_delta(inner[body][:, 0], local_hue)))
                if abs(body_hue) > 15:
                    refit.add((cx, cy))
        badge = False
        if cloth_fraction > .35:
            # A solid ball the same colour as the cloth (purple 4 on a purple
            # table, red 3 on red) looks like cloth inside, but its white
            # number badge fills the centre when it faces up. Hollow rings,
            # lettering and shadows have cloth there.
            badge_v = 150.0
            # A pale cloth (any colour) passes the default badge-white test
            # itself; the badge must then be brighter than the cloth.
            if len(ring) and float(np.median(ring[:, 1])) < 90:
                badge_v = max(badge_v, float(np.percentile(ring[:, 2], 80)) + 10)
            badge_s = 90.0
            if len(ring):
                ring_s = float(np.median(ring[:, 1]))
                if ring_s >= 25:
                    badge_s = min(badge_s, ring_s * .6)
            badge = has_number_badge(patch, xx, yy, cx, cy, r, badge_v, badge_s)
        sat = float(np.median(inner[:, 1]))
        val = float(np.median(inner[:, 2]))
        white_fraction = float(np.mean((inner[:, 1] < 90) & (inner[:, 2] > 150)))
        red_spot = float(np.mean((centre[:, 0] < 30) & (centre[:, 1] > 70) & (centre[:, 2] < 90)))
        if cloth_fraction > .35 and not badge:
            # Judged again below, once the balls show where their
            # highlight sits (a ball the cloth's colour, number turned away).
            cloth_like.append((cloth_fraction, cx, cy, r, sat, val, white_fraction, red_spot))
            continue
        accepted.append((cloth_fraction, cx, cy, r, sat, val, white_fraction, red_spot))
    if cloth_like and accepted and cloth_coloured:
        confident = [item for item in accepted if item[0] <= .35 and (item[1], item[2]) not in from_badge]
        radius = float(np.median([item[3] for item in confident])) if confident else typical
        centres = [(item[1], item[2]) for item in accepted]
        look = None
        for item in sorted(cloth_like):
            cloth_fraction, cx, cy = item[:3]
            if all((cx - ax) ** 2 + (cy - ay) ** 2 >= (radius * CLOTH_COLOURED_GAP) ** 2 for ax, ay in centres):
                if look is None:
                    # (Only now: most frames have no such circle left.)
                    offset = shared.get("highlight") if sites is not None and shared is not None else None
                    look = BallLook(roi, centres, [(c[1], c[2]) for c in confident], radius, offset, roi_hsv)
                    if shared is not None and sites is None:
                        shared["highlight"] = look.offset
                if look.cloth_coloured_ball(float(cx), float(cy)):
                    accepted.append(item)
                    centres.append((cx, cy))
                    continue
            if shared is not None and sites is None:
                # (On a white cloth detect() looks at these again: there
                # the white cue ball itself can look like cloth.)
                shared.setdefault("cloth_like", []).append((cx, cy, item[3], cloth_fraction))
    elif cloth_like and shared is not None and sites is None:
        shared.setdefault("cloth_like", []).extend((item[1], item[2], item[3], item[0]) for item in cloth_like)
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
        # (A badge-only proposal is centred on its white badge, so it looks
        # mostly white too; it is a duplicate of the ball found there.)
        if all(item[6] >= .5 and k[6] >= .5 and (item[1], item[2]) not in from_badge for k in clash):
            kept.append(item)
    # Same size for every ball: a shadow must not make one circle bigger or
    # pull its centre towards the shadow.
    typical = float(np.median([item[3] for item in kept]))
    result = []
    badge_found: list[int] = []
    for _, cx, cy, r, *rest in kept:
        if (cx, cy) in from_badge:
            badge_found.append(len(result))
        # Smaller circles keep their size: the cue-stick filter tells stick
        # rings from balls by it. Every ball is drawn at one size in the end
        # (``uniform_ball_size``).
        # Only a white ball (the cue ball, whose centre sets the aim) has an
        # edge strong enough to beat its shadow in this fit; on a ball close
        # to the cloth colour the shadow would win and pull the centre.
        if (cx, cy) in refit:
            fx, fy = refit_centre(smooth, float(cx), float(cy), typical, reach=typical * .6)
            if np.hypot(fx - cx, fy - cy) <= typical * .6:
                # Landing on another detection: it was that ball's badge.
                if any((fx - k[1]) ** 2 + (fy - k[2]) ** 2 < (diameter * .6) ** 2
                       for k in kept if (k[1], k[2]) != (cx, cy)):
                    continue
                cx, cy = int(round(fx)), int(round(fy))
        white_ball = rest[2] >= .5  # (sat, val, white_fraction, red_spot)
        if white_ball and r > typical * (1 + BALL_SIZE_SPREAD) and rim_open_fraction(smooth, cx, cy, typical) >= .65:
            fx, fy = refit_centre(smooth, float(cx), float(cy), typical)
            cx, cy = int(round(fx)), int(round(fy))
        result.append((cx, cy, r, *rest, rim_open_fraction(smooth, cx, cy, r)))
    if sites is None and len(result) >= 3:
        # Packed balls (a rack) sit on a hexagonal grid one diameter apart.
        # A spot with two or more touching neighbours and no ball found is
        # proposed again; the same checks decide.
        spots = rack_gaps([(b[0], b[1]) for b in result], typical, tw, th, pockets_local)
        if spots:
            extra = find_balls(roi, gray, min_r, max_r,
                               {"distance": distance, "highlight": shared.get("highlight") if shared else None},
                               sites=spots)
            for item in extra:
                if all(np.hypot(item[0] - b[0], item[1] - b[1]) >= typical * 1.5 for b in result):
                    result.append(item)
    if sites is None and len(result) >= 3:
        # A ball found only by its badge sits where the badge is printed, and
        # a packed one shows too little rim to fit; among touching balls the
        # grid they form places it better.
        points = [(b[0], b[1]) for b in result]
        weak = set(badge_found) | {k for k, b in enumerate(result) if b[-1] < .5}
        for i in range(len(result)):
            # A well-fitted ball moves only when the grid is firm about it.
            spot = grid_spot(points, i, typical, need=2 if i in weak else 4)
            if spot is not None:
                result[i] = (int(round(spot[0])), int(round(spot[1])), *result[i][2:])
                points[i] = result[i][:2]
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
        # A piece of the frame is enough to start from: a stick lying across
        # the slot can cut it into two (at full power on a break, 24% and
        # 55% of the table's height). The whole box is checked below.
        if not (.3 * height <= h <= 1.05 * height and .025 * width <= w <= .09 * width and h >= 3 * w):
            continue
        # A framed slot, not a thin edge line (the real frame fills ~17% of
        # its box, a rail edge ~5%).
        if area < .1 * w * h:
            continue
        # The cue stick can lie across the slot and cut its frame in two:
        # carry on over a short gap in the same columns.
        # Only a piece that is the frame again counts (both side edges for a
        # stretch); a stick passing just past the slot's end is not.
        rows = mask[:, x:x + w].sum(axis=1) >= max(2, w * .2)
        edge = max(2, int(round(w * .2)))
        sides = (mask[:, x:x + edge].any(axis=1) & mask[:, x + w - edge:x + w].any(axis=1))
        run = max(3, int(round(height * .03)))
        gap = int(round(height * .08))

        def framed(a: int, b: int) -> bool:
            longest = current = 0
            for s in sides[a:b]:
                current = current + 1 if s else 0
                longest = max(longest, current)
            return longest >= run

        top_row, bottom_row = int(y), int(y + h - 1)
        while True:
            above = np.nonzero(rows[max(0, top_row - gap):top_row])[0]
            if above.size == 0:
                break
            reach = max(0, top_row - gap) + int(above[0])
            if bottom_row - reach > 1.05 * height or not framed(reach, reach + run * 4):
                break
            top_row = reach
        while True:
            below = np.nonzero(rows[bottom_row + 1:bottom_row + 1 + gap])[0]
            if below.size == 0:
                break
            reach = bottom_row + 1 + int(below[-1])
            if reach - top_row > 1.05 * height or not framed(max(bottom_row + 1, reach + 1 - run * 4), reach + 1):
                break
            bottom_row = reach
        full = bottom_row - top_row + 1
        if not (.55 * height <= full and full >= 5 * w):
            continue
        if abs((top_row + bottom_row) / 2 - (top + height / 2)) > height * .2:
            continue
        candidates.append((int(x), top_row, int(w), full))
    # Closest to the table first (other tall icons sit further left); it must
    # hold the cue, or be an empty dark slot (cue dragged all the way down).
    for rect in sorted(candidates, key=lambda r: -r[0]):
        if _slider_shaft_top(frame, rect) is not None or _slider_empty(frame, rect):
            return rect
    return None


def _slider_empty(frame: np.ndarray, slider: tuple[int, int, int, int]) -> bool:
    """True when the slot's middle is dark from top to bottom (no cue showing)."""
    x, y, w, h = slider
    centre = _slider_centre(frame, slider)
    half = max(2, int(round(w * .06)))
    strip = frame[int(y + h * .06):int(y + h * .94), int(centre - half):int(centre + half) + 1]
    if strip.size == 0:
        return False
    value = np.median(cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)[:, :, 2], axis=1)
    # Rows where a stick lies across the slot cover its grey frame too: those
    # rows do not count (at full power on a break the stick crosses it).
    rows = slice(int(y + h * .06), int(y + h * .94))
    edge = max(2, int(round(w * .12)))
    sides = cv2.cvtColor(np.hstack([frame[rows, x:x + edge], frame[rows, x + w - edge:x + w]]), cv2.COLOR_BGR2HSV)
    grey = (sides[:, :, 1] < 80) & (sides[:, :, 2] >= 90) & (sides[:, :, 2] <= 190)
    framed = grey.mean(axis=1) >= .5
    n = min(len(value), len(framed))
    value, framed = value[:n], framed[:n]
    if framed.sum() < n * .7:
        return False
    return bool(np.mean(value[framed] < 110) > .95)


def _slider_centre(frame: np.ndarray, slider: tuple[int, int, int, int]) -> float:
    """Column of the cue's shaft in the slot: the box's middle, unless the
    shaft's colour shows clearly elsewhere (the stick's joint touching the
    frame can widen the box)."""
    x, y, w, h = slider
    box = frame[y:y + h, x:x + w]
    if box.size == 0:
        return x + w / 2
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    shaft = ((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 1] <= 190)
             & (hsv[:, :, 2] >= 170))
    columns = shaft.mean(axis=0)
    if columns.max() < .05:
        return x + w / 2
    strong = np.nonzero(columns >= columns.max() * .6)[0]
    return x + float(strong.mean()) + .5


def _slider_shaft_top(frame: np.ndarray, slider: tuple[int, int, int, int]) -> float | None:
    """Row (fraction of the slot height from its top) where the cue's shaft starts."""
    x, y, w, h = slider
    centre = _slider_centre(frame, slider)
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
    # The cue stick can lie across the slider (aiming from near the left
    # cushion): a longer gap is skipped when the shaft clearly goes on above.
    jump = int(round(h * .07))
    resume = max(3, int(round(h * .03)))
    top_row = bottom
    misses = 0
    r = bottom
    while r >= 0:
        if filled[r]:
            top_row, misses = r, 0
        else:
            misses += 1
            if misses > gap_ok:
                above = filled[max(0, r - jump):r]
                runs = np.nonzero(above)[0]
                if runs.size == 0:
                    break
                # The lowest row above the gap that starts a long enough run.
                start = None
                for k in runs[::-1]:
                    k0 = max(0, r - jump) + int(k)
                    if k0 - resume + 1 >= 0 and filled[k0 - resume + 1:k0 + 1].all():
                        start = k0
                        break
                if start is None:
                    break
                r, misses = start, 0
                continue
        r -= 1
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
    # Balls a ball-sized rim was fitted around (white cloth: striped balls).
    confirmed_balls: set = set()
    white_cloth = None
    shared: dict = {}
    for cx, cy, r, sat, val, white_fraction, red_spot, open_rim in find_balls(roi, gray, min_r, max_r, shared):
        ball = (x + cx, y + cy, r)
        rim_open[(x + cx, y + cy)] = open_rim
        # Keep every detection eligible as an object ball until the actual
        # cue is identified from its shaft (white stripes and number badges
        # are common on object balls too).
        cue_candidates.append((*ball, sat, val, white_fraction, red_spot))
        balls.append((*ball, "ball"))
    if shared.get("cloth_like") and balls:
        # On a white cloth the cue ball can look like cloth to circle
        # finding: look at what it rejected again.
        typical = int(round(float(np.median([b[2] for b in balls]))))
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        smooth = cv2.GaussianBlur(shared["distance"], (0, 0), 1.0)
        for fx, fy in white_cloth_missed_balls(roi, balls, (x, y), shared["cloth_like"]):
            cx, cy = int(round(fx)), int(round(fy))
            yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            inner = roi_hsv[d2 <= (typical * .62) ** 2].astype(np.int16)
            centre = roi_hsv[d2 <= (typical * .4) ** 2].astype(np.int16)
            ball = (x + cx, y + cy, typical)
            rim_open[(x + cx, y + cy)] = rim_open_fraction(smooth, cx, cy, typical)
            cue_candidates.append((*ball, float(np.median(inner[:, 1])), float(np.median(inner[:, 2])),
                                   float(np.mean((inner[:, 1] < 90) & (inner[:, 2] > 150))),
                                   float(np.mean((centre[:, 0] < 30) & (centre[:, 1] > 70) & (centre[:, 2] < 90)))))
            balls.append((*ball, "ball"))
    cue = None
    # The ghost ring when it was looked for anywhere on the table (frame px).
    ghost_ring = None
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
        # A striped ball turned end-on is as white inside, but its rim is
        # coloured (``coloured_rim_share``): it only counts when nothing else
        # is left (synthetic white cloth with turned balls, where the cue
        # ball's own ray is often not confirmed: cue ball wrong in 39 of 150
        # frames before).
        lab_roi = None
        striped = set()
        typical_r = float(np.median([b[2] for b in balls]))
        for c in cue_candidates:
            if c[5] >= .52:
                if lab_roi is None:
                    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                if coloured_rim_share(lab_roi, c[0] - x, c[1] - y, typical_r) >= STRIPE_RIM:
                    striped.add(id(c))
        plain = [c for c in cue_candidates if id(c) not in striped]
        white_candidates = ([c for c in plain if c[5] >= .68]
                            or [c for c in cue_candidates if c[5] >= .68])
        shaft_candidates = (white_candidates or [c for c in plain if c[5] >= .52 and c[3] < 80]
                            or [c for c in cue_candidates if c[5] >= .52 and c[3] < 80])
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
        far_ghost = {}   # candidates whose ray ends in a ghost a few radii out: that ghost
        roi_hsv = None
        if ray_candidates:
            local_all = [(bx - x, by - y, br, kind) for bx, by, br, kind in balls]
            radius = float(np.median([b[2] for b in balls]))
            ghost_mask = white_guide_mask(roi, local_all, ball_scale=.9)
            guide = white_guide_mask(roi, local_all, ridges=True)
            verified = []
            for candidate in ray_candidates:
                local = (candidate[0] - x, candidate[1] - y, candidate[2])
                directions = aim_candidates(roi, local, local_all, guide=guide)
                if not directions:
                    continue
                ghost = None
                for direction in directions:
                    ghost = find_ghost_ball(ghost_mask, local, direction, radius, roi)
                    if ghost is not None:
                        break
                verified.append((ghost is not None, id(candidate) not in striped, candidate is cue, candidate[5],
                                 candidate))
                if ghost is not None and np.hypot(ghost[0] - local[0], ghost[1] - local[1]) >= radius * RAY_GHOST_REACH:
                    far_ghost[id(candidate)] = ghost
            if verified:
                best = max(verified, key=lambda item: item[:4])
                if best[0] or cue is None:
                    cue = best[4]
            # The ray and ghost of a struck ball right next to the cue ball
            # are hidden under the ring, while a striped ball's own object
            # line can pass for a ray ending in a "ghost". The cue ball is
            # the purest white ball: a much whiter, less saturated candidate
            # wins over a striped one.
            # (On a colourless light cloth a striped ball's white half is as
            # unsaturated as the cue ball, 4 against 4 on synthetic seed
            # 40003: there the whiteness alone decides.)
            whitest = max(plain or cue_candidates, key=lambda c: c[5])
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            colourless = _light_cloth(roi_hsv) and _cloth_saturation(roi_hsv) < 40
            if (cue is not None and whitest is not cue and whitest[5] >= .75
                    and whitest[5] >= cue[5] + .15 and (colourless or whitest[3] <= cue[3] - 10)):
                cue = whitest
        radius = float(np.median([b[2] for b in balls]))
        seen = far_ghost.get(id(cue)) if cue is not None else None
        # A ghost at the end of a ray counts once it passes the strict ring
        # test (a few ms around it): the game's object line from a struck
        # ball can end at a striped ball, and the ring overlapping the cue
        # ball can read as a ring further on (synthetic near shots 40053,
        # 40004 on blue).
        ray_to_ghost = seen is not None and find_ghost_ring(roi, radius, hsv=roi_hsv,
                                                            around=(seen[0], seen[1], radius * .5)) is not None
        if not ray_to_ghost:
            # No ray was read out to a ghost: the struck ball may lie right
            # next to the cue ball, its ring on top of the cue ball and the
            # ray hidden under both. Only the cue ball can overlap the ghost
            # ring: it marks where the cue ball will be at contact, and the
            # struck ball only touches it. So the ball under the ring found
            # anywhere on the table is the cue ball (synthetic near shots on
            # blue: wrong cue ball in 9 of 100 before).
            ring = find_ghost_ring(roi, radius, hsv=roi_hsv)
            if ring is not None:
                ghost_ring = (ring[0] + x, ring[1] + y, ring[2])
                lines = guide if ray_candidates else white_guide_mask(
                    roi, [(bx - x, by - y, br, kind) for bx, by, br, kind in balls], ridges=True)
                under = ball_under_ring(cue_candidates, ghost_ring[:2], radius, lines, (x, y))
                if under is None or np.hypot(under[0] - ghost_ring[0], under[1] - ghost_ring[1]) < radius * RING_ITSELF:
                    # No ball found under the ring, or only the ring itself:
                    # look for the ball again with the ring painted out. Only
                    # a ball circle finding missed counts (else it is a ball
                    # touching the ring, such as the struck one, measured a
                    # little close; the cue ball is then elsewhere).
                    found = ball_beside_ring(roi, ring, radius, min_r, max_r)
                    if found is not None and any(np.hypot(c[0] - x - found[0], c[1] - y - found[1]) < radius * .6
                                                 for c in cue_candidates if c is not under):
                        found = None
                    if found is not None:
                        bx, by = found[0] + x, found[1] + y
                        under = (bx, by, *found[2:7])
                        rim_open[(bx, by)] = found[7]
                        cue_candidates.append(under)
                        balls.append((bx, by, found[2], "ball"))
                if under is not None:
                    cue = under
        if cue:
            cue_index = min(range(len(balls)), key=lambda i: np.hypot(balls[i][0]-cue[0], balls[i][1]-cue[1]))
            # With the ghost ring on the cue ball, its rim is fitted on the
            # picture with the ring painted out (white cloth, synthetic near
            # shots: 0.3 px typical against 0.8, 1.2 at the 90th percentile
            # against 5.6).
            ring_on_ball = (ghost_ring is not None
                            and np.hypot(cue[0] - ghost_ring[0], cue[1] - ghost_ring[1]) <= radius * RING_OVERLAP)
            fit_roi = paint_out_ring(roi, (ghost_ring[0] - x, ghost_ring[1] - y, ghost_ring[2]), radius) if ring_on_ball else roi
            white_cloth = _WhiteCloth.of(fit_roi, balls, (x, y))
            moved = (white_cloth_cue_centre(fit_roi, (cue[0] - x, cue[1] - y), balls, (x, y), white_cloth)
                     if white_cloth is not None else None)
            if moved is None and white_cloth is None:
                # A light tinted or light grey cloth: the rim read on the lit
                # side (the usual fit slides towards the shadow). With the
                # ghost ring on the ball only on light grey (synthetic near
                # shots 68 -> 76 of 100; on tinted cloths, even with rim
                # points on the ring left out, 7 of 400 got worse).
                ring = (ghost_ring[0] - x, ghost_ring[1] - y, ghost_ring[2]) if ring_on_ball else None
                moved = pale_cloth_cue_centre(fit_roi, (cue[0] - x, cue[1] - y), balls, (x, y), ring=ring)
            if moved is None and ring_on_ball and white_cloth is None and not _light_cloth(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)):
                # A coloured cloth: the usual rim fit, without the ring
                # (synthetic near shots on blue: 0.75 px at the 90th
                # percentile against 1.4).
                clean = cv2.GaussianBlur(felt_distance_image(fit_roi, cv2.cvtColor(fit_roi, cv2.COLOR_BGR2HSV), as_float=True),
                                         (0, 0), 1.0)
                fx, fy, _ = refine_circle(clean, cue[0] - x, cue[1] - y, radius)
                if np.hypot(fx - (cue[0] - x), fy - (cue[1] - y)) <= radius * .5:   # (the ring pulls a fit 6-7 px)
                    moved = (fx, fy)
            if moved is not None:
                # (The candidate tuple carries the colour measurements on.)
                cue = (int(round(moved[0] + x)), int(round(moved[1] + y)), *cue[2:])
            if white_cloth is not None:
                moved_balls = white_cloth_stripe_centres(white_cloth, balls, (x, y), cue[:2], confirmed_balls)
                cue_candidates = _carry_ball_keys(balls, moved_balls, rim_open, cue_candidates, confirmed_balls)
                balls = moved_balls
            cue_position = cue[:3]
            bx, by, br, _ = balls[cue_index]
            balls[cue_index] = (cue[0], cue[1], br, "cue")
            balls = separate_cue_overlap_detections(balls, cue_position)
            if white_cloth is not None:
                balls = merge_duplicate_balls(balls, confirmed_balls)
            balls = drop_second_circles(balls, shared.get("distance"), (x, y))
            local_balls = [(bx-x, by-y, br, kind) for bx, by, br, kind in balls]
            aim = aim_direction(roi, (cue_position[0]-x, cue_position[1]-y, cue_position[2]), local_balls)
            stick_aim = aim
            if not stick_aim and stick_matches and stick_matches_cue(stick_matches, cue):
                # No aim ray read, but the stick still lies behind the cue
                # ball: filter its rings along the stick itself.
                stick_aim = stick_direction(stick_lines, (cue_position[0] - x, cue_position[1] - y),
                                            cue_position[2], cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
            if stick_aim:
                aim_for_stick = stick_aim
                # Without a ray read out to a ghost the aim is rough (a ghost
                # on top of the cue ball, or none): the stick's own line, where
                # one is found, gives its direction better (``stick_axis``),
                # and then its rings can be told by lying on its axis. A ray
                # to a ghost gives it best (0.1 degrees against ~0.3, with
                # lines now and then 3-10 degrees off: real balls beside the
                # stick were dropped).
                exact = None if ray_to_ghost else stick_axis(stick_lines, (cue_position[0] - x, cue_position[1] - y),
                                                             float(np.median([b[2] for b in balls])), stick_aim)
                if exact is not None:
                    aim_for_stick = exact
                # Read from the stick alone the side can be wrong; a struck
                # ball right in front must not be dropped then.
                min_behind = 1.35 if aim else 3.5
                distance = shared.get("distance")
                if distance is None:
                    distance = felt_distance_image(roi, cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), as_float=True)
                body = {(c[0], c[1]): (c[3], c[4], c[5]) for c in cue_candidates}
                balls = separate_cue_stick_detections(balls, cue_position, aim_for_stick, rim_open,
                                                      distance=distance, offset=(x, y), body=body,
                                                      min_behind=min_behind, axis_rule=bool(aim) or exact is not None,
                                                      confirmed=confirmed_balls,
                                                      white_cloth=white_cloth is not None)
            # Exactly one ball is the cue ball; other white/striped detections
            # remain valid object-ball targets under 8-ball rules.
            cue_index = min(range(len(balls)), key=lambda i: np.hypot(balls[i][0]-cue[0], balls[i][1]-cue[1]))
            bx, by, br, _ = balls[cue_index]
            balls[cue_index] = (bx, by, br, "cue")
            # Striped balls on the two arcs where their band meets the cloth
            # (after the stick filter: a ball under the stick, moved onto
            # its axis, read as one of its rings).
            ghost_xy = (ghost_ring[:2] if ghost_ring is not None
                        else (seen[0] + x, seen[1] + y) if ray_to_ghost else None)
            moved_balls = stripe_centres(roi, balls, (x, y), cue[:2],
                                         stick=(cue_position[:2], aim_for_stick) if stick_aim else None,
                                         confirmed=confirmed_balls, ghost=ghost_xy)
            cue_candidates = _carry_ball_keys(balls, moved_balls, rim_open, cue_candidates, confirmed_balls)
            balls = moved_balls
    if len(balls) > 24:
        return Detection(table=table, balls=[], power_slider=slider, power_percent=read_power_slider(frame, slider),
                         note=f"Table found · {len(balls)} circle candidates · too many to track confidently")
    if cue:
        shot = predict_shot(roi, table, balls, cue[:3])

        def stick_pass(balls, shot, aim_before):
            """Filter the stick's rings along the aim just read when that
            differs from the aim used before, or, with the ghost on top of
            the cue ball (its aim degrees off), along the stick's own line
            where one is found (see ``stick_axis``); predict again if any
            went."""
            aim_now = shot.get("aim_dir")
            ghost_now = shot.get("ghost")
            on_cue = (ghost_now is not None and balls
                      and np.hypot(ghost_now[0] - cue[0], ghost_now[1] - cue[1]) <= cue[2] * RING_OVERLAP)
            axis = (stick_axis(stick_lines, (cue[0] - x, cue[1] - y), float(np.median([b[2] for b in balls])), aim_now)
                    if on_cue else None)
            turn = None if aim_now is None or aim_before is None else aim_now[0] * aim_before[0] + aim_now[1] * aim_before[1]
            fresh = axis is not None or aim_before is None or (turn is not None and turn < .985)
            if aim_now is not None and (fresh or turn < STICK_REFILTER_COS):
                # (Only a small turn from the first rough reading: only the
                # rings centred on the axis go.)
                filtered = separate_cue_stick_detections(
                    balls, cue[:3], axis or aim_now, rim_open,
                    distance=shared.get("distance") if shared.get("distance") is not None
                    else felt_distance_image(roi, cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), as_float=True),
                    offset=(x, y), body={(c[0], c[1]): (c[3], c[4], c[5]) for c in cue_candidates},
                    confirmed=confirmed_balls, white_cloth=white_cloth is not None, rings_only=not fresh)
                if len(filtered) != len(balls):
                    return filtered, predict_shot(roi, table, filtered, cue[:3])
            return balls, shot

        # The aim used for the cue-stick filter above can be wrong (the ghost
        # ring on top of the cue ball hid the ray).
        balls, shot = stick_pass(balls, shot, aim)
        def drop_ring_ball(balls, shot):
            """Nothing can occupy the spot where the cue ball will be at
            impact: a "ball" centred on the ghost ring is the ring itself
            (the ray and deflection line inside it can make it look filled),
            and one centred within GHOST_OVERLAP ghost radii of it is a
            circle on its stroke and lines (the struck ball only touches it,
            two radii out; a gap in a rack can be proposed right there).
            Drop it and predict again so it is not masked or used as an
            obstacle."""
            if not shot["ghost"]:
                return balls, shot
            gx, gy, gr = shot["ghost"]
            cleaned = [b for b in balls if b[3] == "cue" or np.hypot(b[0] - gx, b[1] - gy) > gr * GHOST_OVERLAP]
            if len(cleaned) == len(balls):
                return balls, shot
            return cleaned, predict_shot(roi, table, cleaned, cue[:3])

        balls, shot = drop_ring_ball(balls, shot)
        ring_on_cue = None
        if balls:
            radius = float(np.median([b[2] for b in balls]))
            if ghost_ring is not None and np.hypot(cue[0] - ghost_ring[0], cue[1] - ghost_ring[1]) <= radius * RING_OVERLAP:
                # The ring found on the table lies on the cue ball. Only a
                # "cue ball" fitted on the ring itself is fitted again: one
                # beside it is the ball (light cloths, synthetic near shots:
                # 7-10 px off on the ring, 0.1-2 after; beside it the fit
                # moved good ones by 3-4 px, on a white cloth 5-19).
                if np.hypot(cue[0] - ghost_ring[0], cue[1] - ghost_ring[1]) <= radius * RING_ITSELF:
                    ring_on_cue = ghost_ring
            elif shot["ghost"] and np.hypot(cue[0] - shot["ghost"][0], cue[1] - shot["ghost"][1]) < radius * 1.5:
                ring_on_cue = shot["ghost"]
            elif not shot["ghost"]:
                # No ghost found: the "cue ball" may be the ring itself.
                white = white_guide_mask(roi, []).astype(np.float32) / 255
                lx, ly = cue[0] - x, cue[1] - y
                k = _ring_kernel(radius)
                y0_, x0_ = max(0, ly - k.shape[0]), max(0, lx - k.shape[1])
                local = white[y0_:ly + k.shape[0] + 1, x0_:lx + k.shape[1] + 1]
                score = cv2.filter2D(local, -1, k, borderType=cv2.BORDER_CONSTANT)
                sy, sx = ly - y0_, lx - x0_
                near = score[max(0, sy - 2):sy + 3, max(0, sx - 2):sx + 3]
                if near.size and float(near.max()) >= RING_ON_CUE_SCORE:
                    ring_on_cue = (cue[0], cue[1], radius)
                elif _is_ghost_ring(roi, lx, ly, radius):
                    # That score needs cloth inside the ring; with the white
                    # cue ball under it there is none (synthetic near shots:
                    # -0.21). A thin bright line with dark outlines along the
                    # whole circle is the ring all the same.
                    ring_on_cue = (cue[0], cue[1], radius)
        if ring_on_cue is not None:
            # The struck ball right next to the cue ball puts the ghost ring
            # on top of the cue ball, and circle finding can lock onto the
            # ring (a crisp white circle) instead of the ball under it. Fit
            # the ball again with the ring's pixels left out.
            gx, gy, gr = ring_on_cue
            distance = shared.get("distance")
            if distance is None:
                distance = felt_distance_image(roi, cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), as_float=True)
            fx, fy = refit_beside_ring(cv2.GaussianBlur(distance, (0, 0), 1.0), cue[0] - x, cue[1] - y,
                                       radius, gx - x, gy - y, gr)
            moved = (int(round(fx + x)), int(round(fy + y)))
            if moved != tuple(cue[:2]) and np.hypot(moved[0] - cue[0], moved[1] - cue[1]) <= radius * 1.2:
                balls = [(moved[0], moved[1], b[2], b[3]) if b[3] == "cue" else b for b in balls]
                cue = (moved[0], moved[1], *cue[2:])
                shot = predict_shot(roi, table, balls, cue[:3])
                # The aim is read from the real cue ball now (it was read
                # from the ring before): its stick's rings go along it.
                balls, shot = stick_pass(balls, shot, None)
                # (Only now is the ghost found when the "cue ball" was the
                # ring: a second circle on the ring stayed as a ball.)
                balls, shot = drop_ring_ball(balls, shot)
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
