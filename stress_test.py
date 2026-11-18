"""Measure the detector on many synthetic frames with known geometry.

    python stress_test.py [--count 200] [--start 0] [--cloth white] [--harsh] [--flat] [--save-failures DIR]

Prints error statistics and every failed check per seed. A frame passes when
the table, cue ball, balls, ghost ball, aim angle, target and predicted
directions all match the truth within the tolerances below. The balls are
turned at random as the game's balls are (``fake_game.draw_ball``);
``--flat`` draws them the old way (number up, straight band) to compare
with results from before.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

import fake_game as fg
import main

# Tolerances in pixels / degrees.
TABLE_PX = 3.0
CUE_PX = 3.0
GHOST_PX = 2.5
AIM_DEG = .6
AIM_PX = 1.0       # ...unless the offset at the ghost ball stays under this
OBJECT_DEG = 2.0
CUE_PATH_DEG = 4.0
BOUNCE_PX = 4.0    # bounce points vs an exact trace from the detector's own start and direction


def angle_of(vec):
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def seg_angle(segment):
    (x1, y1), (x2, y2) = segment
    return angle_of((x2 - x1, y2 - y1))


def _to_polyline(point, points):
    """Shortest distance from a point to a polyline."""
    best = float("inf")
    px, py = point
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 < 1e-9 else min(max(((px - ax) * dx + (py - ay) * dy) / length2, 0.0), 1.0)
        best = min(best, float(np.hypot(px - ax - t * dx, py - ay - t * dy)))
    return best


def bounce_check(segments, direction, truth, balls, true_start=None, true_dir=None):
    """Compare a drawn multi-bounce path with fake_game's exact bounces.

    Returns (mechanics_px, true_deg, n_bounces): how far the drawn bounce
    points lie off an exact trace from the detector's own start and direction
    (does the tracer bounce off the right cushions), and the worst sideways
    drift from the true path as degrees of travel (the angle error that the
    extra bounces add up to). Distances are measured to the path, not between
    matching bounce points, which slide far along the cushion on glancing hits.
    """
    if not segments or direction is None:
        return None
    drawn = [end for _, end in segments[:-1]]  # ends of all but the last segment are bounces
    if not drawn:
        return None
    exact, _ = fg.truth_path(segments[0][0], direction, truth.table, truth.radius,
                             bounces=len(segments) + 1)
    mech = max(_to_polyline(p, exact) for p in drawn)
    true_deg = None
    if true_start is not None and true_dir is not None:
        real, _ = fg.truth_path(true_start, true_dir, truth.table, truth.radius,
                                bounces=len(segments) + 1)
        travelled, worst = 0.0, 0.0
        previous = segments[0][0]
        for p in drawn:
            travelled += float(np.hypot(p[0] - previous[0], p[1] - previous[1]))
            previous = p
            # Right next to the start the drift is the start offset, not an angle.
            if travelled > truth.radius * 4:
                worst = max(worst, float(np.degrees(np.arctan2(_to_polyline(p, real), travelled))))
        true_deg = worst
    return mech, true_deg, len(drawn)


def evaluate(seed: int, kind: str | None = None, cloth: str | int = "blue", harsh: bool = False,
             flat: bool = False):
    scene = fg.random_scene(seed, kind, cloth)
    if flat:
        scene = replace(scene, turned=False)
    if harsh:
        scene = fg.harsh(scene)
    return check_scene(scene)


def check_scene(scene, frame=None, truth=None, d=None):
    """Every check on one rendered scene (``frame``/``truth``/``d`` may be
    passed when already made). Returns (scene, frame, truth, d, problems,
    stats) like ``evaluate``."""
    if frame is None:
        frame, truth = fg.render(scene)
    if d is None:
        d = main.detect(frame)
    r = truth.radius
    problems = []
    stats = {}
    # Table.
    if d.table is None:
        problems.append("table: not found")
    else:
        tx, ty, tw, th = truth.table
        edges = np.abs(np.array([d.table[0] - tx, d.table[1] - ty,
                                 d.table[0] + d.table[2] - (tx + tw), d.table[1] + d.table[3] - (ty + th)]))
        stats["table"] = float(edges.max())
        if edges.max() > TABLE_PX:
            problems.append(f"table: edge off by {edges.max():.1f}px")
    # Balls.
    detected = [(bx, by, br, kind) for bx, by, br, kind in d.balls or []]
    truth_points = {n: p for n, p in truth.balls.items()}
    missing = [n for n, (bx, by) in truth_points.items()
               if not any(np.hypot(bx - x, by - y) <= r * .5 for x, y, _, k in detected if k != "cue")]
    extra = [(x, y) for x, y, _, k in detected if k != "cue" and
             not any(np.hypot(bx - x, by - y) <= r * .5 for bx, by in truth_points.values())]
    stats["missing"], stats["extra"] = len(missing), len(extra)
    if missing:
        problems.append(f"balls: missing {sorted(missing)}")
    if extra:
        problems.append(f"balls: {len(extra)} false at {[(int(x), int(y)) for x, y in extra][:4]}")
    # Cue ball.
    if d.cue is None:
        problems.append("cue: not found")
    else:
        err = float(np.hypot(d.cue[0] - truth.cue[0], d.cue[1] - truth.cue[1]))
        stats["cue"] = err
        if err > CUE_PX:
            problems.append(f"cue: off by {err:.1f}px at {d.cue[:2]}")
    # Ghost ball and aim.
    if truth.ghost is not None:
        if d.ghost is None:
            problems.append("ghost: not found")
        else:
            err = float(np.hypot(d.ghost[0] - truth.ghost[0], d.ghost[1] - truth.ghost[1]))
            stats["ghost"] = err
            if err > GHOST_PX:
                problems.append(f"ghost: off by {err:.1f}px")
        if not d.aim_segments:
            problems.append("aim: no ray")
        else:
            direction = d.aim_dir or (lambda s: (s[1][0] - s[0][0], s[1][1] - s[0][1]))(d.aim_segments[0])
            err = angle_diff(angle_of(direction), truth.aim_degrees)
            stats["aim"] = err
            # What the eye sees is the offset where the aim meets the ghost.
            reach = float(np.hypot(truth.ghost[0] - truth.cue[0], truth.ghost[1] - truth.cue[1]))
            offset = reach * float(np.sin(np.radians(err)))
            stats["aim_px"] = offset
            if err > AIM_DEG and offset > AIM_PX:
                problems.append(f"aim: off by {err:.2f} deg ({offset:.1f}px at the ghost)")
            # The drawn ray must end on the ghost ball.
            end = d.aim_segments[0][1]
            if truth.ghost and np.hypot(end[0] - truth.ghost[0], end[1] - truth.ghost[1]) > GHOST_PX + 1:
                problems.append("aim: drawn ray does not end on the ghost ball")
    # Target and directions.
    if truth.target is not None:
        tx, ty = truth.balls[truth.target]
        if d.target is None:
            problems.append(f"target: none (truth ball {truth.target})")
        elif np.hypot(d.target[0] - tx, d.target[1] - ty) > r * .5:
            problems.append(f"target: wrong ball at {d.target[:2]} (truth {truth.target})")
        else:
            if d.object_dir is None or not d.object_segments:
                problems.append("object path: missing")
            else:
                err = angle_diff(angle_of(d.object_dir), angle_of(truth.object_dir))
                stats["object"] = err
                start = d.object_segments[0][0]
                stats["object_start"] = float(np.hypot(start[0] - tx, start[1] - ty))
                if err > OBJECT_DEG:
                    problems.append(f"object path: off by {err:.1f} deg")
                # The drawn path must follow that direction, unless the ball is
                # frozen on a cushion (then it rebounds at once) or blocked
                # within a radius (segment too short to have an angle).
                bx0, by0, bw0, bh0 = truth.table
                on_cushion = min(tx - bx0, bx0 + bw0 - tx, ty - by0, by0 + bh0 - ty) < r * 1.3
                (sx, sy), (ex, ey) = d.object_segments[0]
                if not on_cushion and np.hypot(ex - sx, ey - sy) > r * 2:
                    drawn = angle_diff(seg_angle(d.object_segments[0]), angle_of(d.object_dir))
                    if drawn > 1.5:
                        problems.append(f"object path: drawn {drawn:.1f} deg off its own direction")
                if stats["object_start"] > r * .3:
                    problems.append(f"object path: starts {stats['object_start']:.1f}px from the ball")
            if truth.cue_dir is not None:
                cos_cut = abs(float(np.dot(truth.object_dir, (np.cos(np.radians(truth.aim_degrees)),
                                                              np.sin(np.radians(truth.aim_degrees))))))
                if cos_cut < .99:
                    if d.cue_dir is None:
                        problems.append("cue path: missing")
                    else:
                        err = angle_diff(angle_of(d.cue_dir), angle_of(truth.cue_dir))
                        stats["cue_path"] = err
                        if err > CUE_PATH_DEG:
                            problems.append(f"cue path: off by {err:.1f} deg")
    else:
        if d.target is not None:
            problems.append(f"target: reported {d.target[:2]} but the aim hits no ball")
    # Stick pull-back (raw measure for shot power): reported, not a pass/fail
    # check yet; the watermark under the stick can still cut it short.
    if d.cue is not None and getattr(d, "stick_pull", None) is not None:
        stats["stick_pull"] = abs(d.stick_pull - (scene.stick_pull + fg.STICK_REST))
    # Pot or miss: only judged when the truth is clear of a decision boundary
    # by a quarter radius (a pixel of angle error can flip a near-jaw call).
    target_found = (truth.target is not None and d.target is not None and
                    np.hypot(d.target[0] - truth.balls[truth.target][0],
                             d.target[1] - truth.balls[truth.target][1]) <= r * .5)
    if target_found and truth.pot_margin >= r * .25:
        said = getattr(d, "pot", None)
        if truth.pot is None and said == "pot" or truth.pot == "pot" and said != "pot"                 or truth.pot == "jaw" and said is None:
            problems.append(f"pot: said {said}, truth {truth.pot}")
        stats["pot_agree"] = float(said == truth.pot)
    # Paths through several cushion bounces.
    others = [p for n, p in truth.balls.items() if n != truth.target]
    checks = []
    target_ok = (truth.target is not None and d.target is not None and
                 np.hypot(d.target[0] - truth.balls[truth.target][0],
                          d.target[1] - truth.balls[truth.target][1]) <= r * .5)
    if target_ok:
        checks.append(("object", d.object_segments, d.object_dir, others,
                       truth.balls[truth.target], truth.object_dir))
    if d.cue_segments and (target_ok or truth.target is None):
        a = np.radians(truth.aim_degrees)
        true_cue_dir = truth.cue_dir
        if truth.target is None and truth.ghost is not None:
            # Cue ball against an empty cushion: mirrored aim.
            x0, y0, w0, h0 = truth.table
            ux, uy = float(np.cos(a)), float(np.sin(a))
            gx, gy = truth.ghost
            if min(abs(gx - x0 - r), abs(x0 + w0 - r - gx)) < r * .2:
                ux = -ux
            if min(abs(gy - y0 - r), abs(y0 + h0 - r - gy)) < r * .2:
                uy = -uy
            true_cue_dir = (ux, uy)
        checks.append(("cue", d.cue_segments, d.cue_dir, others, truth.ghost, true_cue_dir))
    for name, segments, direction, balls, start, true_dir in checks:
        found = bounce_check(segments, direction, truth, balls, start, true_dir)
        if found is None:
            continue
        mech, true_deg, count = found
        stats.setdefault("bounce_px", 0.0)
        stats["bounce_px"] = max(stats["bounce_px"], mech)
        if true_deg is not None:
            stats["bounce_deg"] = max(stats.get("bounce_deg", 0.0), true_deg)
        stats["bounces"] = max(stats.get("bounces", 0), count)
        if mech > BOUNCE_PX:
            problems.append(f"bounces: {name} path off the exact bounces by {mech:.1f}px")
    return scene, frame, truth, d, problems, stats


def annotate(frame, truth, d):
    img = frame.copy()
    for n, (x, y) in truth.balls.items():
        cv2.circle(img, (int(x), int(y)), int(truth.radius), (0, 255, 0), 1)
    if truth.ghost:
        cv2.circle(img, (int(truth.ghost[0]), int(truth.ghost[1])), 3, (0, 255, 0), -1)
    for x, y, r, kind in d.balls or []:
        cv2.circle(img, (x, y), r + 3, (0, 0, 255) if kind != "cue" else (255, 255, 0), 1)
    for segs, colour in ((d.aim_segments, (255, 255, 0)), (d.object_segments, (0, 200, 255)),
                         (d.cue_segments, (255, 0, 255))):
        for a, b in segs or []:
            cv2.line(img, a, b, colour, 1)
    if d.table:
        x, y, w, h = d.table
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return img


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--kind")
    parser.add_argument("--cloth", default="blue", help="cloth colour: " + ", ".join(list(fg.CLOTH_HUES) + list(fg.CLOTH_HSV)) + ", black, or h,s,v")
    parser.add_argument("--harsh", action="store_true",
                        help="show each scene as a real mirror might: stream scale, 4:2:0 colour, display tone")
    parser.add_argument("--flat", action="store_true",
                        help="balls drawn flat (number up, straight band), as before they were turned")
    parser.add_argument("--save-failures")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    failures = 0
    categories = Counter()
    collected = {}
    for seed in range(args.start, args.start + args.count):
        cloth = tuple(int(v) for v in args.cloth.split(",")) if args.cloth.count(",") == 2 else args.cloth
        scene, frame, truth, d, problems, stats = evaluate(seed, args.kind, cloth, args.harsh, args.flat)
        for key, value in stats.items():
            collected.setdefault(key, []).append(value)
        if problems:
            failures += 1
            for p in problems:
                categories[p.split(":")[0]] += 1
            if not args.quiet:
                print(f"seed {seed}: " + "; ".join(problems))
            if args.save_failures:
                os.makedirs(args.save_failures, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_failures, f"seed{seed}.png"), annotate(frame, truth, d))
    print(f"\n{args.count - failures}/{args.count} frames passed every check")
    print("failed checks:", dict(categories))
    for key, values in sorted(collected.items()):
        v = np.array(values)
        print(f"  {key:13s} mean {v.mean():6.2f}  p95 {np.percentile(v, 95):6.2f}  max {v.max():6.2f}")


if __name__ == "__main__":
    main_cli()
