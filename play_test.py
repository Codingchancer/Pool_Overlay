"""Play whole 8-ball games (``sim_game``) and test the overlay on every shot.

    python play_test.py [--games 8] [--start 1] [--cloth blue] [--harsh] [--workers 8]

For each shot:
  * aiming: the stick turns onto the shot over a few frames, run through the
    live path (``ShotTracker`` + ``GuideStabilizer``) like the app does; the
    final aiming frame gets every stress-test check against the exact truth;
  * prediction vs physics: the struck ball, its direction, the cue ball's
    direction just after contact and the pot call are compared with where the
    physics actually sent the balls;
  * rolling: the frames while the balls move go through the live path too
    (a guide drawn then would be stale: the game shows none).

The cue ball's direction after contact is reported but not judged: the
overlay draws the game's own tangent line, while the physics rolls the cue
ball off it (follow, draw and natural roll).
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

AIM_FRAMES = 6          # stick turning onto the shot
MOTION_EVERY = 6        # every 6th physics frame (10 per second) while rolling


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def direction_after(trail, start_index, radius, travel=2.0):
    """Direction of a trail from ``start_index`` once it has moved ``travel``
    radii (None if it never does)."""
    x0, y0 = trail[start_index]
    for x, y in trail[start_index + 1:]:
        if np.hypot(x - x0, y - y0) >= travel * radius:
            return float(np.degrees(np.arctan2(y - y0, x - x0)))
    return None


def play_game(job):
    seed, cloth, harsh, max_shots = job
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import sim_game as sg
    import stress_test as st
    from overlay import GuideStabilizer
    from tracking import ShotTracker

    game = sg.SimGame(seed, cloth, harsh)
    tracker, stabilizer = ShotTracker(), GuideStabilizer()
    now = 0.0
    log = []
    while not game.over and len(game.shots) < max_shots:
        plan = game.plan_shot()
        entry = {"seed": seed, "cloth": cloth, "shot": len(game.shots) + 1, "kind": plan.kind, "power": plan.power,
                 "problems": [],
                 "live_errors": [], "stale_guide_frames": 0, "motion_frames": 0}
        if plan.kind != "break":
            # Aiming: the stick turns onto the shot.
            for k in range(AIM_FRAMES):
                angle = plan.aim_degrees - 3.0 * (AIM_FRAMES - 1 - k) / (AIM_FRAMES - 1)
                frame, truth = game.render_aim(plan, aim_degrees=angle, pull=.5 + .1 * k)
                shown = stabilizer.update(tracker.detect(frame, now=now), now=now)
                now += 1 / 30
                if k >= 2 and shown.aim_dir is not None:
                    drawn = float(np.degrees(np.arctan2(shown.aim_dir[1], shown.aim_dir[0])))
                    entry["live_errors"].append(angle_diff(drawn, angle))
                elif k >= 2:
                    entry["live_errors"].append(None)
            scene = game.scene(aim_degrees=plan.aim_degrees, guide=True, stick=True, power=min(1.0, plan.power),
                               pull=1.0)
            *_, truth, d, problems, stats = st.check_scene(scene)
            entry["problems"] = problems
            entry["power_err"] = (None if d.power_percent is None
                                  else abs(d.power_percent - round(min(1.0, plan.power) * 100)))
        shot = game.shoot(plan, every=MOTION_EVERY)
        entry["first_contact"] = shot.first_contact
        entry["pocketed"] = shot.pocketed
        entry["message"] = shot.message
        if plan.kind != "break" and shot.first_contact not in (None, 0):
            r = game.r
            tx, ty = shot.start_balls[shot.first_contact]
            entry["target_found"] = d.target is not None and np.hypot(d.target[0] - tx, d.target[1] - ty) <= r * .5
            # Time-aligned 60 Hz positions: contact is the first frame the
            # struck ball moves.
            track = [balls.get(shot.first_contact) for _, balls in shot.positions]
            cue_track = [cue for cue, _ in shot.positions]
            hit = next((i for i, p in enumerate(track) if p is None or np.hypot(p[0] - tx, p[1] - ty) > .05 * r), None)
            if entry["target_found"] and hit is not None and d.object_dir is not None:
                obj = [p for p in track[hit - 1 if hit else 0:] if p is not None]
                real = direction_after(obj, 0, r) if obj else None
                if real is not None:
                    entry["object_err"] = angle_diff(real, float(np.degrees(np.arctan2(d.object_dir[1],
                                                                                         d.object_dir[0]))))
                cues = [p for p in cue_track[max(0, hit - 1):] if p is not None]
                if d.cue_dir is not None and len(cues) > 1:
                    real_cue = direction_after(cues, 0, r, travel=1.5)
                    if real_cue is not None:
                        entry["cue_err"] = angle_diff(real_cue, float(np.degrees(np.arctan2(d.cue_dir[1],
                                                                                               d.cue_dir[0]))))
                        entry["spin"] = plan.spin
            potted = shot.first_contact in shot.pocketed
            entry["pot_real"] = potted
            entry["pot_said"] = d.pot
        # Rolling: the live path while the balls move (and after they stop).
        for cue, balls in shot.frames:
            frame, _ = game.render_motion(cue, balls)
            shown = stabilizer.update(tracker.detect(frame, now=now), now=now)
            now += MOTION_EVERY / 60
            entry["motion_frames"] += 1
            entry["stale_guide_frames"] += bool(shown.aim_segments)
        log.append(entry)
    return {"seed": seed, "cloth": cloth, "winner": game.rules.winner, "shots": log}


def summarize(games):
    shots = [s for g in games for s in g["shots"] if s["kind"] != "break"]
    out = []
    out.append(f"{len(games)} games, {len(shots)} aimed shots, "
               f"{sum(g['winner'] is not None for g in games)} played to the end")
    clean = sum(not s["problems"] for s in shots)
    out.append(f"aiming frame: every check passed on {clean}/{len(shots)}")
    cats = Counter(p.split(":")[0] for s in shots for p in s["problems"])
    out.append(f"  failed checks: {dict(cats.most_common())}")
    live = [e for s in shots for e in s["live_errors"]]
    lost = sum(e is None for e in live)
    errs = np.array([e for e in live if e is not None])
    if len(errs):
        out.append(f"live (stick turning): drawn aim off by median {np.median(errs):.2f} deg, "
                   f"p90 {np.percentile(errs, 90):.2f}, frames without a guide {lost}/{len(live)}")
    hit = [s for s in shots if "target_found" in s]
    out.append(f"struck ball (physics' first contact) found: {sum(s['target_found'] for s in hit)}/{len(hit)}")
    obj = np.array([s["object_err"] for s in hit if "object_err" in s])
    if len(obj):
        out.append(f"struck ball's direction vs physics: median {np.median(obj):.2f} deg, p90 "
                   f"{np.percentile(obj, 90):.2f}, within 2 deg {np.mean(obj <= 2) * 100:.0f}%")
    by_spin = defaultdict(list)
    for s in hit:
        if "cue_err" in s:
            by_spin["draw" if s["spin"][1] < 0 else "follow" if s["spin"][1] > 0 else "centre"].append(s["cue_err"])
    for k, v in sorted(by_spin.items()):
        v = np.array(v)
        out.append(f"cue ball's direction after contact vs physics ({k} hit): median {np.median(v):.1f} deg, "
                   f"p90 {np.percentile(v, 90):.1f} (n={len(v)})")
    for name, low, high in (("normal power (<= 80%)", 0, .8), ("break power (> 80%)", .8, 1.01)):
        part = np.array([s["object_err"] for s in hit if "object_err" in s and low <= s["power"] < high])
        called = [s for s in hit if s.get("target_found") and low <= s["power"] < high and s["pot_said"] == "pot"]
        if len(part):
            out.append(f"  {name}: direction within 2 deg {np.mean(part <= 2) * 100:.0f}% (n={len(part)}); "
                       f"called pots that dropped {sum(s['pot_real'] for s in called)}/{len(called)}")
    pots = [s for s in hit if s.get("target_found")]
    agree = sum((s["pot_said"] == "pot") == s["pot_real"] for s in pots)
    said_pot = [s for s in pots if s["pot_said"] == "pot"]
    out.append(f"pot call vs physics: agree {agree}/{len(pots)}; said pot {len(said_pot)}, "
               f"of those really potted {sum(s['pot_real'] for s in said_pot)}")
    power = [s["power_err"] for s in shots if s.get("power_err") is not None]
    if power:
        out.append(f"power slider read: median error {np.median(power):.1f}%, max {max(power):.0f}% (n={len(power)})")
    motion = sum(s["motion_frames"] for s in shots)
    stale = sum(s["stale_guide_frames"] for s in shots)
    out.append(f"frames while balls roll with a guide still drawn: {stale}/{motion}")
    return "\n".join(out)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--cloth", default="blue")
    parser.add_argument("--harsh", action="store_true")
    parser.add_argument("--max-shots", type=int, default=40)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--json")
    args = parser.parse_args()
    jobs = [(seed, args.cloth, args.harsh, args.max_shots) for seed in range(args.start, args.start + args.games)]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(min(args.workers, len(jobs))) as pool:
            games = pool.map(play_game, jobs)
    else:
        games = [play_game(job) for job in jobs]
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(games, f, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print(summarize(games))


if __name__ == "__main__":
    main_cli()
