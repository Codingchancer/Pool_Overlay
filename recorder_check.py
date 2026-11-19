"""Checks ``shot_recorder`` on whole synthetic games against the exact truth.

    python recorder_check.py [--games 3] [--cloth blue] [--fps 60] [--harsh] [--snap 0]
                             [--max-shots 12] [--workers 4] [--all-cloths] [--flat]

Each shot is streamed as the mirror would show it (``SimGame.shot_stream``:
aiming, dragging the slider, the release, the balls rolling, rest) and fed to
the recorder frame by frame. Every log is compared with what really
happened: the power set, the start positions, the cue ball's and the struck
ball's positions frame by frame, where the balls stopped and what was potted.
The balls are turned and roll as the game's do; ``--flat`` gives the old look
(number up, straight band).
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import fake_game as fg
import sim_game as sg
from shot_recorder import ShotRecorder

# What counts as right (pixels on a ~1000-1300 px wide table; percent of
# power). Absolute positions come from the detector (whole pixels, typically
# 0.5 px off, up to ~3); movement along a track is what the power fit uses
# (a constant offset cancels), and is held to a tighter bound.
POWER_TOL = 1.5
START_TOL = 3.0
TRACK_TOL = 3.0
MOVE_TOL = 1.5
END_TOL = 3.0


def check_game(job):
    seed, cloth, fps, harsh, snap, max_shots, slider_every, flat = job
    game = sg.SimGame(seed, cloth=cloth, harsh=harsh, turned=not flat)
    rec = ShotRecorder(None)
    truth_at = {}
    results = []
    t = 0.0
    shots = 0
    while not game.over and shots < max_shots:
        plan = game.plan_shot()
        start_cue, start_balls = game.cue, dict(game.balls)
        slider = not (slider_every and shots % slider_every == slider_every - 1)
        logs = []
        stream_t0 = t
        for ft, frame, phase in game.shot_stream(plan, fps=fps, t0=t, snap_frames=snap, slider=slider):
            t = ft + 1.0 / fps
            log = rec.feed(frame, ft)
            if phase == "roll":
                truth_at[round(ft, 6)] = None  # filled below
            if log is not None:
                logs.append(log)
        record = game.shots[-1]
        # Truth positions for every roll frame, by time.
        roll_times = sorted(k for k in truth_at if truth_at[k] is None)
        ratio = 60.0 / fps
        for i, rt in enumerate(roll_times):
            truth_at[rt] = record.positions[min(int(i * ratio), len(record.positions) - 1)]
        # Keep the recorder going until the shot is logged (rest frames).
        results.append(score_shot(plan, record, start_cue, start_balls, logs, truth_at, game, slider, fps))
        truth_at.clear()
        shots += 1
        del stream_t0
    return {"seed": seed, "cloth": cloth, "shots": results, "notes": list(rec.notes)}


def score_shot(plan, record, start_cue, start_balls, logs, truth_at, game, slider, fps):
    out = {"kind": plan.kind, "power_set": round(min(1.0, plan.power) * 100, 2), "problems": []}
    if len(logs) != 1:
        out["problems"].append(f"{len(logs)} logs for one shot")
        if not logs:
            return out
    d = logs[-1].data
    r = d["ball_radius"]
    if slider:
        if d["power"] is None or abs(d["power"] - out["power_set"]) > POWER_TOL:
            out["problems"].append(f"power {d['power']} vs {out['power_set']}")
    elif d["power"] is not None:
        out["problems"].append(f"power {d['power']} with no slider")
    out["power"] = d["power"]
    bc = d["before"]["cue"]
    if np.hypot(bc[0] - start_cue[0], bc[1] - start_cue[1]) > START_TOL:
        out["problems"].append(f"start cue off {np.hypot(bc[0] - start_cue[0], bc[1] - start_cue[1]):.1f}px")
    # Cue track vs truth, frame by frame: positions, and movement from the start.
    errs, moves = [], []
    x0, y0 = d["cue_track"][0][1:]
    for rel, x, y in d["cue_track"][1:]:
        key = round(d["t_start"] + rel, 6)
        tr = truth_at.get(key)
        if tr is None or tr[0] is None:
            continue
        errs.append(float(np.hypot(x - tr[0][0], y - tr[0][1])))
        moves.append(float(np.hypot((x - x0) - (tr[0][0] - start_cue[0]), (y - y0) - (tr[0][1] - start_cue[1]))))
    if moves:
        out["cue_move_max"] = float(np.max(moves))
        if np.max(moves) > MOVE_TOL:
            out["problems"].append(f"cue movement max {np.max(moves):.1f}px")
    out["cue_track_n"] = len(errs)
    if errs:
        out["cue_track_med"] = float(np.median(errs))
        out["cue_track_max"] = float(np.max(errs))
        if np.max(errs) > TRACK_TOL:
            out["problems"].append(f"cue track max {np.max(errs):.1f}px")
    cue_potted = record.positions[-1][0] is None if record.positions else False
    if cue_potted != d["cue_potted"]:
        out["problems"].append(f"cue potted {d['cue_potted']} vs {cue_potted}")
    if cue_potted != (d["cue_lost_at"] is not None):
        out["problems"].append(f"cue lost at {d['cue_lost_at']} (potted {cue_potted})")
    if not cue_potted and d["after"]["cue"] is not None:
        fc = record.positions[-1][0]
        e = np.hypot(d["after"]["cue"][0] - fc[0], d["after"]["cue"][1] - fc[1])
        if e > END_TOL:
            out["problems"].append(f"cue end off {e:.1f}px")
    # The struck ball.
    first = record.first_contact
    oi = d["before"]["object_ball"]
    if first is not None and plan.kind != "break":
        if oi is None:
            out["problems"].append("no struck ball logged")
        else:
            ob = d["before"]["balls"][oi]
            fx, fy = start_balls[first]
            if np.hypot(ob["x"] - fx, ob["y"] - fy) > r:
                out["problems"].append("logged struck ball is not the one hit")
            else:
                oerrs, omoves = [], []
                ox0, oy0 = d["object_track"][0][1:]
                for rel, x, y in d["object_track"][1:]:
                    tr = truth_at.get(round(d["t_start"] + rel, 6))
                    if tr is None or first not in tr[1]:
                        continue
                    oerrs.append(float(np.hypot(x - tr[1][first][0], y - tr[1][first][1])))
                    omoves.append(float(np.hypot((x - ox0) - (tr[1][first][0] - fx), (y - oy0) - (tr[1][first][1] - fy))))
                if omoves and np.max(omoves) > MOVE_TOL:
                    out["problems"].append(f"object movement max {np.max(omoves):.1f}px")
                out["object_track_n"] = len(oerrs)
                if oerrs:
                    out["object_track_max"] = float(np.max(oerrs))
                    if np.max(oerrs) > TRACK_TOL:
                        out["problems"].append(f"object track max {np.max(oerrs):.1f}px")
                potted = first in record.pocketed
                if potted != (d["object_lost_at"] is not None):
                    out["problems"].append(f"object lost at {d['object_lost_at']} (potted {potted})")
    # Potted balls and where the rest stopped, through the colour matching.
    numbers = []
    for b in d["before"]["balls"]:
        near = min(start_balls.items(), key=lambda kv: np.hypot(kv[1][0] - b["x"], kv[1][1] - b["y"]))
        numbers.append(near[0] if np.hypot(near[1][0] - b["x"], near[1][1] - b["y"]) < r else None)
    logged_potted = {numbers[i] for i in d["potted"] if numbers[i] is not None}
    real_potted = {n for n in record.pocketed if n != 0}
    if logged_potted != real_potted:
        out["problems"].append(f"potted {sorted(logged_potted)} vs {sorted(real_potted)}")
    final_balls = record.positions[-1][1] if record.positions else start_balls
    wrong_end = 0
    for b, n in zip(d["before"]["balls"], numbers):
        if n is None or b["after"] is None or n not in final_balls:
            continue
        a = d["after"]["balls"][b["after"]]
        if np.hypot(a["x"] - final_balls[n][0], a["y"] - final_balls[n][1]) > END_TOL:
            wrong_end += 1
    if wrong_end:
        out["problems"].append(f"{wrong_end} balls' end spot off by more than {END_TOL}px")
    return out


def summarize(games, label):
    shots = [s for g in games for s in g["shots"]]
    ok = sum(not s["problems"] for s in shots)
    print(f"== {label}: {ok}/{len(shots)} shots logged right")
    counts = {}
    for s in shots:
        for p in s["problems"]:
            key = p.split(" ")[0] + " " + (p.split(" ")[1] if len(p.split(" ")) > 1 else "")
            counts[key] = counts.get(key, 0) + 1
    if counts:
        print("   problems:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    med = [s["cue_track_med"] for s in shots if "cue_track_med" in s]
    mx = [s["cue_track_max"] for s in shots if "cue_track_max" in s]
    mv = [s["cue_move_max"] for s in shots if "cue_move_max" in s]
    if med:
        print(f"   cue track error: median {np.median(med):.2f}px, worst {np.max(mx):.2f}px; "
              f"movement from the start worst {np.max(mv):.2f}px")
    pe = [abs(s["power"] - s["power_set"]) for s in shots if s.get("power") is not None]
    if pe:
        print(f"   power error: median {np.median(pe):.2f}%, worst {np.max(pe):.2f}%")
    for g in games:
        for i, s in enumerate(g["shots"]):
            if s["problems"]:
                print(f"   seed {g['seed']} shot {i} ({s['kind']}, power {s['power_set']}): {'; '.join(s['problems'])}")
    return ok, len(shots)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--cloth", default="blue")
    ap.add_argument("--all-cloths", action="store_true")
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--harsh", action="store_true")
    ap.add_argument("--snap", type=int, default=0)
    ap.add_argument("--max-shots", type=int, default=12)
    ap.add_argument("--no-slider-every", type=int, default=0, help="hide the slider on every n-th shot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--flat", action="store_true", help="balls not turned (number up, straight band)")
    args = ap.parse_args(argv)
    cloths = list(fg.CLOTH_HUES) + list(fg.CLOTH_HSV) if args.all_cloths else [args.cloth]
    total = [0, 0]
    for cloth in cloths:
        jobs = [(s, cloth, args.fps, args.harsh, args.snap, args.max_shots, args.no_slider_every, args.flat)
                for s in range(args.start, args.start + args.games)]
        with Pool(min(args.workers, len(jobs))) as pool:
            games = pool.map(check_game, jobs)
        ok, n = summarize(games, f"{cloth}{' harsh' if args.harsh else ''}{' flat' if args.flat else ''} {args.fps:.0f} fps")
        total[0] += ok
        total[1] += n
    if len(cloths) > 1:
        print(f"== all: {total[0]}/{total[1]}")
    return 0 if total[0] == total[1] else 1


if __name__ == "__main__":
    sys.exit(main())
