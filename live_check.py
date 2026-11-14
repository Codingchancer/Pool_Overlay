"""End-to-end live check against the synthetic table (Windows only).

    python live_check.py [--scenes 12] [--interval 3500]

Starts ``fake_game.py`` in a window that shows a new synthetic shot every few
seconds and writes its truth to a JSON file, starts the overlay app with its
debug dump on, presses Fit table and Enable guide through UI Automation (the
app's own buttons only), and then, for every scene, compares the overlay's
analysis frame with the truth. The guide is visible on screen the whole time,
so this exercises the capture-feedback path (draw -> capture -> subtract).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

import fake_game as fg
import main

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


def press(button: str, window: str = "Pool Practice Overlay") -> bool:
    """Invoke one of the overlay app's own buttons by its label."""
    script = (
        "Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes;"
        "$r=[System.Windows.Automation.AutomationElement]::RootElement;"
        f"$w=$r.FindFirst('Children',(New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty,'{window}')));"
        "if(-not $w){exit 2};"
        f"$b=$w.FindFirst('Descendants',(New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty,'{button}')));"
        "if(-not $b){exit 3};"
        "$b.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()")
    return subprocess.run(["powershell", "-NoProfile", "-Command", script]).returncode == 0


def overlay_rect():
    script = (
        "Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes;"
        "$r=[System.Windows.Automation.AutomationElement]::RootElement;"
        "$g=$r.FindFirst('Children',(New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty,'Pool Practice Guide')));"
        "$b=$g.Current.BoundingRectangle; \"$($b.X),$($b.Y),$($b.Width),$($b.Height)\"")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True).stdout
    return tuple(int(float(v)) for v in out.strip().split(","))


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--interval", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=500)
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="pool-live-"))
    truth_file = work / "truth.json"
    env = dict(os.environ, POOL_OVERLAY_DEBUG_DIR=str(work))
    env.pop("QT_QPA_PLATFORM", None)
    table = subprocess.Popen([PYTHON, str(HERE / "fake_game.py"), "--seed", str(args.seed),
                              "--interval", str(args.interval), "--truth", str(truth_file)], env=env)
    time.sleep(2.5)
    app = subprocess.Popen([PYTHON, str(HERE / "main.py")], env=env)
    results = []
    try:
        time.sleep(4)
        press("Fit table")
        time.sleep(1.5)
        if not press("Enable guide"):
            print("could not press Enable guide")
            return 1
        ox, oy, ow, oh = overlay_rect()
        print(f"overlay at {ox},{oy} {ow}x{oh}; dumps in {work}")
        rendered = {}

        def true_crop(seed, window):
            # The exact image the window shows for this scene, cut to the
            # overlay's capture rectangle.
            if seed not in rendered:
                frame, truth = fg.render(fg.live_scene(seed))
                rendered[seed] = (frame, truth)
            frame, truth = rendered[seed]
            x0, y0 = ox - window[0], oy - window[1]
            return frame[y0:y0 + oh, x0:x0 + ow], truth

        judged = set()
        deadline = time.time() + args.scenes * args.interval / 1000 + 8
        while time.time() < deadline and len(judged) < args.scenes:
            time.sleep(.3)
            try:
                info = json.loads(truth_file.read_text())
            except Exception:
                continue
            cleaned = cv2.imread(str(work / "cleaned.png"))
            capture = cv2.imread(str(work / "capture.png"))
            layer = cv2.imread(str(work / "layer.png"), cv2.IMREAD_UNCHANGED)
            if cleaned is None or capture is None or layer is None or cleaned.shape[:2] != (oh, ow):
                continue
            # Which scene does this analysed frame show? Only judge exact matches.
            match = None
            for seed in (info["seed"], info["seed"] - 1):
                crop, truth = true_crop(seed, info["window"])
                if crop.shape == cleaned.shape and float(np.abs(crop.astype(int) - cleaned).mean()) < 2:
                    match = (seed, crop, truth)
                    break
            if match is None or match[0] in judged:
                continue
            seed, crop, truth = match
            judged.add(seed)
            covered = layer[:, :, 3] > 0
            before = np.abs(capture.astype(int) - crop).max(axis=2)[covered]
            after = np.abs(cleaned.astype(int) - crop).max(axis=2)[covered]
            d = main.detect(cleaned)
            shift = np.array([info["window"][0] - ox, info["window"][1] - oy], float)
            problems = []
            r = truth.radius
            if truth.ghost is not None:
                g = np.array(truth.ghost) + shift
                if d.ghost is None or np.hypot(*(np.array(d.ghost[:2]) - g)) > 2.5:
                    problems.append(f"ghost {d.ghost} vs {g.round(1)}")
                if d.aim_dir is None:
                    problems.append("aim missing")
                else:
                    err = angle_diff(np.degrees(np.arctan2(d.aim_dir[1], d.aim_dir[0])), truth.aim_degrees)
                    reach = np.hypot(truth.ghost[0] - truth.cue[0], truth.ghost[1] - truth.cue[1])
                    if err > .6 and reach * np.sin(np.radians(err)) > 1:
                        problems.append(f"aim off {err:.2f} deg")
            if truth.target is not None:
                t = np.array(truth.balls[truth.target]) + shift
                if d.target is None or np.hypot(*(np.array(d.target[:2]) - t)) > r * .5:
                    problems.append(f"target {d.target} vs {t.round(1)}")
            elif d.target is not None:
                problems.append(f"target {d.target} reported, aim hits no ball")
            results.append((seed, problems))
            print(f"scene {seed}: {'ok' if not problems else '; '.join(problems)} | under the guide: "
                  f"capture off by up to {int(before.max())} levels, after subtraction p99 "
                  f"{np.percentile(after, 99):.0f} / max {int(after.max())}")
    finally:
        app.terminate()
        table.terminate()
    passed = sum(1 for _, p in results if not p)
    print(f"\n{passed}/{len(results)} live scenes matched the truth")
    return 0 if results and passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main_cli())
