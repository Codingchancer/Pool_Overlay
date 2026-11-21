"""Train the ball-proposal network of ``ball_net.py`` on synthetic frames.

    python train_ball_net.py render --count 16000          (app's .venv is enough)
    .venv-train\\Scripts\\python train_ball_net.py train --epochs 35
    .venv-train\\Scripts\\python train_ball_net.py export   (-> models/balls.onnx)

``render`` draws random ``fake_game`` scenes (every cloth in CLOTH_HUES /
CLOTH_HSV, "black" and random colours; turned and flat balls; some through
``fake_game.harsh``; some without guide or stick) and stores the table crop
(the truth's cushion-nose rectangle, each edge jittered by a few pixels as
the detector's table can be, sometimes scaled a few per cent) at the
network's input size with every ball's exact centre and class. Some frames
get what the random scenes never show: the cue ball frozen on a cushion,
round ornaments along the stick, faint ball-sized rings on the cloth
(``add_distractors``). Seeds start at 1,000,000: never the stress test's
(0-299) or the sweeps' (40000-40099). The data and the checkpoints go to
``outputs/nn/`` (not in git); only the exported ONNX model is kept.

Training needs PyTorch (the separate ``.venv-train``); the app itself never
imports it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import ball_net as bn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "nn" / "data"
CHECKPOINT = ROOT / "outputs" / "nn" / "balls.pt"
SEED_BASE = 1_000_000
VAL_SEED_BASE = 2_000_000
KINDS = ["scatter", "scatter", "rack", "rack", "cushion", "miss", "pot", "near"]
# Share of the crop width each table edge is jittered by (detector's table
# is within ~3 px; more on the cloth-colour fallback).
TABLE_JITTER = .004


def cloth_names():
    import fake_game as fg
    return list(fg.CLOTH_HUES) + list(fg.CLOTH_HSV) + ["black"]


def make_scene(seed: int):
    import fake_game as fg
    rng = np.random.default_rng(seed + 77)
    kind = str(rng.choice(KINDS))
    if rng.random() < .15:
        cloth = (int(rng.integers(0, 180)), int(rng.integers(0, 230)), int(rng.integers(60, 236)))
    else:
        cloth = str(rng.choice(cloth_names()))
    scene = fg.random_scene(seed, kind, cloth)
    scene = replace(scene, turned=bool(rng.random() < .85))
    if rng.random() < .1:
        scene = replace(scene, show_guide=False)
    if rng.random() < .1:
        scene = replace(scene, show_stick=False)
    if rng.random() < CUE_ON_CUSHION:
        scene = cue_on_cushion(scene, rng)
    if rng.random() < .35:
        scene = fg.harsh(scene)
    return scene, rng


# Shares of training frames with extras the random scenes never show: the
# cue ball frozen on a cushion (``_random_scene`` keeps it two radii in), a
# stick with round ornaments like the real cue's butt, and faint ball-sized
# rings and discs in the cloth's darker shade (the real table's logo).
CUE_ON_CUSHION = .2
STICK_ORNAMENTS = .6
CLOTH_RINGS = .4
# Share of frames whose crop is scaled about its centre (a table fit a few
# per cent off shows the balls a little bigger or smaller), and by how much.
CROP_SCALE = (.3, .06)


def cue_on_cushion(scene, rng):
    import fake_game as fg
    x0, y0, w, h = scene.table
    r = scene.radius
    pockets = fg.pocket_centres(scene.table)
    for _ in range(200):
        side = int(rng.integers(4))
        if side < 2:
            px, py = rng.uniform(x0 + 4 * r, x0 + w - 4 * r), (y0 + r if side == 0 else y0 + h - r)
        else:
            px, py = (x0 + r if side == 2 else x0 + w - r), rng.uniform(y0 + 4 * r, y0 + h - 4 * r)
        if (min(np.hypot(px - qx, py - qy) for qx, qy in pockets) > 3.2 * r
                and all(np.hypot(px - bx, py - by) >= 2.1 * r for bx, by in scene.balls.values())):
            # Aim into the table, at a ball when one is in front.
            ahead = [(bx, by) for bx, by in scene.balls.values()
                     if (bx - x0 - w / 2) * (px - x0 - w / 2) + (by - y0 - h / 2) * (py - y0 - h / 2) < w * w]
            tx, ty = ahead[int(rng.integers(len(ahead)))] if ahead else (x0 + w / 2, y0 + h / 2)
            aim = float(np.degrees(np.arctan2(ty - py, tx - px)) + rng.uniform(-8, 8))
            return replace(scene, cue=(float(px), float(py)), aim_degrees=aim)
    return scene


def _colour(rng):
    """A random ornament colour: gold, orange, red, black, metal or any."""
    choices = [(40, 170, 230), (30, 120, 240), (40, 40, 200), (25, 25, 30), (190, 195, 200), (60, 90, 140)]
    if rng.random() < .3:
        return tuple(float(v) for v in rng.integers(0, 256, 3))
    return tuple(float(v) for v in choices[int(rng.integers(len(choices)))])


def add_distractors(frame, scene, rng):
    """Paint round things that are not balls onto a rendered frame: ornaments
    along the stick (on top of balls, as the game's stick is) and faint rings
    and discs on the cloth away from the balls. Labels are unchanged."""
    import fake_game as fg
    canvas = frame.astype(np.float32)
    r = scene.radius
    x0, y0, w, h = scene.table
    if scene.show_stick and rng.random() < STICK_ORNAMENTS:
        span = w / (1 - 2 * fg.POCKET_TO_CUSHION)
        a = np.radians(scene.aim_degrees)
        ux, uy = float(np.cos(a)), float(np.sin(a))
        cx, cy = scene.cue
        start, length = r * (1.25 + scene.stick_pull), span * .52
        for _ in range(int(rng.integers(2, 9))):
            t = start + length * rng.uniform(.05, 1.0)
            px, py = cx - ux * t, cy - uy * t
            size = r * rng.uniform(.3, .95)
            fg.fill_disc(canvas, px, py, size, _colour(rng))
            if rng.random() < .6:
                fg.stroke_ring(canvas, px, py, size * rng.uniform(.5, .9), size * rng.uniform(.1, .25), _colour(rng))
            if rng.random() < .5:
                fg.fill_disc(canvas, px, py, size * rng.uniform(.2, .5), _colour(rng))
        for _ in range(int(rng.integers(0, 4))):
            # Flames / bands along the shaft.
            t0 = start + length * rng.uniform(.1, .95)
            t1 = t0 + r * rng.uniform(.8, 3)
            fg.stroke_line(canvas, (cx - ux * t0, cy - uy * t0), (cx - ux * t1, cy - uy * t1),
                           r * rng.uniform(.3, .7), _colour(rng), alpha=float(rng.uniform(.6, 1)))
    if rng.random() < CLOTH_RINGS:
        balls = [scene.cue] + list(scene.balls.values())
        for _ in range(int(rng.integers(1, 4))):
            for _ in range(50):
                px, py = rng.uniform(x0 + r, x0 + w - r), rng.uniform(y0 + r, y0 + h - r)
                if all(np.hypot(px - bx, py - by) >= 1.8 * r for bx, by in balls):
                    break
            else:
                continue
            shade = canvas[int(py), int(px)] * rng.uniform(.7, .9)
            size = r * rng.uniform(.7, 1.3)
            if rng.random() < .5:
                fg.stroke_ring(canvas, px, py, size, r * rng.uniform(.12, .3), tuple(float(v) for v in shade))
            else:
                fg.fill_disc(canvas, px, py, size, tuple(float(v) for v in shade), alpha=float(rng.uniform(.5, 1)))
    return np.clip(np.round(canvas), 0, 255).astype(np.uint8)


def render_one(args):
    seed, out_dir = args
    import fake_game as fg
    scene, rng = make_scene(seed)
    frame, truth = fg.render(scene)
    frame = add_distractors(frame, scene, rng)
    x, y, w, h = truth.table
    j = TABLE_JITTER * w
    x0, x1 = x + rng.uniform(-j, j), x + w + rng.uniform(-j, j)
    y0, y1 = y + rng.uniform(-j, j), y + h + rng.uniform(-j, j)
    if rng.random() < CROP_SCALE[0]:
        k = 1 + rng.uniform(-CROP_SCALE[1], CROP_SCALE[1])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1 = mx + (x0 - mx) * k, mx + (x1 - mx) * k
        y0, y1 = my + (y0 - my) * k, my + (y1 - my) * k
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(frame.shape[1]), x1), min(float(frame.shape[0]), y1)
    ix0, iy0, ix1, iy1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    roi = frame[iy0:iy1, ix0:ix1]
    small = bn.preprocess(roi)[0].transpose(1, 2, 0) * 255
    image = np.clip(np.round(small), 0, 255).astype(np.uint8)
    sx, sy = roi.shape[1] / bn.INPUT_W, roi.shape[0] / bn.INPUT_H
    balls = [(0, *truth.cue)] + [(n, *p) for n, p in truth.balls.items()]
    labels = [(bn.to_input(bx - ix0, sx), bn.to_input(by - iy0, sy), bn.class_of(n)) for n, bx, by in balls]
    name = f"{seed}.png"
    cv2.imwrite(str(out_dir / name), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return name, labels, float(truth.radius / sx)


def render(count: int, split: str, workers: int):
    out_dir = DATA / split
    out_dir.mkdir(parents=True, exist_ok=True)
    base = SEED_BASE if split == "train" else VAL_SEED_BASE
    t = time.time()
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(render_one, [(base + k, out_dir) for k in range(count)],
                                                     chunksize=8)):
            rows.append(row)
            if (i + 1) % 1000 == 0:
                print(f"{split}: {i + 1}/{count} ({time.time() - t:.0f} s)", flush=True)
    rows.sort()
    (out_dir / "labels.json").write_text(json.dumps(rows))
    print(f"{split}: {count} frames in {time.time() - t:.0f} s, radius ~{np.median([r[2] for r in rows]):.1f} px")


# --- real frames ---------------------------------------------------------------

# Real screenshots in public datasets are ~640-760 px wide; the detector is
# made for the mirror's size (~1600-2500 px), so they are scaled up first.
REAL_SCALE = 2.5


def import_real(root: Path, name: str, ball_class: str = "ball"):
    """Real 8 Ball Pool screenshots with hand-drawn ball boxes (a COCO export
    with train/valid/test folders, e.g. from Roboflow Universe) -> table
    crops at the network's input size with every labelled ball's centre
    (class -1: the labels give no class), in DATA/real-<name>-<split>/, and
    frames.json there: each whole frame's path and its ball centres (frame
    px after REAL_SCALE) for measuring the detector on it.

    A frame is left out when no table is found on it (menus, the spin
    screen, a turned copy) or it has no ball label (some exports have
    unlabelled frames, which would teach the network that balls are not
    balls). The splits stay as the export made them (copies of one picture
    are in one split only)."""
    import detector
    for split_dir in sorted(p for p in root.iterdir() if (p / "_annotations.coco.json").is_file()):
        split = {"valid": "val"}.get(split_dir.name, split_dir.name)
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())
        wanted = {c["id"] for c in coco["categories"] if c["name"] == ball_class}
        boxes: dict[int, list] = {}
        for a in coco["annotations"]:
            if a["category_id"] in wanted:
                boxes.setdefault(a["image_id"], []).append(a["bbox"])
        out = DATA / f"real-{name}-{split}"
        out.mkdir(parents=True, exist_ok=True)
        rows, frames, skipped = [], [], {"no table": 0, "no labels": 0}
        for im in coco["images"]:
            bxs = boxes.get(im["id"], [])
            if not bxs:
                skipped["no labels"] += 1
                continue
            frame = cv2.imread(str(split_dir / im["file_name"]), cv2.IMREAD_COLOR)
            big = cv2.resize(frame, None, fx=REAL_SCALE, fy=REAL_SCALE, interpolation=cv2.INTER_CUBIC)
            table = detector.find_table_bounds(big)
            if table is None or not 1.7 <= table[2] / max(1, table[3]) <= 2.3 or table[2] < big.shape[1] * .4:
                skipped["no table"] += 1
                continue
            x, y, w, h = table
            # Box centre in pixel-centre coordinates of the scaled frame.
            centres = [((bx + bw / 2) * REAL_SCALE - .5, (by + bh / 2) * REAL_SCALE - .5) for bx, by, bw, bh in bxs]
            radius = float(np.median([(bw + bh) / 4 for _, _, bw, bh in bxs])) * REAL_SCALE
            centres = [(cx, cy) for cx, cy in centres if x - radius <= cx <= x + w + radius
                       and y - radius <= cy <= y + h + radius]
            if not centres:
                skipped["no labels"] += 1
                continue
            roi = big[y:y + h, x:x + w]
            small = np.clip(np.round(bn.preprocess(roi)[0].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
            stem = Path(im["file_name"]).stem
            cv2.imwrite(str(out / f"{stem}.png"), small, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            sx, sy = w / bn.INPUT_W, h / bn.INPUT_H
            rows.append((f"{stem}.png", [(bn.to_input(cx - x, sx), bn.to_input(cy - y, sy), -1) for cx, cy in centres],
                         radius / sx))
            frames.append({"file": str(split_dir / im["file_name"]), "scale": REAL_SCALE, "table": list(table),
                           "radius": radius, "balls": centres})
        (out / "labels.json").write_text(json.dumps(rows))
        (out / "frames.json").write_text(json.dumps(frames))
        print(f"{out.name}: {len(rows)} frames, {sum(len(r[1]) for r in rows)} balls; left out {skipped}")


def _layout_match(a, b, reach: float = 10.0) -> float:
    """Share of balls two frames have in the same spots (input px), the
    smaller of the two directions."""
    pa, pb = np.array([(u, v) for u, v, _ in a]), np.array([(u, v) for u, v, _ in b])
    d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
    return float(min((d.min(1) < reach).mean(), (d.min(0) < reach).mean()))


def split_real(name: str, test_share: float = .3):
    """Re-split an imported real set by scene: public sets are frames of a
    few recorded games, and the export's picture-by-picture split puts the
    same layout in train and test (8 ball detector: 414 of 577 train frames).
    Frames with half their balls in the same spots are one scene (joined
    transitively); whole scenes go to DATA/real-<name>-scenes-test until it
    holds ``test_share`` of each table colour's frames, the rest to
    -scenes-train (labels only; the crops stay where import_real put them).
    Frames whose crop is no cloth (the spin screen's huge cue ball) are
    dropped."""
    rows, frames = [], []
    for split in ("train", "val", "test"):
        d = DATA / f"real-{name}-{split}"
        if not d.is_dir():
            continue
        for row, frame in zip(json.loads((d / "labels.json").read_text()), json.loads((d / "frames.json").read_text())):
            image = cv2.imread(str(d / row[0]), cv2.IMREAD_COLOR)
            hsv = cv2.cvtColor(image[100:220, 200:440], cv2.COLOR_BGR2HSV).reshape(-1, 3)
            h, sat, val = np.median(hsv, axis=0)
            if sat < 40 and val > 200:
                continue
            rows.append([f"../{d.name}/{row[0]}", row[1], row[2]])
            frames.append({**frame, "colour": f"{int(h) // 12}-{int(val) // 64}"})
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if root(i) != root(j) and _layout_match(rows[i][1], rows[j][1]) >= .5:
                parent[root(i)] = root(j)
    scenes: dict[int, list[int]] = {}
    for i in range(len(rows)):
        scenes.setdefault(root(i), []).append(i)
    by_colour: dict[str, list[list[int]]] = {}
    for members in scenes.values():
        colour = max({frames[i]["colour"] for i in members}, key=[frames[i]["colour"] for i in members].count)
        by_colour.setdefault(colour, []).append(members)
    test: set[int] = set()
    rng = np.random.default_rng(7)
    for colour, groups in sorted(by_colour.items()):
        total = sum(len(g) for g in groups)
        for k in rng.permutation(len(groups)):
            if sum(len(g) for g in groups if set(g) <= test) + len(groups[k]) <= total * test_share * 1.15:
                test.update(groups[k])
    for part, keep in (("train", lambda i: i not in test), ("test", lambda i: i in test)):
        out = DATA / f"real-{name}-scenes-{part}"
        out.mkdir(parents=True, exist_ok=True)
        idx = [i for i in range(len(rows)) if keep(i)]
        (out / "labels.json").write_text(json.dumps([rows[i] for i in idx]))
        (out / "frames.json").write_text(json.dumps([frames[i] for i in idx]))
        print(f"{out.name}: {len(idx)} frames, {sum(len(rows[i][1]) for i in idx)} balls, "
              f"{len({root(i) for i in idx})} scenes, colours {sorted({frames[i]['colour'] for i in idx})}")


def _eval_real_frame(job):
    mode, frame = job
    import os as _os
    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    cv2.setNumThreads(1)
    import detector
    detector.NN_BALLS = "" if mode == "off" else mode
    image = cv2.imread(frame["file"], cv2.IMREAD_COLOR)
    big = cv2.resize(image, None, fx=frame["scale"], fy=frame["scale"], interpolation=cv2.INTER_CUBIC)
    d = detector.detect(big)
    found = [(x, y) for x, y, _, _ in d.balls or []]
    r = frame["radius"]
    used, errors = set(), []
    for bx, by in frame["balls"]:
        dist = [np.hypot(x - bx, y - by) if k not in used else 1e9 for k, (x, y) in enumerate(found)]
        k = int(np.argmin(dist)) if dist else -1
        if k >= 0 and dist[k] <= r * .5:
            used.add(k)
            errors.append(dist[k])
    return len(frame["balls"]), len(errors), len(found) - len(used), errors


def eval_real(split: str, modes, workers: int = 14):
    """Whole real frames through detect() for each NN_BALLS mode: balls found
    (within half a radius of a labelled centre), false balls (not near any
    label; a ball the labels missed counts here too), frames with every ball
    right, distance to the label's box centre (frame px after REAL_SCALE)."""
    frames = json.loads((DATA / split / "frames.json").read_text())
    with Pool(workers) as pool:
        for mode in modes:
            res = pool.map(_eval_real_frame, [(mode, f) for f in frames], chunksize=2)
            total, hit, false = (sum(r[i] for r in res) for i in range(3))
            perfect = sum(1 for r in res if r[0] == r[1] and r[2] == 0)
            err = np.concatenate([np.array(r[3], float) for r in res]) if hit else np.zeros(1)
            print(f"{mode:12s} {split}: balls found {hit}/{total} ({hit / total:.1%}), false {false}, "
                  f"frames all right {perfect}/{len(frames)}, centre vs box median {np.median(err):.2f} "
                  f"p90 {np.percentile(err, 90):.2f} px", flush=True)


# --- training (PyTorch) ----------------------------------------------------

def build_model():
    import torch
    from torch import nn

    def conv(cin, cout, stride=1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))

    class Separable(nn.Module):
        """Depthwise 3x3 + pointwise 1x1 (MobileNet), residual when the shape stays."""

        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                                      nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                                      nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
            self.residual = stride == 1 and cin == cout
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            y = self.body(x)
            return self.act(y + x if self.residual else y)

    class BallNet(nn.Module):
        """Input /2 /4 /8 /16, back up to /8 (STRIDE) with a skip; heads: C
        heatmaps (sigmoid) and 2 offsets (sigmoid, 0..1 of a cell)."""

        def __init__(self, c=(16, 32, 64, 96)):
            super().__init__()
            self.s1 = conv(3, c[0], 2)                                                              # /2
            self.s2 = nn.Sequential(Separable(c[0], c[1], 2), Separable(c[1], c[1]))               # /4
            self.s3 = nn.Sequential(Separable(c[1], c[2], 2), Separable(c[2], c[2]), Separable(c[2], c[2]))  # /8
            self.s4 = nn.Sequential(Separable(c[2], c[3], 2), Separable(c[3], c[3]), Separable(c[3], c[3]))  # /16
            self.lateral = nn.Sequential(nn.Conv2d(c[3], c[2], 1, bias=False), nn.BatchNorm2d(c[2]))
            self.up = nn.Sequential(Separable(c[2], c[2]), Separable(c[2], c[2]))
            self.heat = nn.Conv2d(c[2], len(bn.CLASSES), 1)
            self.offset = nn.Conv2d(c[2], 2, 1)
            self.heat.bias.data.fill_(-4.0)

        def forward(self, x):
            f3 = self.s3(self.s2(self.s1(x)))
            f4 = self.s4(f3)
            u = nn.functional.interpolate(self.lateral(f4), scale_factor=2.0, mode="nearest")
            u = self.up(torch.relu(u + f3))
            return torch.sigmoid(self.heat(u)), torch.sigmoid(self.offset(u))

    return BallNet()


SIGMA = .7    # heatmap gaussian (output cells; a ball radius is ~1.25 cells)


def targets(labels, h, w):
    """Heatmaps (C + 1, h, w), offsets (2, h, w) and their mask (h, w).

    The last heatmap is every ball whatever its class: a real frame's labels
    (class -1) say only where balls are. Offsets are learnt from synthetic
    labels only (a hand-drawn box's centre is good to a pixel or two)."""
    heat = np.zeros((len(bn.CLASSES) + 1, h, w), np.float32)
    offset = np.zeros((2, h, w), np.float32)
    mask = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for u, v, c in labels:
        gu, gv = u / bn.STRIDE, v / bn.STRIDE
        jx, jy = int(np.floor(gu)), int(np.floor(gv))
        if not (0 <= jx < w and 0 <= jy < h):
            continue
        g = np.exp(-((xx - gu + .5) ** 2 + (yy - gv + .5) ** 2) / (2 * SIGMA ** 2))
        for k in ((c, -1) if c >= 0 else (-1,)):
            heat[k] = np.maximum(heat[k], g)
            heat[k, jy, jx] = 1.0
        if c >= 0:
            offset[:, jy, jx] = (gu - jx, gv - jy)
            mask[jy, jx] = 1.0
    return heat, offset, mask


class Frames:
    def __init__(self, split: str, augment: bool):
        self.dir = DATA / split
        self.rows = json.loads((self.dir / "labels.json").read_text())
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, labels, _ = self.rows[i]
        image = cv2.imread(str(self.dir / name), cv2.IMREAD_COLOR).astype(np.float32)
        real = any(c < 0 for _, _, c in labels)
        if self.augment:
            rng = np.random.default_rng()
            # Display tone / white balance a little off.
            image = image * rng.uniform(.9, 1.1) + rng.uniform(-12, 12)
            image = image * rng.uniform(.95, 1.05, 3)[None, None, :]
            if rng.random() < .3:
                image = image + rng.normal(0, rng.uniform(1, 4), image.shape)
            image = np.clip(image, 0, 255)
            if real and rng.random() < .5:
                # (Only class-less labels: a mirrored ball is still a ball.)
                image = image[:, ::-1]
                labels = [(bn.INPUT_W - 1 - u, v, c) for u, v, c in labels]
        x = np.ascontiguousarray(image.transpose(2, 0, 1)) / 255.0
        h, w = bn.INPUT_H // bn.STRIDE, bn.INPUT_W // bn.STRIDE
        heat, offset, mask = targets(labels, h, w)
        return x.astype(np.float32), heat, offset, mask, np.float32(real)


_FRAMES = {}


def _load_batch(job):
    augment, samples = job
    items = []
    for split, i in samples:
        key = (split, augment)
        if key not in _FRAMES:
            _FRAMES[key] = Frames(split, augment)
        items.append(_FRAMES[key][i])
    return tuple(np.stack([item[k] for item in items]) for k in range(5))


def batches(pool, split: str, augment: bool, batch: int, shuffle: bool, real=(), real_share: float = 0.0):
    """Batches as numpy arrays, decoded by a plain multiprocessing pool
    (PyTorch's DataLoader workers die on Windows with Python 3.15). With
    ``real`` splits, ``real_share`` of each batch is drawn from them (with
    repeats: there are far fewer real frames)."""
    n = len(Frames(split, False))
    order = np.random.permutation(n) if shuffle else np.arange(n)
    k = int(round(batch * real_share)) if real else 0
    step = batch - k
    stop = n - n % step if shuffle else n
    pool_real = [(s, i) for s in real for i in range(len(Frames(s, False)))]
    jobs = []
    for start in range(0, stop, step):
        samples = [(split, int(i)) for i in order[start:start + step]]
        if k:
            picks = np.random.randint(len(pool_real), size=k)
            samples += [pool_real[p] for p in picks]
        jobs.append((augment, samples))
    yield from pool.imap(_load_batch, jobs, chunksize=1)


def focal_loss(pred, gt):
    import torch
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def heat_loss(ph, heat, real):
    """Focal loss: synthetic frames per class, real frames (class-less labels)
    on the strongest class at each spot against the any-ball heatmap."""
    import torch
    loss = ph.new_zeros(())
    synth = real < .5
    if synth.any():
        loss = loss + focal_loss(ph[synth], heat[synth, :-1])
    if (~synth).any():
        loss = loss + focal_loss(ph[~synth].max(dim=1).values, heat[~synth, -1])
    return loss


def train(epochs: int, batch: int, lr: float, workers: int, real=(), real_share: float = .25,
          init: str | None = None, val_real=()):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    if init:
        model.load_state_dict(torch.load(init, map_location=device))
        print(f"starting from {init}")
    print(f"{sum(p.numel() for p in model.parameters())} weights on {device}; real {list(real)} "
          f"({real_share:.0%} of each batch)")
    step = batch - (int(round(batch * real_share)) if real else 0)
    steps = len(Frames("train", False)) // step
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps, pct_start=.1)
    best = float("inf")
    pool = Pool(workers)

    def to_device(arrays):
        return (torch.from_numpy(v).to(device, non_blocking=True) for v in arrays)

    def loss_of(arrays):
        x, heat, offset, mask, is_real = to_device(arrays)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            ph, po = model(x)
        ph, po = ph.float(), po.float()
        return heat_loss(ph, heat, is_real) + ((po - offset).abs() * mask[:, None]).sum() / mask.sum().clamp(min=1)

    for epoch in range(epochs):
        model.train()
        t, total = time.time(), 0.0
        for arrays in batches(pool, "train", True, batch, True, real, real_share):
            loss = loss_of(arrays)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        model.eval()
        vals = []
        with torch.no_grad():
            for split in ["val", *val_real]:
                losses = [float(loss_of(arrays)) for arrays in batches(pool, split, False, batch, False)]
                vals.append(float(np.mean(losses)))
        val = float(sum(vals))
        print(f"epoch {epoch + 1}/{epochs}: train {total / steps:.4f} val {' + '.join(f'{v:.4f}' for v in vals)} "
              f"({time.time() - t:.0f} s)", flush=True)
        # Fine-tuning on real frames keeps the last epoch: the synthetic
        # validation loss would favour forgetting them, and the real test
        # frames must not choose the model.
        if val < best or real:
            best = min(best, val)
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT)
    pool.close()
    print(f"best val {best:.4f} -> {CHECKPOINT}")


def export(path: Path):
    import torch
    model = build_model()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, bn.INPUT_H, bn.INPUT_W)
    torch.onnx.export(model, (dummy,), str(path), input_names=["image"], output_names=["heat", "offset"],
                      opset_version=17, dynamo=False)
    print(f"{path}: {path.stat().st_size / 1e6:.2f} MB")


def evaluate_proposals(split: str = "val", threshold: float = bn.THRESHOLD, limit: int = 2000):
    """Recall / false proposals of the exported model on rendered crops."""
    rows = json.loads((DATA / split / "labels.json").read_text())[:limit]
    found = missed = false = 0
    errs = []
    t = 0.0
    per_class_miss = [0] * len(bn.CLASSES)
    for name, labels, radius in rows:
        image = cv2.imread(str(DATA / split / name), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        props = bn.propose(image, threshold)
        t += time.perf_counter() - t0
        used = set()
        for u, v, c in labels:
            d = [np.hypot(p[0] - u, p[1] - v) for p in props]
            k = int(np.argmin(d)) if d else -1
            if k >= 0 and d[k] <= radius * .5 and k not in used:
                used.add(k)
                found += 1
                errs.append(d[k])
            else:
                missed += 1
                per_class_miss[c] += 1
        false += len(props) - len(used)
    n = found + missed
    print(f"{split}: recall {found}/{n} ({found / n:.4f}), false {false} in {len(rows)} frames, "
          f"centre error median {np.median(errs):.2f} p95 {np.percentile(errs, 95):.2f} px (input), "
          f"missed by class {dict(zip(bn.CLASSES, per_class_miss))}, {t / len(rows) * 1000:.1f} ms/frame")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--count", type=int, default=16000)
    r.add_argument("--val", type=int, default=1000)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=35)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--real", default="", help="real splits to mix in, comma-separated (e.g. real-8bd-train)")
    t.add_argument("--real-share", type=float, default=.25, help="share of each batch from the real splits")
    t.add_argument("--val-real", default="", help="real splits to report validation loss on")
    t.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    i = sub.add_parser("import-real")
    i.add_argument("root", help="COCO export folder with train/valid/test subfolders")
    i.add_argument("--name", required=True)
    i.add_argument("--ball-class", default="ball")
    er = sub.add_parser("eval-real")
    er.add_argument("--split", default="real-8bd-scenes-test")
    er.add_argument("--modes", default="off,only+trust")
    sp = sub.add_parser("split-real")
    sp.add_argument("--name", required=True)
    sp.add_argument("--test-share", type=float, default=.3)
    e = sub.add_parser("export")
    e.add_argument("--out", default=str(bn.MODEL_PATH))
    v = sub.add_parser("eval")
    v.add_argument("--threshold", type=float, default=bn.THRESHOLD)
    args = parser.parse_args()
    if args.cmd == "render":
        if args.count:
            render(args.count, "train", args.workers)
        if args.val:
            render(args.val, "val", args.workers)
    elif args.cmd == "train":
        split_list = lambda text: [p for p in text.split(",") if p]
        train(args.epochs, args.batch, args.lr, args.workers, split_list(args.real), args.real_share, args.init,
              split_list(args.val_real))
    elif args.cmd == "import-real":
        import_real(Path(args.root), args.name, args.ball_class)
    elif args.cmd == "eval-real":
        eval_real(args.split, args.modes.split(","))
    elif args.cmd == "split-real":
        split_real(args.name, args.test_share)
    elif args.cmd == "export":
        export(Path(args.out))
    elif args.cmd == "eval":
        evaluate_proposals(threshold=args.threshold)


if __name__ == "__main__":
    sys.exit(main())
