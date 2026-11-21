"""Neural-network ball proposals (proof of concept, off by default).

A small fully convolutional network, trained only on synthetic frames from
``fake_game.py`` (``train_ball_net.py``), looks at the table crop (cushion
nose rectangle, resized to INPUT_W x INPUT_H) and outputs one heatmap of ball
centres per class (cue ball, solid, 8 ball, stripe) at an eighth of that
size, plus a sub-cell offset. Its peaks are only PROPOSALS: ``find_balls``
judges and refines them exactly like circle-finding proposals (rim fit,
cloth test, stripe and white-cloth fits), so the aim precision still comes
from the geometric fits, not from the network.

Runs with OpenCV's own ``cv2.dnn`` (no extra package; onnxruntime has no
wheel for the app's Python). Pure numpy/OpenCV, no Qt.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(os.environ.get("POOL_OVERLAY_NN_MODEL", Path(__file__).with_name("models") / "balls.onnx"))
INPUT_W, INPUT_H = 640, 320
STRIDE = 8
CLASSES = ("cue", "solid", "eight", "stripe")
# Peak score a proposal needs.
THRESHOLD = .3


def class_of(number: int) -> int:
    """Class index of a ball number (0 the cue ball)."""
    return 0 if number == 0 else 2 if number == 8 else 1 if number < 8 else 3


def preprocess(roi: np.ndarray) -> np.ndarray:
    """Table crop (BGR uint8) -> network input (1, 3, INPUT_H, INPUT_W) float32, BGR / 255."""
    interp = cv2.INTER_AREA if roi.shape[1] > INPUT_W else cv2.INTER_LINEAR
    small = cv2.resize(roi, (INPUT_W, INPUT_H), interpolation=interp)
    return np.ascontiguousarray(small.transpose(2, 0, 1)[None], np.float32) / 255.0


def to_input(v: float, scale: float) -> float:
    """ROI pixel coordinate -> input pixel coordinate (pixel centres at
    whole numbers; ``scale`` = ROI size / input size, as cv2.resize maps)."""
    return (v + .5) / scale - .5


def to_roi(u: float, scale: float) -> float:
    return (u + .5) * scale - .5


def decode(heat: np.ndarray, offset: np.ndarray, threshold: float = THRESHOLD):
    """Peaks of (C, h, w) heatmaps (already sigmoid) with (2, h, w) offsets ->
    [(x, y, score, class)] in input pixels. One peak per spot: a 3x3 local
    maximum over all classes together (a ball has one class)."""
    best = heat.max(axis=0)
    cls = heat.argmax(axis=0)
    peak = (best >= threshold) & (best >= cv2.dilate(best, np.ones((3, 3), np.float32)))
    ys, xs = np.nonzero(peak)
    out = []
    for y, x in zip(ys, xs):
        # Cell (x, y) holds a centre at input pixel (x + offset) * STRIDE.
        px = (x + float(offset[0, y, x])) * STRIDE
        py = (y + float(offset[1, y, x])) * STRIDE
        out.append((px, py, float(best[y, x]), int(cls[y, x])))
    return out


_local = threading.local()


def _net():
    net = getattr(_local, "net", None)
    if net is None:
        net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _local.net = net
    return net


def available() -> bool:
    return MODEL_PATH.is_file()


def run(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw network output for a table crop: heatmaps (C, h, w), offsets (2, h, w)."""
    net = _net()
    net.setInput(preprocess(roi))
    heat, offset = net.forward(["heat", "offset"])
    return heat[0], offset[0]


def propose(roi: np.ndarray, threshold: float = THRESHOLD):
    """Ball proposals in ``roi`` pixels: [(x, y, score, class)], best first."""
    heat, offset = run(roi)
    sx, sy = roi.shape[1] / INPUT_W, roi.shape[0] / INPUT_H
    found = [(to_roi(x, sx), to_roi(y, sy), s, c) for x, y, s, c in decode(heat, offset, threshold)]
    return sorted(found, key=lambda p: -p[2])
