"""Guide drawing, overlay un-blending, and screen-coordinate helpers.

The guide is visible in screen captures, so every analysed frame contains it.
``remove_overlay`` subtracts the exact layer that was painted before detection."""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QImage, QGuiApplication

from detector import Detection


# Every guide stroke stays at or below this opacity. The overlay is visible in
# screenshots and screen shares, so its own drawing is in every analysis frame.
# Partial opacity lets ``remove_overlay`` recover the game pixels underneath
# exactly instead of having to guess them.
GUIDE_ALPHA = 150
# Seconds to wait after an overlay repaint before reading the screen, so the
# desktop compositor is showing that repaint (the layer we subtract).
PAINT_SETTLE = .07

# Colours the user can pick for the guide lines. White and grey are left out on
# purpose: the detector reads Miniclip's own white lines, so the guide must
# never look like them, even where un-blending is not perfect.
GUIDE_COLOURS = {
    "cyan": (45, 225, 255),
    "yellow": (255, 210, 55),
    "magenta": (255, 70, 180),
    "green": (60, 230, 90),
    "orange": (255, 140, 40),
    "red": (240, 60, 60),
    "blue": (70, 120, 255),
    "purple": (170, 90, 255),
}
GUIDE_THICKNESS = {"Thin": .75, "Normal": 1.0, "Thick": 1.5, "Extra thick": 2.0}


@dataclass(frozen=True)
class GuideStyle:
    """How the guide looks. Opacity is fixed (``GUIDE_ALPHA``) for un-blending."""
    aim_colour: str = "cyan"
    object_colour: str = "yellow"
    cue_colour: str = "magenta"
    thickness: float = 1.0
    show_bounce: bool = True

    def colour(self, which: str, alpha: int) -> QColor:
        name = getattr(self, f"{which}_colour")
        return QColor(*GUIDE_COLOURS.get(name, GUIDE_COLOURS[getattr(DEFAULT_STYLE, f"{which}_colour")]), alpha)


DEFAULT_STYLE = GuideStyle()


def load_guide_style(settings) -> GuideStyle:
    """Read the saved style from QSettings; unknown or broken values fall back to defaults."""
    def text(key, default):
        value = settings.value(f"guide/{key}", default)
        return value if isinstance(value, str) and value in GUIDE_COLOURS else default
    try:
        thickness = float(settings.value("guide/thickness", DEFAULT_STYLE.thickness))
    except (TypeError, ValueError):
        thickness = DEFAULT_STYLE.thickness
    if thickness not in GUIDE_THICKNESS.values():
        thickness = DEFAULT_STYLE.thickness
    bounce = settings.value("guide/show_bounce", DEFAULT_STYLE.show_bounce)
    if isinstance(bounce, str):
        bounce = bounce.lower() not in ("false", "0", "no")
    return GuideStyle(text("aim_colour", DEFAULT_STYLE.aim_colour),
                      text("object_colour", DEFAULT_STYLE.object_colour),
                      text("cue_colour", DEFAULT_STYLE.cue_colour),
                      thickness, bool(bounce))


def save_guide_style(settings, style: GuideStyle):
    settings.setValue("guide/aim_colour", style.aim_colour)
    settings.setValue("guide/object_colour", style.object_colour)
    settings.setValue("guide/cue_colour", style.cue_colour)
    settings.setValue("guide/thickness", style.thickness)
    settings.setValue("guide/show_bounce", style.show_bounce)


def paint_guide(p: QPainter, d: Detection, width: int, height: int, adjusting: bool,
                active: bool, ui_scale: float = 1.0, style: GuideStyle = DEFAULT_STYLE):
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
    thick = style.thickness
    for bx, by, br, kind in d.balls or []:
        color = style.colour("aim", a) if kind == "cue" else QColor(255, 255, 255, 110)
        p.setPen(pen(color, 2 * thick if kind == "cue" else 2))
        # Ring just outside the ball so the ball itself stays clear.
        p.drawEllipse(QPointF(bx, by), br + 3, br + 3)
    for segments, color, line, width_px in (
        (d.aim_segments, style.colour("aim", a), Qt.PenStyle.SolidLine, 3),
        (d.object_segments, style.colour("object", a), Qt.PenStyle.SolidLine, 3),
        (d.cue_segments, style.colour("cue", a), Qt.PenStyle.DashLine, 2.5),
    ):
        # Later segments are the path after a cushion bounce.
        for index, (start, end) in enumerate((segments or [])[:None if style.show_bounce else 1]):
            segment_color = QColor(color)
            if index:
                segment_color.setAlpha(round(a * .75))
            p.setPen(pen(segment_color, width_px * thick, Qt.PenStyle.DashLine if index else line))
            p.drawLine(QPoint(*start), QPoint(*end))
    for bx, by, br in d.blockers or []:
        p.setPen(pen(QColor(255, 140, 40, a), 2, Qt.PenStyle.DotLine))
        p.drawEllipse(QPointF(bx, by), br + 5, br + 5)
    if d.target:
        tx, ty, tr = d.target
        p.setPen(pen(style.colour("object", a), 3 * thick))
        p.drawEllipse(QPointF(tx, ty), tr + 5, tr + 5)


def render_guide_layer(d: Detection, width: int, height: int, adjusting: bool, active: bool,
                       ui_scale: float = 1.0, style: GuideStyle = DEFAULT_STYLE) -> np.ndarray:
    """The overlay's drawing as premultiplied BGRA pixels (what the compositor blends)."""
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    p = QPainter(image)
    paint_guide(p, d, width, height, adjusting, active, ui_scale, style)
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
