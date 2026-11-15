"""The live overlay: transparent guide window, control panel, cue tracking and background analysis."""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRect, QSettings, QObject, Signal, QRunnable, QThreadPool
from PySide6.QtGui import QPainter, QGuiApplication
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QDialog,
                               QComboBox, QCheckBox, QFormLayout)

from capture import (GUIDE_COLOURS, GUIDE_THICKNESS, PAINT_SETTLE, DEFAULT_STYLE, GuideStyle, _debug_dump,
                     load_guide_style, logical_to_physical, paint_guide, physical_to_logical, remove_overlay,
                     render_guide_layer, save_guide_style, shown_layer)
from detector import Detection, find_table_bounds
from tracking import ShotTracker
from simulator_ui import ScreenshotTester, ShotSimulator


class CueTracker:
    """Keep cue-ball identity stable across brief visual occlusions."""
    def __init__(self):
        self.last: tuple[float, float, int] | None = None
        self.last_seen = 0.0
        self.geometry: tuple[int, int, int, int] | None = None

    def reset(self):
        self.last = None
        self.last_seen = 0.0
        self.geometry = None

    def update(self, detection: Detection, geometry: QRect) -> Detection:
        rect = geometry.getRect()
        if self.geometry != rect:
            self.reset()
            self.geometry = rect
        now = time.monotonic()
        if detection.cue:
            current = detection.cue
            if self.last:
                distance = float(np.hypot(current[0] - self.last[0], current[1] - self.last[1]))
                if distance <= max(12, current[2] * 2.0):
                    current = (round(self.last[0] * .3 + current[0] * .7),
                               round(self.last[1] * .3 + current[1] * .7), current[2])
            self.last = current
            self.last_seen = now
            detection.cue = current
            detection.balls = [(current[0], current[1], current[2], "cue") if kind == "cue"
                               else (bx, by, br, kind)
                               for bx, by, br, kind in detection.balls or []]
        elif self.last and now - self.last_seen <= .30:
            detection.cue = self.last
            detection.balls = [(bx, by, br, kind) for bx, by, br, kind in detection.balls or []
                               if kind != "cue"]
            detection.balls.append((*self.last, "cue"))
            detection.note = "Cue briefly obscured · holding its last position"
        else:
            self.last = None
        return detection


def _near(a, b, tolerance: float) -> bool:
    """Both missing, or both present with centres within ``tolerance`` px."""
    if a is None or b is None:
        return a is None and b is None
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) <= tolerance


def _angle_between(u, v) -> float:
    if u is None or v is None:
        return 0.0 if u is None and v is None else 180.0
    dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return float(np.degrees(np.arccos(dot)))


class GuideStabilizer:
    """Keep the drawn guide still while the shot on screen hasn't changed.

    Every analysed frame gives slightly different numbers (stream noise moves
    the fitted lines by a fraction of a pixel), and a frame can miss the aim
    line now and then. Redrawing for each of those makes the guide shimmer.
    This keeps showing the same guide while the new frame agrees with it
    within the detector's own noise, and bridges a short gap with no aim line.
    The shot is decided by the table, the cue ball, the ghost ball, the target
    and the aim. While those agree, the whole guide stays as drawn, including
    the rings around other balls and the short after-impact lines: those follow
    from the shot but jitter more (a boxed-in rack ball by up to ~10 px, the
    cue-ball line by a few degrees). A real change of the shot is shown at once.
    """
    # Frame-to-frame noise on noisy, JPEG-compressed synthetic frames of a
    # still shot (max): table 1 px, cue ball 2.8 px, ghost 1.4 px, target 2.8 px,
    # aim 0.1 deg. Tolerances sit just above that and below a visible change.
    CUE_PX = 3.0
    GHOST_PX = 2.0
    TARGET_PX = 3.0
    TABLE_PX = 2
    AIM_DEG = .3
    HOLD_SECONDS = .5

    def __init__(self):
        self.reset()

    def reset(self):
        self.shown: Detection | None = None
        self.confirmed_at = 0.0

    @staticmethod
    def has_shot(d: Detection | None) -> bool:
        return d is not None and d.cue is not None and bool(d.aim_segments)

    def same_shot(self, a: Detection, b: Detection) -> bool:
        if a.table is None or b.table is None or \
                max(abs(p - q) for p, q in zip(a.table, b.table)) > self.TABLE_PX:
            return False
        if not _near(a.cue, b.cue, self.CUE_PX) or not _near(a.ghost, b.ghost, self.GHOST_PX):
            return False
        if not _near(a.target, b.target, self.TARGET_PX):
            return False
        return _angle_between(a.aim_dir, b.aim_dir) <= self.AIM_DEG

    def update(self, detection: Detection, now: float | None = None) -> Detection:
        """The detection to draw: the one already shown if nothing really changed."""
        now = time.monotonic() if now is None else now
        shown = self.shown
        if self.has_shot(detection):
            if self.has_shot(shown) and self.same_shot(shown, detection):
                self.confirmed_at = now
                return shown
            self.shown, self.confirmed_at = detection, now
            return detection
        # No aim line in this frame: keep the last guide briefly, unless the
        # cue ball visibly moved (then the shot is over or being set up anew).
        if self.has_shot(shown) and now - self.confirmed_at <= self.HOLD_SECONDS and \
                (detection.cue is None or _near(detection.cue, shown.cue, shown.cue[2])):
            return shown
        self.shown = detection
        return detection


@dataclass
class GrabPlan:
    """What to capture and how to clean it, gathered on the window thread.

    ``paints`` are the guide drawings that may be on screen, newest first
    (empty when the overlay is hidden), as ``Overlay.paintEvent`` records them.
    """
    region: dict
    overlay: dict
    panel: dict | None
    paints: list


def grab_clean(capture, plan: GrabPlan) -> np.ndarray | None:
    """Capture ``plan.region`` (physical pixels) and undo the guide's own drawing.

    Safe to run off the window thread. Returns None when the capture shows
    none of the recent guide drawings exactly (the screen was mid-update), so
    the subtraction would not match.
    """
    region = plan.region
    shot = capture.grab(region)
    # Straight from mss's BGRA bytes (a third of the time of np.asarray(shot)).
    frame = cv2.cvtColor(np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4),
                         cv2.COLOR_BGRA2BGR)
    # The control panel is always on top and dark; where it overlaps the
    # capture (often right above the table) it can merge with a pocket.
    # Neutral grey is neither dark, felt nor white, so analysis ignores it.
    if plan.panel is not None:
        panel = plan.panel
        shadow = 8  # Windows 11 draws a soft drop shadow outside the frame.
        px0 = max(0, panel["left"] - shadow - region["left"])
        py0 = max(0, panel["top"] - shadow - region["top"])
        px1 = min(frame.shape[1], panel["left"] + panel["width"] + shadow - region["left"])
        py1 = min(frame.shape[0], panel["top"] + panel["height"] + shadow - region["top"])
        if px1 > px0 and py1 > py0:
            frame[py0:py1, px0:px1] = 128
    if not plan.paints:
        return frame
    overlay = plan.overlay
    size = (overlay["width"], overlay["height"])
    # Drawings made at another overlay size cannot be what is on screen now.
    paints = [paint for paint in plan.paints if (paint[3], paint[4]) == size]
    if not paints or paints[0] is not plan.paints[0]:
        return None
    layers = [_layer_for(paint) for paint in paints]
    # Place the layer where the overlay sits inside the captured region.
    x0 = overlay["left"] - region["left"]
    y0 = overlay["top"] - region["top"]
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1 = min(frame.shape[1], x0 + size[0])
    fy1 = min(frame.shape[0], y0 + size[1])
    if fx1 <= fx0 or fy1 <= fy0:
        return frame
    parts = [layer[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] for layer in layers]
    index = shown_layer(frame[fy0:fy1, fx0:fx1], parts)
    debug_dir = os.environ.get("POOL_OVERLAY_DEBUG_DIR")
    if debug_dir:
        # Opt-in diagnostics: what the screen showed, the guide layer that
        # is subtracted, and the frame the detector actually analyses.
        _debug_dump(debug_dir, "capture.png", frame)
        _debug_dump(debug_dir, "layer.png", layers[0 if index is None else index])
    if index is None:
        return None
    frame[fy0:fy1, fx0:fx1] = remove_overlay(frame[fy0:fy1, fx0:fx1], parts[index])
    if debug_dir:
        _debug_dump(debug_dir, "cleaned.png", frame)
    return frame


_layers_lock = threading.Lock()
_layers: dict[int, tuple] = {}


def _layer_for(paint: tuple) -> np.ndarray:
    """The rendered pixels of a recorded drawing, kept between frames.

    The same drawing stays on screen for many frames; drawing it again for
    each capture was a large part of the per-frame time.
    """
    with _layers_lock:
        cached = _layers.get(id(paint))
        if cached is not None and cached[0] is paint:
            return cached[1]
    d, adjusting, active, w, h, ui_scale, style = paint
    layer = render_guide_layer(d, w, h, adjusting, active, ui_scale, style)
    layer.setflags(write=False)
    with _layers_lock:
        _layers[id(paint)] = (paint, layer)
        while len(_layers) > 6:
            _layers.pop(next(iter(_layers)))
    return layer


def change_thumbnail(frame: np.ndarray) -> np.ndarray:
    """A quarter-size grey copy for telling whether the screen changed."""
    h, w = frame.shape[:2]
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (max(1, w // 4), max(1, h // 4)),
                      interpolation=cv2.INTER_AREA)


def screen_changed(thumbnail: np.ndarray, last: np.ndarray | None) -> bool:
    """True unless the frame is the same as the last analysed one.

    Looks for any small spot that changed, not an average over the whole
    frame: turning the aim a fraction of a degree only moves a thin line and
    the cue stick, and that must still be analysed. Stream noise averages out
    over each 4x4 block.
    """
    if last is None or last.shape != thumbnail.shape:
        return True
    return int(np.count_nonzero(cv2.absdiff(thumbnail, last) > 8)) >= 3


# One mss capture per worker thread, kept by the thread's id. (A
# threading.local does not work here: PySide clears it after every
# QRunnable, so a new capture was opened for each frame and never closed;
# its screen DC, memory DC and bitmap leaked until Windows' limit of 10,000
# GDI objects was reached after a few minutes and the guide froze.)
_captures: dict[int, object] = {}
_captures_lock = threading.Lock()


def _capture_for_this_thread():
    """An mss capture must stay on the thread that opened it: one per worker."""
    key = threading.get_ident()
    with _captures_lock:
        capture = _captures.get(key)
        if capture is None:
            capture = _captures[key] = mss.MSS()
    return capture


def close_captures():
    """Close every worker's capture (on exit)."""
    with _captures_lock:
        for capture in _captures.values():
            try:
                capture.close()
            except Exception:
                pass
        _captures.clear()


class AnalysisSignals(QObject):
    completed = Signal(object, object, object)
    skipped = Signal()
    failed = Signal(str)


class AnalysisTask(QRunnable):
    """Capture, clean and analyse one frame away from the window thread.

    The window thread only gathers the ``GrabPlan`` (a few rectangles and the
    recent drawings), so the controls and the guide stay smooth while frames
    are processed back to back.
    """
    def __init__(self, plan: GrabPlan, geometry: QRect, tracker: ShotTracker, last_thumbnail):
        super().__init__()
        self.plan = plan
        self.geometry = QRect(geometry)
        self.tracker = tracker
        self.last_thumbnail = last_thumbnail
        self.signals = AnalysisSignals()

    def run(self):
        try:
            shot = grab_clean(_capture_for_this_thread(), self.plan)
            if shot is None:
                self.signals.skipped.emit()
                return
            thumbnail = change_thumbnail(shot)
            if not screen_changed(thumbnail, self.last_thumbnail):
                self.signals.skipped.emit()
                return
            self.signals.completed.emit(self.tracker.detect(shot), self.geometry, thumbnail)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class Overlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.settings = QSettings("Codex", "PoolPracticeOverlay")
        # The earlier geometry was saved against a smaller AirPlay window. Use
        # a fresh key so the current full-screen mirror starts over the whole
        # table once, then remember the user's later adjustments.
        saved_geometry = self.settings.value("overlay/geometry_v2")
        if not saved_geometry or not self.restoreGeometry(saved_geometry):
            # Match the complete table visible in the current 1808x1080 mirror.
            self.setGeometry(308, 280, 1190, 660)
        self.active = False
        self.adjusting = True
        self.detection = Detection()
        self.guide_style = load_guide_style(self.settings)
        self._drag = None
        self._resize = False
        # What the last paintEvent drew, so analysis can subtract exactly that.
        self._painted = None
        self.painted_at = 0.0
        # Recent drawings, oldest first, each with the time it was replaced
        # (None for the current one): the compositor may still show a replaced one.
        self._paint_history: list[list] = []
        self.setWindowTitle("Pool Practice Guide")
        self.show()
        self._allow_capture()

    def _allow_capture(self):
        """Keep the guide in screenshots, recordings and screen shares.

        WDA_NONE explicitly; the analysis removes the guide's own pixels with
        ``remove_overlay`` instead of hiding the window from capture.
        """
        if sys.platform == "win32":
            try:
                ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x00000000)
            except Exception:
                pass

    def scale(self) -> float:
        screen = self.screen()
        return float(screen.devicePixelRatio()) if screen else 1.0

    def capture_rect(self) -> dict:
        """The overlay's area in physical screen pixels (what mss captures)."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        return logical_to_physical(self.geometry(), screen)

    def render_layer(self) -> np.ndarray | None:
        """Pixels the overlay currently shows on screen, or None if it is hidden."""
        if not self.isVisible() or self._painted is None:
            return None
        detection, adjusting, active, width, height, ui_scale, style = self._painted
        return render_guide_layer(detection, width, height, adjusting, active, ui_scale, style)

    def set_style(self, style: GuideStyle):
        """Change how the guide looks and remember it for the next start."""
        self.guide_style = style
        save_guide_style(self.settings, style)
        self.update()

    def set_active(self, active: bool):
        self.active = active
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self.adjusting)
        self.update()

    def set_adjusting(self, adjusting: bool):
        self.adjusting = adjusting
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not adjusting)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, not adjusting)
        self.show()
        if not adjusting:
            self.settings.setValue("overlay/geometry_v2", self.saveGeometry())
        self.update()

    def closeEvent(self, event):
        self.settings.setValue("overlay/geometry_v2", self.saveGeometry())
        super().closeEvent(event)

    def mousePressEvent(self, e):
        if not self.adjusting or e.button() != Qt.MouseButton.LeftButton:
            return
        self._resize = e.position().x() > self.width()-26 and e.position().y() > self.height()-26
        self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if not self.adjusting or self._drag is None:
            return
        p = e.globalPosition().toPoint()
        if self._resize:
            self.resize(max(300, p.x()-self.x()), max(200, p.y()-self.y()))
        else:
            self.move(p-self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None
        self._resize = False

    def paintEvent(self, _):
        # Detection coordinates are physical capture pixels; draw in that space
        # so the guide lines up at any Windows display scaling.
        scale = self.scale()
        width, height = round(self.width() * scale), round(self.height() * scale)
        p = QPainter(self)
        p.scale(1 / scale, 1 / scale)
        style = self.guide_style
        paint_guide(p, self.detection, width, height, self.adjusting, self.active, scale, style)
        p.end()
        # The style is part of what was painted: the subtracted layer must match it.
        painted = (self.detection, self.adjusting, self.active, width, height, scale, style)
        now = time.monotonic()
        if self._painted is None or painted != self._painted:
            if self._paint_history:
                self._paint_history[-1][1] = now
            self._paint_history = self._paint_history[-3:] + [[painted, None]]
        self._painted = painted
        self.painted_at = now

    # How long a replaced drawing may still be on screen (measured up to
    # ~70 ms; a little wider to be safe, each extra candidate costs ~3 ms).
    REPLACED_VISIBLE = .12

    def recent_paints(self) -> list:
        """Drawings that may be on screen right now, newest first."""
        if not self.isVisible() or self._painted is None:
            return []
        now = time.monotonic()
        paints = [self._painted]
        for painted, replaced_at in reversed(self._paint_history[:-1]):
            if replaced_at is None or now - replaced_at > self.REPLACED_VISIBLE:
                break
            paints.append(painted)
        return paints


def _swatch(name: str) -> QIcon:
    pixmap = QPixmap(14, 14)
    pixmap.fill(QColor(*GUIDE_COLOURS[name]))
    return QIcon(pixmap)


class GuideSettings(QDialog):
    """Colours, thickness and bounce for the guide; changes show at once and are saved."""

    def __init__(self, overlay, parent=None, on_change=None):
        super().__init__(parent)
        self.setWindowTitle("Guide style")
        self.overlay = overlay
        self.on_change = on_change
        self.colour_boxes = {}
        form = QFormLayout()
        for which, label in (("aim", "Aim line (cue ball → ghost)"), ("object", "Object ball path"),
                             ("cue", "Cue ball after impact")):
            box = QComboBox()
            for name in GUIDE_COLOURS:
                box.addItem(_swatch(name), name.capitalize(), name)
            box.currentIndexChanged.connect(self.apply)
            self.colour_boxes[which] = box
            form.addRow(label, box)
        self.thickness = QComboBox()
        for name, value in GUIDE_THICKNESS.items():
            self.thickness.addItem(name, value)
        self.thickness.currentIndexChanged.connect(self.apply)
        form.addRow("Line thickness", self.thickness)
        self.bounce = QCheckBox("Show the path after a cushion bounce")
        self.bounce.toggled.connect(self.apply)
        form.addRow("Length", self.bounce)
        reset = QPushButton("Reset to default"); reset.clicked.connect(lambda: self.show_style(DEFAULT_STYLE, apply=True))
        note = QLabel("Opacity stays fixed so the guide can be removed from screen captures. "
                      "White is not offered because the game's own lines are white.")
        note.setWordWrap(True); note.setStyleSheet("color:#667085")
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(reset); layout.addWidget(note)
        self.show_style(overlay.guide_style)

    def show_style(self, style: GuideStyle, apply: bool = False):
        widgets = list(self.colour_boxes.values()) + [self.thickness, self.bounce]
        for widget in widgets:
            widget.blockSignals(True)
        for which, box in self.colour_boxes.items():
            box.setCurrentIndex(box.findData(getattr(style, f"{which}_colour")))
        self.thickness.setCurrentIndex(self.thickness.findData(style.thickness))
        self.bounce.setChecked(style.show_bounce)
        for widget in widgets:
            widget.blockSignals(False)
        if apply:
            self.apply()

    def current_style(self) -> GuideStyle:
        return GuideStyle(self.colour_boxes["aim"].currentData(), self.colour_boxes["object"].currentData(),
                          self.colour_boxes["cue"].currentData(), float(self.thickness.currentData()),
                          self.bounce.isChecked())

    def apply(self, *_):
        self.overlay.set_style(self.current_style())
        if self.on_change:
            self.on_change()


class Control(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowTitle("Pool Practice Overlay")
        self.setMinimumWidth(350)
        self.overlay = Overlay()
        self.capture = mss.MSS()
        self.last_thumbnail = None
        self.analysis_pending = False
        self.analysis_task = None
        self.thread_pool = QThreadPool.globalInstance()
        self.tracker = ShotTracker()
        self.cue_tracker = CueTracker()
        self.stabilizer = GuideStabilizer()
        self._stabilized_rect = None
        self._analysed_at: list[float] = []
        self.label = QLabel("Position the guide over the complete pool table, then enable it.")
        self.label.setWordWrap(True)
        self.toggle = QPushButton("Enable guide")
        self.adjust = QPushButton("Finish positioning")
        self.fit = QPushButton("Fit table")
        self.simulator = QPushButton("Shot simulator")
        self.simulator.clicked.connect(self.open_simulator)
        self.simulator_window = None
        self.screenshot_test = QPushButton("Test a screenshot")
        self.screenshot_test.clicked.connect(self.open_screenshot_tester)
        self.screenshot_window = None
        self.synthetic = QPushButton("Synthetic test table")
        self.synthetic.setToolTip("A generated pool table with a white aim guide, for checking the overlay "
                                  "without AirPlay. Space or → shows the next shot.")
        self.synthetic.clicked.connect(self.open_synthetic_table)
        self.synthetic_window = None
        self.redetect = QPushButton("Re-detect")
        self.redetect.setEnabled(False)
        self.style_button = QPushButton("Guide style…")
        self.style_button.setToolTip("Line colours, thickness and length. Saved for the next start.")
        self.style_button.clicked.connect(self.open_guide_settings)
        self.style_window = None
        self.toggle.clicked.connect(self.toggle_guide)
        self.adjust.clicked.connect(self.toggle_adjust)
        self.fit.clicked.connect(self.fit_table)
        self.redetect.clicked.connect(self.redetect_now)
        row = QHBoxLayout(); row.addWidget(self.toggle); row.addWidget(self.adjust); row.addWidget(self.fit)
        layout = QVBoxLayout(); layout.addWidget(QLabel("Miniclip 8 Ball Pool · practice overlay")); layout.addLayout(row)
        layout.addWidget(self.simulator)
        layout.addWidget(self.screenshot_test)
        layout.addWidget(self.synthetic)
        tools = QHBoxLayout(); tools.addWidget(self.redetect); tools.addWidget(self.style_button)
        layout.addLayout(tools); layout.addWidget(self.label)
        note = QLabel("Visual estimates only. No gameplay inputs are sent.")
        note.setStyleSheet("color:#667085")
        layout.addWidget(note)
        root = QWidget(); root.setLayout(layout); self.setCentralWidget(root)
        # Frames are analysed back to back (each finished analysis starts the
        # next one); this timer only restarts the chain after a pause.
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.analyze)
        self.resize(420, 190)
        # Keep the controls in the black letterbox area above the table so
        # they cannot hide cue/object balls from the screen analysis.
        self.move(12, 18)
        QTimer.singleShot(700, self.fit_table)

    def open_simulator(self):
        if self.simulator_window is None:
            self.simulator_window=ShotSimulator()
            self.simulator_window.closed.connect(self.restore_overlay)
        self.suspend_overlay()
        self.simulator_window.show();self.simulator_window.raise_();self.simulator_window.activateWindow()

    def open_synthetic_table(self):
        """Open a generated table to try the live overlay without a phone.

        The overlay stays visible (that is the point): press Fit table, then
        Enable guide, and compare the guide with the white game-style lines.
        """
        import random
        import fake_game
        if self.synthetic_window is None:
            screen = QGuiApplication.primaryScreen().availableGeometry()
            width = int(min(1836, screen.width() * .85))
            self.synthetic_window = fake_game.make_table_window(seed=random.randint(0, 10 ** 6),
                                                                interval_ms=4000, width=width)
            self.synthetic_window.move(screen.x() + (screen.width() - width) // 2,
                                       screen.y() + max(0, (screen.height() - int(width * .6)) // 2))
        self.synthetic_window.show()
        self.synthetic_window.raise_()
        self.label.setText("Synthetic test table open · press Fit table, then Enable guide")

    def open_guide_settings(self):
        if self.style_window is None:
            self.style_window = GuideSettings(self.overlay, self, on_change=self._style_changed)
        self.style_window.show(); self.style_window.raise_(); self.style_window.activateWindow()

    def _style_changed(self):
        if self.screenshot_window is not None:
            self.screenshot_window.canvas.guide_style = self.overlay.guide_style
            self.screenshot_window.canvas.update()

    def open_screenshot_tester(self):
        if self.screenshot_window is None:
            self.screenshot_window = ScreenshotTester()
            self.screenshot_window.closed.connect(self.restore_overlay)
        # The tester draws exactly what the live guide would, in the same style.
        self.screenshot_window.canvas.guide_style = self.overlay.guide_style
        self.screenshot_window.canvas.update()
        self.suspend_overlay()
        self.screenshot_window.show(); self.screenshot_window.raise_(); self.screenshot_window.activateWindow()

    def suspend_overlay(self):
        self.overlay.hide()
        if self.overlay.active:
            self.timer.stop()

    def restore_overlay(self):
        self.overlay.show()
        self.overlay.raise_()
        if self.overlay.active:
            self.timer.start()
            self.analyze()

    def _grab_plan(self, region: dict) -> GrabPlan:
        panel = None
        if self.isVisible() and not self.isMinimized():
            panel = logical_to_physical(self.frameGeometry(), self.screen() or QGuiApplication.primaryScreen())
        return GrabPlan(region, self.overlay.capture_rect(), panel, self.overlay.recent_paints())

    def _grab_without_overlay(self, region: dict) -> np.ndarray | None:
        """Capture ``region`` (physical pixels) and undo the guide's own drawing.

        Returns None when the capture matches none of the overlay's recent
        drawings, so the subtraction would not match what the screen shows.
        """
        return grab_clean(self.capture, self._grab_plan(region))

    def _place_overlay_on_table(self, x: int, y: int, width: int, height: int):
        """Cover a table found at a physical desktop rect, plus a pocket margin.

        The margin keeps the six pockets inside each analysis frame, so every
        frame refits the table from its pockets instead of the felt colour.
        """
        margin = round(width * .06)
        desktop = self.capture.monitors[0]
        left = max(desktop["left"], x - margin)
        top = max(desktop["top"], y - margin)
        right = min(desktop["left"] + desktop["width"], x + width + margin)
        bottom = min(desktop["top"] + desktop["height"], y + height + margin)
        self.overlay.setGeometry(physical_to_logical(left, top, right - left, bottom - top))
        self.overlay.settings.setValue("overlay/geometry_v2", self.overlay.saveGeometry())
        self.overlay.update()

    def fit_table(self):
        """Find the table on the full desktop and resize the guide around it."""
        if time.monotonic() - self.overlay.painted_at < PAINT_SETTLE:
            QTimer.singleShot(100, self.fit_table)
            return
        try:
            desktop = self.capture.monitors[0]
            frame = self._grab_without_overlay(desktop)
            if frame is None:
                QTimer.singleShot(100, self.fit_table)
                return
            table = find_table_bounds(frame)
            if table is None:
                if self._fit_airplay_layout():
                    self.label.setText("Table fitted to AirPlay view · check alignment, then enable guide")
                else:
                    self.label.setText("Table not found on screen · move the mirrored pool table into view")
                return
            x, y, width, height = table
            self._place_overlay_on_table(desktop["left"] + x, desktop["top"] + y, width, height)
            self.label.setText("Table fitted · finish positioning, then enable the guide")
            if self.overlay.active:
                self.last_thumbnail = None
                QTimer.singleShot(150, self.analyze)
        except Exception as exc:
            if self._fit_airplay_layout():
                self.label.setText("Table fitted to AirPlay view · check alignment, then enable guide")
                if self.overlay.active:
                    QTimer.singleShot(150, self.analyze)
            else:
                self.label.setText(f"Could not fit table: {exc}")

    def _fit_airplay_layout(self) -> bool:
        """Fallback fit for the known UxPlay window when desktop capture fails."""
        if sys.platform != "win32":
            return False
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
            user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
            user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
            user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
            found = []
            enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def visit(hwnd, _):
                size = user32.GetWindowTextLengthW(hwnd)
                if size:
                    title = ctypes.create_unicode_buffer(size + 1)
                    user32.GetWindowTextW(hwnd, title, size + 1)
                    if title.value in ("AirPlay Video Stream", "uxplay-windows") and not user32.IsIconic(hwnd):
                        found.append(hwnd)
                return True
            user32.EnumWindows(enum_proc(visit), 0)
            if not found:
                return False
            valid=[]
            for hwnd in found:
                client=RECT()
                if user32.GetClientRect(hwnd,ctypes.byref(client)):
                    width,height=client.right-client.left,client.bottom-client.top
                    if width>=900 and height>=500:valid.append((width*height,hwnd,width,height))
            if not valid:return False
            _,hwnd,width,height=max(valid,key=lambda item:item[0])
            origin = POINT(0, 0)
            if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
                return False
            # UxPlay's Miniclip view places the felt inside fixed dark side
            # panels. These proportions follow the visible 2:1 playing area.
            x, y = origin.x + int(width*.205), origin.y + int(height*.225)
            table_w, table_h = int(width*.594), int(height*.642)
            # Win32 client rects are physical pixels, like the mss capture.
            self._place_overlay_on_table(x, y, table_w, table_h)
            return True
        except Exception:
            return False

    def toggle_guide(self):
        active = not self.overlay.active
        if active and self.overlay.adjusting:
            self.overlay.set_adjusting(False)
            self.adjust.setText("Adjust overlay")
        self.overlay.set_active(active)
        self.toggle.setText("Disable guide" if active else "Enable guide")
        self.redetect.setEnabled(active)
        if active:
            self.last_thumbnail = None
            self.analyze(); self.timer.start()
        else:
            self.timer.stop(); self.last_thumbnail = None; self.overlay.detection = Detection(note="Guide is off")
            self.cue_tracker.reset(); self.stabilizer.reset(); self.tracker = ShotTracker()
            self.overlay.update(); self.label.setText("Guide is off")

    def toggle_adjust(self):
        adjusting = not self.overlay.adjusting
        self.overlay.set_adjusting(adjusting)
        self.adjust.setText("Finish positioning" if adjusting else "Adjust overlay")

    def analyze(self):
        if not self.overlay.active:
            return
        if self.analysis_pending:
            return
        if self.overlay.adjusting:
            self.label.setText("Positioning mode · analysis paused")
            return
        # No waiting for the compositor after a repaint: the worker checks
        # which recent drawing the capture shows and subtracts that one.
        try:
            # Capture in physical pixels; detection coordinates stay in that
            # space and the overlay paints in it too (see Overlay.paintEvent).
            plan = self._grab_plan(self.overlay.capture_rect())
            self.analysis_pending = True
            self.analysis_task = AnalysisTask(plan, self.overlay.geometry(), self.tracker, self.last_thumbnail)
            self.analysis_task.signals.completed.connect(self._analysis_completed)
            self.analysis_task.signals.skipped.connect(self._analysis_skipped)
            self.analysis_task.signals.failed.connect(self._analysis_failed)
            self.thread_pool.start(self.analysis_task)
        except Exception as exc:
            self.analysis_pending = False
            self.label.setText(f"Screen capture failed: {exc}")

    def redetect_now(self):
        """Find the table and every ball again on the next frame."""
        self.tracker = ShotTracker()
        self.last_thumbnail = None
        self.analyze()

    # Pause before trying again when a frame was skipped (screen unchanged or
    # mid-update), so an idle table does not keep a CPU core busy.
    IDLE_RETRY_MS = 16

    def _next_frame(self, delay_ms: int = 0):
        if self.overlay.active and not self.overlay.adjusting:
            QTimer.singleShot(delay_ms, self.analyze)

    def _analysis_skipped(self):
        self.analysis_pending = False
        self.analysis_task = None
        self._next_frame(self.IDLE_RETRY_MS)

    def _analysis_completed(self, detection: Detection, geometry: QRect, thumbnail):
        self.analysis_pending = False
        self.analysis_task = None
        if not self.overlay.active or self.overlay.adjusting or self.overlay.geometry() != geometry:
            return
        self.last_thumbnail = thumbnail
        detection = self.cue_tracker.update(detection, geometry)
        if geometry.getRect() != self._stabilized_rect:
            # The overlay moved or was resized: old coordinates no longer apply.
            self.stabilizer.reset()
            if self._stabilized_rect is not None:
                self.tracker = ShotTracker()
            self._stabilized_rect = geometry.getRect()
        shown = self.stabilizer.update(detection)
        # Analysed frames in the last second, shown so the speed is visible.
        now = time.monotonic()
        self._analysed_at = [t for t in self._analysed_at if now - t < 1.0] + [now]
        self.label.setText(f"{shown.note} · {len(self._analysed_at)} updates/s")
        # Same shot as on screen: no repaint, so the guide stays perfectly still.
        if shown is not self.overlay.detection:
            self.overlay.detection = shown
            # Paint now rather than later, so the next capture's plan already
            # knows this drawing (the screen may show it by then).
            self.overlay.repaint()
        self._next_frame()

    def _analysis_failed(self, message: str):
        self.analysis_pending = False
        self.analysis_task = None
        if self.overlay.active:
            self.label.setText(f"Image analysis failed: {message}")
            self._next_frame(100)

    def closeEvent(self, e):
        self.timer.stop()
        self.overlay.close()
        if self.synthetic_window is not None:
            self.synthetic_window.close()
        self.capture.close()
        self.thread_pool.waitForDone(2000)
        close_captures()
        super().closeEvent(e)
