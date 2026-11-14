"""Shot simulator and screenshot tester windows (no AirPlay needed)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPoint, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QImage
from PySide6.QtWidgets import (QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QSlider,
                               QFileDialog, QCheckBox)
from pool_physics import Ball as PhysicsBall, simulate_shot
from pool_rules import EIGHT, EightBallGame, ShotFacts

from capture import paint_guide
from detector import Detection, collision_paths, detect, trace_bounces


def describe_spin(side: float, top: float) -> str:
    """Plain words for a spin setting, e.g. "40% draw · 20% right"."""
    parts = []
    if abs(top) >= .05:
        parts.append(f"{round(abs(top) * 100)}% {'follow' if top > 0 else 'draw'}")
    if abs(side) >= .05:
        parts.append(f"{round(abs(side) * 100)}% {'right' if side > 0 else 'left'}")
    return " · ".join(parts) if parts else "centre"


class SpinPad(QWidget):
    """The cue ball seen from the shooter: click where the tip should hit it."""
    changed = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setFixedSize(76, 76)
        self.side = 0.0
        self.top = 0.0
        self.setToolTip("Spin: click or drag where the cue tip hits the cue ball.\n"
                        "Up = follow, down = draw, left/right = english. Double-click to reset.")

    def _radius(self) -> float:
        return min(self.width(), self.height()) / 2 - 4

    def set_spin(self, side: float, top: float):
        length = float(np.hypot(side, top))
        if length > 1:
            side, top = side / length, top / length
        self.side, self.top = float(side), float(top)
        self.changed.emit(self.side, self.top)
        self.update()

    def _set_from(self, x: float, y: float):
        r = self._radius()
        self.set_spin((x - self.width() / 2) / r, (self.height() / 2 - y) / r)

    def mousePressEvent(self, e):
        self._set_from(e.position().x(), e.position().y())

    def mouseMoveEvent(self, e):
        self._set_from(e.position().x(), e.position().y())

    def mouseDoubleClickEvent(self, e):
        self.set_spin(0.0, 0.0)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, r = self.width() / 2, self.height() / 2, self._radius()
        p.setPen(QPen(QColor(120, 128, 140), 1.5)); p.setBrush(QColor(248, 248, 242))
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
        p.setPen(QPen(QColor(190, 196, 205), 1, Qt.PenStyle.DotLine))
        p.drawLine(QPoint(round(cx - r), round(cy)), QPoint(round(cx + r), round(cy)))
        p.drawLine(QPoint(round(cx), round(cy - r)), QPoint(round(cx), round(cy + r)))
        p.setPen(QPen(QColor(120, 20, 20), 1.5)); p.setBrush(QColor(220, 40, 40))
        p.drawEllipse(QRectF(cx + self.side * r - 5, cy - self.top * r - 5, 10, 10))
        p.end()


class SimCanvas(QWidget):
    # Emitted when a shot's animation has finished and the table is settled.
    shot_finished = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(860, 500)
        # None = free practice (drag anything); an EightBallGame = two-player rules.
        self.game: EightBallGame | None = None
        self._outcome = None
        self._finished = True
        self.cue = [0.0, 0.0, 12, QColor(248, 248, 242), 0]
        self.balls = []
        self.angle = 0
        self.power = 0.68
        # Where the cue tip hits the cue ball: (side, top), each -1..1.
        self.spin = (0.0, 0.0)
        self._preview_key = None
        self._preview_path = []
        self.dragging = None
        self.result = None
        self.frame_index = 0
        self.pocketed = set()
        self._last_table = None
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.advance)
        self.reset_rack()

    def table(self):
        margin = 38
        available_width = max(200, self.width() - margin * 2)
        available_height = max(100, self.height() - margin * 2)
        if available_width / available_height > 2.0:
            height = available_height
            width = height * 2.0
        else:
            width = available_width
            height = width / 2.0
        left = (self.width() - width) / 2
        top = (self.height() - height) / 2
        return (int(round(left)), int(round(top)), int(round(width)), int(round(height)))

    def reset_rack(self):
        left, top, width, height = self.table()
        radius = max(9, min(14, int(height * .031)))
        center_y = top + height / 2
        self.cue = [left + width * .22, center_y, radius, QColor(248, 248, 242), 0]
        palette = {
            1: (246, 190, 31), 2: (43, 108, 196), 3: (204, 54, 55),
            4: (122, 70, 167), 5: (230, 123, 36), 6: (42, 142, 68),
            7: (133, 55, 45), 8: (28, 30, 36), 9: (246, 190, 31),
            10: (43, 108, 196), 11: (204, 54, 55), 12: (122, 70, 167),
            13: (230, 123, 36), 14: (42, 142, 68), 15: (133, 55, 45),
        }
        # Standard five-row triangle with the 8 ball in the center.
        rack_order = (1, 9, 2, 10, 8, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)
        rack_tip_x = left + width * .72
        index = 0
        self.balls = []
        for row in range(5):
            x = rack_tip_x + row * radius * 1.78
            for slot in range(row + 1):
                number = rack_order[index]
                index += 1
                y = center_y + (slot - row / 2) * radius * 2.04
                self.balls.append([x, y, radius, QColor(*palette[number]), number])
        self.result = None
        self.frame_index = 0
        self.pocketed.clear()
        self.timer.stop()
        self._finished = True
        self._outcome = None
        if self.game is not None:
            self.game.reset()
        self._last_table = (left, top, width, height)
        self.update()

    def set_rules(self, enabled: bool):
        """Switch between free practice and a two-player 8-ball game."""
        self.game = EightBallGame() if enabled else None
        self.reset_rack()

    def head_string_x(self) -> float:
        left, _, width, _ = self.table()
        return left + width * .25

    def _overlaps(self, x: float, y: float, r: float, skip=None) -> bool:
        for ball in self.balls:
            if ball is skip or ball[4] in self.pocketed:
                continue
            if np.hypot(x - ball[0], y - ball[1]) < r + ball[2] + 1:
                return True
        if skip is not self.cue and np.hypot(x - self.cue[0], y - self.cue[1]) < r + self.cue[2] + 1:
            return True
        return False

    def _free_spot(self, x: float, y: float, r: float, skip) -> tuple[float, float]:
        """The nearest free place to (x, y) along the table's long axis."""
        left, top, width, height = self.table()
        for step in range(40):
            for direction in (1, -1):
                tx = x + direction * step * r * 1.1
                if left + r <= tx <= left + width - r and not self._overlaps(tx, y, r, skip):
                    return tx, y
        return x, y

    def _finish_shot(self):
        """Settle the table after a shot: re-spot balls and apply the rules."""
        self._finished = True
        left, top, width, height = self.table()
        if self.result is not None and self.result.frames and self.result.frames[-1][0].pocketed:
            self.cue[0], self.cue[1] = self._free_spot(left + width * .22, top + height / 2, self.cue[2], self.cue)
        if self._outcome is not None and self._outcome.respot_eight:
            eight = next(ball for ball in self.balls if ball[4] == EIGHT)
            self.pocketed.discard(EIGHT)
            eight[0], eight[1] = self._free_spot(left + width * .72, top + height / 2, eight[2], eight)
        if self.game is not None:
            # Back to aiming: the preview lines return for the next player.
            self.result = None
        self.update()
        self.shot_finished.emit()

    def _jump_to_end(self):
        if self.result is not None and self.result.frames and not self._finished:
            self.frame_index = len(self.result.frames) - 1
            self.advance()
            self.advance()

    def resizeEvent(self, event):
        new_table = self.table()
        old_table = self._last_table
        if self.game is not None:
            # A game keeps its state: finish a running shot before rescaling.
            self._jump_to_end()
        if old_table and old_table[2] > 0 and old_table[3] > 0:
            old_left, old_top, old_width, old_height = old_table
            new_left, new_top, new_width, new_height = new_table
            sx, sy = new_width / old_width, new_height / old_height
            for ball in [self.cue] + self.balls:
                ball[0] = new_left + (ball[0] - old_left) * sx
                ball[1] = new_top + (ball[1] - old_top) * sy
                ball[2] = max(8, round(ball[2] * min(sx, sy)))
            self.result = None
            if self.game is None:
                self.pocketed.clear()
            self.timer.stop()
        self._last_table = new_table
        super().resizeEvent(event)

    def first_hit(self):
        theta = np.radians(self.angle)
        direction = (float(np.cos(theta)), float(np.sin(theta)))
        cx, cy, cr = self.cue[:3]
        hits = []
        for i, ball in enumerate(self.balls):
            if ball[4] in self.pocketed:
                continue
            bx, by, br = ball[:3]
            dx, dy = bx - cx, by - cy
            along = dx * direction[0] + dy * direction[1]
            across = abs(dx * direction[1] - dy * direction[0])
            if along > cr and across < cr + br:
                lead = along - np.sqrt(max(0, (cr + br) ** 2 - across ** 2))
                if lead > 0:
                    hits.append((lead, i))
        if not hits:
            return direction, None, None
        _, index = min(hits)
        left, top, width, height = self.table()
        paths = collision_paths(tuple(map(int, self.cue[:3])),
                                tuple(map(int, self.balls[index][:3])), direction,
                                (left, top, width, height))
        return direction, index, paths

    def spin_cue_path(self, target_index):
        """Where the cue ball goes after hitting the target, with the chosen spin.

        Runs the physics for just the cue ball and the target ball (other balls
        are ignored, like the straight-line preview) and returns the cue ball's
        positions from the moment of contact until it stops.
        """
        cx, cy, cr = self.cue[:3]
        bx, by, br = self.balls[target_index][:3]
        left, top, width, height = self.table()
        key = (round(cx, 1), round(cy, 1), cr, round(bx, 1), round(by, 1), br,
               self.angle, round(self.power, 3), self.spin, (left, top, width, height))
        if key != self._preview_key:
            result = simulate_shot([PhysicsBall(0, cx, cy, cr), PhysicsBall(1, bx, by, br)],
                                   (left, top, left + width, top + height),
                                   self.angle, self.power, spin=self.spin)
            start = next((k for k, frame in enumerate(result.frames)
                          if np.hypot(frame[1].x - bx, frame[1].y - by) > .5), None)
            path = []
            if start is not None:
                for frame in result.frames[max(0, start - 1):]:
                    if frame[0].pocketed:
                        break
                    point = (round(frame[0].x), round(frame[0].y))
                    if not path or point != path[-1]:
                        path.append(point)
            self._preview_key, self._preview_path = key, path
        return self._preview_path

    def shoot(self):
        if self.timer.isActive() or (self.game is not None and self.game.winner is not None):
            return
        left, top, width, height = self.table()
        physics_balls = [PhysicsBall(0, self.cue[0], self.cue[1], self.cue[2])]
        physics_balls.extend(PhysicsBall(ball[4], ball[0], ball[1], ball[2])
                             for ball in self.balls if ball[4] not in self.pocketed)
        self._physics_ball_order = physics_balls[1:]
        self.pocketed_at_shot = set(self.pocketed)
        self.result = simulate_shot(physics_balls, (left, top, left + width, top + height),
                                    self.angle, self.power, spin=self.spin)
        self._outcome = None
        if self.game is not None:
            self._outcome = self.game.apply(ShotFacts(self.result.first_contact, self.result.pocket_order,
                                                      self.result.rail_after_contact))
        self.frame_index = 0
        self._finished = False
        if self.result.frames:
            self.timer.start()
        else:
            self._finish_shot()
        self.update()

    def advance(self):
        if self.result is None or not self.result.frames:
            self.timer.stop()
            return
        if self.frame_index >= len(self.result.frames):
            self.timer.stop()
            if not self._finished:
                self._finish_shot()
            self.update()
            return
        frame = self.result.frames[self.frame_index]
        self.cue[0], self.cue[1] = frame[0].x, frame[0].y
        self.pocketed = set(getattr(self, "pocketed_at_shot", set()))
        by_number = {ball[4]: ball for ball in self.balls}
        for i, physics_ball in enumerate(self._physics_ball_order, start=1):
            state = frame[i]
            target = by_number.get(physics_ball.number)
            if target is not None:
                target[0], target[1] = state.x, state.y
                if state.pocketed:
                    self.pocketed.add(physics_ball.number)
        self.frame_index += 1
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, top, width, height = self.table(); right, bottom = left + width, top + height
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(112, 61, 36))
        p.drawRoundedRect(left - 14, top - 14, width + 28, height + 28, 20, 20)
        p.setBrush(QColor(36, 138, 148)); p.drawRoundedRect(left, top, width, height, 12, 12)
        if self.game is not None and self.game.ball_in_hand and self.game.kitchen_only:
            # Head string: the cue ball must start behind it on the break.
            p.setPen(QPen(QColor(255, 255, 255, 110), 1.5, Qt.PenStyle.DashLine))
            hx = round(self.head_string_x())
            p.drawLine(QPoint(hx, top + 4), QPoint(hx, bottom - 4))
            p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 14, 20, 245))
        pocket_centers = [(left, top), ((left + right)//2, top), (right, top),
                          (left, bottom), ((left + right)//2, bottom), (right, bottom)]
        for px, py in pocket_centers:
            p.drawEllipse(QPoint(px, py), 16, 16)

        direction, target_index, paths = self.first_hit()
        ux, uy = direction; cx, cy, cue_radius = self.cue[:3]
        if self.result is not None:
            for i, trail in enumerate(self.result.trails):
                color = QColor(255, 255, 255, 95) if i == 0 else QColor(255, 213, 70, 100)
                if len(trail) > 1:
                    p.setPen(QPen(color, 1.5, Qt.PenStyle.DotLine))
                    for a, b in zip(trail, trail[1:]):
                        p.drawLine(QPoint(round(a[0]), round(a[1])), QPoint(round(b[0]), round(b[1])))
        elif paths:
            approach, cue_path, object_path, _ = paths
            # The cue ball's path after impact depends on spin, so it comes from
            # the physics (draw, follow, english off the cushion), not a 90° line.
            spun = self.spin_cue_path(target_index)
            if len(spun) > 1:
                cue_path = list(zip(spun, spun[1:]))
            for segments, color, style, stroke in (
                (approach, QColor(45, 225, 255, 225), Qt.PenStyle.SolidLine, 3),
                (object_path, QColor(255, 210, 55, 225), Qt.PenStyle.SolidLine, 3),
                (cue_path, QColor(255, 70, 180, 220), Qt.PenStyle.DashLine, 2),
            ):
                if segments is cue_path and len(spun) > 1:
                    p.setPen(QPen(color, stroke, Qt.PenStyle.DashLine))
                    for a, b in segments:
                        p.drawLine(QPoint(*a), QPoint(*b))
                    continue
                for index, (a, b) in enumerate(segments):
                    segment_color = QColor(color)
                    if index: segment_color.setAlpha(145)
                    p.setPen(QPen(segment_color, stroke, Qt.PenStyle.DashLine if index else style))
                    p.drawLine(QPoint(*a), QPoint(*b))
        else:
            bounds = (left + cue_radius, top + cue_radius, right - cue_radius, bottom - cue_radius)
            ray = trace_bounces((cx, cy), direction, bounds, cue_radius, max_bounces=1)
            p.setPen(QPen(QColor(255, 255, 255, 220), 2, Qt.PenStyle.DashLine))
            for a, b in ray: p.drawLine(QPoint(*a), QPoint(*b))

        by_number = {ball[4]: ball for ball in self.balls}
        if self.result is not None and self.result.frames:
            for i, physics_ball in enumerate(self._physics_ball_order, start=1):
                state = self.result.frames[min(self.frame_index, len(self.result.frames)-1)][i]
                if state.pocketed: self.pocketed.add(physics_ball.number)
        for ball in self.balls:
            bx, by, radius, color, number = ball
            if number in self.pocketed: continue
            p.setPen(QPen(QColor(245, 245, 245, 230), 1.5)); p.setBrush(color)
            p.drawEllipse(QPoint(round(bx), round(by)), radius, radius)
            if 9 <= number <= 15:
                p.setBrush(QColor(245, 245, 238)); p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(round(bx-radius), round(by-radius*.42), radius*2, round(radius*.84), radius*.3, radius*.3)
            # Miniclip-style number badges keep each ball identifiable even in a
            # compact window; the 8-ball gets the same white badge as the rest.
            p.setBrush(QColor(248, 248, 242)); p.setPen(QPen(QColor(25, 28, 32, 210), 0.8))
            badge_radius = max(5, round(radius * .48))
            p.drawEllipse(QPoint(round(bx), round(by)), badge_radius, badge_radius)
            p.setPen(QPen(QColor(12, 14, 17), 1))
            p.setFont(QFont("Arial", max(8, round(radius * .82)), QFont.Weight.Bold))
            p.drawText(QRectF(bx - badge_radius, by - badge_radius, badge_radius * 2, badge_radius * 2),
                       Qt.AlignmentFlag.AlignCenter, str(number))
            if target_index is not None and ball is self.balls[target_index] and self.result is None:
                p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor(255, 230, 85), 2.5))
                p.drawEllipse(QPoint(round(bx), round(by)), radius + 4, radius + 4)

        p.setPen(QPen(QColor(65, 245, 255), 2.5)); p.setBrush(QColor(248, 248, 242))
        p.drawEllipse(QPoint(round(cx), round(cy)), cue_radius, cue_radius)
        p.setPen(QPen(QColor(120, 30, 30), 2)); p.drawPoint(QPoint(round(cx), round(cy)))
        if self.result is not None and self.result.frames:
            moving = self.frame_index < len(self.result.frames)
            if not moving:
                p.setPen(QColor(235, 240, 245)); p.setFont(QFont("Arial", 10))
                p.drawText(left + 10, bottom - 10,
                           f"{self.result.ball_collisions} ball hits · {self.result.cushion_collisions} cushion hits · potted {len(self.result.pocketed)}")
        p.end()

    def mousePressEvent(self,e):
        x,y=e.position().x(),e.position().y()
        if self.game is not None:
            # In a game only the cue ball moves by hand, and only with ball in hand.
            if (self.game.ball_in_hand and self.game.winner is None and not self.timer.isActive()
                    and np.hypot(x-self.cue[0],y-self.cue[1])<25):
                self.dragging='cue'
            return
        if np.hypot(x-self.cue[0],y-self.cue[1])<25:self.dragging='cue';return
        for i,b in enumerate(self.balls):
            if np.hypot(x-b[0],y-b[1])<25:self.dragging=i;return

    def mouseMoveEvent(self,e):
        if self.dragging is None:return
        left,top,w,h=self.table();x=np.clip(e.position().x(),left+24,left+w-24);y=np.clip(e.position().y(),top+24,top+h-24)
        if self.game is not None:
            if self.game.kitchen_only:
                x=min(x,self.head_string_x())
            if not self._overlaps(x,y,self.cue[2],self.cue):
                self.cue[0],self.cue[1]=x,y
            self.result=None
            self.update()
            return
        if self.dragging=='cue':self.cue[0],self.cue[1]=x,y
        else:self.balls[self.dragging][0],self.balls[self.dragging][1]=x,y
        self.result=None;self.pocketed.clear();self.timer.stop()
        self.update()

    def mouseReleaseEvent(self,e):self.dragging=None


class ShotSimulator(QMainWindow):
    closed = Signal()
    def __init__(self):
        super().__init__();self.setWindowTitle('Pool Shot Simulator');self.resize(1000,700)
        self.canvas=SimCanvas();self.angle_text=QLabel('Aim: 0°');self.power_text=QLabel('Power: 68%')
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setRange(-180,180);self.slider.setValue(0)
        self.slider.valueChanged.connect(self.set_angle)
        self.power_slider=QSlider(Qt.Orientation.Horizontal);self.power_slider.setRange(10,100);self.power_slider.setValue(68)
        self.power_slider.valueChanged.connect(self.set_power)
        self.shoot_button=QPushButton('Shoot');self.shoot_button.clicked.connect(self.shoot)
        self.reset=QPushButton('Reset rack');self.reset.clicked.connect(self.reset_scene)
        self.spin_pad=SpinPad();self.spin_text=QLabel('Spin: centre');self.spin_text.setMinimumWidth(150)
        self.spin_pad.changed.connect(self.set_spin)
        self.rules_box=QCheckBox('8-ball game (2 players)');self.rules_box.toggled.connect(self.set_rules)
        self.game_text=QLabel('');self.game_text.setWordWrap(True);self.game_text.setStyleSheet('font-weight:600');self.game_text.hide()
        self.canvas.shot_finished.connect(self.shot_finished)
        controls=QHBoxLayout();controls.addWidget(self.angle_text);controls.addWidget(self.slider);controls.addWidget(self.power_text);controls.addWidget(self.power_slider);controls.addWidget(self.spin_pad);controls.addWidget(self.spin_text);controls.addWidget(self.shoot_button);controls.addWidget(self.reset)
        intro=QLabel('A full 15-ball practice table. Drag balls, set aim, power and spin, then shoot. '
                     'Click the white spin ball where the cue tip should hit (up = follow, down = draw, '
                     'sides = english; double-click to reset). The pink dashes show where the cue ball goes.')
        intro.setWordWrap(True)
        limits=QLabel('Approximate 2D physics. Spin covers follow, draw and english off cushions; not modelled: '
                      'swerve/masse, squirt, throw and jump. The 8-ball game uses simplified 8-ball rules: '
                      'no called pockets, push-outs or three-foul rule. '
                      'Use it to practise angles and cue-ball control, not to predict exact results.')
        limits.setWordWrap(True);limits.setStyleSheet('color:#667085')
        layout=QVBoxLayout();layout.addWidget(intro);layout.addWidget(limits);game_row=QHBoxLayout();game_row.addWidget(self.rules_box);game_row.addWidget(self.game_text,1)
        layout.addLayout(game_row);layout.addWidget(self.canvas);layout.addLayout(controls)
        root=QWidget();root.setLayout(layout);self.setCentralWidget(root)
    def set_angle(self,value):self.canvas.angle=value;self.angle_text.setText(f'Aim: {value}°');self.canvas.update()
    def set_power(self,value):self.canvas.power=value/100;self.power_text.setText(f'Power: {value}%'+(' (break)' if value>80 else ''));self.canvas.update()
    def set_spin(self,side,top):self.canvas.spin=(side,top);self.spin_text.setText(f'Spin: {describe_spin(side,top)}');self.canvas.update()
    def shoot(self):
        self.canvas.shoot()
        self.shoot_button.setEnabled(False)
        if self.canvas.result is not None:
            QTimer.singleShot(max(500, len(self.canvas.result.frames)*16 + 80),
                              lambda:self.shoot_button.setEnabled(self.canvas.game is None or self.canvas.game.winner is None))
    def reset_scene(self):
        self.canvas.reset_rack();self.slider.setValue(0);self.power_slider.setValue(68);self.spin_pad.set_spin(0.0,0.0);self.shoot_button.setEnabled(True)
        self.update_game_text()
    def set_rules(self,enabled):
        self.canvas.set_rules(enabled);self.reset_scene()
    def shot_finished(self):
        self.update_game_text()
        game=self.canvas.game
        self.shoot_button.setEnabled(game is None or game.winner is None)
    def update_game_text(self):
        game=self.canvas.game
        if game is None:
            self.game_text.hide();return
        lines=[game.last.message] if game.last else []
        lines.append(game.status())
        self.game_text.setText('\n'.join(lines));self.game_text.show()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class ScreenshotCanvas(QWidget):
    """Show detector output on a saved frame without needing a live mirror."""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 420)
        self.image = None
        self.detection = Detection()

    def load_image(self, path: str) -> Detection | None:
        frame = cv2.imread(path)
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.image = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        self.detection = detect(frame)
        self.update()
        return self.detection

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 28, 34))
        if self.image is None:
            p.setPen(QColor(225, 230, 235))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a saved pool screenshot to inspect the detected lines")
            p.end()
            return
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        ox, oy = (self.width() - iw * scale) / 2, (self.height() - ih * scale) / 2
        p.translate(ox, oy)
        p.scale(scale, scale)
        p.drawImage(0, 0, self.image)
        # Same drawing as the live overlay, with stroke widths kept constant on screen.
        paint_guide(p, self.detection, iw, ih, adjusting=False, active=True, ui_scale=1 / scale)
        p.end()


class ScreenshotTester(QMainWindow):
    closed = Signal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test a Pool Screenshot")
        self.resize(1100, 760)
        self.canvas = ScreenshotCanvas()
        self.status = QLabel("Cyan: cue approach to the game's ghost ball · yellow: object ball · "
                             "magenta dashes: cue ball after impact · orange dots: a ball in the way.")
        self.status.setWordWrap(True)
        self.official_button = QPushButton("Miniclip aim sample")
        self.official_button.clicked.connect(lambda: self.load_sample("miniclip_example_aiming.png", "Miniclip aiming-guide sample"))
        self.earlier_button = QPushButton("Earlier table sample")
        self.earlier_button.clicked.connect(lambda: self.load_sample("example_pool_screenshot.png", "Earlier table screenshot"))
        self.open_button = QPushButton("Open screenshot…")
        self.open_button.clicked.connect(self.open_image)
        bar = QHBoxLayout(); bar.addWidget(self.official_button); bar.addWidget(self.earlier_button); bar.addWidget(self.open_button); bar.addWidget(self.status, 1)
        layout = QVBoxLayout(); layout.addLayout(bar); layout.addWidget(self.canvas, 1)
        root = QWidget(); root.setLayout(layout); self.setCentralWidget(root)
        self.load_sample("miniclip_example_aiming.png", "Miniclip aiming-guide sample")

    def load_sample(self, filename: str, label: str):
        sample = Path(__file__).resolve().parent / "outputs" / filename
        detection = self.canvas.load_image(str(sample))
        if detection:
            self.status.setText(f"{label} · {detection.note}")
        else:
            self.status.setText("The built-in screenshot sample is missing. Open a saved screenshot instead.")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open a pool screenshot", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        detection = self.canvas.load_image(path)
        if detection is None:
            self.status.setText("That image could not be opened. Choose a PNG, JPG, or BMP screenshot.")
        else:
            self.status.setText(detection.note)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
