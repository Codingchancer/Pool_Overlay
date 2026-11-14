"""Small, deterministic 2D pool simulator for the offline shot lab.

The model uses equal-mass ball impulses, modest restitution, rolling drag,
linear cushion rebounds, and six pocket mouths. It intentionally omits spin.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, radians, sin
from typing import Sequence


@dataclass(frozen=True)
class Ball:
    number: int
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class BallFrame:
    x: float
    y: float
    pocketed: bool


@dataclass(frozen=True)
class SimulationResult:
    frames: tuple[tuple[BallFrame, ...], ...]
    trails: tuple[tuple[tuple[float, float], ...], ...]
    pocketed: frozenset[int]
    ball_collisions: int
    cushion_collisions: int
    duration: float


def simulate_shot(
    balls: Sequence[Ball],
    bounds: tuple[float, float, float, float],
    angle_degrees: float,
    power: float,
    *,
    dt: float = 1 / 240,
    max_seconds: float = 8.0,
    cloth_deceleration: float = 125.0,
    ball_restitution: float = 0.93,
    cushion_restitution: float = 0.84,
) -> SimulationResult:
    """Simulate a shot and return 60 Hz frames and trails for visualization.

    Bounds are the inside edges of the playing area. Angles use screen space:
    zero points right and positive values point down.
    """
    if not balls:
        return SimulationResult((), (), frozenset(), 0, 0, 0.0)
    if dt <= 0 or max_seconds <= 0:
        raise ValueError("dt and max_seconds must be positive")

    left, top, right, bottom = map(float, bounds)
    if right <= left or bottom <= top:
        raise ValueError("table bounds must have positive width and height")
    n = len(balls)
    positions = [[float(ball.x), float(ball.y)] for ball in balls]
    radii = [float(ball.radius) for ball in balls]
    pocketed = [False] * n
    velocities = [[0.0, 0.0] for _ in balls]
    theta = radians(angle_degrees)
    speed = 150.0 + 480.0 * min(1.0, max(0.0, power))
    velocities[0] = [cos(theta) * speed, sin(theta) * speed]

    pockets = ((left, top), ((left + right) / 2, top), (right, top),
               (left, bottom), ((left + right) / 2, bottom), (right, bottom))
    max_radius = max(radii)
    pocket_capture = max_radius * 1.8
    mouth_half_width = max_radius * 1.55
    frames: list[tuple[BallFrame, ...]] = []
    trails: list[list[tuple[float, float]]] = [[(x, y)] for x, y in positions]
    ball_hits = rail_hits = 0
    counted_pairs: set[tuple[int, int]] = set()
    steps = int(max_seconds / dt)
    sample_every = max(1, round(1 / (60 * dt)))
    elapsed = 0.0

    def pocket_at_edge(i: int, crossed_x: bool, crossed_y: bool) -> bool:
        x, y = positions[i]
        r = radii[i]
        candidates = []
        if crossed_x:
            candidates.extend(((left, top), (left, bottom),
                               ((left + right) / 2, top), ((left + right) / 2, bottom),
                               (right, top), (right, bottom)))
        if crossed_y:
            candidates.extend(((left, top), (right, top), (left, bottom), (right, bottom),
                               ((left + right) / 2, top), ((left + right) / 2, bottom)))
        for px, py in candidates:
            if hypot(x - px, y - py) <= pocket_capture:
                return True
        if crossed_x and (y < top + mouth_half_width or y > bottom - mouth_half_width):
            return True
        if crossed_y and min(abs(x - left), abs(x - (left + right) / 2), abs(x - right)) <= mouth_half_width:
            return True
        return False

    for step in range(steps):
        elapsed = (step + 1) * dt
        moving = False
        for i in range(n):
            if pocketed[i]:
                continue
            vx, vy = velocities[i]
            speed_now = hypot(vx, vy)
            if speed_now > 0:
                moving = True
                next_speed = max(0.0, speed_now - cloth_deceleration * dt)
                scale = next_speed / speed_now
                vx *= scale
                vy *= scale
                velocities[i] = [vx, vy]
                positions[i][0] += vx * dt
                positions[i][1] += vy * dt

            x, y = positions[i]
            r = radii[i]
            low_x, high_x = left + r, right - r
            low_y, high_y = top + r, bottom - r
            crossed_x = x < low_x or x > high_x
            crossed_y = y < low_y or y > high_y
            if (crossed_x or crossed_y) and pocket_at_edge(i, crossed_x, crossed_y):
                pocketed[i] = True
                velocities[i] = [0.0, 0.0]
                continue
            if x < low_x:
                positions[i][0] = low_x
                velocities[i][0] = abs(velocities[i][0]) * cushion_restitution
                rail_hits += 1
            elif x > high_x:
                positions[i][0] = high_x
                velocities[i][0] = -abs(velocities[i][0]) * cushion_restitution
                rail_hits += 1
            if y < low_y:
                positions[i][1] = low_y
                velocities[i][1] = abs(velocities[i][1]) * cushion_restitution
                rail_hits += 1
            elif y > high_y:
                positions[i][1] = high_y
                velocities[i][1] = -abs(velocities[i][1]) * cushion_restitution
                rail_hits += 1

        for i in range(n):
            if pocketed[i]:
                continue
            for j in range(i + 1, n):
                if pocketed[j]:
                    continue
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                distance = hypot(dx, dy)
                minimum = radii[i] + radii[j]
                if distance >= minimum:
                    continue
                if distance < 1e-9:
                    nx, ny, distance = 1.0, 0.0, 1e-9
                else:
                    nx, ny = dx / distance, dy / distance
                overlap = minimum - distance
                positions[i][0] -= nx * overlap * 0.5
                positions[i][1] -= ny * overlap * 0.5
                positions[j][0] += nx * overlap * 0.5
                positions[j][1] += ny * overlap * 0.5
                relative_normal = ((velocities[j][0] - velocities[i][0]) * nx
                                   + (velocities[j][1] - velocities[i][1]) * ny)
                if relative_normal < 0:
                    impulse = -(1 + ball_restitution) * relative_normal * 0.5
                    velocities[i][0] -= impulse * nx
                    velocities[i][1] -= impulse * ny
                    velocities[j][0] += impulse * nx
                    velocities[j][1] += impulse * ny
                    pair = (min(balls[i].number, balls[j].number), max(balls[i].number, balls[j].number))
                    if pair not in counted_pairs:
                        counted_pairs.add(pair)
                        ball_hits += 1

        if step % sample_every == 0 or step == steps - 1:
            frames.append(tuple(BallFrame(positions[i][0], positions[i][1], pocketed[i])
                                for i in range(n)))
            for i, (x, y) in enumerate(positions):
                if not pocketed[i] and hypot(x - trails[i][-1][0], y - trails[i][-1][1]) >= 3.0:
                    trails[i].append((x, y))

        if not moving and step > 0:
            break
        if all(pocketed[i] or hypot(*velocities[i]) < 1.0 for i in range(n)):
            break

    # Always include the resting/last state so animation ends exactly at rest.
    final_frame = tuple(BallFrame(positions[i][0], positions[i][1], pocketed[i]) for i in range(n))
    if not frames or frames[-1] != final_frame:
        frames.append(final_frame)
    return SimulationResult(tuple(frames), tuple(tuple(points) for points in trails),
                            frozenset(balls[i].number for i in range(n) if pocketed[i]),
                            ball_hits, rail_hits, elapsed)
