"""Headless 8-ball test game: the program plays whole games against itself.

A test harness for the overlay, not a game for people (and not a copy of any
game): it racks the balls on a synthetic table (``fake_game``), breaks, plans
real shots (ghost-ball aiming at a pocket, or at least a legal hit), shoots
them through the repo's physics (``pool_physics``) under the 8-ball rules
(``pool_rules``), and renders every frame with exact truth. ``play_test.py``
runs the overlay's detector and live tracker on those frames.

    game = SimGame(seed=1, cloth="blue")
    while not game.over:
        plan = game.plan_shot()
        aim_frame, truth = game.render_aim(plan)     # the game's guide shown
        shot = game.shoot(plan)                      # ShotRecord: motion frames, facts
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

import fake_game as fg
from pool_physics import Ball as PhysicsBall, simulate_shot
from pool_rules import CUE, EIGHT, EightBallGame, ShotFacts

# The physics' speeds suit a table about this wide (px); shots are simulated
# at this scale and mapped back to the frame.
PHYSICS_WIDTH = 800.0
# Standard 8-ball rack order (apex first, row by row): the 8 in the middle of
# the third row, a solid and a stripe in the back corners.
RACK_ORDER = (1, 9, 2, 10, 8, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)


@dataclass
class ShotPlan:
    aim_degrees: float
    power: float                     # 0..1 (above 0.8 is break power)
    spin: tuple[float, float] = (0.0, 0.0)
    target: int | None = None        # the ball the plan means to hit first
    pocket: int | None = None        # index into the table's six pockets
    kind: str = "pot"                # "break", "pot", "safety"


@dataclass
class ShotRecord:
    plan: ShotPlan
    start_balls: dict[int, tuple[float, float]]
    start_cue: tuple[float, float]
    frames: list[tuple[tuple[float, float] | None, dict[int, tuple[float, float]]]] = field(default_factory=list)
    # Every 60 Hz physics frame (``frames`` keeps every ``every``-th).
    positions: list[tuple[tuple[float, float] | None, dict[int, tuple[float, float]]]] = field(default_factory=list)
    trails: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    # How each ball (0: the cue ball) is turned on each of ``positions``:
    # rolled along its path, so the number badge and stripe turn with it.
    turns: list[dict[int, tuple[float, ...]]] = field(default_factory=list)
    first_contact: int | None = None
    pocketed: tuple[int, ...] = ()
    message: str = ""


class SimGame:
    def __init__(self, seed: int, cloth: str | tuple = "blue", harsh: bool = False, aim_error: float = .25,
                 turned: bool | None = None):
        self.seed = seed
        self.aim_error = aim_error      # degrees (1 sigma) the planner misses its aim by
        self.rng = np.random.default_rng(seed)
        base = fg.random_scene(seed, "scatter", cloth)
        self.harsh = harsh
        # ``turned`` False: the balls are drawn flat and do not turn (the old
        # look); None: as ``fake_game.TURN_BALLS``.
        self.base = replace(base, balls={}, show_stick=True, show_guide=True,
                            turned=fg.TURN_BALLS if turned is None else turned)
        self.table = base.table
        self.r = base.radius
        self.pockets = fg.pocket_centres(self.table)
        self.rules = EightBallGame()
        self.balls: dict[int, tuple[float, float]] = {}
        self.cue: tuple[float, float] | None = None
        # How the balls are turned (number -> 3x3 rotation, 0: the cue ball);
        # one not in it yet is turned at random by its position.
        self.turns: dict[int, np.ndarray] = {}
        self.shots: list[ShotRecord] = []
        self._rack()

    # --- state ---------------------------------------------------------------

    @property
    def over(self) -> bool:
        return self.rules.winner is not None

    def _rack(self):
        x0, y0, w, h = self.table
        gap = 2.02 * self.r
        apex = (x0 + w * .74, y0 + h / 2)
        spots = []
        for row in range(5):
            for slot in range(row + 1):
                spots.append((apex[0] + row * gap * np.sqrt(3) / 2, apex[1] + (slot - row / 2) * gap))
        self.balls = {n: (float(sx), float(sy)) for n, (sx, sy) in zip(RACK_ORDER, spots)}
        self.cue = (x0 + w * .25, y0 + h / 2 + float(self.rng.uniform(-.15, .15)) * h)

    def _free(self, x, y, skip=None) -> bool:
        x0, y0, w, h = self.table
        if not (x0 + self.r <= x <= x0 + w - self.r and y0 + self.r <= y <= y0 + h - self.r):
            return False
        if min(np.hypot(x - px, y - py) for px, py in self.pockets) < self.r * 3:
            return False
        others = [p for n, p in self.balls.items() if n != skip]
        return all(np.hypot(x - px, y - py) >= 2.05 * self.r for px, py in others)

    def place_cue(self):
        """Ball in hand: the cue ball where a good shot is (else anywhere free)."""
        x0, y0, w, h = self.table
        best = None
        for _ in range(60):
            x = float(self.rng.uniform(x0 + self.r, x0 + w - self.r))
            y = float(self.rng.uniform(y0 + self.r, y0 + h - self.r))
            if self.rules.kitchen_only and x > x0 + w * .25:
                continue
            if not self._free(x, y):
                continue
            self.cue = (x, y)
            plan = self._best_pot()
            score = -1.0 if plan is None else plan[0]
            if best is None or score > best[0]:
                best = (score, (x, y))
        self.cue = best[1] if best else (x0 + w * .25, y0 + h / 2)

    # --- planning ------------------------------------------------------------

    def _clear(self, a, b, ignore=()) -> bool:
        """Can a ball travel from a to b without touching another ball?"""
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            return True
        ux, uy = dx / length, dy / length
        for n, (px, py) in self.balls.items():
            if n in ignore:
                continue
            t = (px - ax) * ux + (py - ay) * uy
            if 0 < t < length and abs((px - ax) * uy - (py - ay) * ux) < 2 * self.r:
                return False
        return True

    def _best_pot(self):
        """(score, plan) of the easiest legal pot, or None."""
        if self.cue is None:
            return None
        legal = self.rules.targets(self.rules.turn)
        best = None
        for n in legal:
            if n not in self.balls:
                continue
            bx, by = self.balls[n]
            for k, (px, py) in enumerate(self.pockets):
                dx, dy = px - bx, py - by
                dist = float(np.hypot(dx, dy))
                ux, uy = dx / dist, dy / dist
                gx, gy = bx - 2 * self.r * ux, by - 2 * self.r * uy
                ax, ay = gx - self.cue[0], gy - self.cue[1]
                reach = float(np.hypot(ax, ay))
                if reach < self.r:
                    continue
                cut = float(np.degrees(np.arccos(np.clip((ax * ux + ay * uy) / reach, -1, 1))))
                if cut > 72:
                    continue
                if not self._clear(self.cue, (gx, gy), ignore=(n,)) or not self._clear((bx, by), (px, py), ignore=(n,)):
                    continue
                score = np.cos(np.radians(cut)) ** 2 / (1 + (reach + dist) / (self.table[2] * .8))
                if best is None or score > best[0]:
                    power = self._power_for(reach, dist, cut)
                    best = (score, ShotPlan(float(np.degrees(np.arctan2(ay, ax))), power, (0.0, 0.0), n, k, "pot"))
        return best

    def _power_for(self, reach: float, dist: float, cut: float) -> float:
        """Power that sends the struck ball ~1.4x the way to the pocket, from
        the physics' own numbers (rolling drag 125 px/s^2 at PHYSICS_WIDTH,
        start speed 150 + 480 x power, ~0.97 of the speed along the line of
        centres passed on, a fifth lost to sliding first)."""
        scale = PHYSICS_WIDTH / self.table[2]
        drag = 125.0
        v_object = np.sqrt(2 * drag * dist * scale * 1.4)
        v_contact = v_object / (.965 * max(.25, np.cos(np.radians(cut))))
        v_start = np.sqrt(v_contact ** 2 + 2 * drag * reach * scale) * 1.2
        # Invert the physics' start speed: 150 + 480 p, plus break speed
        # (2200 ((p - 0.8) / 0.2)^2) above 80%.
        if v_start <= 150 + 480 * .8:
            return float(np.clip((v_start - 150) / 480, .2, .8))
        powers = np.linspace(.8, 1.0, 201)
        speeds = 150 + 480 * powers + 2200 * ((powers - .8) / .2) ** 2
        return float(powers[min(len(powers) - 1, int(np.searchsorted(speeds, v_start)))])

    def plan_shot(self) -> ShotPlan:
        if self.rules.ball_in_hand and self.rules.break_shot:
            self._rack_cue_for_break()
        elif self.rules.ball_in_hand:
            self.place_cue()
        if self.rules.break_shot:
            x0, y0, w, h = self.table
            apex = self.balls[RACK_ORDER[0]]
            aim = float(np.degrees(np.arctan2(apex[1] - self.cue[1], apex[0] - self.cue[0])))
            return ShotPlan(aim + float(self.rng.uniform(-.6, .6)), 1.0, (0.0, 0.0), RACK_ORDER[0], None, "break")
        found = self._best_pot()
        if found is not None:
            plan = found[1]
            plan.spin = (float(self.rng.uniform(-.3, .3)), float(self.rng.choice([-.5, 0, 0, .5])))
            self._tune_power(plan)
            # Aim is not perfect, like a player's.
            plan.aim_degrees += float(self.rng.normal(0, self.aim_error)) if self.aim_error else 0.0
            return plan
        # No pot: hit a legal ball full, softly (a safety).
        legal = [n for n in self.rules.targets(self.rules.turn) if n in self.balls]
        legal.sort(key=lambda n: np.hypot(self.balls[n][0] - self.cue[0], self.balls[n][1] - self.cue[1]))
        for n in legal:
            if self._clear(self.cue, self.balls[n], ignore=(n,)):
                bx, by = self.balls[n]
                return ShotPlan(float(np.degrees(np.arctan2(by - self.cue[1], bx - self.cue[0]))), .4,
                                (0.0, 0.0), n, None, "safety")
        n = legal[0] if legal else next(iter(self.balls))
        bx, by = self.balls[n]
        return ShotPlan(float(np.degrees(np.arctan2(by - self.cue[1], bx - self.cue[0]))), .45, (0.0, 0.0), n, None,
                        "safety")

    def _rack_cue_for_break(self):
        x0, y0, w, h = self.table
        self.cue = (x0 + w * .25, y0 + h / 2 + float(self.rng.uniform(-.2, .2)) * h)

    # --- rendering -----------------------------------------------------------

    def _turn_of(self, number: int, point) -> np.ndarray:
        turn = self.turns.get(number)
        return turn if turn is not None else fg.ball_turn(self.base, number, *point)

    def scene(self, balls=None, cue=None, aim_degrees=0.0, guide=False, stick=False,
              power: float | None = None, pull: float = 0.0, turns=None) -> fg.Scene:
        """The table as the mirror shows it; ``turns`` (as ``ShotRecord.turns``)
        defaults to how the balls are turned now."""
        balls = dict(self.balls if balls is None else balls)
        cue = self.cue if cue is None else cue
        if turns is None and self.base.turned:
            turns = {n: tuple(self._turn_of(n, p).ravel()) for n, p in [(0, cue), *balls.items()] if p is not None}
        s = replace(self.base, balls=balls, cue=cue, aim_degrees=float(aim_degrees),
                    show_guide=guide, show_stick=stick, power=power, stick_pull=float(pull), turns=turns)
        return fg.harsh(s) if self.harsh else s

    def render_aim(self, plan: ShotPlan, aim_degrees: float | None = None, pull: float = 1.0):
        """The aiming frame: stick, the game's guide and the power slider."""
        return fg.render(self.scene(aim_degrees=plan.aim_degrees if aim_degrees is None else aim_degrees,
                                    guide=True, stick=True, power=min(1.0, plan.power), pull=pull))

    def render_motion(self, cue, balls):
        """A frame while the balls roll: no stick, no guide."""
        return fg.render(self.scene(balls=balls, cue=cue if cue is not None else (-100.0, -100.0)))

    def shot_stream(self, plan: ShotPlan, fps: float = 60.0, t0: float = 0.0, aim: float = .4,
                    drag: float = .5, hold: float = .3, rest: float = 1.0, snap_frames: int = 0,
                    slider: bool = True):
        """Frames of one whole shot as the mirror shows them: aiming, dragging
        the power slider down to ``plan.power``, holding, the release (the
        slider snaps back over ``snap_frames`` frames while the stick strikes),
        the balls rolling, and the table at rest. Yields (time, frame, phase);
        the shot is played (``shoot``) between the aiming and the rolling
        frames, so ``self.shots[-1]`` is its record afterwards. ``slider``
        False hides the slider (as on the opponent's turn)."""
        step = 1.0 / fps
        t = t0
        power = min(1.0, plan.power)

        def aim_frame(p, pull):
            return fg.render(self.scene(aim_degrees=plan.aim_degrees, guide=True, stick=True,
                                        power=p if slider else None, pull=pull))[0]

        for _ in range(int(round(aim * fps))):
            yield t, aim_frame(0.0, .3), "aim"
            t += step
        n = max(1, int(round(drag * fps)))
        for k in range(1, n + 1):
            p = power * k / n
            yield t, aim_frame(p, .3 + 2.2 * p), "drag"
            t += step
        for _ in range(int(round(hold * fps))):
            yield t, aim_frame(power, .3 + 2.2 * power), "hold"
            t += step
        # The release: the slider snaps back while the stick strikes.
        snaps = max(1, snap_frames)
        for k in range(1, snaps + 1):
            p = power * (1 - k / snaps)
            pull = (.3 + 2.2 * power) * (1 - k / snaps)
            yield t, aim_frame(p, pull), "release"
            t += step
        record = self.shoot(plan, every=1)
        # The physics runs at 60 Hz; at a lower frame rate frames are skipped.
        ratio = 60.0 / fps
        k = 0.0
        while int(k) < len(record.positions):
            cue, balls = record.positions[int(k)]
            yield t, fg.render(self.scene(balls=balls, cue=cue if cue is not None else (-100.0, -100.0),
                                          power=0.0 if slider else None, turns=record.turns[int(k)]))[0], "roll"
            t += step
            k += ratio
        cue, balls = record.positions[-1] if record.positions else (self.cue, self.balls)
        still = fg.render(self.scene(balls=balls, cue=cue if cue is not None else (-100.0, -100.0),
                                     power=0.0 if slider else None,
                                     turns=record.turns[-1] if record.turns else None))[0]
        for _ in range(int(round(rest * fps))):
            yield t, still, "rest"
            t += step

    # --- shooting ------------------------------------------------------------

    def _simulate(self, plan: ShotPlan):
        """Run the physics for ``plan`` from the current table. Returns (result,
        60 Hz positions, ball numbers in the order they dropped)."""
        x0, y0, w, h = self.table
        scale = PHYSICS_WIDTH / w
        order = [CUE] + sorted(self.balls)
        points = [self.cue] + [self.balls[n] for n in order[1:]]
        physics = [PhysicsBall(n, (px - x0) * scale, (py - y0) * scale, self.r * scale)
                   for n, (px, py) in zip(order, points)]
        result = simulate_shot(physics, (0.0, 0.0, w * scale, h * scale), plan.aim_degrees, plan.power,
                               spin=plan.spin)

        def back(fx, fy):
            return (x0 + fx / scale, y0 + fy / scale)

        dropped: set[int] = set()
        order_dropped: list[int] = []
        positions = []
        for frame in result.frames:
            for n, f in zip(order, frame):
                if n in dropped:
                    continue
                # The physics drops a ball at a rail within two radii of a
                # pocket's mouth, the drawn table's own rule.
                if f.pocketed:
                    dropped.add(n)
                    order_dropped.append(n)
            cue = None if CUE in dropped else back(frame[0].x, frame[0].y)
            balls = {n: back(f.x, f.y) for n, f in zip(order[1:], frame[1:]) if n not in dropped}
            positions.append((cue, balls))
        result_trails = {n: [back(x, y) for x, y in trail] for n, trail in zip(order, result.trails)}
        return result, positions, tuple(order_dropped), result_trails

    def _tune_power(self, plan: ShotPlan) -> None:
        """Like a player who knows the table: the softest of a few powers that
        drops the planned ball (plus a margin), else the one closest to it."""
        hole = fg.pocket_holes(self.table)[plan.pocket]
        best = None
        for power in (.3, .4, .5, .6, .7, .8, .86, .92, 1.0):
            trial = ShotPlan(plan.aim_degrees, power, plan.spin, plan.target, plan.pocket, plan.kind)
            _, positions, dropped, _ = self._simulate(trial)
            if plan.target in dropped and CUE not in dropped:
                plan.power = min(1.0, power * 1.08)
                return
            track = [balls.get(plan.target) for _, balls in positions]
            gap = min((np.hypot(p[0] - hole[0], p[1] - hole[1]) for p in track if p is not None), default=1e9)
            if best is None or gap < best[0]:
                best = (gap, power)
        plan.power = best[1]

    def shoot(self, plan: ShotPlan, every: int = 2) -> ShotRecord:
        """Simulate the shot, apply the rules, keep every ``every``-th 60 Hz frame."""
        x0, y0, w, h = self.table
        result, positions, dropped, trails = self._simulate(plan)
        record = ShotRecord(plan, dict(self.balls), self.cue)
        record.positions = positions
        # Each ball rolls along its path (a pocketed one keeps its last turn).
        turning = ({n: self._turn_of(n, p) for n, p in [(0, self.cue), *self.balls.items()] if p is not None}
                   if self.base.turned else {})
        last_at = {0: self.cue, **self.balls}
        for cue, balls in positions:
            for n, p in [(0, cue), *balls.items()]:
                q = last_at.get(n)
                if p is not None and q is not None and n in turning:
                    turning[n] = fg.roll(turning[n], p[0] - q[0], p[1] - q[1], self.r)
                last_at[n] = p
            record.turns.append({n: tuple(t.ravel()) for n, t in turning.items()})
        self.turns = turning
        order_dropped = list(dropped)
        last = len(record.positions) - 1
        record.frames = [p for i, p in enumerate(record.positions) if i % every == 0 or i == last]
        record.trails = trails
        # The physics reports ball numbers (the ``number`` passed in).
        first = result.first_contact
        record.first_contact = first
        record.pocketed = tuple(order_dropped)
        outcome = self.rules.apply(ShotFacts(first, record.pocketed, result.rail_after_contact))
        record.message = outcome.message
        # The table after the shot.
        final_cue, final_balls = record.frames[-1] if record.frames else (self.cue, self.balls)
        self.balls = dict(final_balls)
        self.cue = final_cue
        if outcome.respot_eight and EIGHT not in self.balls:
            spot = (x0 + w * .74, y0 + h / 2)
            while not self._free(*spot):
                spot = (spot[0] + self.r * .5, spot[1])
            self.balls[EIGHT] = spot
        if self.cue is None:
            self.cue = (x0 + w * .25, y0 + h / 2)
        self.shots.append(record)
        return record
