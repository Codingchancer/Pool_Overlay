"""8-ball rules for the shot simulator.

A simplified version of the common (WPA-style) rules that mobile 8-ball games
also use:

- Break: the cue ball starts behind the head string. Balls potted on the break
  keep the turn but the table stays open. An 8 potted on the break is spotted
  back. A scratch on the break is a foul.
- Open table: the first ball potted legally after the break gives the shooter
  that group (solids 1-7 or stripes 9-15). Hitting the 8 first is a foul.
- The shooter keeps the turn after legally potting one of their own balls
  (any ball while the table is open).
- Fouls: cue ball potted, no ball hit, the wrong ball hit first, or no ball
  potted and no ball touching a cushion after the contact. The opponent then
  gets ball in hand anywhere on the table.
- The 8: potting it after clearing your group, without a foul, wins. Potting it
  earlier, or with a foul, loses.

Not included: calling shots and pockets, push-outs, the three-foul rule,
re-racks, time limits.
"""
from __future__ import annotations

from dataclasses import dataclass

SOLIDS = frozenset(range(1, 8))
STRIPES = frozenset(range(9, 16))
EIGHT = 8
CUE = 0
GROUP_NAMES = {SOLIDS: "solids", STRIPES: "stripes"}


@dataclass(frozen=True)
class ShotFacts:
    """What happened in one shot, as reported by the physics."""
    first_contact: int | None
    pocketed: tuple[int, ...]  # ball numbers in the order they dropped; 0 = cue ball
    rail_after_contact: bool


@dataclass(frozen=True)
class ShotOutcome:
    fouls: tuple[str, ...]
    message: str
    keeps_turn: bool
    respot_eight: bool
    winner: int | None


class EightBallGame:
    """Turn, groups, fouls and ball in hand for a two-player 8-ball game."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.turn = 0
        self.groups: list[frozenset[int] | None] = [None, None]
        self.on_table: set[int] = set(range(1, 16))
        self.break_shot = True
        self.ball_in_hand = True
        self.kitchen_only = True  # break: cue ball behind the head string
        self.winner: int | None = None
        self.last: ShotOutcome | None = None

    @staticmethod
    def player_name(player: int) -> str:
        return f"Player {player + 1}"

    def remaining(self, player: int) -> set[int]:
        group = self.groups[player]
        return set(group & self.on_table) if group else set()

    def targets(self, player: int) -> set[int]:
        """Balls the player may legally hit first."""
        group = self.groups[player]
        if group is None:
            return self.on_table - {EIGHT}
        left = group & self.on_table
        return set(left) if left else {EIGHT}

    def apply(self, facts: ShotFacts) -> ShotOutcome:
        if self.winner is not None:
            raise ValueError("the game is over; reset it to play again")
        player, other = self.turn, 1 - self.turn
        name, other_name = self.player_name(player), self.player_name(other)
        was_break = self.break_shot
        targets = self.targets(player)
        potted = [ball for ball in facts.pocketed if ball != CUE]

        fouls: list[str] = []
        if CUE in facts.pocketed:
            fouls.append("cue ball potted")
        if facts.first_contact is None:
            fouls.append("no ball hit")
        elif not was_break and facts.first_contact not in targets:
            if self.groups[player] is None and facts.first_contact == EIGHT:
                fouls.append("hit the 8 first on an open table")
            else:
                fouls.append(f"hit the {facts.first_contact} first")
        if (not was_break and facts.first_contact is not None and not potted
                and not facts.rail_after_contact):
            fouls.append("no ball reached a cushion")

        self.on_table.difference_update(potted)
        self.break_shot = False

        respot = False
        if EIGHT in potted:
            if was_break:
                self.on_table.add(EIGHT)
                respot = True
            else:
                won = targets == {EIGHT} and not fouls
                self.winner = player if won else other
                if won:
                    message = f"{name} potted the 8 and wins!"
                elif fouls:
                    message = f"{name} potted the 8 with a foul ({', '.join(fouls)}). {other_name} wins."
                else:
                    message = f"{name} potted the 8 too early. {other_name} wins."
                self.ball_in_hand = self.kitchen_only = False
                self.last = ShotOutcome(tuple(fouls), message, False, False, self.winner)
                return self.last

        # Groups are decided by the first ball potted legally after the break.
        if not was_break and not fouls and self.groups[player] is None:
            first = next((ball for ball in potted if ball != EIGHT), None)
            if first is not None:
                mine = SOLIDS if first in SOLIDS else STRIPES
                self.groups[player] = mine
                self.groups[other] = STRIPES if mine is SOLIDS else SOLIDS

        group = self.groups[player]
        scoring = [ball for ball in potted if ball != EIGHT and (group is None or ball in group)]
        keeps_turn = not fouls and bool(scoring)

        if fouls:
            self.turn = other
            self.ball_in_hand, self.kitchen_only = True, False
            message = f"Foul: {', '.join(fouls)}. {other_name} has ball in hand."
        elif keeps_turn:
            self.ball_in_hand = self.kitchen_only = False
            message = f"{name} potted {', '.join(map(str, scoring))} and shoots again."
        else:
            self.turn = other
            self.ball_in_hand = self.kitchen_only = False
            message = f"No ball potted. {other_name} to shoot."
        if respot:
            message += " The 8 was spotted back."
        if keeps_turn and group is not None and not self.remaining(player):
            message += f" {name} now plays the 8."
        self.last = ShotOutcome(tuple(fouls), message, keeps_turn, respot, None)
        return self.last

    def status(self) -> str:
        """One line for the window: whose turn, their group, ball in hand."""
        if self.winner is not None:
            return f"{self.player_name(self.winner)} wins. Press Reset rack for a new game."
        player = self.turn
        name = self.player_name(player)
        if self.break_shot:
            return f"{name} to break. Place the cue ball behind the line, then shoot."
        group = self.groups[player]
        if group is None:
            aim = "open table"
        elif self.remaining(player):
            aim = f"{GROUP_NAMES[group]} ({len(self.remaining(player))} left)"
        else:
            aim = "on the 8"
        hand = " · ball in hand: drag the cue ball anywhere" if self.ball_in_hand else ""
        return f"{name} to shoot · {aim}{hand}"
