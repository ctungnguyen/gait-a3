"""Assessment-faithful Gridworld mechanics for Part I."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .levels import LevelDefinition

Position = tuple[int, int]
State = tuple[int, int, int, int, int, tuple[Position, ...]]

ACTIONS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_NAMES: tuple[str, ...] = ("UP", "RIGHT", "DOWN", "LEFT")
ALL_ACTIONS: tuple[int, ...] = tuple(range(len(ACTIONS)))


@dataclass(frozen=True)
class StepResult:
    next_state: State
    reward: float
    done: bool
    info: dict


class GridWorld:
    """A seedable stochastic environment with the exact assignment rewards.

    Static level geometry is not repeated in the state. Dynamic state contains
    the player, remaining-apple mask, key/chest flags, and all current monster
    positions. Including monster positions makes the tabular state Markov.
    """

    def __init__(
        self,
        level: LevelDefinition,
        monster_move_chance: float = 0.40,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= monster_move_chance <= 1.0:
            raise ValueError("monster_move_chance must be in [0, 1]")
        self.level = level
        self.layout = level.layout
        self.width = max(map(len, self.layout))
        self.height = len(self.layout)
        self.monster_move_chance = float(monster_move_chance)
        self.rng = random.Random(seed)

        self.rocks: set[Position] = set()
        self.fires: set[Position] = set()
        self.apples: list[Position] = []
        self.apple_index: dict[Position, int] = {}
        self.monster_starts: list[Position] = []
        self.key_pos: Position | None = None
        self.chest_pos: Position | None = None
        self.start: Position | None = None
        self._parse_layout()
        self.reset()

    @property
    def w(self) -> int:  # compatibility with the original student renderer
        return self.width

    @property
    def h(self) -> int:
        return self.height

    def _parse_layout(self) -> None:
        recognised = {" ", "S", "A", "R", "F", "K", "C", "M"}
        for y, row in enumerate(self.layout):
            for x, symbol in enumerate(row.ljust(self.width)):
                if symbol not in recognised:
                    raise ValueError(f"Unsupported level symbol {symbol!r} at {(x, y)}")
                position = (x, y)
                if symbol == "S":
                    if self.start is not None:
                        raise ValueError("A level must contain exactly one start")
                    self.start = position
                elif symbol == "A":
                    self.apple_index[position] = len(self.apples)
                    self.apples.append(position)
                elif symbol == "R":
                    self.rocks.add(position)
                elif symbol == "F":
                    self.fires.add(position)
                elif symbol == "K":
                    if self.key_pos is not None:
                        raise ValueError("A level can contain at most one key")
                    self.key_pos = position
                elif symbol == "C":
                    if self.chest_pos is not None:
                        raise ValueError("A level can contain at most one chest")
                    self.chest_pos = position
                elif symbol == "M":
                    self.monster_starts.append(position)
        if self.start is None:
            raise ValueError("A level must contain a start tile")
        if self.chest_pos is not None and self.key_pos is None:
            raise ValueError("A chest level must also contain a key")
        if not self.apples and self.chest_pos is None:
            raise ValueError("A level needs at least one collectible reward")

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self.rng.seed(seed)
        assert self.start is not None
        self.agent: Position = self.start
        self.alive = True
        self.has_key = 0
        self.chest_opened = 0
        self.step_count = 0
        self.apple_mask = (1 << len(self.apples)) - 1
        self.monsters: list[Position] = list(self.monster_starts)
        self.last_event = "reset"
        return self.encode_state()

    def encode_state(self) -> State:
        # Monster identity does not affect the transition, so canonical sorting
        # avoids wasting Q-table entries when two monsters exchange positions.
        return (
            self.agent[0],
            self.agent[1],
            self.apple_mask,
            self.has_key,
            self.chest_opened,
            tuple(sorted(self.monsters)),
        )

    def in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def legal_destination(self, position: Position) -> bool:
        return self.in_bounds(position) and position not in self.rocks

    def try_move(self, position: Position, action: int) -> Position:
        if action not in ALL_ACTIONS:
            raise ValueError(f"action must be one of {ALL_ACTIONS}")
        dx, dy = ACTIONS[action]
        candidate = (position[0] + dx, position[1] + dy)
        return candidate if self.legal_destination(candidate) else position

    def _move_monsters(self) -> tuple[int, tuple[tuple[Position, Position], ...]]:
        """Give each monster one independent 40% move opportunity."""

        occupied = set(self.monsters)
        positions: list[Position] = []
        transitions: list[tuple[Position, Position]] = []
        for origin in self.monsters:
            occupied.discard(origin)
            destination = origin
            if self.rng.random() < self.monster_move_chance:
                candidates = []
                for dx, dy in ACTIONS:
                    candidate = (origin[0] + dx, origin[1] + dy)
                    if self.legal_destination(candidate) and candidate not in occupied:
                        candidates.append(candidate)
                if candidates:
                    destination = self.rng.choice(candidates)
            occupied.add(destination)
            positions.append(destination)
            if destination != origin:
                transitions.append((origin, destination))
        self.monsters = positions
        return len(transitions), tuple(transitions)

    def _objectives_complete(self) -> bool:
        apples_complete = self.apple_mask == 0
        chest_complete = self.chest_pos is None or self.chest_opened == 1
        return apples_complete and chest_complete

    def remaining_apples(self) -> int:
        return self.apple_mask.bit_count()

    def step(self, action: int) -> StepResult:
        """Apply one player action, then the stochastic monster transition."""

        if not self.alive:
            raise RuntimeError("Cannot step a terminal Gridworld; call reset()")
        self.step_count += 1
        previous_agent = self.agent
        self.agent = self.try_move(self.agent, int(action))
        blocked = self.agent == previous_agent
        reward = 0.0
        events: list[str] = []

        # Entering a hazard or current monster is immediate death.
        if self.agent in self.fires or self.agent in self.monsters:
            self.alive = False
            self.last_event = "death"
            return StepResult(
                self.encode_state(), 0.0, True,
                {"event": "death", "events": ("death",), "blocked": blocked, "monster_moves": 0},
            )

        apple_index = self.apple_index.get(self.agent)
        if apple_index is not None and (self.apple_mask >> apple_index) & 1:
            self.apple_mask &= ~(1 << apple_index)
            reward += 1.0  # fixed assignment reward
            events.append("apple_collected")

        if self.key_pos == self.agent and not self.has_key:
            self.has_key = 1
            events.append("key_collected")  # deliberately +0 reward

        if self.chest_pos == self.agent and self.has_key and not self.chest_opened:
            self.chest_opened = 1
            reward += 2.0  # fixed assignment reward
            events.append("chest_opened")

        if self._objectives_complete():
            events.append("win")
            self.last_event = "win"
            return StepResult(
                self.encode_state(), reward, True,
                {"event": "win", "events": tuple(events), "blocked": blocked, "monster_moves": 0},
            )

        monster_moves = 0
        monster_transitions: tuple[tuple[Position, Position], ...] = ()
        if self.monsters:
            monster_moves, monster_transitions = self._move_monsters()
            if monster_moves:
                events.append("monster_moved")
            if self.agent in self.monsters:
                self.alive = False
                events.append("death")
                self.last_event = "death"
                return StepResult(
                    self.encode_state(), reward, True,
                    {
                        "event": "death",
                        "events": tuple(events),
                        "blocked": blocked,
                        "monster_moves": monster_moves,
                        "monster_transitions": monster_transitions,
                    },
                )

        self.last_event = events[-1] if events else ("blocked" if blocked else "moved")
        return StepResult(
            self.encode_state(), reward, False,
            {
                "event": self.last_event,
                "events": tuple(events),
                "blocked": blocked,
                "monster_moves": monster_moves,
                "monster_transitions": monster_transitions,
            },
        )

    def traversable_positions(self) -> Iterable[Position]:
        for y in range(self.height):
            for x in range(self.width):
                position = (x, y)
                if position not in self.rocks:
                    yield position
