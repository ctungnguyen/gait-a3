"""The seven assessment Gridworld layouts and presentation metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LevelDefinition:
    number: int
    name: str
    task: str
    layout: tuple[str, ...]
    description: str


def _level(number: int, name: str, task: str, rows: list[str], description: str) -> LevelDefinition:
    width = max(map(len, rows))
    return LevelDefinition(
        number=number,
        name=name,
        task=task,
        layout=tuple(row.ljust(width) for row in rows),
        description=description,
    )


LEVELS: dict[int, LevelDefinition] = {
    0: _level(
        0, "Apple Run", "Task 1 - Q-learning",
        [
            "S           ",
            "            ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
        ],
        "Only apples appear on the right; discounting favours the shortest collection route.",
    ),
    1: _level(
        1, "Hazard Choice", "Task 2 - SARSA",
        [
            "            ",
            "          A ",
            "   FFFFFFF  ",
            "            ",
            "            ",
            "            ",
            "S           ",
            "            ",
        ],
        "A short route passes close to fire; SARSA can learn a safer on-policy route.",
    ),
    2: _level(
        2, "Collection Order", "Task 3 - extended state",
        [
            "S    A      ",
            "            ",
            "    K       ",
            "         A  ",
            "            ",
            "        C   ",
            "   A        ",
            "            ",
        ],
        "Multiple apples plus key and chest require collectible state memory.",
    ),
    3: _level(
        3, "Rock and Fire", "Task 3 - extended state",
        [
            "S   R   A   ",
            "    R       ",
            "    R   FFF ",
            "  K R   F C ",
            "    R   F   ",
            "    RRRRR   ",
            "        A   ",
            "        A   ",
        ],
        "Rocks constrain movement while fire makes unsafe transitions terminal.",
    ),
    4: _level(
        4, "Moving Threat", "Task 4 - stochastic monsters",
        [
            "S           ",
            "     R      ",
            "     R  M   ",
            "     R      ",
            "     R      ",
            "            ",
            "        A   ",
            "        A   ",
        ],
        "A monster independently attempts a random legal move after every player action.",
    ),
    5: _level(
        5, "Monster Heist", "Task 4 - stochastic monsters",
        [
            "S     R     ",
            "      R   M ",
            "  K   R     ",
            "      R   A ",
            "          C ",
            "      RRRR  ",
            "        A   ",
            "   A        ",
        ],
        "The stochastic threat is combined with apples, a key, a chest, and rocks.",
    ),
    6: _level(
        6, "Sparse Maze", "Task 5 - intrinsic reward",
        [
            "S           ",
            "RRRRRRRRRR R",
            "            ",
            " RRRRRRRRRRR",
            "            ",
            "RRRRRRRRRR R",
            "            ",
            " RRRRRRRRRRR",
            "            ",
            "RRRRRRRRRR R",
            "K         CA",
        ],
        "A long alternating corridor delays all external reward, exposing the benefit of count-based exploration.",
    ),
}


def get_level(number: int) -> LevelDefinition:
    try:
        return LEVELS[number]
    except KeyError as exc:
        raise ValueError(f"Unknown level {number}; choose one of {sorted(LEVELS)}") from exc
