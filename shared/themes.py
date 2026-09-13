"""Colour themes used only by renderers; gameplay and RL state stay unchanged."""

from __future__ import annotations

from dataclasses import dataclass

Colour = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    name: str
    background: Colour
    background_alt: Colour
    grid: Colour
    edge: Colour
    panel: Colour
    text: Colour
    muted: Colour
    primary: Colour
    secondary: Colour
    success: Colour
    danger: Colour
    warning: Colour
    accent: Colour


THEMES: dict[str, Theme] = {
    "neon": Theme(
        "Neon",
        (12, 18, 30), (18, 27, 43), (29, 48, 67), (67, 106, 136),
        (9, 15, 26), (239, 246, 251), (153, 174, 192), (83, 215, 239),
        (102, 145, 255), (91, 231, 151), (242, 99, 113), (255, 198, 75),
        (166, 117, 255),
    ),
    "forest": Theme(
        "Forest",
        (13, 27, 24), (20, 40, 33), (37, 67, 55), (83, 132, 102),
        (11, 24, 21), (239, 247, 237), (157, 181, 164), (106, 219, 139),
        (85, 173, 191), (171, 225, 105), (231, 104, 85), (245, 194, 92),
        (192, 135, 220),
    ),
    "sunset": Theme(
        "Sunset",
        (31, 18, 34), (48, 26, 48), (78, 47, 72), (152, 82, 105),
        (28, 15, 31), (255, 244, 232), (201, 164, 166), (255, 132, 113),
        (238, 113, 155), (126, 224, 178), (242, 82, 92), (255, 194, 87),
        (169, 130, 255),
    ),
}


def get_theme(name: str | None) -> Theme:
    """Return a known palette and reject typos at the command-line boundary."""

    key = (name or "neon").lower()
    try:
        return THEMES[key]
    except KeyError as exc:
        choices = ", ".join(THEMES)
        raise ValueError(f"Unknown theme {name!r}; choose one of: {choices}") from exc


def theme_names() -> tuple[str, ...]:
    return tuple(THEMES)
