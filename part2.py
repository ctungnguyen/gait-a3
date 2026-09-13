#!/usr/bin/env python3
"""Open the Part II style -> Human/PPO visual menu directly."""

from __future__ import annotations

import argparse

from main import _visual_menu
from shared import theme_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=theme_names(), default="neon")
    args = parser.parse_args()
    _visual_menu(args.theme, initial_page="arena")


if __name__ == "__main__":
    main()
