#!/usr/bin/env python3
"""Visual evaluation entry point for the rotation/thrust agent only."""

from part2.arena import ControlStyle
from evaluate_agent import default_model_path, evaluate_style


if __name__ == "__main__":
    evaluate_style(
        ControlStyle.ROTATION,
        default_model=default_model_path(ControlStyle.ROTATION),
    )
