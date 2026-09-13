#!/usr/bin/env python3
"""Train the required rotation/thrust PPO model."""

from part2.arena import ControlStyle
from train_agent import main


if __name__ == "__main__":
    main(default_style=ControlStyle.ROTATION)
