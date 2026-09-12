#!/usr/bin/env python3
"""Generate report-ready curves from periodic PPO evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from arena import ControlStyle
from training.plots import plot_evaluation_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Block 6 PPO training curves")
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--style", choices=("both", "rotation", "direct"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    styles = (
        (ControlStyle.ROTATION, ControlStyle.DIRECT)
        if args.style == "both"
        else (ControlStyle(args.style),)
    )
    destination = plot_evaluation_curves(
        styles,
        run_name=args.run_name,
        output=args.output,
    )
    print(f"Saved training curves: {destination}")


if __name__ == "__main__":
    main()
