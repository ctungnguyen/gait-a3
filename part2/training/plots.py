"""Create report-ready learning curves from SB3 EvalCallback logs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..arena import ControlStyle
from .artifacts import PROJECT_ROOT


def plot_evaluation_curves(
    styles: Iterable[ControlStyle | str] = (ControlStyle.ROTATION, ControlStyle.DIRECT),
    *,
    run_name: str = "final",
    output: str | Path | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required to create training curves") from exc

    datasets = []
    for raw_style in styles:
        style = ControlStyle(raw_style)
        source = (
            PROJECT_ROOT
            / "logs"
            / "evaluations"
            / style.value
            / run_name
            / "evaluations.npz"
        )
        if not source.exists():
            continue
        data = np.load(source)
        datasets.append(
            (
                style,
                np.asarray(data["timesteps"]),
                np.asarray(data["results"], dtype=np.float64),
                np.asarray(data["ep_lengths"], dtype=np.float64),
            )
        )

    if not datasets:
        raise FileNotFoundError(
            "No EvalCallback logs found. Train an agent before plotting curves."
        )

    destination = (
        Path(output)
        if output is not None
        else PROJECT_ROOT / "reports" / f"training_curves_{run_name}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        ControlStyle.ROTATION: "#8B5CF6",
        ControlStyle.DIRECT: "#06B6D4",
    }
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for style, timesteps, returns, lengths in datasets:
        mean_return = returns.mean(axis=1)
        std_return = returns.std(axis=1)
        mean_length = lengths.mean(axis=1)
        color = colors[style]
        label = style.value.title()
        axes[0].plot(timesteps, mean_return, label=label, color=color, linewidth=2)
        axes[0].fill_between(
            timesteps,
            mean_return - std_return,
            mean_return + std_return,
            color=color,
            alpha=0.18,
        )
        axes[1].plot(timesteps, mean_length, label=label, color=color, linewidth=2)

    axes[0].set_title("PPO Evaluation Reward During Training")
    axes[0].set_ylabel("Mean episode return")
    axes[1].set_title("Evaluation Episode Length")
    axes[1].set_ylabel("Mean steps")
    axes[1].set_xlabel("Training timesteps")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
