"""Artifact paths and reproducibility metadata for each independent agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Any

from ..arena import ControlStyle, action_names
from ..arena.adapters import (
    OBSERVATION_LABELS,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not cleaned:
        raise ValueError("run name must contain a letter, number, '_' or '-'")
    return cleaned


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    style: ControlStyle
    run_name: str
    promote: bool = True

    @property
    def model_dir(self) -> Path:
        return PROJECT_ROOT / "models" / self.style.value

    @property
    def final_model(self) -> Path:
        return self.model_dir / f"{self.run_name}_final.zip"

    @property
    def best_dir(self) -> Path:
        return self.model_dir / f"{self.run_name}_best"

    @property
    def best_model(self) -> Path:
        return self.best_dir / "best_model.zip"

    @property
    def output_model(self) -> Path:
        if self.promote:
            return PROJECT_ROOT / "models" / f"{self.style.value}_agent.zip"
        return PROJECT_ROOT / "models" / "tuning" / f"{self.style.value}_{self.run_name}.zip"

    @property
    def metadata(self) -> Path:
        return self.output_model.with_suffix(".metadata.json")

    @property
    def checkpoint_dir(self) -> Path:
        return PROJECT_ROOT / "checkpoints" / self.style.value / self.run_name

    @property
    def tensorboard_dir(self) -> Path:
        return PROJECT_ROOT / "logs" / "tensorboard" / self.style.value

    @property
    def monitor_dir(self) -> Path:
        return PROJECT_ROOT / "logs" / "monitor" / self.style.value / self.run_name

    @property
    def evaluation_dir(self) -> Path:
        return PROJECT_ROOT / "logs" / "evaluations" / self.style.value / self.run_name

    @property
    def summary(self) -> Path:
        return PROJECT_ROOT / "reports" / f"{self.style.value}_{self.run_name}_summary.json"

    def create_directories(self) -> None:
        for path in (
            self.model_dir,
            self.output_model.parent,
            self.best_dir,
            self.checkpoint_dir,
            self.tensorboard_dir,
            self.monitor_dir,
            self.evaluation_dir,
            self.summary.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_metadata(
    *,
    style: ControlStyle,
    algorithm: str,
    model_path: Path,
    settings: dict[str, Any],
    arena_config: dict[str, Any],
    evaluation: dict[str, Any],
    dependency_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "control_style": style.value,
        "action_count": len(action_names(style)),
        "action_meanings": list(action_names(style)),
        "observation_size": OBSERVATION_SIZE,
        "observation_labels": list(OBSERVATION_LABELS),
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": dependency_versions,
        "training": settings,
        "arena": arena_config,
        "final_evaluation": evaluation,
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_metadata_for_model(model_path: str | Path) -> dict[str, Any] | None:
    path = Path(model_path).with_suffix(".metadata.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
