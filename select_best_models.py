#!/usr/bin/env python3
"""Audit real trained PPO checkpoints and promote the strongest per style.

All candidates are evaluated deterministically on the same unseen episode seeds.
Ranking prioritises the rubric objective (phase advancement), then spawners
destroyed, then return. The generated report records every candidate and hash.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from part2.arena import ArenaConfig, ControlStyle
from part2.training.artifacts import (
    PROJECT_ROOT,
    load_metadata_for_model,
    model_metadata,
    sha256_file,
    write_json,
)
from part2.training.config import load_training_settings
from part2.training.evaluation import (
    evaluate_model,
    load_ppo_model,
    validate_model_contract,
    write_episode_csv,
)


def candidate_paths(style: ControlStyle) -> dict[str, Path]:
    base = PROJECT_ROOT / "models"
    return {
        "full_run_best": base / style.value / "final_best" / "best_model.zip",
        "full_run_last": base / style.value / "final_final.zip",
        "tune_baseline": base / "tuning" / f"{style.value}_tune_baseline.zip",
        "tune_exploratory": base / "tuning" / f"{style.value}_tune_exploratory.zip",
        "tune_stable": base / "tuning" / f"{style.value}_tune_stable.zip",
    }


def _source_metadata(style: ControlStyle, label: str, source: Path) -> dict:
    metadata = load_metadata_for_model(source)
    if metadata is not None:
        return metadata
    if label.startswith("full_run"):
        canonical = load_metadata_for_model(
            PROJECT_ROOT / "models" / f"{style.value}_agent.zip"
        )
        if canonical is not None:
            return canonical
    return {}


def _rank(summary: dict) -> tuple[float, float, float, float]:
    return (
        float(summary["success_rate"]),
        float(summary["mean_spawners_destroyed"]),
        float(summary["mean_enemies_destroyed"]),
        float(summary["mean_return"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    arena_config = ArenaConfig.from_json(PROJECT_ROOT / "config" / "arena.json")
    settings = load_training_settings(PROJECT_ROOT / "config" / "training.json")
    reports_dir = PROJECT_ROOT / "reports" / "model_selection"
    reports_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    selection_report: dict[str, dict] = {}
    promoted_files: list[tuple[Path, Path, str]] = []

    for style in (ControlStyle.ROTATION, ControlStyle.DIRECT):
        candidates: list[dict] = []
        episode_rows_by_label: dict[str, list[dict]] = {}
        for label, path in candidate_paths(style).items():
            if not path.exists():
                continue
            model = load_ppo_model(path, device=args.device)
            validate_model_contract(model, style, path, warn_missing_metadata=False)
            episode_rows, summary = evaluate_model(
                model,
                style=style,
                arena_config=arena_config,
                reward_config=settings.reward,
                episodes=args.episodes,
                seed=args.seed,
                deterministic=True,
            )
            source_metadata = _source_metadata(style, label, path)
            row = {
                "control_style": style.value,
                "candidate": label,
                "source": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
                "training_timesteps": source_metadata.get("training", {}).get("actual_timesteps"),
                **summary,
            }
            candidates.append(row)
            all_rows.append(row)
            episode_rows_by_label[label] = episode_rows
            print(
                f"{style.value:8s} {label:18s} success={summary['success_rate']:.1%} "
                f"spawners={summary['mean_spawners_destroyed']:.2f} "
                f"return={summary['mean_return']:.2f}"
            )

        if not candidates:
            raise SystemExit(f"No trained candidates found for {style.value}")
        selected = max(candidates, key=_rank)
        selection_report[style.value] = {
            "ranking_order": [
                "success_rate",
                "mean_spawners_destroyed",
                "mean_enemies_destroyed",
                "mean_return",
            ],
            "selected": selected,
            "candidates": candidates,
        }

        if args.no_promote:
            continue

        source = PROJECT_ROOT / selected["source"]
        canonical = PROJECT_ROOT / "models" / f"{style.value}_agent.zip"
        source_metadata = _source_metadata(style, selected["candidate"], source)
        if source.resolve() != canonical.resolve():
            temporary = canonical.with_suffix(".selection.tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(canonical)
        expected_hash = selected["sha256"]
        if sha256_file(canonical) != expected_hash:
            raise RuntimeError(f"Promotion checksum failed for {style.value}")
        promoted_files.append((source, canonical, expected_hash))

        selected_model = load_ppo_model(canonical, device=args.device)
        selected_episodes = episode_rows_by_label[selected["candidate"]]
        selected_summary = {
            key: selected[key]
            for key in (
                "episodes", "mean_return", "std_return", "min_return", "max_return",
                "mean_episode_steps", "mean_max_phase", "highest_phase", "success_rate",
                "mean_enemies_destroyed", "mean_spawners_destroyed", "mean_damage_dealt",
                "mean_damage_taken", "mean_shots_fired",
            )
        }
        training_metadata = dict(source_metadata.get("training", settings.to_dict()))
        training_metadata.update(
            {
                "selection_method": "deterministic holdout checkpoint selection",
                "selection_seed": args.seed,
                "selection_episodes": args.episodes,
                "selected_candidate": selected["candidate"],
                "selected_checkpoint": selected["source"],
            }
        )
        metadata = model_metadata(
            style=style,
            algorithm="PPO",
            model_path=canonical,
            settings=training_metadata,
            arena_config=arena_config.to_dict(),
            evaluation=selected_summary,
            dependency_versions=source_metadata.get("dependency_versions", {}),
        )
        metadata["model_selection"] = selection_report[style.value]
        write_json(canonical.with_suffix(".metadata.json"), metadata)
        validate_model_contract(selected_model, style, canonical)

        episode_csv = PROJECT_ROOT / "reports" / f"{style.value}_final_episodes.csv"
        write_episode_csv(episode_csv, selected_episodes)
        final_summary = {
            "algorithm": "PPO",
            "control_style": style.value,
            "model": str(canonical.relative_to(PROJECT_ROOT)),
            "metadata": str(canonical.with_suffix('.metadata.json').relative_to(PROJECT_ROOT)),
            "episode_csv": str(episode_csv.relative_to(PROJECT_ROOT)),
            "selection": selection_report[style.value],
            "settings": training_metadata,
            "final_evaluation": selected_summary,
        }
        write_json(PROJECT_ROOT / "reports" / f"{style.value}_final_summary.json", final_summary)

    # A final atomic copy/verification prevents a later style's processing from
    # ever leaving an earlier canonical artifact out of sync with its sidecar.
    for source, canonical, expected_hash in promoted_files:
        temporary = canonical.with_suffix(".selection.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(canonical)
        if sha256_file(canonical) != expected_hash:
            raise RuntimeError(f"Final promotion checksum failed: {canonical}")

    selection_payload = {
        "algorithm": "PPO",
        "holdout_seed_start": args.seed,
        "episodes_per_candidate": args.episodes,
        "deterministic": True,
        "promotion_enabled": not args.no_promote,
        "selection": selection_report,
    }
    write_json(reports_dir / "model_selection.json", selection_payload)
    with (reports_dir / "model_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({style: data["selected"]["candidate"] for style, data in selection_report.items()}, indent=2))


if __name__ == "__main__":
    main()
