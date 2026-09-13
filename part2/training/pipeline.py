"""Stable Baselines3 PPO training pipeline shared by both control styles."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ..arena import ArenaConfig, ControlStyle, action_names, make_direct_env, make_rotation_env
from ..arena.rewards import ProgressionReward
from .artifacts import ArtifactPaths, model_metadata, safe_run_name, write_json
from .config import TrainingSettings
from .evaluation import EPISODE_INFO_KEYS, evaluate_model, validate_model_contract, write_episode_csv


def linear_schedule(initial_value: float):
    """Linearly reduce PPO's learning rate as training progresses."""

    initial_value = float(initial_value)

    def schedule(progress_remaining: float) -> float:
        return max(0.0, float(progress_remaining)) * initial_value

    return schedule


def _require_training_dependencies():
    try:
        import gymnasium
        import stable_baselines3
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "Block 6 dependencies are missing. Run "
            "'python -m pip install -r requirements-rl.txt'."
        ) from exc
    return {
        "gymnasium": gymnasium,
        "stable_baselines3": stable_baselines3,
        "torch": torch,
        "PPO": PPO,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "EvalCallback": EvalCallback,
        "Monitor": Monitor,
        "set_random_seed": set_random_seed,
        "DummyVecEnv": DummyVecEnv,
    }


def _tensorboard_callback(base_callback, style: ControlStyle):
    class ArenaTensorboardCallback(base_callback):
        def __init__(self):
            super().__init__(verbose=0)
            self.event_totals: Counter[str] = Counter()
            self.action_counts: Counter[int] = Counter()

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", ())
            for info in infos:
                self.logger.record_mean("arena/phase", float(info.get("phase", 1)))
                self.logger.record_mean(
                    "arena/player_health", float(info.get("player_health", 0.0))
                )
                self.logger.record_mean("arena/enemy_count", float(info.get("enemy_count", 0)))
                self.logger.record_mean("arena/spawner_count", float(info.get("spawner_count", 0)))
                for event in info.get("events", ()):
                    self.event_totals[str(event.get("name", "unknown"))] += 1
                for name, value in info.get("reward_components", {}).items():
                    self.logger.record_mean(f"reward/{name}", float(value))

            actions = np.asarray(self.locals.get("actions", ())).reshape(-1)
            for action in actions:
                self.action_counts[int(action)] += 1
            return True

        def _on_rollout_end(self) -> None:
            for event_name, count in sorted(self.event_totals.items()):
                self.logger.record(f"events/{event_name}_total", count)

            total_actions = sum(self.action_counts.values())
            if total_actions:
                names = action_names(style)
                for action_index, name in enumerate(names):
                    fraction = self.action_counts[action_index] / total_actions
                    self.logger.record(f"actions/{action_index}_{name.lower()}", fraction)
            self.action_counts.clear()

    return ArenaTensorboardCallback()


def train_style(
    *,
    style: ControlStyle | str,
    arena_config: ArenaConfig,
    settings: TrainingSettings,
    run_name: str = "final",
    promote: bool = True,
    resume_model: str | Path | None = None,
    device: str = "auto",
    progress_bar: bool = True,
    verbose: int = 1,
) -> dict[str, Any]:
    """Train, checkpoint, evaluate, save, and document one PPO agent."""

    style = ControlStyle(style)
    run_name = safe_run_name(run_name)
    dependencies = _require_training_dependencies()
    PPO = dependencies["PPO"]
    Monitor = dependencies["Monitor"]
    DummyVecEnv = dependencies["DummyVecEnv"]
    paths = ArtifactPaths(style=style, run_name=run_name, promote=promote)
    artifact_root = paths.model_dir.parents[1]
    paths.create_directories()
    dependencies["set_random_seed"](settings.seed)

    factory = make_rotation_env if style is ControlStyle.ROTATION else make_direct_env

    def build_training_env(rank: int):
        def initialize():
            env = factory(
                config=arena_config,
                reward_fn=ProgressionReward(settings.reward),
                render_mode=None,
                seed=settings.seed + rank,
            )
            return Monitor(
                env,
                filename=str(paths.monitor_dir / f"train_{rank:02d}"),
                info_keywords=EPISODE_INFO_KEYS,
                override_existing=resume_model is None,
            )

        return initialize

    train_env = DummyVecEnv([build_training_env(rank) for rank in range(settings.n_envs)])
    evaluation_env = Monitor(
        factory(
            config=arena_config,
            reward_fn=ProgressionReward(settings.reward),
            render_mode=None,
            seed=settings.seed + 50_000,
        ),
        filename=str(paths.monitor_dir / "periodic_evaluation"),
        info_keywords=EPISODE_INFO_KEYS,
        override_existing=resume_model is None,
    )

    checkpoint_callback = dependencies["CheckpointCallback"](
        save_freq=max(settings.checkpoint_freq // settings.n_envs, 1),
        save_path=str(paths.checkpoint_dir),
        name_prefix=f"ppo_{style.value}_{run_name}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    evaluation_callback = dependencies["EvalCallback"](
        evaluation_env,
        best_model_save_path=str(paths.best_dir),
        log_path=str(paths.evaluation_dir),
        eval_freq=max(settings.eval_freq // settings.n_envs, 1),
        n_eval_episodes=settings.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    metrics_callback = _tensorboard_callback(dependencies["BaseCallback"], style)
    callbacks = dependencies["CallbackList"](
        [checkpoint_callback, evaluation_callback, metrics_callback]
    )

    ppo_kwargs = dict(settings.ppo)
    use_linear_schedule = bool(ppo_kwargs.pop("linear_learning_rate"))
    initial_learning_rate = float(ppo_kwargs["learning_rate"])
    if use_linear_schedule:
        ppo_kwargs["learning_rate"] = linear_schedule(initial_learning_rate)

    activation_class = {
        "tanh": dependencies["torch"].nn.Tanh,
        "relu": dependencies["torch"].nn.ReLU,
    }[settings.activation]
    policy_kwargs = {
        "net_arch": list(settings.net_arch),
        "activation_fn": activation_class,
    }

    try:
        if resume_model is None:
            model = PPO(
                "MlpPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(paths.tensorboard_dir),
                seed=settings.seed,
                device=device,
                verbose=verbose,
                stats_window_size=100,
                **ppo_kwargs,
            )
            reset_num_timesteps = True
        else:
            resume_path = Path(resume_model)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume model not found: {resume_path}")
            model = PPO.load(
                str(resume_path),
                env=train_env,
                device=device,
                tensorboard_log=str(paths.tensorboard_dir),
                print_system_info=True,
            )
            validate_model_contract(model, style, resume_path)
            reset_num_timesteps = False

        model.learn(
            total_timesteps=settings.total_timesteps,
            callback=callbacks,
            tb_log_name=f"ppo_{style.value}_{run_name}",
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        model.save(str(paths.final_model))
        actual_timesteps = int(model.num_timesteps)
    finally:
        train_env.close()
        evaluation_env.close()

    selected_source = paths.best_model if paths.best_model.exists() else paths.final_model
    shutil.copy2(selected_source, paths.output_model)

    # The copied model replaces the previous artifact, so its old checksum
    # metadata is no longer valid.
    paths.metadata.unlink(missing_ok=True)

    evaluation_model = PPO.load(str(paths.output_model), device=device)
    validate_model_contract(
        evaluation_model,
        style,
        paths.output_model,
        warn_missing_metadata=False,
    )
    episode_rows, evaluation_summary = evaluate_model(
        evaluation_model,
        style=style,
        arena_config=arena_config,
        reward_config=settings.reward,
        episodes=settings.final_eval_episodes,
        seed=settings.seed + 100_000,
        deterministic=True,
    )
    episode_csv = paths.summary.with_name(paths.summary.stem.replace("_summary", "_episodes") + ".csv")
    write_episode_csv(episode_csv, episode_rows)

    versions = {
        "stable_baselines3": dependencies["stable_baselines3"].__version__,
        "gymnasium": dependencies["gymnasium"].__version__,
        "torch": dependencies["torch"].__version__,
        "numpy": np.__version__,
    }
    settings_dict = settings.to_dict()
    settings_dict.update(
        {
            "actual_timesteps": actual_timesteps,
            "run_name": run_name,
            "promoted_to_canonical_path": promote,
            "selected_checkpoint": str(selected_source.relative_to(artifact_root)),
        }
    )
    metadata = model_metadata(
        style=style,
        algorithm=settings.algorithm,
        model_path=paths.output_model,
        settings=settings_dict,
        arena_config=arena_config.to_dict(),
        evaluation=evaluation_summary,
        dependency_versions=versions,
    )
    write_json(paths.metadata, metadata)
    
    validate_model_contract(
        evaluation_model,
        style,
        paths.output_model,
    )

    summary = {
        "algorithm": settings.algorithm,
        "control_style": style.value,
        "model": str(paths.output_model.relative_to(artifact_root)),
        "metadata": str(paths.metadata.relative_to(artifact_root)),
        "episode_csv": str(episode_csv.relative_to(artifact_root)),
        "tensorboard_directory": str(paths.tensorboard_dir.relative_to(artifact_root)),
        "monitor_directory": str(paths.monitor_dir.relative_to(artifact_root)),
        "periodic_evaluation_directory": str(paths.evaluation_dir.relative_to(artifact_root)),
        "settings": settings_dict,
        "final_evaluation": evaluation_summary,
    }
    write_json(paths.summary, summary)
    return summary
