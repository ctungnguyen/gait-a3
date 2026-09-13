# Final restructuring and optimization notes

## New project structure and presentation

- `main.py`: added a unified visual menu for choosing the two games, manual or
  trained Arena modes, and three colour themes.
- `shared/themes.py`: added Neon, Forest, and Sunset render-only palettes. Theme
  changes do not alter observations, rewards, physics, or saved models.
- Moved Arena implementation/artifacts into `part2/`; moved the preserved lab
  Gridworld into `part1/reference/`; kept convenient run/train/evaluate scripts
  at repository root.
- Updated all imports, tests, default paths, and verification paths for the new
  package structure.

## Part I, especially Tasks 4 and 5

- `part1/config/gridworld.json`, `part1/config.py`: moved training parameters to
  a validated config file and activated per-level overrides.
- `part1/levels.py`: organized seven explicit layouts and task descriptions;
  Level 6 is a long sparse-reward maze suitable for a meaningful intrinsic
  comparison.
- `part1/environment.py`: separated exact mechanics from rendering; added seeded
  stochastic movement, live monster positions in the Markov state, move-event
  evidence, and both player-to-monster and monster-to-player death handling.
- `part1/agents.py`: isolated Q-learning and SARSA targets, exact linear epsilon
  schedule, required random tie-breaking, per-episode count bonus, and saved
  Q-table support.
- `part1/training.py`: added fast headless training, deterministic repeated
  evaluation, separate environment/learning/intrinsic returns, BFS shortest-path
  proof, policy traces, CSV summaries, and model metadata.
- `part1/evidence.py`: added reproducible runs for both algorithms across the
  required levels and report-ready curves, including mandatory Levels 4–5 and
  Level 6 with/without evidence.
- `part1/renderer.py`, `part1/app.py`, `part1.py`: added an interactive Pygame
  policy selector/demo with live Q-values, epsilon-zero label, events, pause,
  single-step, replay, speed, and theme controls.
- `tests/test_part1_gridworld.py`: added direct tests for fixed rewards, rocks,
  fire, monster transitions/state, both update equations, epsilon/ties,
  intrinsic formula, shortest Level 0 learning, and stochastic learning.

## Part II, especially Block 6

- `part2/arena/env.py`, `part2/arena/renderer.py`: integrated presentation-only
  themes while keeping the trained model contract unchanged.
- `train_agents.py`: replaced the older competing DQN/PPO script with a safe
  compatibility wrapper around the canonical, metadata-aware PPO pipeline.
- `select_best_models.py`: added deterministic holdout checkpoint evaluation,
  rubric-aligned ranking, atomic promotion, hashes, and full selection reports.
- Promoted the genuine rotation tuning checkpoint because it outperformed the
  later regressed checkpoint; retained the strongest direct full-run checkpoint.
- Re-generated `part2/reports/control_style_comparison.*` on identical holdout
  seeds and revalidated both canonical model checksums.
- `run_manual.py`, `evaluate_agent.py`, `run_api_demo.py`: added theme arguments
  and updated paths without changing actions or gameplay.
- `verify_block6_submission.py`: updated package/script paths and retained strict
  model/log/report validation.

## Documentation/evidence

- Replaced the conflicted/stale README (including incorrect 21-observation and
  missing-model claims) with current run, training, evaluation, and video guidance.
- Updated Block checklists and added this changelog plus `RUBRIC_AUDIT.md`.
- Retained the teammate's original Gridworld byte-for-byte under
  `part1/reference/gridworld.py` for provenance.
