# Part I Gridworld

This package is the active Part I implementation:

- `levels.py`: Levels 0–6 and task descriptions.
- `environment.py`: exact fixed mechanics and stochastic monsters.
- `agents.py`: Q-learning, SARSA, epsilon-greedy selection, and count bonus.
- `training.py`: headless training, evaluation, logs, Q-tables, and BFS proof.
- `renderer.py` / `app.py`: interactive Pygame learned-policy presentation with
  supplied pixel-art sprites, safe primitive fallbacks, clickable controls,
  deliberate `R`-only reset after terminal states, themes, and menu return.
- `evidence.py`: reproducible report curves and summaries.
- `config/gridworld.json`: all supplied-style training parameters and overrides.

`reference/` preserves older supplied/merged code for provenance; it is not
imported by the active implementation. Run from repository root with
`python part1.py`.
