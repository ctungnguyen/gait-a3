# GAIT A3 rubric and requirement audit

Audit date: 12 September 2026. This document distinguishes implemented code and
generated evidence from work that only the student team can complete.

## Interpretation

The overview explicitly says to create **two interactive environments**. Part I
is a tabular Gridworld; Part II is a different real-time Arena. Part II requires
both action sets and two separate trained models. It permits either PPO or DQN
as the shared deep-RL algorithm; this submission consistently uses PPO.

## Code/evidence coverage

| Rubric area | Status | Inspectable evidence |
| --- | --- | --- |
| Part I A: Pygame Gridworld and exact mechanics | Complete | `part1/environment.py`, `part1/renderer.py`, Part I tests |
| Part I B: Level 0 Q-learning | Complete | epsilon-greedy, linear decay, random ties, exact Q update; 15/15 optimal steps in `part1_evaluation_summary.json` |
| Part I C: Level 1 SARSA | Complete | on-policy update; route figure shows 19-step/fire-clearance-2 SARSA versus 15-step/clearance-1 Q-learning |
| Part I D: Levels 2–3 | Complete | both Q-learning and SARSA policies/logs/evaluations for both levels |
| Part I Task 4: Levels 4–5 monsters | Complete | exact live monster positions in state, 40% random legal movement, both collision directions kill, both algorithms, curves |
| Part I F/Task 5: Level 6 intrinsic reward | Complete | exact per-episode formula, unchanged environment return, with/without logs and comparison figure |
| Part II G: Arena simulation | Complete | continuous player/ship, pursuit enemies, periodic spawners, health, projectiles, phase progression, two episode endings |
| Part II H: API and observation | Complete | modern Gymnasium API plus exact four-value adapter; fixed normalized `(25,) float32` vector |
| Part II I: controls and models | Complete | exact `Discrete(5)` and `Discrete(6)` contracts, two canonical PPO models, two visual evaluators |
| Part II J: reward/training | Complete | auditable progression reward, PPO MLP `[128,128]`, TensorBoard, 3 measured presets, checkpoints, evaluation and comparison |
| Creativity/originality in code | Strong | themed two-game launcher, three palettes, pursuit prediction debugger, action/evidence HUDs, reproducible model selection |

## Quantitative evidence snapshot

- Level 0: 100% evaluation success, mean 15 steps, exact BFS optimum 15.
- Level 1: Q-learning 15 steps at minimum fire clearance 1; SARSA 19 steps at
  minimum clearance 2. This is a concrete conservative on-policy difference.
- Levels 4 and 5: Q-learning and SARSA each reached 100% success in the saved
  30-episode evaluation while monster movement remained stochastic.
- Level 6 equal-budget baseline: 0% evaluation success without intrinsic reward;
  100% with intrinsic reward, using environment return only for comparison.
- Arena holdout (30 identical seeds starting at 12000): rotation destroys 1.00
  spawner per episode on average; direct reaches phase 2 with 23.3% phase-advance
  success and mean return 15.00. See `part2/reports/control_style_comparison.json`.

## Honest limitations and remaining team obligations

- The rotation model demonstrates learned aiming/combat and reliably destroys a
  spawner, but it did not advance phase on the 30-episode holdout. The direct
  model supplies the required visible phase-progression clip. Additional rotation
  training/curriculum could improve quality, but is not needed to prove the two
  required action/model pipelines exist.
- The source ZIP cannot complete the separately submitted report or group video.
  The team must create a maximum-10-page PDF, include student numbers and
  contribution summary, add the video URL, keep the video under 10 minutes, have
  every member appear/present, show actual saved-model gameplay for both controls,
  and visibly show at least one phase transition.
- Do not describe `run_api_demo.py` as learned behaviour; it intentionally uses
  random actions for API testing.

## Best video seed/command strategy

Use `evaluate_direct.py --episodes 3 --debug --seed 12000` to capture a phase
transition from the included direct model. Use `evaluate_rotation.py --episodes
3 --debug --seed 12000` to show its distinct thrust/turn/shoot policy and
spawner destruction. If a stochastic clip is unclear, record another real
episode rather than editing or scripting model actions.
