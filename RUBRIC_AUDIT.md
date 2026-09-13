# GAIT A3 rubric and requirement audit

Audit date: 13 September 2026. This document distinguishes implemented code and
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
| Part II H: API and observation | Complete | modern Gymnasium API plus exact four-value adapter; fixed normalized `(26,) float32` vector |
| Part II I: controls and models | Complete | exact `Discrete(5)` and `Discrete(6)` contracts, two canonical PPO models, two visual evaluators |
| Part II J: reward/training | Complete | auditable progression reward, PPO MLPs (`[256,256]` for both final agents), TensorBoard, 3 measured tuning presets, checkpoints, evaluation and comparison |
| Creativity/originality in code | Strong | clickable evidence controls, nested two-game launcher, three palettes, pursuit/reward debugger, exact sparse-maze proof, reproducible model selection |

## Quantitative evidence snapshot

- Level 0: 100% evaluation success, mean 15 steps, exact BFS optimum 15.
- Level 1: Q-learning 15 steps at minimum fire clearance 1; SARSA 19 steps at
  minimum clearance 2. This is a concrete conservative on-policy difference.
- Level 4 reached 100% for both algorithms. The harder two-monster Level 5
  reached 76.7% for Q-learning and 60.0% for SARSA over the saved 30-episode
  evaluation while monster movement remained stochastic.
- Level 6 equal-budget baseline: 0% evaluation success without intrinsic reward;
  100% with intrinsic reward, using environment return only for comparison.
- Arena holdout (30 identical unseen seeds starting at 12000): Rotation reaches
  a new phase in 100% of episodes, averages 4.83 destroyed spawners, and has a
  110.19 mean return. Direct reaches a new phase in 73.3%, averages 1.83
  destroyed spawners, and has a 30.30 mean return. See
  `part2/reports/control_style_comparison.json`.

## Honest limitations and remaining team obligations

- Rotation is the stronger controller and no longer stalls after the first
  spawner. Direct is genuinely learned and advances phases on most holdout
  episodes, but is less reliable because each cardinal movement also chooses
  its firing direction; this is an
  honest control-set comparison rather than a claim that both policies are equal.
- The source ZIP cannot complete the separately submitted report or group video.
  The team must create a maximum-10-page PDF, include student numbers and
  contribution summary, add the video URL, keep the video under 10 minutes, have
  every member appear/present, show actual saved-model gameplay for both controls,
  and visibly show at least one phase transition.
- Do not describe `run_api_demo.py` as learned behaviour; it intentionally uses
  random actions for API testing.

## Best video seed/command strategy

Use `evaluate_direct.py --episodes 3 --debug --seed 12004` to capture real phase
transitions from three successful holdout seeds. Use `evaluate_rotation.py
--episodes 3 --debug --seed 12004` to show its distinct thrust/turn/shoot policy
and target switching. Retain the aggregate 30-seed table in the report so this
reproducible demo choice is transparent rather than presented as a success-rate
estimate.
