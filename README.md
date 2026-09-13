# GAIT Assignment 3 — Reinforcement Learning Arcade

This submission contains **two different games**, as required by the brief:

- `part1/`: a tile-based Pygame Gridworld trained with tabular Q-learning and SARSA.
- `part2/`: a real-time continuous Pygame combat Arena trained with Stable Baselines3 PPO.

Part II is not an upgraded version of Part I. The environments, states, actions,
rewards, and learning algorithms are intentionally separate. Root-level scripts
are launchers so the implementation remains easy to run from PowerShell.

## Start here

Windows PowerShell (using `-LiteralPath` safely handles the `&` in a parent path):

```powershell
Set-Location -LiteralPath "D:\Study\Coding Workspace\Python Workspace\Games&AITech\A3\gait-a3"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
.\.venv\Scripts\python.exe main.py
```

`main.py` opens the unified Pygame menu. Use Up/Down and Enter (or the mouse),
press `T` to cycle Neon, Forest, and Sunset themes, and Esc to go back.

## Assessment interpretation: PPO/DQN versus the two controls

The brief requires **both control styles** and a separate trained model and
visual evaluator for each. The phrase **“either DQN or PPO”** refers only to the
choice of deep-RL algorithm. This project uses PPO for both controls so the
comparison changes only the action representation.

PPO (Proximal Policy Optimization) and DQN (Deep Q-Network) are neural-network
RL algorithms available in Stable Baselines3. PPO learns a policy/value model
from rollout batches and constrains update size; DQN learns action values using
a replay buffer and target network. Both support discrete actions here, but PPO
was selected consistently for a controlled experiment and stable training API.

| Required control | Exact space | Saved model | Visual evaluator |
| --- | --- | --- | --- |
| Rotation/thrust | `Discrete(5)` | `part2/models/rotation_agent.zip` | `evaluate_rotation.py` |
| Direct direction | `Discrete(6)` | `part2/models/direct_agent.zip` | `evaluate_direct.py` |

## Part I — classical RL Gridworld

The main implementation is split into level definitions, mechanics, algorithms,
training/evaluation, rendering, and evidence generation. Training parameters
come from `part1/config/gridworld.json`, including linear epsilon decay.

```powershell
# Open the visual Part I policy selector
.\.venv\Scripts\python.exe part1.py

# Directly show a saved learned policy
.\.venv\Scripts\python.exe part1.py --level 4 --algorithm q_learning --theme forest
.\.venv\Scripts\python.exe part1.py --level 6 --algorithm q_learning --intrinsic

# Retrain one policy, save its log/Q-table, then animate epsilon-zero evaluation
.\.venv\Scripts\python.exe part1.py --level 1 --algorithm sarsa --retrain

# Reproduce every Part I CSV, model, summary, and report figure
.\.venv\Scripts\python.exe part1.py --generate-evidence
```

The visual HUD shows the greedy action, `epsilon = 0`, live environment return,
remaining objectives, current `Q(s,a)` values, stochastic monster events, and
the update target. Controls: `P` pause, `N` single step, `R` replay, `T` theme,
`+/-` speed, and Esc back.

Key evidence already included:

- Level 0 learns the exact 15-step shortest collection policy.
- Level 1 Q-learning takes 15 steps at fire clearance 1; on-policy SARSA takes
  19 steps at clearance 2, visibly demonstrating the safer policy.
- Both algorithms successfully solve Levels 2 and 3.
- Levels 4 and 5 use a Markov state containing live monster positions; each
  monster independently gets a 40% random legal-move opportunity after every
  player action. Required learning curves are included.
- Level 6 keeps apple/key/chest environment rewards unchanged. Only learning
  updates add `intrinsicRewardStrength / sqrt(n(s) + 1)` using a counter reset
  every episode. The included comparison shows the sparse maze is learned with
  the bonus and not learned by the equal-budget baseline.

## Part II — deep-RL Pursuit Arena

### Manual control/API checks

```powershell
.\.venv\Scripts\python.exe run_manual.py --style rotation --theme neon
.\.venv\Scripts\python.exe run_manual.py --style direct --theme sunset
.\.venv\Scripts\python.exe run_api_demo.py --legacy --steps 50
```

| Style | Keyboard | Assignment action order |
| --- | --- | --- |
| Rotation | W/Up thrust, A/D turn, Space shoot | No-op, thrust, left, right, shoot |
| Direct | WASD/arrows move, Space shoot | No-op, up, down, left, right, shoot |

For manual Arena play: `P` pause, `F3` debug, `R` reset, Esc exit. `D` remains
Move/Rotate Right; debug is only `F3`. Firing preserves momentum, and held Space
interleaves shooting with movement during the weapon cooldown.

`run_api_demo.py` sends random actions to smoke-test API integration. It is not
a trained agent and must not be used as video evidence of learned behaviour.

### PPO training and evidence

```powershell
# Validate both environments without expensive training
.\.venv\Scripts\python.exe train_both.py --dry-run

# Re-run measured 50k-step hyperparameter trials
.\.venv\Scripts\python.exe tune_hyperparameters.py --style both --trial-timesteps 50000 --eval-episodes 10

# Train both mandatory models (300k timesteps each by default)
.\.venv\Scripts\python.exe train_both.py --preset baseline --timesteps 300000

# Audit checkpoints on identical holdout seeds and promote the best real model
.\.venv\Scripts\python.exe select_best_models.py --episodes 30 --seed 12000

# Fair comparison, figures, and strict artifact verification
.\.venv\Scripts\python.exe compare_agents.py --episodes 30 --seed 12000
.\.venv\Scripts\python.exe verify_block6_submission.py
```

The pipeline uses PPO `MlpPolicy` with `[128, 128]` hidden layers, four seeded
headless training environments, periodic evaluation, best checkpoints, Monitor
CSV logs, TensorBoard, and SHA-256 model metadata. Baseline, exploratory, and
stable presets provide measured hyperparameter exploration.

The final model-selection report uses unseen seeds and prioritises phase
advancement, then spawners destroyed, then enemies and return. It selected the
early-stopped rotation tuning checkpoint (reliably destroys one spawner) because
the longer rotation run regressed, and retained the full-run best direct model.
This process and every candidate metric are in
`part2/reports/model_selection/`.

```powershell
# Actual learned gameplay for the required video
.\.venv\Scripts\python.exe evaluate_rotation.py --episodes 3 --debug
.\.venv\Scripts\python.exe evaluate_direct.py --episodes 3 --debug

# Inspect PPO and custom Arena metrics
.\.venv\Scripts\python.exe -m tensorboard.main --logdir part2\logs\tensorboard
```

The 25-value normalized observation contains player position/velocity/heading,
nearest-enemy and nearest-spawner relative features, health, phase, cooldown,
entity counts, and progress features. The reward adapter exposes auditable
components for combat, spawners, phase progression, damage/death, approach
progress, shot alignment, and time cost without changing game mechanics.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_submission.py
```

The automated suite covers Part I rules/formulas/learning, Arena physics and
API, exact action contracts, reward isolation, model metadata, and the training
pipeline. `verify_submission.py` checks generated code artifacts; the team must
still provide its own final PDF report and real group video link.

## Project layout

```text
main.py                         unified themed Pygame launcher
part1.py                        Part I compatibility launcher
part1/                          Gridworld source, config, policies, logs, figures
part2/arena/                    Arena simulation, API, observation, reward, renderer
part2/training/                 PPO configuration, pipeline, evaluation, plots
part2/models/                   two canonical models plus real checkpoints
part2/logs/                     TensorBoard, Monitor, and periodic evaluations
part2/reports/                  tuning, comparison, selection, and figures
train_*.py / evaluate_*.py      root-level convenience launchers
tests/                          53 automated tests
RUBRIC_AUDIT.md                 evidence mapping and honest remaining obligations
CHANGELOG_FINAL.md              file-by-file update notes
```

Do not remove the tests from the GitHub repository. They are small, demonstrate
correctness, and are useful to markers. The strict submission asks for a link to
the complete code ZIP/repository, so tests are appropriate inside that ZIP.
