# GAIT Assignment 3 — Gridworld and Deep-RL Arena (v2.0.0)

This merged project contains the Part I tabular Gridworld and the Part II
**Neon Pursuit Arena**. Part II now has complete code paths for environment,
Gymnasium API, observations, two required action spaces, reward integration,
PPO training, TensorBoard logging, hyperparameter trials, saved models,
headless comparison, learning curves, and visual evaluation.

## Important interpretation of the brief

The brief requires **both** control styles and a separately trained model for
each style. The word **either** applies to the deep-RL algorithm: the team may
choose either DQN or PPO. This project chooses **PPO for both agents** so the
control-set comparison is controlled and defensible.

| Required output | Implementation |
| --- | --- |
| Rotation/thrust actions | `Discrete(5)` and `models/rotation_agent.zip` |
| Direct-direction actions | `Discrete(6)` and `models/direct_agent.zip` |
| Neural-network algorithm | Stable Baselines3 PPO with a two-layer MLP |
| Headless training | Four `DummyVecEnv` Arena instances, no rendering |
| Logs | Monitor CSV, periodic evaluation NPZ, TensorBoard event files |
| Evaluation | Separate `evaluate_rotation.py` and `evaluate_direct.py` |
| Comparison evidence | CSV, JSON, and combined training-curve PNG |

The source package does not include fake model or log files. Run the final
training commands before submission so the two real model files and their
matching evidence are generated from this exact code.

## Installation on Windows PowerShell

Paths containing `&` should be opened with `-LiteralPath`:
agent, log = train_tabular(
    level_id=6,
    algorithm="q_learning",
    cfg={"useIntrinsicReward": True},
)
```
# GAIT Part II - Blocks 1, 2, and 4 controls (v1.3.0)

This project implements the real-time **Neon Pursuit Arena** and its Gym-style
API. It is intentionally separate from the Part I GridWorld: Part II uses
continuous positions, velocities, steering, projectiles, enemy spawners, and
phase progression rather than a tile map.

## What is complete

- A controllable player ship with health, movement, rotation, thrust, and shooting
- Periodic enemy spawners with health
- Enemies with health and smooth pursuit steering toward the player's predicted position
- Projectiles and enemy/spawner collision damage
- Contact damage when an enemy reaches the player
- Phase progression when every active spawner is destroyed
- Increasing enemy/spawner difficulty across phases
- Death and maximum-step episode endings
- Pygame human and RGB-array rendering
- Headless deterministic simulation for fast training
- Modern Gymnasium API and the assignment's legacy four-value API
- A normalized, fixed-size 21-value numeric observation covering every minimum
  feature named in the assessment
- Final, style-locked Block 4 action contracts for rotation/thrust
  (`Discrete(5)`) and direct movement (`Discrete(6)`)
- Separate visual model-evaluation entry points for the two action spaces
- Event-based progression reward: enemy `+10`, spawner `+50`, phase `+100`,
  damage `-5`, and death `-100`

The provided `gridworld.py` is preserved byte-for-byte under
`part1_reference/`; the Arena never imports or modifies it.

## Install

From this folder:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
Set-Location -LiteralPath "D:\Study\Coding Workspace\Python Workspace\Games&AITech\A3\gait-a3"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
```

`requirements.txt` contains only the Arena dependencies. `requirements-rl.txt`
adds Stable Baselines3, PyTorch, TensorBoard, plotting, and progress-bar tools.

## Verify before expensive training

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe train_both.py --dry-run --timesteps 1000
.\.venv\Scripts\python.exe verify_block6_submission.py --code-only
```

The unit suite checks the environment, exact action semantics, reward isolation,
training configuration, artifact separation, evaluator mismatch protection,
and an end-to-end mocked training pipeline.

## Manual verification of both Block 4 styles

```powershell
.\.venv\Scripts\python.exe run_manual.py --style rotation
.\.venv\Scripts\python.exe run_manual.py --style direct
```

| Style | Keyboard | Exact actions |
| --- | --- | --- |
| Rotation | W/Up thrust, A/D turn, Space shoot | `0 NOOP`, `1 THRUST_FORWARD`, `2 ROTATE_LEFT`, `3 ROTATE_RIGHT`, `4 SHOOT` |
| Direct | WASD/arrows move, Space shoot | `0 NOOP`, `1 MOVE_UP`, `2 MOVE_DOWN`, `3 MOVE_LEFT`, `4 MOVE_RIGHT`, `5 SHOOT` |

For both: `P` pauses, `F3` toggles debug, `R` resets, and `Esc` exits. Shooting
preserves momentum, so holding movement and Space does not freeze the ship.

## Block 6 training workflow

### 1. Run measured hyperparameter trials

```powershell
.\.venv\Scripts\python.exe tune_hyperparameters.py --style both --trial-timesteps 50000 --eval-episodes 10
```

Three presets are compared rather than claiming tuning without evidence:

- `baseline`: learning rate `3e-4`, entropy `0.01`, MLP `[128, 128]`.
- `exploratory`: higher learning rate and entropy for broader exploration.
- `stable`: lower learning rate, higher gamma, and MLP `[256, 128]`.

The output under `reports/tuning/` ranks each style and also recommends one
shared preset for the fairest control-style comparison. Use the measured shared
winner for both final models.

### 2. Train both final models

The default is 300,000 aggregate timesteps per style, which falls inside the
assignment's feasibility guide:

```powershell
.\.venv\Scripts\python.exe train_both.py --preset baseline --timesteps 300000
```

Replace `baseline` with the shared winning preset from the tuning JSON. You may
also train each model independently:

```powershell
.\.venv\Scripts\python.exe train_rotation.py --preset baseline --timesteps 300000
.\.venv\Scripts\python.exe train_direct.py --preset baseline --timesteps 300000
```

Training automatically:

- uses `MlpPolicy` with two hidden layers;
- runs headless with four seeded environments;
- evaluates deterministically on a separate environment every 50,000 steps;
- saves checkpoints and the best evaluation checkpoint;
- promotes the best checkpoint to the canonical style-specific model path;
- writes a SHA-256 model checksum and complete reproducibility metadata;
- runs a final 20-episode headless evaluation.

Resume an interrupted run with, for example:

```powershell
.\.venv\Scripts\python.exe train_rotation.py --resume models\rotation_agent.zip --timesteps 100000
```

### 3. Inspect TensorBoard

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir logs\tensorboard
```

Open the local URL printed by TensorBoard. Besides PPO losses and episode
return, custom logs show phase, health, entity counts, reward components,
gameplay events, and the action distribution for each control style.

### 4. Compare both agents on identical seeds

```powershell
.\.venv\Scripts\python.exe compare_agents.py --episodes 30
```

This produces:

- `reports/control_style_comparison.csv`
- `reports/control_style_comparison.json`
- `reports/training_curves_final.png`

The comparison uses the same Arena configuration, reward, deterministic mode,
episode count, and episode seeds. Report mean return, standard deviation,
success rate, phase reached, episode length, destroyed entities, damage, and
shots—not just one lucky gameplay clip.

### 5. Visually evaluate the trained agents

```powershell
.\.venv\Scripts\python.exe evaluate_rotation.py --debug
.\.venv\Scripts\python.exe evaluate_direct.py --debug
```

These scripts load different canonical models and reject the wrong action space
or incompatible metadata. They show actual Pygame gameplay suitable for the
required video. `run_api_demo.py` is random and must not be presented as learned
behaviour.

### 6. Run the strict submission check

```powershell
.\.venv\Scripts\python.exe verify_block6_submission.py
```

It fails until both real models, metadata files, TensorBoard events, periodic
evaluations, final summaries, comparison tables, and training curves exist.

## Reward used for training

`arena/rewards.py` converts Arena events into a learning signal without changing
health, damage, spawning, collisions, phase rules, or episode termination.

| Signal | Purpose |
| --- | --- |
| Enemy damage/destruction | Rewards effective combat |
| Spawner damage/destruction | Gives a larger reward for the main objective |
| Phase advancement | Largest positive progression event |
| Player damage and death | Encourages hazard avoidance and survival |
| Small step cost | Prevents camping for time |
| Distance-difference shaping | Reduces early sparse exploration toward spawners |
| Shot alignment | Rewards intentional firing and discourages blind spam |

All weights are visible in `config/training.json`; each step exposes
`info["reward_components"]` for auditing. If those weights change, both models
must be retrained before comparison.

## Generated artifact layout
- Rotation: `0 NOOP`, `1 THRUST_FORWARD`, `2 ROTATE_LEFT`,
  `3 ROTATE_RIGHT`, `4 SHOOT`
- Direct: `0 NOOP`, `1 MOVE_UP`, `2 MOVE_DOWN`, `3 MOVE_LEFT`,
  `4 MOVE_RIGHT`, `5 SHOOT`

`arena/controls.py` is the single authoritative mapping. `ArenaEnv`, the
manual player, the HUD, the style-locked factories, evaluation scripts, and
tests all read the same contract. Both agents therefore use identical Arena
physics and differ only in their required action space.

### Continuous-fire behaviour

`SHOOT` remains one separate discrete action, exactly as required. It no longer
acts as an accidental brake: the ship preserves its existing momentum during a
shooting step. In manual mode, holding Space with a movement/thrust key fires
whenever the cooldown is ready and uses the intervening frames for that valid
movement action. This produces continuous movement and firing while still
submitting only one rubric-defined action per simulation step.

## Separate trained-model evaluation entry points

Block 4 supplies separate, style-locked visual evaluators:

```bash
python evaluate_rotation.py --model models/rotation_agent.zip --algorithm ppo
python evaluate_direct.py --model models/direct_agent.zip --algorithm ppo
```

Install the optional deep-RL dependencies first with
`python -m pip install -r requirements-rl.txt`. Both evaluators display actual
Pygame gameplay, support `P` pause and `F3` debug, use deterministic prediction
by default, and reject a model whose action-space size belongs to the other
style.

The evaluator code and output paths are complete. The actual two trained model
files must still be generated by Block 6 after Block 5 provides the final
reward; this package does not present an untrained placeholder as assessment
evidence.

## Deep-RL training and reward design

`ArenaEnv` now defaults to `progression_reward`, which assigns `+10` for an
enemy destroyed, `+50` for a spawner destroyed, `+100` for phase advancement,
`-5` for each damage event, and `-100` on death. These are event rewards only;
there is no unvalidated distance-shaping term.

Train either style (or both) with Stable Baselines3 PPO:

```bash
python train_agents.py --style both --algorithm ppo --preset short --timesteps 100000
tensorboard --logdir runs
```

The `short` and `long` presets intentionally vary learning rate, rollout
horizon, batch size, discount factor, and entropy coefficient. The selected
configuration is written beside each model as `*_agent.json`, providing
reproducible evidence of hyperparameter exploration. DQN is also supported
with `--algorithm dqn`; it automatically uses replay-buffer settings suitable
for DQN.

## Visual debugger

The debugger is **off by default**. Press `F3` in `run_manual.py` to toggle it.
It is a read-only visualization and does not change physics, observations,
rewards, or episode state.

`D` is reserved exclusively for moving/turning right. Resetting with `R` also
returns the debugger to its default off state.

It displays:

- Player position, velocity, speed, heading, fire cooldown, and last action
- Collision radii for the player, enemies, spawners, and projectiles
- Player/enemy/projectile velocity arrows and the player's facing arrow
- Exact pursuit rays and predicted player targets for the nearest eight enemies
- Nearest enemy/spawner links, IDs, health, distance, and spawn timers
- Entity counts and current step versus the maximum-step limit

The same overlay can be requested programmatically:

```python
env.render(debug=True)
```

## Gym-style APIs

Modern Gymnasium/Stable Baselines3 interface:

```python
from arena import ArenaEnv

env = ArenaEnv(control_style="direct", render_mode=None)
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)
env.close()
```

Exact four-value interface printed in the assignment:

```python
from arena import LegacyArenaAdapter

env = LegacyArenaAdapter(control_style="direct", render_mode="human")
observation = env.reset(seed=42)
observation, reward, done, info = env.step(0)
env.render()
env.close()
```

The modern interface is required by current Stable Baselines3. The legacy
adapter exists so the submitted code also demonstrates the exact rubric API.

## What `run_api_demo.py` is

`run_api_demo.py` is a developer smoke test, not a trained AI and not the final
video demonstration. It sends random actions through the environment to check
that reset, step, info, termination, and rendering remain connected:

```bash
python run_api_demo.py --style direct --steps 300
python run_api_demo.py --style rotation --steps 300 --render --debug
python run_api_demo.py --legacy --steps 50
```

Console output from this utility does not replace the required Pygame display;
the actual Arena is rendered by `render()` and `run_manual.py`.

## Handoff to Blocks 3–6

`arena/adapters.py` is the integration point:

- `build_baseline_observation(core)` currently returns a normalized 21-value
  vector covering player position/velocity/orientation, nearest enemy and
  spawner relative information, health, cooldown, counts, and phase. This
  already meets the minimum observation fields in the rubric; the Block 3
  owner can refine or justify the final selection.
- `progression_reward(core, outcome)` is the default and consumes the explicit
  simulation events. `neutral_reward` remains available for API smoke tests and
  ablation experiments. There is no per-step distance shaping, so the agent
  cannot earn reward by orbiting without destroying targets.

Every simulation step places facts in `info["events"]`, including:

```text
models/
  rotation_agent.zip
  rotation_agent.metadata.json
  direct_agent.zip
  direct_agent.metadata.json
checkpoints/<style>/final/
logs/tensorboard/<style>/
logs/monitor/<style>/final/
logs/evaluations/<style>/final/evaluations.npz
reports/<style>_final_summary.json
reports/<style>_final_episodes.csv
reports/control_style_comparison.csv
reports/control_style_comparison.json
reports/training_curves_final.png
```

## Part I

`part1.py` contains the visual Gridworld, Q-learning, SARSA, Levels 0–6, and
intrinsic-reward support. Part I and its existing logs were preserved while
Block 6 was added.

See `BLOCK4_CHECKLIST.md`, `BLOCK6_CHECKLIST.md`, and `SOURCES_AND_REUSE.md` for
the exact assessment mapping and remaining evidence checklist.
