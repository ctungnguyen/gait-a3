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
