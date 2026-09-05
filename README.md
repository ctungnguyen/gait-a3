# gait-a3

## Part I tabular reinforcement learning

`part1_tasks1_3.py` contains the visual GridWorld and both epsilon-greedy
tabular agents:

- `QLearningAgent` uses the off-policy maximum next-state value.
- `SARSAAgent` uses the value of the action selected by its next epsilon-greedy
  decision.
- `train_tabular(level, algorithm, cfg)` runs a headless experiment for
  Levels 0-3 and Level 6, which is useful for reproducible evaluation.

The shared configuration controls the linear epsilon schedule. Level 6
intrinsic reward can be enabled with `"useIntrinsicReward": True`; the
environment return remains unchanged while the update reward is
`environment_reward + intrinsicRewardStrength / sqrt(n(s) + 1)`. Visit counts
are reset at every episode.

For example:

```python
from part1_tasks1_3 import train_tabular

agent, log = train_tabular(
    level_id=6,
    algorithm="q_learning",
    cfg={"useIntrinsicReward": True},
)
```
# GAIT Part II - Blocks 1 and 2 (v1.2.2)

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
- Replaceable observation and reward hooks for the other team members

The provided `gridworld.py` is preserved byte-for-byte under
`part1_reference/`; the Arena never imports or modifies it.

## Install

From this folder:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the visual Arena manually

These commands are human-controlled QA tools for the same Arena. The `--style`
option selects one of the **Block 4 action schemes**; it does not mean Blocks 1
and 2 are two play styles.

Direct directional movement:

```bash
python run_manual.py --style direct
```

Rotation and thrust:

```bash
python run_manual.py --style rotation
```

Controls:

| Style | Controls |
| --- | --- |
| Direct | WASD/arrows move, Space shoots in the last movement direction |
| Rotation | W/Up thrusts, A/D or Left/Right rotates, Space shoots forward |
| Both | P pauses, F3 toggles debugger, R resets, Esc exits |

The discrete action sets exactly match the assignment:

- Rotation: `0 NOOP`, `1 THRUST`, `2 LEFT`, `3 RIGHT`, `4 SHOOT`
- Direct: `0 NOOP`, `1 UP`, `2 DOWN`, `3 LEFT`, `4 RIGHT`, `5 SHOOT`

They are included now because `step(action)` needs a defined action meaning and
because both future agents must use identical Arena physics. Their inclusion
does **not** claim that Block 4 is complete: separate trained models and visual
model-evaluation scripts still have to be produced by the Block 4/6 owners.

### Continuous-fire behaviour

`SHOOT` remains one separate discrete action, exactly as required. It no longer
acts as an accidental brake: the ship preserves its existing momentum during a
shooting step. In manual mode, holding Space with a movement/thrust key fires
whenever the cooldown is ready and uses the intervening frames for that valid
movement action. This produces continuous movement and firing while still
submitting only one rubric-defined action per simulation step.

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
- `neutral_reward(core, outcome)` intentionally returns `0.0`; the Block 5
  owner must supply the final justified reward function.

Every simulation step places facts in `info["events"]`, including:

```text
projectile_fired
enemy_spawned
enemy_damaged
enemy_destroyed
spawner_damaged
spawner_destroyed
player_damaged
player_died
phase_advanced
time_limit_reached
```

The reward owner should convert those events into reward values. Gameplay code
must not contain reward weights.

One Arena should be shared by both models. Do not copy the Arena and change its
physics for each control style; otherwise the control-set comparison will not
be fair.

## Tests

The core tests run without Pygame or Gymnasium:

```bash
python -m unittest discover -s tests -v
python -m compileall -q arena run_manual.py run_api_demo.py
```

After installing Gymnasium, the same tests use its real action and observation
spaces automatically.

## Windows startup note

Version 1.1.0 and later render once before reading the Pygame event queue. This fixes the
older startup error:

```text
pygame.error: video system not initialized
```

Run the command from the extracted project directory. Because the default
configuration is resolved relative to `run_manual.py`, paths containing spaces
or `&` are supported.

## Rubric scope

| Rubric item | Status in this package |
| --- | --- |
| G1–G3 Arena environment | Complete and tested |
| H1 Gym-style API | Complete: modern and exact four-value compatibility APIs |
| H2 minimum observation vector | Complete baseline; final explanation/refinement can be owned by Block 3 |
| I1–I2 control mappings | Included as integration scaffolding |
| I3–I4 trained models/evaluation | Not part of Blocks 1–2; still required later |
| J reward and deep-RL training | Not part of Blocks 1–2; still required later |

## File ownership

| Files | Recommended owner |
| --- | --- |
| `arena/core.py`, `entities.py`, `renderer.py`, `config.py` | Person 3 - Blocks 1 and 2 |
| `arena/adapters.py` final observation justification/refinement | Person 4 - Block 3 |
| `arena/controls.py` rotation mapping | Person 1 - Block 4 Style 1 |
| `arena/controls.py` direct mapping | Person 2 - Block 4 Style 2 |
| Reward function plugged into `ArenaEnv` | Person 4 - Block 5 |
| Training scripts, logs, and final models | Persons 1, 2, and 4 - Block 6 |

See `BLOCK1_2_CHECKLIST.md`, `TEAM_HANDOFF.md`, and `SOURCES_AND_REUSE.md` for
the assessment mapping and integration boundaries.
