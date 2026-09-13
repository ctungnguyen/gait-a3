# Part II Blocks 1–2 checklist

## Block 1 - Arena environment

- [x] Real-time Pygame scene with continuous positions and velocities
- [x] Player ship movement and shooting
- [x] Periodic enemy spawners
- [x] Enemies pursue the player's predicted position
- [x] Player health and enemy health
- [x] Spawner health
- [x] Projectile movement and collision handling
- [x] Enemy contact damage with a cooldown
- [x] Destroying all active spawners advances the phase
- [x] Later phases increase spawner health, enemy health, and enemy speed
- [x] Episode terminates on player death
- [x] Episode truncates at the maximum step count
- [x] Rendering is separate from headless simulation
- [x] Open Arena visual design rather than a tile-based GridWorld

## Block 2 - Gym-style API

- [x] Modern Gymnasium `reset()` returns `(observation, info)`
- [x] Modern Gymnasium `step()` returns five values
- [x] `LegacyArenaAdapter.reset()` returns the observation
- [x] `LegacyArenaAdapter.step()` returns `(observation, reward, done, info)`
- [x] `render()` displays the Pygame evaluation scene
- [x] `rgb_array` rendering is supported
- [x] Fixed simulation timestep for repeatable training
- [x] Seeded randomness
- [x] `info` exposes gameplay events for reward design and diagnostics

## Observation coverage required alongside the API by rubric H2

- [x] Fixed-size numeric feature vector (`float32`, shape `(25,)`)
- [x] Player position
- [x] Player velocity
- [x] Player orientation represented by sine and cosine
- [x] Relative X/Y direction and distance to the nearest enemy
- [x] Relative X/Y direction and distance to the nearest spawner
- [x] Player health
- [x] Current phase
- [x] Observation values fit the declared `Box(-1, 1)` space

## Scope boundary

- The two manual `--style` options select the now-final Block 4 action
  contracts; they are not two versions of Blocks 1 and 2.
- Both mappings and style-locked factories share exactly one Arena so the two
  agents can be compared fairly.
- Separate trained-agent evaluation scripts are included. The actual trained
  model files remain Block 6 outputs and must not be claimed until generated.
- `run_api_demo.py` uses random actions only to smoke-test the API. It is not a
  learned agent and is not assessment evidence of learned behaviour.

## Debug and demonstration support

- [x] Debugger is off by default and toggled with `F3`
- [x] `D` remains exclusively available for direct Move Right / rotation Right
- [x] Shooting preserves existing momentum instead of applying hidden braking
- [x] Manual held-fire interleaves movement during weapon-cooldown frames
- [x] Live player, enemy, spawner, projectile, phase, and step statistics
- [x] Collision-radius visualization
- [x] Velocity and facing vectors
- [x] Exact enemy pursuit prediction targets and rays
- [x] Entity IDs, spawner timers, nearest-target distance, and health
- [x] Debug rendering is read-only and does not alter simulation results

The debugger is useful evidence and an originality feature, but it does not
replace the required trained-agent clips, TensorBoard results, or report.

## Later-block integration status

- Final Block 3 observation explanation, optional refinement, and ablation
- Block 5 reward weights are implemented in `part2/arena/rewards.py`; the team
  should review and justify them in the report.
- Block 6 PPO code, real TensorBoard logs, final models, tuning results, and
  comparison evidence are included and pass strict verification.
