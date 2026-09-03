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