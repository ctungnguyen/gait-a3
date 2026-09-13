import numpy as np
from part2.arena import make_direct_env, make_rotation_env

for name, env_factory in [("Direct", make_direct_env), ("Rotation", make_rotation_env)]:
    env = env_factory()
    obs, info = env.reset()
    print(f"[{name}] Space shape: {env.observation_space.shape} | Vector shape: {obs.shape}")
    assert obs.shape == (25,), f"Error: Vector does not have 25 dimensions in {name}"
    assert env.observation_space.contains(obs), f"Error: Observation value out of Box bounds in {name}"

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        assert not np.isnan(obs).any(), f"Error: NaN detected in {name} observation"
        if term or trunc:
            obs, info = env.reset()

print("\n>> ALL OBSERVATION TESTS (BLOCK 3) PASSED SUCCESSFULLY!")
