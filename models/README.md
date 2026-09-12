# Separate Block 4 model outputs

The two agents must be trained and saved separately because their action spaces
are not interchangeable:

- `rotation_agent.zip` — rotation/thrust policy using `Discrete(5)`
- `direct_agent.zip` — direct-direction policy using `Discrete(6)`

The shared Arena factories are `make_rotation_env()` and `make_direct_env()`.
The matching visual entry points are `evaluate_rotation.py` and
`evaluate_direct.py`. Each evaluator rejects a model with the wrong action
space before gameplay begins.

Generate both `.zip` files with `python train_both.py`. The pipeline now uses
the event-based reward in `arena/rewards.py`, selects the best periodic PPO
checkpoint, and writes a matching `.metadata.json` file beside each model.
Do not rename one model to impersonate the other style; the evaluators validate
both the action space and metadata.
