# Sources and implementation provenance

The Arena, observations, control adapters, reward logic, training orchestration,
evaluation metrics, and project-specific tests are implemented for this project.
The training lifecycle follows public APIs and patterns documented by the
official framework maintainers:

- Stable Baselines3 training, saving, loading, and evaluation examples:
  https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
- Stable Baselines3 callbacks, including periodic evaluation and checkpointing:
  https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
- Stable Baselines3 TensorBoard integration:
  https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
- Stable Baselines3 Monitor wrapper:
  https://stable-baselines3.readthedocs.io/en/master/common/monitor.html
- Gymnasium environment API (`reset`, `step`, termination and truncation):
  https://gymnasium.farama.org/api/env/

No pretrained third-party model is supplied. The final two models must be
trained from this submitted Arena and configuration so the video, models, logs,
and source code all match.
