# Report and video handoff guide

This is a planning aid, not a replacement for the team's own writing and live
presentation. Keep the PDF at 10 pages maximum including every image.

## Suggested 10-page report allocation

1. Project overview, two-environment distinction, team contributions.
2. Gridworld layout/state/action design and exact environment rewards.
3. Q-learning/SARSA equations, epsilon schedule, Level 0 optimum evidence.
4. Level 1 policy-route comparison and Levels 2–3 state extension.
5. Levels 4–5 stochastic monsters and `task4_monster_levels4_5.png`.
6. Level 6 formula and `task5_level6_intrinsic_comparison.png`.
7. Arena mechanics, phase loop, Pygame screenshots, 25-value observation.
8. Exact two controls, progression reward table and shaping justification.
9. PPO `[128,128]`, three presets, TensorBoard/training curve/model selection.
10. Fair control comparison, originality, limitations, video URL and references.

For distance shaping, explain that the reward uses the change in distance to
the current spawner, is bounded per step, is symmetric when moving away, and is
combined with a step cost; it guides sparse early exploration without changing
Arena mechanics.

## Suggested video (target 9:15, never exceed 10:00)

| Time | Presenter | Show and say |
| --- | --- | --- |
| 0:00–0:35 | Member 1 | Open `main.py`; state clearly that Part I and Part II are two different games; cycle one theme. |
| 0:35–1:30 | Member 1 | Gridworld rules and fixed rewards; show config-driven epsilon schedule and current Q-values. |
| 1:30–2:15 | Member 1 | Level 0 Q-learning, epsilon zero, exact 15-step shortest policy. |
| 2:15–3:05 | Member 1 | Level 1 Q versus SARSA route figure: off-policy maximum versus chosen-next-action target; clearance 1 versus 2. |
| 3:05–4:10 | Member 2 | Level 4 or 5 learned policy; pause/single-step so a 40% monster move and collision avoidance are visible. |
| 4:10–4:55 | Member 2 | Level 6 formula and curve: per-episode visit counter, environment reward unchanged, 0% versus 100% equal-budget result. |
| 4:55–5:45 | Member 3 | Arena player/enemies/spawners/health/projectiles/phase and `F3` pursuit/collision debugger. |
| 5:45–6:45 | Member 3 | `evaluate_rotation.py --debug --seed 12000`: actual PPO model, exact five actions, learned spawner attack. |
| 6:45–8:05 | Member 3 | `evaluate_direct.py --debug --seed 12000`: exact six actions; capture a real phase transition. |
| 8:05–8:50 | Member 3 | Training curve/TensorBoard, PPO MLP, three presets, identical-seed model/control comparison. |
| 8:50–9:15 | All | Originality, limitations, contribution summary, closing. |

Use real saved-model footage. Do not use manual control or `run_api_demo.py` as
evidence of learned behaviour. Every member must appear and speak for at least
one section.
