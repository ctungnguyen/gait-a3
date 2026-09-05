#!/usr/bin/env python3
"""Short API smoke test using random actions; optionally display it."""

from __future__ import annotations

import argparse

from arena import ArenaEnv, ControlStyle, LegacyArenaAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=("rotation", "direct"), default="direct")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--legacy", action="store_true", help="Use the assignment's four-value API")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Show visual debugger (requires --render)")
    args = parser.parse_args()
    if args.debug and not args.render:
        parser.error("--debug requires --render")

    render_mode = "human" if args.render else None
    modern = ArenaEnv(control_style=ControlStyle(args.style), render_mode=render_mode, seed=args.seed)
    env = LegacyArenaAdapter(modern) if args.legacy else modern
    env.reset(seed=args.seed)

    for _ in range(args.steps):
        action = env.action_space.sample()
        result = env.step(action)
        if args.legacy:
            _obs, _reward, done, _info = result
        else:
            _obs, _reward, terminated, truncated, _info = result
            done = terminated or truncated
        if args.render:
            env.render(debug=args.debug)
        if done:
            break

    modern.core.assert_invariants()
    print(modern.core.info())
    env.close()


if __name__ == "__main__":
    main()
