#!/usr/bin/env python3
# =====================================================================
# q_learning_level0_standalone.py
# Single file demo for Task 1 - Basic Q-learning with a visual GridWorld
# =====================================================================

import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pygame

CONFIG = {
    "episodes": 1000,
    "alpha": 0.2,
    "gamma": 0.95,
    "epsilonStart": 1.0,
    "epsilonEnd": 0.05,
    "epsilonDecayEpisodes": 700,
    "maxStepsPerEpisode": 400,
    "fpsVisual": 25,
    "fpsFast": 240,
    "tileSize": 48,
    "seed": 42
}

ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ALL_ACTIONS = [0, 1, 2, 3]

MAPS = {
    0: [
        "S           ",
        "            ",
        "        A   ",
        "        A   ",
        "        A   ",
        "        A   ",
        "        A   ",
        "        A   "
    ],
    1: [
        "S        FFF",
        "         F A",
        "    F    F  ",
        "            ",
        "            ",
        "            ",
        "            ",
        "            "
    ],
    2: [
        "S    A      ",
        "            ",
        "    K       ",
        "         A  ",
        "            ",
        "        C   ",
        "   A        ",
        "            "
    ],
    3: [
        "S   R   A   ",
        "    R       ",
        "    R   FFF ",
        "  K R   F C ",
        "    R   F   ",
        "    RRRRR   ",
        "        A   ",
        "        A   "
    ]
}


@dataclass
class StepResult:
    next_state: Tuple
    reward: float
    done: bool
    info: dict


class GridWorld:
    def __init__(self, layout: List[str]):
        self.layout = layout
        self.w = max(len(row) for row in layout)
        self.h = len(layout)
        self.rocks = set()
        self.fires = set()
        self.key_pos: Optional[Tuple[int, int]] = None
        self.chest_pos: Optional[Tuple[int, int]] = None
        self.apples: List[Tuple[int, int]] = []
        self.apple_index: Dict[Tuple[int, int], int] = {}
        self.start = (0, 0)

        for y, row in enumerate(layout):
            padded = row.ljust(self.w)
            for x, ch in enumerate(padded):
                p = (x, y)
                if ch == "S": self.start = p
                elif ch == "A":
                    self.apple_index[p] = len(self.apples)
                    self.apples.append(p)
                elif ch == "R": self.rocks.add(p)
                elif ch == "F": self.fires.add(p)
                elif ch == "K": self.key_pos = p
                elif ch == "C": self.chest_pos = p
        self.reset()

    def reset(self) -> Tuple:
        self.agent = self.start
        self.alive = True
        self.has_key = 0
        self.chest_opened = 0
        self.step_count = 0
        self.apple_mask = 0
        for i in range(len(self.apples)):
            self.apple_mask |= (1 << i)
        return self.encode_state()

    def encode_state(self) -> Tuple:
        return (
            self.agent[0],
            self.agent[1],
            self.apple_mask,
            self.has_key,
            self.chest_opened
        )

    def in_bounds(self, p: Tuple[int, int]) -> bool:
        return 0 <= p[0] < self.w and 0 <= p[1] < self.h

    def try_move(self, p: Tuple[int, int], a: int) -> Tuple[int, int]:
        dx, dy = ACTIONS[a]
        np = (p[0] + dx, p[1] + dy)
        if not self.in_bounds(np) or np in self.rocks:
            return p
        return np

    def step(self, action: int) -> StepResult:
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}

        self.agent = self.try_move(self.agent, action)

        if self.agent in self.fires:
            self.alive = False
            return StepResult(self.encode_state(), 0.0, True, {"event": "death"})

        if self.agent in self.apple_index:
            idx = self.apple_index[self.agent]
            if (self.apple_mask >> idx) & 1:
                self.apple_mask &= ~(1 << idx)
                reward += 1.0

        if self.key_pos and self.agent == self.key_pos and self.has_key == 0:
            self.has_key = 1
            info["event"] = "key_collected"

        if self.chest_pos and self.agent == self.chest_pos and self.has_key == 1 and self.chest_opened == 0:
            self.chest_opened = 1
            reward += 2.0
            info["event"] = "chest_opened"

        apples_done = (self.apple_mask == 0)
        chest_done = True if not self.chest_pos else (self.chest_opened == 1)

        if apples_done and chest_done:
            done = True
            info["event"] = "win"

        return StepResult(self.encode_state(), reward, done, info)


class QTable:
    def __init__(self):
        self.q: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0] * len(ALL_ACTIONS))

    def get(self, s: Tuple, a: int) -> float:
        return self.q[s][a]

    def set(self, s: Tuple, a: int, val: float):
        self.q[s][a] = val

    def best_value(self, s: Tuple) -> float:
        return max(self.q[s])

    def best_actions(self, s: Tuple) -> List[int]:
        vals = self.q[s]
        m = max(vals)
        return [a for a, v in enumerate(vals) if math.isclose(v, m, abs_tol=1e-7)]


class BaseTabularAgent:
    def __init__(self, cfg: dict):
        self.alpha = float(cfg["alpha"])
        self.gamma = float(cfg["gamma"])
        self.eps_start = float(cfg["epsilonStart"])
        self.eps_end = float(cfg["epsilonEnd"])
        self.decay_ep = int(cfg["epsilonDecayEpisodes"])
        self.qtable = QTable()

    def get_epsilon(self, ep: int) -> float:
        if self.decay_ep <= 0: return self.eps_end
        t = min(ep / float(self.decay_ep), 1.0)
        return self.eps_start + t * (self.eps_end - self.eps_start)

    def choose_action(self, state: Tuple, eps: float, evaluate: bool = False) -> int:
        if not evaluate and random.random() < eps:
            return random.choice(ALL_ACTIONS)
        return random.choice(self.qtable.best_actions(state))


class QLearningAgent(BaseTabularAgent):
    def update(self, s: Tuple, a: int, r: float, sp: Tuple, done: bool):
        cur_q = self.qtable.get(s, a)
        target = r if done else (r + self.gamma * self.qtable.best_value(sp))
        self.qtable.set(s, a, cur_q + self.alpha * (target - cur_q))


class SARSAAgent(BaseTabularAgent):
    def update(self, s: Tuple, a: int, r: float, sp: Tuple, ap: int, done: bool):
        cur_q = self.qtable.get(s, a)
        target = r if done else (r + self.gamma * self.qtable.get(sp, ap))
        self.qtable.set(s, a, cur_q + self.alpha * (target - cur_q))


def save_log(log_rows: list, level_id: int, algorithm: str) -> str:
    os.makedirs("logs", exist_ok=True)
    path = f"logs/level{level_id}_{algorithm}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "steps", "epsilon", "died"])
        writer.writerows(log_rows)
    return path


def print_summary(log_rows: list):
    last = log_rows[-100:]
    avg_return = sum(r[1] for r in last) / len(last)
    death_count = sum(1 for r in last if r[4] == 1)
    print(f">> Last 100 episodes -> Avg Return: {avg_return:.2f} | Deaths: {death_count}/100")


def render_frame(screen, font, env: GridWorld, level_id: int, algo_name: str,
                 ep: int, total_ep: int, step: int, eps: float, score: float, is_eval: bool = False):
    ts = CONFIG["tileSize"]
    screen.fill((28, 30, 38))

    for x in range(env.w):
        for y in range(env.h):
            pygame.draw.rect(screen, (45, 50, 62), (x * ts, y * ts, ts, ts), 1)

    for (rx, ry) in env.rocks:
        pygame.draw.rect(screen, (100, 110, 125), (rx * ts + 2, ry * ts + 2, ts - 4, ts - 4), border_radius=4)
    for (fx, fy) in env.fires:
        pygame.draw.circle(screen, (239, 68, 68), (fx * ts + ts // 2, fy * ts + ts // 2), ts // 3)

    if env.key_pos and env.has_key == 0:
        kx, ky = env.key_pos
        pygame.draw.circle(screen, (59, 130, 246), (kx * ts + ts // 2, ky * ts + ts // 2), ts // 4)
    if env.chest_pos:
        cx, cy = env.chest_pos
        col = (107, 114, 128) if env.chest_opened else (168, 85, 247)
        pygame.draw.rect(screen, col, (cx * ts + 8, cy * ts + 8, ts - 16, ts - 16), border_radius=4)

    for p, idx in env.apple_index.items():
        if (env.apple_mask >> idx) & 1:
            ax, ay = p
            pygame.draw.circle(screen, (250, 204, 21), (ax * ts + ts // 2, ay * ts + ts // 2), ts // 3)

    px, py = env.agent
    pygame.draw.rect(screen, (34, 197, 94), (px * ts + 6, py * ts + 6, ts - 12, ts - 12), border_radius=8)

    status = "EVALUATION (Learned Policy)" if is_eval else f"TRAINING (Ep {ep+1}/{total_ep})"
    hud = [
        f"Lvl: {level_id} | Algo: {algo_name.upper()} | {status}",
        f"Step: {step} | Eps: {eps:.3f} | Return: {score:.1f}",
        f"Apples: {bin(env.apple_mask).count('1')} | Key: {env.has_key} | Chest: {env.chest_opened}",
        "V: Fast Mode | R: Reset Q-table | Esc: Next Mode"
    ]
    for i, line in enumerate(hud):
        surf = font.render(line, True, (248, 250, 252))
        screen.blit(surf, (8, 6 + i * 18))

    pygame.display.flip()


def run_experiment(level_id: int = 0, algorithm: str = "q_learning"):
    pygame.init()
    layout = MAPS[level_id]
    env = GridWorld(layout)
    screen = pygame.display.set_mode((env.w * CONFIG["tileSize"], env.h * CONFIG["tileSize"]))
    pygame.display.set_caption(f"Task 1-3 Demo - Level {level_id} ({algorithm.upper()})")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    agent = QLearningAgent(CONFIG) if algorithm == "q_learning" else SARSAAgent(CONFIG)
    episodes = CONFIG["episodes"]
    max_steps = CONFIG["maxStepsPerEpisode"]
    is_fast = False
    running = True
    log_rows = []

    print(f"\n>> RUNNING: Level {level_id} | Algorithm: {algorithm.upper()}")

    for ep in range(episodes):
        s = env.reset()
        eps = agent.get_epsilon(ep)
        ep_score = 0.0
        steps = 0
        res = None

        a = agent.choose_action(s, eps)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False; break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v: is_fast = not is_fast
                    elif event.key == pygame.K_ESCAPE: running = False; break
                    elif event.key == pygame.K_r:
                        agent.qtable = QTable()
                        s = env.reset()
                        ep_score, steps = 0.0, 0
            if not running: break

            if algorithm == "q_learning":
                a = agent.choose_action(s, eps)

            res = env.step(a)
            sp, r, done = res.next_state, res.reward, res.done
            ep_score += r
            steps += 1

            if algorithm == "q_learning":
                agent.update(s, a, r, sp, done)
            elif algorithm == "sarsa":
                ap = agent.choose_action(sp, eps)
                agent.update(s, a, r, sp, ap, done)
                a = ap

            s = sp

            render_frame(screen, font, env, level_id, algorithm, ep, episodes, steps, eps, ep_score, is_eval=False)
            clock.tick(CONFIG["fpsFast"] if is_fast else CONFIG["fpsVisual"])

            if done or steps >= max_steps:
                break

        died = 1 if (res and res.info.get("event") == "death") else 0
        log_rows.append([ep + 1, ep_score, steps, eps, died])

        if (ep + 1) % 100 == 0:
            print(f"Ep {ep+1:4d}/{episodes} | Return: {ep_score:.1f} | Epsilon: {eps:.3f}")

    if log_rows:
        path = save_log(log_rows, level_id, algorithm)
        print(f">> Log saved to {path}")
        print_summary(log_rows)

    # Evaluation Mode
    print("\n>> Huấn luyện hoàn tất! Bắt đầu Evaluation Mode (Chính sách tối ưu eps = 0.0)...")
    eval_running = running
    while eval_running:
        s = env.reset()
        ep_score, steps = 0.0, 0
        while eval_running:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    eval_running = False; break
            if not eval_running: break

            a = agent.choose_action(s, eps=0.0, evaluate=True)
            res = env.step(a)
            s = res.next_state
            ep_score += res.reward
            steps += 1

            render_frame(screen, font, env, level_id, algorithm, 0, 0, steps, 0.0, ep_score, is_eval=True)
            clock.tick(CONFIG["fpsVisual"])

            if res.done or steps >= max_steps:
                pygame.time.delay(400)
                break

    pygame.quit()


if __name__ == "__main__":
    run_experiment(level_id=3, algorithm="q_learning")

    # Task 2: Q-learning né lửa (Level 1) - Mở để so sánh
    # run_experiment(level_id=1, algorithm="q_learning")

    # Task 3: Bitmask nhiều táo + Key + Chest (Level 2)
    # run_experiment(level_id=2, algorithm="q_learning")

    # Task 3: Mê cung đá + lửa (Level 3)
    # run_experiment(level_id=3, algorithm="q_learning")