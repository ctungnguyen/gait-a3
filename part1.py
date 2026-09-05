#!/usr/bin/env python3

import csv
import argparse
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
    "fpsFast": 1000, # fast mode (originally 240)
    "tileSize": 48,
    "seed": 42,
    "intrinsicRewardStrength": 0.1,
    "useIntrinsicReward": False,
    "monsterMoveChance": 0.4

}

ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ALL_ACTIONS = [0, 1, 2, 3]
MONSTER_LEVELS = {4, 5}
INTRINSIC_LEVELS = {6}

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
    ],
    4: [
        "S           ",
        "            ",
        "      M     ",
        "        A   ",
        "            ",
        "  M         ",
        "        A   ",
        "        A   "
    ],
    5: [
        "S     R     ",
        "      R   M ",
        "  K   R     ",
        "      R   A ",
        "  M       C ",
        "      RRRRR ",
        "        A   ",
        "   A        "
    ],
    6: [
        "S    R      R    ",
        "     R      R    ",
        "     R      R    ",
        "     RRRRRR R    ",
        "            R    ",
        "  RRRRRRRR  R    ",
        "  R         R    ",
        "  R   RRRRRRR    ",
        "  R              ",
        "  RRRRRRRRRR K   ",
        "             R   ",
        "             R  C",
        "             R   ",
        "             R  A"
    ]
    # change this to whatever you prefer

}



@dataclass
class StepResult:
    next_state: Tuple
    reward: float
    done: bool
    info: dict


class GridWorld:
    def __init__(self, layout: List[str], monster_move_chance: float = 0.4):
        self.layout = layout
        self.w = max(len(row) for row in layout)
        self.h = len(layout)
        self.monster_move_chance = monster_move_chance
        self.rocks = set()
        self.fires = set()
        self.key_pos: Optional[Tuple[int, int]] = None
        self.chest_pos: Optional[Tuple[int, int]] = None
        self.apples: List[Tuple[int, int]] = []
        self.apple_index: Dict[Tuple[int, int], int] = {}
        self.monster_starts: List[Tuple[int, int]] = []
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
                elif ch == "M": self.monster_starts.append(p)
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
        self.monsters: List[Tuple[int, int]] = list(self.monster_starts)
        return self.encode_state()

    def encode_state(self) -> Tuple:
        return (
            self.agent[0],
            self.agent[1],
            self.apple_mask,
            self.has_key,
            self.chest_opened,
            tuple(self.monsters)
        )

    def in_bounds(self, p: Tuple[int, int]) -> bool:
        return 0 <= p[0] < self.w and 0 <= p[1] < self.h

    def try_move(self, p: Tuple[int, int], a: int) -> Tuple[int, int]:
        dx, dy = ACTIONS[a]
        np = (p[0] + dx, p[1] + dy)
        if not self.in_bounds(np) or np in self.rocks:
            return p
        return np

    def _move_monsters(self):
        occupied = set(self.monsters)
        new_positions = []
        for m in self.monsters:
            occupied.discard(m)
            new_pos = m
            if random.random() < self.monster_move_chance:
                candidates = []
                for a in ALL_ACTIONS:
                    dx, dy = ACTIONS[a]
                    cand = (m[0] + dx, m[1] + dy)
                    if self.in_bounds(cand) and cand not in self.rocks and cand not in occupied:
                        candidates.append(cand)
                if candidates:
                    new_pos = random.choice(candidates)
            occupied.add(new_pos)
            new_positions.append(new_pos)
        self.monsters = new_positions


    def step(self, action: int) -> StepResult:
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}

        self.agent = self.try_move(self.agent, action)

        if self.agent in self.fires or self.agent in self.monsters:
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

        if self.monsters:
            self._move_monsters()
            if self.agent in self.monsters:
                self.alive = False
                return StepResult(self.encode_state(), reward, True, {"event": "death"})


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

class VisitCounter:
    def __init__(self):
        self.counts: Dict[Tuple, int] = defaultdict(int)
 
    def reset(self):
        self.counts.clear()
 
    def visit_and_bonus(self, s: Tuple, strength: float) -> float:
        n_s = self.counts[s]
        bonus = strength / math.sqrt(n_s + 1)
        self.counts[s] += 1
        return bonus


class BaseTabularAgent:
    def __init__(self, cfg: dict):
        self.alpha = float(cfg["alpha"])
        self.gamma = float(cfg["gamma"])
        self.eps_start = float(cfg["epsilonStart"])
        self.eps_end = float(cfg["epsilonEnd"])
        self.decay_ep = int(cfg["epsilonDecayEpisodes"])
        self.intrinsic_strength = float(cfg.get("intrinsicRewardStrength", 0.0))
        self.qtable = QTable()
        self.visit_counts: Dict[Tuple, int] = {}

    def start_episode(self, initial_state: Tuple):
        """Reset the per-episode exploration counts and record the start state."""
        self.visit_counts = {initial_state: 1}

    def intrinsic_reward(self, state: Tuple) -> float:
        """Record a visit and return its novelty bonus for this episode."""
        visits = self.visit_counts.get(state, 0)
        self.visit_counts[state] = visits + 1
        return self.intrinsic_strength / math.sqrt(visits + 1)

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


def save_log(log_rows: list, level_id: int, algorithm: str, suffix: str="") -> str:
    os.makedirs("logs", exist_ok=True)
    path = f"logs/level{level_id}_{algorithm}{suffix}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "steps", "epsilon", "died"])
        writer.writerows(log_rows)
    return path


def print_summary(log_rows: list):
    last = log_rows[-100:]
    avg_return = sum(r[1] for r in last) / len(last)
    death_count = sum(1 for r in last if r[4] == 1)
    print(f">> Last {len(last)} episodes -> Avg Return: {avg_return:.2f} | "
          f"Deaths: {death_count}/{len(last)}")


def train_tabular(level_id: int, algorithm: str, cfg: Optional[dict] = None,
                  use_intrinsic_reward: Optional[bool] = None) -> Tuple[BaseTabularAgent, list]:
    """Train without rendering, making experiments and automated evaluation reproducible."""
    settings = dict(CONFIG)
    if cfg:
        settings.update(cfg)
    if level_id not in MAPS:
        raise ValueError(f"Unknown level {level_id}; available levels: {sorted(MAPS)}")
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
    agent = QLearningAgent(settings) if algorithm == "q_learning" else SARSAAgent(settings)
    intrinsic = settings["useIntrinsicReward"] if use_intrinsic_reward is None else use_intrinsic_reward
    env = GridWorld(MAPS[level_id])
    rows = []
    random.seed(settings.get("seed", 42))

    for episode in range(int(settings["episodes"])):
        state = env.reset()
        agent.start_episode(state)
        epsilon = agent.get_epsilon(episode)
        action = agent.choose_action(state, epsilon)
        episode_return = 0.0
        result = None

        for step in range(int(settings["maxStepsPerEpisode"])):
            if algorithm == "q_learning":
                action = agent.choose_action(state, epsilon)
            result = env.step(action)
            next_state, env_reward, done = result.next_state, result.reward, result.done
            episode_return += env_reward
            if step + 1 >= int(settings["maxStepsPerEpisode"]) and not done:
                done = True
            learning_reward = env_reward
            if intrinsic:
                learning_reward += agent.intrinsic_reward(next_state)

            if algorithm == "q_learning":
                agent.update(state, action, learning_reward, next_state, done)
            else:
                next_action = agent.choose_action(next_state, epsilon) if not done else 0
                agent.update(state, action, learning_reward, next_state, next_action, done)
                action = next_action
            state = next_state
            if done:
                break
        rows.append([episode + 1, episode_return, step + 1, epsilon,
                     int(result is not None and result.info.get("event") == "death")])
    return agent, rows


def render_frame(screen, font, env: GridWorld, level_id: int, algo_name: str,
                 ep: int, total_ep: int, step: int, eps: float, score: float,
                 use_intrinsic: bool = False, is_eval: bool = False):

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


    for (mx, my) in env.monsters:
        cx, cy = mx * ts + ts // 2, my * ts + ts // 2
        r = ts // 3
        pygame.draw.polygon(screen, (185, 28, 28),
                             [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)])
 
    px, py = env.agent
    pygame.draw.rect(screen, (34, 197, 94), (px * ts + 6, py * ts + 6, ts - 12, ts - 12), border_radius=8)


    status = "EVALUATION (Learned Policy)" if is_eval else f"TRAINING (Ep {ep+1}/{total_ep})"
    intrinsic_tag = " | Intrinsic: ON" if use_intrinsic else ""
    hud = [
        f"Lvl: {level_id} | Algo: {algo_name.upper()}{intrinsic_tag} | {status}",
        f"Step: {step} | Eps: {eps:.3f} | Return: {score:.1f}",
         f"Apples: {bin(env.apple_mask).count('1')} | Key: {env.has_key} | Chest: {env.chest_opened} | Monsters: {len(env.monsters)}",
        "V: Fast Mode | R: Reset Q-table | Esc: Next Mode"
    ]
    for i, line in enumerate(hud):
        surf = font.render(line, True, (248, 250, 252))
        screen.blit(surf, (8, 6 + i * 18))

    pygame.display.flip()


def run_experiment(
    level_id: int = 0,
    algorithm: str = "q_learning",
    use_intrinsic_reward: Optional[bool] = None,
    show_evaluation: bool = True):
    if level_id not in MAPS:
        raise ValueError(
            f"Unknown level {level_id}; available levels: {sorted(MAPS)}"
        )

    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError("algorithm must be 'q_learning' or 'sarsa'")

    random.seed(CONFIG.get("seed", 42))

    use_intrinsic = (
        bool(CONFIG["useIntrinsicReward"])
        if use_intrinsic_reward is None
        else use_intrinsic_reward
    )

    pygame.init()
    layout = MAPS[level_id]
    env = GridWorld(layout, monster_move_chance=CONFIG["monsterMoveChance"])
    screen = pygame.display.set_mode((env.w * CONFIG["tileSize"], env.h * CONFIG["tileSize"]))
    pygame.display.set_caption(f"Part 1 Demo - Level {level_id} ({algorithm.upper()})")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    agent = QLearningAgent(CONFIG) if algorithm == "q_learning" else SARSAAgent(CONFIG)
    visits = VisitCounter()
    intrinsic_strength = CONFIG["intrinsicRewardStrength"]
    episodes = CONFIG["episodes"]
    max_steps = CONFIG["maxStepsPerEpisode"]
    is_fast = False
    running = True
    log_rows = []

    print(f"\n>> RUNNING: Level {level_id} | Algorithm: {algorithm.upper()} | Intrinsic: {use_intrinsic}")

    for ep in range(episodes):
        s = env.reset()
        visits.reset()
        eps = agent.get_epsilon(ep)
        ep_score = 0.0
        steps = 0
        res = None
 
        if use_intrinsic:
            visits.visit_and_bonus(s, intrinsic_strength)
 
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
                        visits.reset()
                        ep_score, steps = 0.0, 0
            if not running: break

            if algorithm == "q_learning":
                a = agent.choose_action(s, eps)

            res = env.step(a)
            sp, r, done = res.next_state, res.reward, res.done
            ep_score += r
            steps += 1
            if steps >= max_steps and not done:
                done = True

            update_r = r
            if use_intrinsic:
                update_r = r + visits.visit_and_bonus(sp, intrinsic_strength)

            if algorithm == "q_learning":
                agent.update(s, a, update_r, sp, done)
            elif algorithm == "sarsa":
                ap = agent.choose_action(sp, eps) if not done else 0
                agent.update(s, a, update_r, sp, ap, done)
                a = ap

            s = sp

            render_frame(screen, font, env, level_id, algorithm, ep, episodes, steps, eps, ep_score,
                         use_intrinsic=use_intrinsic, is_eval=False)

            clock.tick(CONFIG["fpsFast"] if is_fast else CONFIG["fpsVisual"])

            if done or steps >= max_steps:
                break

        died = 1 if (res and res.info.get("event") == "death") else 0
        log_rows.append([ep + 1, ep_score, steps, eps, died])

        if (ep + 1) % 100 == 0:
            print(f"Ep {ep+1:4d}/{episodes} | Return: {ep_score:.1f} | Epsilon: {eps:.3f}")

    if log_rows:
        suffix = ""
        if level_id in INTRINSIC_LEVELS:
            suffix = "_intrinsic" if use_intrinsic else "_no_intrinsic"
        path = save_log(log_rows, level_id, algorithm, suffix=suffix)
        print(f">> Log saved to {path}")
        print_summary(log_rows)

    if not show_evaluation:
        pygame.quit()
        return

    # Evaluation Mode
    print("\n>> Training complete! Starting Evaluation Mode (optimal policy, eps = 0.0)...")
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

            render_frame(screen, font, env, level_id, algorithm, 0, 0, steps, 0.0, ep_score,
                         use_intrinsic=use_intrinsic, is_eval=True)

            clock.tick(CONFIG["fpsVisual"])

            if res.done or steps >= max_steps:
                pygame.time.delay(400)
                break

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and visualize a tabular GridWorld agent.")
    parser.add_argument("--level", type=int, choices=sorted(MAPS), default=3)
    parser.add_argument("--algorithm", choices=("q_learning", "sarsa"), default="q_learning")
    parser.add_argument("--episodes", type=int, help="Override the configured training episode count.")
    parser.add_argument("--intrinsic", action="store_true",
                        help="Add Level 6 intrinsic reward to learning updates.")
    parser.add_argument("--no-evaluation", action="store_true",
                        help="Exit after training instead of opening evaluation mode.")
    args = parser.parse_args()
    if args.episodes is not None:
        if args.episodes < 1:
            parser.error("--episodes must be at least 1")
        CONFIG["episodes"] = args.episodes
    run_experiment(
        level_id=args.level,
        algorithm=args.algorithm,
        use_intrinsic_reward=True if args.intrinsic else None,
        show_evaluation=not args.no_evaluation,
    )
    

