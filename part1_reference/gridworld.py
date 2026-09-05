#!/usr/bin/env python3
# =====================================================================
# q_learning_level0_standalone.py
# Single file demo for Task 1 - Basic Q-learning with a visual GridWorld
# =====================================================================

import json, os, random
from dataclasses import dataclass
from typing import Tuple, List, Dict
import pygame

# -----------------------------
# Configuration loader
# -----------------------------
DEFAULT_CFG = {
    "episodes": 800,
    "alpha": 0.2,
    "gamma": 0.95,
    "epsilonStart": 1.0,
    "epsilonEnd": 0.05,
    "epsilonDecayEpisodes": 700,
    "maxStepsPerEpisode": 400,
    "fpsVisual": 30,
    "fpsFast": 240,
    "tileSize": 48,
    "seed": 42
}
def load_config():
    cfg = DEFAULT_CFG.copy()
    path = os.path.join(os.path.dirname(__file__), "config_level0.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
        print("Loaded config_level0.json")
    return cfg
CFG = load_config()

# Unpack
EPISODES = int(CFG["episodes"]); ALPHA = float(CFG["alpha"]); GAMMA = float(CFG["gamma"])
EPS_START = float(CFG["epsilonStart"]); EPS_END = float(CFG["epsilonEnd"])
EPS_DECAY_EP = int(CFG["epsilonDecayEpisodes"]); MAX_STEPS = int(CFG["maxStepsPerEpisode"])
FPS_VISUAL = int(CFG["fpsVisual"]); FPS_FAST = int(CFG["fpsFast"])
TILE_SIZE = int(CFG["tileSize"]); random.seed(int(CFG["seed"]))

# -----------------------------
# Pygame window
# -----------------------------
WIDTH_TILES, HEIGHT_TILES = 12, 8
WIDTH, HEIGHT = WIDTH_TILES * TILE_SIZE, HEIGHT_TILES * TILE_SIZE
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GridWorld - Task 1 Q-learning, Level 0")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

# Colors
COL_BG = (25, 28, 34); COL_GRID = (45, 50, 58)
COL_AGENT = (74, 222, 128); COL_APPLE = (252, 92, 101); COL_TEXT = (240, 240, 240)

# Actions: 0 up, 1 right, 2 down, 3 left
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ALL_ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]

# Level 0 layout
LEVEL0 = [
    "S           ",
    "            ",
    "        A   ",
    "        A   ",
    "        A   ",
    "        A   ",
    "        A   ",
    "        A   "
]
LEVEL0 = [row.ljust(WIDTH_TILES)[:WIDTH_TILES] for row in LEVEL0]

# -----------------------------
# Environment
# -----------------------------
@dataclass
class StepResult:
    next_state: Tuple[int, int, int]  # (agent_x, agent_y, apple_mask)
    reward: float
    done: bool
    info: dict

class GridWorld:
    def __init__(self, layout: List[str]):
        self.layout = layout
        self.w, self.h = len(layout[0]), len(layout)
        # object sets
        self.rocks, self.fires, self.keys, self.chests = set(), set(), set(), set()
        self.apples, self.apple_index = [], {}
        self.monsters = []
        self.start = (0, 0)
        # parse layout
        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                p = (x, y)
                if ch == "A":
                    self.apple_index[p] = len(self.apples)
                    self.apples.append(p)
                elif ch == "S":
                    self.start = p
                elif ch == "R": self.rocks.add(p)
                elif ch == "F": self.fires.add(p)
                elif ch == "K": self.keys.add(p)
                elif ch == "C": self.chests.add(p)
                elif ch == "M": self.monsters.append(p)
        self.reset()

    def reset(self) -> Tuple[int, int, int]:
        self.agent = self.start
        self.collected_keys = 0
        self.opened_chests = set()
        self.alive = True
        # build apple mask with bits set for all apples
        self.apple_mask = 0
        for i in range(len(self.apples)):
            self.apple_mask |= (1 << i)
        self.step_count = 0
        return self.encode_state()

    def encode_state(self) -> Tuple[int, int, int]:
        return (self.agent[0], self.agent[1], self.apple_mask)

    # movement helpers
    def in_bounds(self, p): return 0 <= p[0] < self.w and 0 <= p[1] < self.h
    def blocked(self, p): return p in self.rocks
    def cell_contains_monster(self, p): return p in self.monsters

    def try_move(self, p, a):
        dx, dy = ACTIONS[a]
        np = (p[0] + dx, p[1] + dy)
        if not self.in_bounds(np): return p
        if self.blocked(np): return p
        return np

    def step(self, action: int) -> StepResult:
        self.step_count += 1
        reward, done = 0.0, False
        # 1) move
        self.agent = self.try_move(self.agent, action)
        # 2) death checks for later levels
        if self.agent in self.fires or self.cell_contains_monster(self.agent):
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"event": "death"})
        # 3) apple collection
        if self.agent in self.apple_index:
            idx = self.apple_index[self.agent]
            if (self.apple_mask >> idx) & 1:
                self.apple_mask &= ~(1 << idx)
                reward += 1.0
        # 4) end when all apples collected
        if self.apple_mask == 0:
            done = True
        return StepResult(self.encode_state(), reward, done, {})

# -----------------------------
# Q-table and learning helpers
# -----------------------------
class QTable:
    def __init__(self):
        self.q: Dict[Tuple[Tuple[int,int,int], int], float] = {}
    def get(self, s, a): return self.q.get((s, a), 0.0)
    def set(self, s, a, v): self.q[(s, a)] = v
    def best_value(self, s): return max(self.get(s, a) for a in ALL_ACTIONS)
    def best_actions(self, s):
        vals = [self.get(s, a) for a in ALL_ACTIONS]
        m = max(vals)
        return [a for a, v in zip(ALL_ACTIONS, vals) if v == m]

def linear_epsilon(ep, start, end, decay_ep):
    if decay_ep <= 0: return end
    t = min(ep / decay_ep, 1.0)
    return start + t * (end - start)

def epsilon_greedy(qtab: QTable, s, eps):
    if random.random() < eps:
        return random.choice(ALL_ACTIONS)
    best = qtab.best_actions(s)
    return random.choice(best)

def q_learning_update(qtab: QTable, s, a, r, sp, alpha, gamma):
    current = qtab.get(s, a)
    target = r + gamma * qtab.best_value(sp)
    qtab.set(s, a, current + alpha * (target - current))

# -----------------------------
# Drawing
# -----------------------------
def draw_grid(env: GridWorld, episode, step, epsilon, total_reward):
    screen.fill(COL_BG)
    # grid lines
    for x in range(env.w):
        for y in range(env.h):
            pygame.draw.rect(screen, COL_GRID, pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)
    # apples from mask
    for p, idx in env.apple_index.items():
        if (env.apple_mask >> idx) & 1:
            cx, cy = p[0]*TILE_SIZE + TILE_SIZE//2, p[1]*TILE_SIZE + TILE_SIZE//2
            pygame.draw.circle(screen, COL_APPLE, (cx, cy), TILE_SIZE//3)
    # agent
    ax, ay = env.agent
    pygame.draw.rect(screen, COL_AGENT, (ax*TILE_SIZE+8, ay*TILE_SIZE+8, TILE_SIZE-16, TILE_SIZE-16), border_radius=6)
    # HUD
    hud = [
        f"Ep {episode+1}/{EPISODES}  step {step}  eps {epsilon:.3f}",
        f"Apples left {bin(env.apple_mask).count('1')}",
        f"Return {total_reward:.2f}  Task 1 Q-learning Level 0",
        "V toggles fast mode. R resets."
    ]
    for i, t in enumerate(hud):
        screen.blit(font.render(t, True, COL_TEXT), (10, 8 + i*20))
    pygame.display.flip()

# -----------------------------
# Training loop and controls
# -----------------------------
def run_training():
    env = GridWorld(LEVEL0)
    qtab = QTable()
    visualize, running = True, True

    for ep in range(EPISODES):
        s = env.reset()
        ep_reward, steps = 0.0, 0
        eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)

        while running:
            # input
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False; break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v: visualize = not visualize
                    if event.key == pygame.K_r:
                        qtab = QTable()
                        s = env.reset()
                        ep_reward, steps = 0.0, 0
                        eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
            if not running: break

            # pick action, step, learn
            a = epsilon_greedy(qtab, s, eps)
            res = env.step(a)
            q_learning_update(qtab, s, a, res.reward, res.next_state, ALPHA, GAMMA)
            s = res.next_state
            ep_reward += res.reward
            steps += 1

            # draw and pace
            if visualize:
                draw_grid(env, ep, steps, eps, ep_reward)
                clock.tick(FPS_VISUAL)
            else:
                if steps % 5 == 0: draw_grid(env, ep, steps, eps, ep_reward)
                clock.tick(FPS_FAST)

            if res.done or steps >= MAX_STEPS:
                draw_grid(env, ep, steps, eps, ep_reward)
                break

    pygame.quit()

if __name__ == "__main__":
    run_training()