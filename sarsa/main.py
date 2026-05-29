"""
SARSA Taxi-v4 — dashboard temps réel + sauvegarde PNG.
Lancer : python main.py
"""

import os
import random
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from taxi_driver.env import create_environment
from utils.functions import episode_based_epsilon, train_sarsa as shared_train_sarsa
from utils.live_simple_plots import ACTION_ARROWS_TAXI, LiveSimpleDashboard
from utils.taxi_grid import TAXI_COLS, TAXI_ROWS, full_policy, q_to_grids

# --- Hyperparamètres ---
SEED = 50
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99960
MIN_EPSILON = 0.05
EPISODES = 5000
WINDOW = 100
GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
REFRESH_EVERY = 10

np.random.seed(SEED)
random.seed(SEED)

env = create_environment(render_mode="rgb_array")
STATE_SIZE = env.observation_space.n
ACTION_SIZE = env.action_space.n

dashboard = LiveSimpleDashboard(
    algo_label="SARSA",
    env_label="Taxi-v4",
    n_rows=TAXI_ROWS,
    n_cols=TAXI_COLS,
    window=WINDOW,
    graph_dir=GRAPH_DIR,
    file_prefix="sarsa",
    min_epsilon=MIN_EPSILON,
    epsilon_decay=EPSILON_DECAY,
    refresh_every=REFRESH_EVERY,
    action_arrows=ACTION_ARROWS_TAXI,
)


def train_sarsa():
    env.reset(seed=SEED)
    Q = np.zeros((STATE_SIZE, ACTION_SIZE))

    dashboard.setup()
    t0 = time.time()

    Q, visit_counts, episode_rewards, episode_steps, epsilon_values = shared_train_sarsa(
        env=env,
        q_table=Q,
        episodes=EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=MIN_EPSILON,
        seed=SEED,
        epsilon_schedule=episode_based_epsilon,
        dashboard=dashboard,
        refresh_every=REFRESH_EVERY,
        env_for_dashboard=env,
    )

    policy = full_policy(Q)
    print(f"Entraînement terminé en {time.time() - t0:.1f}s")
    return policy, Q, visit_counts, episode_rewards, episode_steps, epsilon_values


if __name__ == "__main__":
    print("=== SARSA Taxi-v4 (temps réel) ===")
    policy, Q, visits, rewards, steps, epsilons = train_sarsa()

    heat_grid, policy_grid = q_to_grids(Q)
    print("\nSauvegarde des graphiques…")
    dashboard.save_individual_plots(
        rewards,
        steps,
        epsilons,
        policy_grid.ravel(),
        heat_grid.ravel(),
        env=env,
        rollout_policy=policy,
    )
    dashboard.close()

    env.close()
    print("\nTerminé.")
