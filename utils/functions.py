import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

# Shared action mappings used across the project.
# FrozenLake: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
ACTION_ARROWS_FROZEN_LAKE = ("\u2190", "\u2193", "\u2192", "\u2191")
# CliffWalking: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTION_ARROWS_CLIFF_WALKING = ("\u2191", "\u2192", "\u2193", "\u2190")
# Taxi-v3: 0=S, 1=N, 2=E, 3=W, 4=Pickup, 5=Dropoff
ACTION_ARROWS_TAXI = ("\u2193", "\u2191", "\u2192", "\u2190", "P", "D")

def epsilon_greedy_action_selection(env, q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        best_actions = np.flatnonzero(q_table[state] == np.max(q_table[state]))
        return np.random.choice(best_actions)
    
def epsilon_decay(epsilon, epsilon_factor, epsilon_min=0.01):
    return max(epsilon * epsilon_factor, epsilon_min)


def epsilon_decay_factor(epsilon_start, epsilon_min, decay_episodes):
    decay_episodes = max(1, int(decay_episodes))
    return (epsilon_min / epsilon_start) ** (1 / max(1, decay_episodes - 1))


def episode_based_epsilon(episode_index, total_episodes, epsilon_start, epsilon_min, decay_fraction=0.9):
    decay_episodes = max(1, int(total_episodes * decay_fraction))
    if episode_index >= decay_episodes:
        return epsilon_min
    decay_factor = epsilon_decay_factor(epsilon_start, epsilon_min, decay_episodes)
    return epsilon_start * (decay_factor ** episode_index)


def random_action_selection(env):
    return env.action_space.sample()


def train_random(env, episodes, rewards=None, lengths=None, seed=None, dashboard=None, refresh_every=None, env_for_dashboard=None):
    if rewards is None:
        rewards = []
    if lengths is None:
        lengths = []
    visit_counts = np.zeros(env.observation_space.n, dtype=int)
    for episode in range(episodes):
        if seed is not None:
            state, _ = env.reset(seed=seed + episode)
        else:
            state, _ = env.reset()
        done = False
        total_reward = 0
        total_step = 0
        while not done:
            visit_counts[state] += 1
            action = random_action_selection(env)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = new_state
            total_reward += reward
            total_step += 1
            if dashboard is not None and env_for_dashboard is not None:
                if hasattr(dashboard, "capture_env_frame"):
                    dashboard.capture_env_frame(env_for_dashboard, store_for_gif=True)
        rewards.append(total_reward)
        lengths.append(total_step)
        if dashboard is not None and refresh_every is not None and env_for_dashboard is not None:
            if hasattr(dashboard, "should_refresh") and dashboard.should_refresh(episode, episodes):
                dashboard.capture_env_frame(env_for_dashboard)
                if hasattr(dashboard, "update"):
                    dashboard.update(rewards, lengths, [], None, visit_counts, q_table=None)
        print(f"Episode {episode}/{episodes} - Reward: {total_reward} - Length: {total_step}")
    return visit_counts, rewards, lengths

def q_learning_update(q_table, state, action, reward, new_state, alpha, gamma):
    q_table[state, action] = q_table[state, action] + alpha * (
        reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
    )

def sarsa_update(q_table, state, action, reward, new_state, new_action, alpha, gamma):
    q_table[state, action] = q_table[state, action] + alpha * (
        reward + gamma * q_table[new_state, new_action] - q_table[state, action]
    )

def train_q_learning(
    env,
    q_table,
    episodes,
    alpha,
    gamma,
    epsilon=None,
    epsilon_factor=None,
    rewards=None,
    lengths=None,
    seed=None,
    epsilon_schedule=None,
    dashboard=None,
    refresh_every=None,
    env_for_dashboard=None,
    epsilon_start=None,
    epsilon_min=None,
):
    if rewards is None:
        rewards = []
    if lengths is None:
        lengths = []
    epsilon_values = []
    visit_counts = np.zeros(q_table.shape[0], dtype=int)
    for episode in range(episodes):
        if seed is not None:
            state, _ = env.reset(seed=seed + episode)
        else:
            state, _ = env.reset()
        if epsilon_schedule is not None:
            start = epsilon_start if epsilon_start is not None else (epsilon if epsilon is not None else 1.0)
            minimum = epsilon_min if epsilon_min is not None else 0.01
            current_epsilon = epsilon_schedule(episode, episodes, start, minimum)
        else:
            current_epsilon = epsilon if epsilon is not None else 1.0
        done = False
        total_reward = 0
        total_step = 0
        while not done:
            visit_counts[state] += 1
            action = epsilon_greedy_action_selection(env, q_table, state, current_epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            q_learning_update(q_table, state, action, reward, new_state, alpha, gamma)
            state = new_state
            total_reward += reward
            total_step += 1
        rewards.append(total_reward)
        lengths.append(total_step)
        epsilon_values.append(current_epsilon)
        if dashboard is not None and refresh_every is not None and env_for_dashboard is not None:
            if hasattr(dashboard, "should_refresh") and dashboard.should_refresh(episode, episodes):
                dashboard.capture_env_frame(env_for_dashboard)
                policy = np.argmax(q_table, axis=1)
                if hasattr(dashboard, "update"):
                    dashboard.update(rewards, lengths, epsilon_values, policy, visit_counts, q_table=q_table)
        print(f"Episode {episode}/{episodes} - Reward: {total_reward} - Length: {total_step} - Epsilon: {current_epsilon:.4f}")
    return q_table, visit_counts, rewards, lengths, epsilon_values

def train_sarsa(
    env,
    q_table,
    episodes,
    alpha,
    gamma,
    epsilon_start,
    epsilon_min,
    rewards=None,
    lengths=None,
    seed=None,
    epsilon_schedule=None,
    dashboard=None,
    refresh_every=None,
    env_for_dashboard=None,
):
    if rewards is None:
        rewards = []
    if lengths is None:
        lengths = []
    epsilon_values = []
    visit_counts = np.zeros(q_table.shape[0], dtype=int)
    for episode in range(episodes):
        if seed is not None:
            state, _ = env.reset(seed=seed + episode)
        else:
            state, _ = env.reset()
        if epsilon_schedule is not None:
            epsilon = epsilon_schedule(episode, episodes, epsilon_start, epsilon_min)
        else:
            epsilon = epsilon_start
        done = False
        total_reward = 0
        total_step = 0
        while not done:
            visit_counts[state] += 1
            action = epsilon_greedy_action_selection(env, q_table, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_action = epsilon_greedy_action_selection(env, q_table, new_state, epsilon)
            sarsa_update(q_table, state, action, reward, new_state, new_action, alpha, gamma)
            state = new_state
            total_reward += reward
            total_step += 1
        rewards.append(total_reward)
        lengths.append(total_step)
        epsilon_values.append(epsilon)
        if dashboard is not None and refresh_every is not None and env_for_dashboard is not None:
            if hasattr(dashboard, "should_refresh") and dashboard.should_refresh(episode, episodes):
                dashboard.capture_env_frame(env_for_dashboard)
                policy = np.argmax(q_table, axis=1)
                if hasattr(dashboard, "update"):
                    dashboard.update(rewards, lengths, epsilon_values, policy, visit_counts, q_table=q_table)
        print(f"Episode {episode}/{episodes} - Reward: {total_reward} - Length: {total_step} - Epsilon: {epsilon:.4f}")
    return q_table, visit_counts, rewards, lengths, epsilon_values

def evaluate_agent(env, q_table, m):
    total_rewards = 0
    for episode in range(m):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(q_table[state])
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
            total_rewards += reward
            done = terminated or truncated
    return total_rewards / m