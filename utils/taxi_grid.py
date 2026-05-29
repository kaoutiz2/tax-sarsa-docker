"""Agrégation des états Taxi-v3 (500) sur la grille 5x5 du taxi."""

from __future__ import annotations

import numpy as np

TAXI_ROWS = 5
TAXI_COLS = 5
TAXI_GRID_SIZE = TAXI_ROWS * TAXI_COLS


def decode_state(state: int) -> tuple[int, int, int, int]:
    """Retourne (taxi_row, taxi_col, passenger_location, destination)."""
    s = int(state)
    dest = s % 4
    s //= 4
    pass_loc = s % 5
    s //= 5
    col = s % 5
    row = s // 5
    return row, col, pass_loc, dest


def visits_to_grid(visit_counts: np.ndarray) -> np.ndarray:
    """Somme des visites par case (row, col), forme (5, 5)."""
    grid = np.zeros(TAXI_GRID_SIZE, dtype=float)
    for state, count in enumerate(np.asarray(visit_counts, dtype=float)):
        row, col, _, _ = decode_state(state)
        grid[row * TAXI_COLS + col] += count
    return grid.reshape(TAXI_ROWS, TAXI_COLS)


def q_to_grids(q_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """max Q et meilleure action par case taxi, formes (5, 5)."""
    q_arr = np.asarray(q_table, dtype=float)
    heat = np.full(TAXI_GRID_SIZE, -np.inf)
    policy = np.zeros(TAXI_GRID_SIZE, dtype=int)

    for state in range(q_arr.shape[0]):
        row, col, _, _ = decode_state(state)
        idx = row * TAXI_COLS + col
        value = float(np.max(q_arr[state]))
        if value > heat[idx]:
            heat[idx] = value
            policy[idx] = int(np.argmax(q_arr[state]))

    heat = np.where(np.isfinite(heat), heat, 0.0)
    return heat.reshape(TAXI_ROWS, TAXI_COLS), policy.reshape(TAXI_ROWS, TAXI_COLS)


def full_policy(q_table: np.ndarray) -> np.ndarray:
    """Politique greedy sur les 500 états."""
    return np.argmax(np.asarray(q_table), axis=1)


def policy_to_grid(policy: np.ndarray, visit_counts: np.ndarray) -> np.ndarray:
    """Meilleure action par case, selon l'état le plus visité à cette position."""
    pol = np.asarray(policy, dtype=int)
    visits = np.asarray(visit_counts, dtype=float)
    grid = np.zeros(TAXI_GRID_SIZE, dtype=int)
    best_visits = np.zeros(TAXI_GRID_SIZE, dtype=float)

    for state, action in enumerate(pol):
        row, col, _, _ = decode_state(state)
        idx = row * TAXI_COLS + col
        if visits[state] >= best_visits[idx]:
            best_visits[idx] = visits[state]
            grid[idx] = action

    return grid.reshape(TAXI_ROWS, TAXI_COLS)
