"""Utilitaires d'environnement pour Taxi-v4."""

from __future__ import annotations

import gymnasium as gym


def create_environment(render_mode: str = "rgb_array"):
    """Crée l'environnement Taxi-v4 avec le mode de rendu demandé."""
    return gym.make("Taxi-v4", render_mode=render_mode)
