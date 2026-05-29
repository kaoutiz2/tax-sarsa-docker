from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.live_simple_plots import (
    ACTION_ARROWS_FROZEN_LAKE,
    rolling_mean,
    rolling_sum,
)


class LiveSimpleDashboard:
    def __init__(
        self,
        *,
        algo_label: str,
        env_label: str,
        n_rows: int,
        n_cols: int,
        window: int,
        graph_dir: str,
        file_prefix: str,
        min_epsilon: float,
        epsilon_decay: float,
        action_arrows: tuple[str, ...] = ACTION_ARROWS_FROZEN_LAKE,
        refresh_every: int = 100,
        max_gif_frames: int = 120,
        gif_duration_ms: int = 100,
        live_env_panel: bool = True,
    ) -> None:
        if len(action_arrows) < 4:
            raise ValueError("action_arrows doit contenir au moins 4 symboles (une par action)")
        self.algo_label = algo_label
        self.env_label = env_label
        self.action_arrows = action_arrows
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.window = window
        self.graph_dir = graph_dir
        self.file_prefix = file_prefix
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.refresh_every = max(1, refresh_every)
        self.max_gif_frames = max_gif_frames
        self.gif_duration_ms = gif_duration_ms
        self.live_env_panel = live_env_panel
        self._layout_done = False

        self.fig = None
        self.ax_reward = None
        self.ax_steps = None
        self.ax_q_heatmap = None
        self.ax_visit_heatmap = None
        self.ax_eps = None
        self.ax_env = None

        self._q_heatmap_im = None
        self._q_heatmap_cbar = None
        self._q_heatmap_texts: list = []
        self._visit_heatmap_im = None
        self._visit_heatmap_cbar = None
        self._visit_heatmap_texts: list = []
        self._env_im = None

        self._last_frame: np.ndarray | None = None
        self._gif_frames: list[np.ndarray] = []

    def setup(self) -> None:
        os.makedirs(self.graph_dir, exist_ok=True)
        plt.ion()
        self.fig = plt.figure(figsize=(18, 9))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25], hspace=0.38, wspace=0.32)
        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_steps = self.fig.add_subplot(gs[0, 1])
        self.ax_eps = self.fig.add_subplot(gs[0, 2])
        self.ax_visit_heatmap = self.fig.add_subplot(gs[1, 0])
        self.ax_q_heatmap = self.fig.add_subplot(gs[1, 1])
        self.ax_env = self.fig.add_subplot(gs[1, 2])
        self.fig.suptitle(f"{self.algo_label} — {self.env_label} (temps réel)", fontsize=13)
        self.ax_env.set_title("Environnement")
        self.ax_env.axis("off")
        self.fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08)
        self._layout_done = True
        plt.show(block=False)
        plt.pause(0.01)

    def capture_env_frame(self, env, *, store_for_gif: bool = False) -> None:
        """Un seul appel render() — à utiliser uniquement lors d'un refresh dashboard."""
        frame = env.render()
        if frame is None:
            return
        frame = np.asarray(frame, dtype=np.uint8)
        self._last_frame = frame
        if store_for_gif and len(self._gif_frames) < self.max_gif_frames:
            self._gif_frames.append(frame.copy())

    def record_rollout_gif(
        self,
        env,
        policy: np.ndarray,
        *,
        max_steps: int = 500,
        frame_stride: int = 1,
    ) -> None:
        """Construit le GIF en fin d'entraînement (un épisode greedy, une frame par pas par défaut)."""
        state, _ = env.reset()
        frames: list[np.ndarray] = []

        for t in range(max_steps):
            if t % frame_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))

            action = int(policy[state])
            state, _reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))
                break

        self._gif_frames = frames[: self.max_gif_frames]
        if frames:
            self._last_frame = frames[-1]

    def should_refresh(self, episode_index: int, total_episodes: int) -> bool:
        ep = episode_index + 1
        return ep == 1 or ep % self.refresh_every == 0 or ep == total_episodes

    def _update_env_panel(self) -> None:
        if self.ax_env is None or self._last_frame is None:
            return
        frame = self._last_frame
        if self._env_im is None:
            self.ax_env.clear()
            self.ax_env.axis("off")
            self._env_im = self.ax_env.imshow(frame)
            self.ax_env.set_title("Environnement (rendu live)")
        else:
            self._env_im.set_data(frame)
            if frame.ndim >= 2:
                h, w = frame.shape[0], frame.shape[1]
                self._env_im.set_extent((0, w, h, 0))

    def _render_heatmap_panel(
        self,
        ax,
        *,
        heat_grid: np.ndarray,
        policy_grid: np.ndarray | None,
        title: str,
        heat_label: str,
        texts: list,
        im,
        cbar,
        show_policy: bool,
    ):
        h_min, h_max = float(heat_grid.min()), float(heat_grid.max())
        h_range = h_max - h_min if h_max != h_min else 1.0

        if im is None:
            ax.clear()
            im = ax.imshow(heat_grid, cmap="Blues", aspect="equal")
            cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(heat_label)
            ax.set_title(title)
            ax.set_xticks(np.arange(self.n_cols))
            ax.set_yticks(np.arange(self.n_rows))
            ax.set_xlabel("Colonne")
            ax.set_ylabel("Ligne")
        else:
            im.set_data(heat_grid)
            im.set_clim(vmin=h_min, vmax=h_max if h_max != h_min else h_min + 1)
            if cbar is not None:
                cbar.set_label(heat_label)
            ax.set_title(title)

        for txt in texts:
            txt.remove()
        texts = []
        arrow_fontsize = max(8, min(18, 100 // max(self.n_rows, self.n_cols)))
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                cell_value = heat_grid[r, c]
                norm_val = (cell_value - h_min) / h_range
                text_color = "white" if norm_val > 0.5 else "black"
                if not show_policy or policy_grid is None:
                    cell_label = ""
                elif np.isclose(cell_value, 0.0):
                    cell_label = ""
                else:
                    action_idx = int(policy_grid[r, c])
                    cell_label = self.action_arrows[action_idx] if action_idx < len(self.action_arrows) else "?"
                txt = ax.text(
                    c,
                    r,
                    cell_label,
                    ha="center",
                    va="center",
                    fontsize=arrow_fontsize,
                    color=text_color,
                    fontweight="bold",
                )
                texts.append(txt)

        return im, cbar, texts

    def _q_table_to_grid(self, q_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_arr = np.asarray(q_table, dtype=float)
        state_count = self.n_rows * self.n_cols
        if q_arr.ndim == 2 and q_arr.shape[0] == state_count:
            return np.max(q_arr, axis=1).reshape(self.n_rows, self.n_cols), np.argmax(q_arr, axis=1).reshape(self.n_rows, self.n_cols)

        try:
            from utils.taxi_grid import q_to_grids

            return q_to_grids(q_arr)
        except Exception:
            flat = np.max(q_arr, axis=1) if q_arr.ndim == 2 else np.asarray(q_arr, dtype=float).ravel()
            if flat.size == state_count:
                return flat.reshape(self.n_rows, self.n_cols), np.zeros((self.n_rows, self.n_cols), dtype=int)
            raise ValueError(
                f"q_table incompatible avec une grille {self.n_rows}x{self.n_cols} (taille {q_arr.shape})"
            )

    def _visit_counts_to_grid(self, visit_counts: np.ndarray) -> np.ndarray:
        visits = np.asarray(visit_counts, dtype=float)
        state_count = self.n_rows * self.n_cols
        if visits.size == state_count:
            return visits.reshape(self.n_rows, self.n_cols)

        try:
            from utils.taxi_grid import visits_to_grid

            return visits_to_grid(visits)
        except Exception:
            raise ValueError(
                f"visit_counts incompatible avec une grille {self.n_rows}x{self.n_cols} (taille {visits.shape})"
            )

    def update(
        self,
        rewards: list,
        steps: list,
        epsilon_values: list,
        policy: np.ndarray | None,
        visit_counts: np.ndarray,
        q_table: np.ndarray | None = None,
    ) -> None:
        if self.fig is None:
            return

        ax_reward, ax_steps, ax_env, ax_visit_heatmap, ax_q_heatmap, ax_eps = (
            self.ax_reward,
            self.ax_steps,
            self.ax_env,
            self.ax_visit_heatmap,
            self.ax_q_heatmap,
            self.ax_eps,
        )

        # --- Récompenses cumulées (fenêtre glissante) ---
        ax_reward.clear()
        if len(rewards) >= self.window:
            x, rolling = rolling_sum(rewards, self.window)
            ax_reward.plot(x, rolling, color="#2563eb", linewidth=1.2)
        elif rewards:
            ax_reward.plot(range(1, len(rewards) + 1), rewards, color="#2563eb", linewidth=1.0, alpha=0.7)
        ax_reward.set_title("Récompenses cumulées (fenêtre glissante)")
        ax_reward.set_xlabel("Épisode")
        ax_reward.set_ylabel(f"Somme sur {self.window} épisodes")
        ax_reward.grid(True, alpha=0.3)

        # --- Pas par épisode (moyenne glissante) ---
        ax_steps.clear()
        if len(steps) >= 2:
            x, ma = rolling_mean(steps, min(self.window, len(steps)))
            ax_steps.plot(x, ma, color="#16a34a", linewidth=1.2)
        ax_steps.set_title("Déplacements moyens par épisode")
        ax_steps.set_xlabel("Épisode")
        ax_steps.set_ylabel(f"Moyenne sur {self.window} épisodes")
        ax_steps.grid(True, alpha=0.3)

        # --- Heatmap Q-table ---
        if q_table is not None:
            q_grid, policy_grid = self._q_table_to_grid(q_table)
            q_title = "Q-table (meilleure action)"
            q_label = "max_a Q(s, a)"
        else:
            q_grid = np.zeros((self.n_rows, self.n_cols), dtype=float)
            q_title = "Q-table absente"
            q_label = "max_a Q(s, a)"
            policy_grid = None

        self._q_heatmap_im, self._q_heatmap_cbar, self._q_heatmap_texts = self._render_heatmap_panel(
            ax_q_heatmap,
            heat_grid=q_grid,
            policy_grid=policy_grid,
            title=q_title,
            heat_label=q_label,
            texts=self._q_heatmap_texts,
            im=self._q_heatmap_im,
            cbar=self._q_heatmap_cbar,
            show_policy=True,
        )

        # --- Heatmap des visites ---
        visit_grid = self._visit_counts_to_grid(visit_counts)
        self._visit_heatmap_im, self._visit_heatmap_cbar, self._visit_heatmap_texts = self._render_heatmap_panel(
            ax_visit_heatmap,
            heat_grid=visit_grid,
            policy_grid=None,
            title="Cases visitées",
            heat_label="Nombre de visites",
            texts=self._visit_heatmap_texts,
            im=self._visit_heatmap_im,
            cbar=self._visit_heatmap_cbar,
            show_policy=False,
        )

        # --- Epsilon ---
        ax_eps.clear()
        if epsilon_values:
            ax_eps.plot(range(1, len(epsilon_values) + 1), epsilon_values, color="#9333ea", linewidth=1.0)
        ax_eps.axhline(
            self.min_epsilon,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"ε_min = {self.min_epsilon}",
        )
        ax_eps.set_title(f"Décroissance ε (decay = {self.epsilon_decay})")
        ax_eps.set_xlabel("Épisode")
        ax_eps.set_ylabel("epsilon")
        ax_eps.set_ylim(0, 1.05)
        ax_eps.legend(fontsize=8)
        ax_eps.grid(True, alpha=0.3)

        if self.live_env_panel:
            self._update_env_panel()

        if not self._layout_done:
            self.fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08)
            self._layout_done = True
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def save_gif(self) -> str | None:
        if not self._gif_frames:
            return None
        path = os.path.join(self.graph_dir, f"{self.file_prefix}_training.gif")
        pil_frames = [Image.fromarray(f) for f in self._gif_frames]
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=self.gif_duration_ms,
            loop=0,
        )
        print(f"  → {path} ({len(self._gif_frames)} frames)")
        return path

    def save_individual_plots(
        self,
        rewards: list,
        steps: list,
        epsilon_values: list,
        policy: np.ndarray | None,
        visit_counts: np.ndarray,
        q_table: np.ndarray | None = None,
        env=None,
        rollout_policy: np.ndarray | None = None,
    ) -> None:
        """Sauvegarde les PNG, le GIF (rollout final) et la figure complète."""
        if env is not None:
            gif_policy = rollout_policy if rollout_policy is not None else policy
            if gif_policy is not None:
                self.record_rollout_gif(env, gif_policy)
                if self.live_env_panel and self.fig is not None:
                    self._update_env_panel()
                    self.fig.canvas.draw_idle()

        prefix = self.file_prefix
        paths: list[str] = []

        arr = np.asarray(rewards, dtype=float)
        if len(arr) >= self.window:
            x, rolling = rolling_sum(rewards, self.window)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, rolling, color="#2563eb", linewidth=1.5)
            ax.set_title(f"{self.algo_label} — récompenses cumulées (fenêtre glissante)")
            ax.set_xlabel("Épisode")
            ax.set_ylabel(f"Somme sur {self.window} épisodes")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = os.path.join(self.graph_dir, f"{prefix}_reward_cumulative_ma.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)

        x, ma = rolling_mean(steps, self.window)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, ma, color="#16a34a", linewidth=1.5)
        ax.set_title(f"{self.algo_label} — déplacements moyens par épisode")
        ax.set_xlabel("Épisode")
        ax.set_ylabel(f"Moyenne sur {self.window} épisodes")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(self.graph_dir, f"{prefix}_steps_per_episode_ma.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        figsize = (18, 9)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25], hspace=0.38, wspace=0.32)
        ax_reward = fig.add_subplot(gs[0, 0])
        ax_steps = fig.add_subplot(gs[0, 1])
        eps_ax = fig.add_subplot(gs[0, 2])
        q_ax = fig.add_subplot(gs[1, 0])
        visit_ax = fig.add_subplot(gs[1, 1])
        ax_env = fig.add_subplot(gs[1, 2])

        if q_table is not None:
            q_grid, policy_grid = self._q_table_to_grid(q_table)
        else:
            q_grid = np.zeros((self.n_rows, self.n_cols), dtype=float)
            policy_grid = None
        visit_grid = self._visit_counts_to_grid(visit_counts)

        ax_reward.plot(x if len(arr) >= self.window else range(1, len(arr) + 1), rolling if len(arr) >= self.window else arr, color="#2563eb", linewidth=1.5)
        ax_reward.set_title(f"{self.algo_label} — récompenses cumulées")
        ax_reward.set_xlabel("Épisode")
        ax_reward.set_ylabel(f"Somme sur {self.window} épisodes")
        ax_reward.grid(True, alpha=0.3)

        ax_steps.plot(x, ma, color="#16a34a", linewidth=1.5)
        ax_steps.set_title(f"{self.algo_label} — déplacements moyens")
        ax_steps.set_xlabel("Épisode")
        ax_steps.set_ylabel(f"Moyenne sur {self.window} épisodes")
        ax_steps.grid(True, alpha=0.3)

        if self._last_frame is not None:
            ax_env.imshow(self._last_frame)
        ax_env.set_title(f"{self.algo_label} — environnement")
        ax_env.axis("off")

        if epsilon_values:
            eps_ax.plot(range(1, len(epsilon_values) + 1), epsilon_values, color="#9333ea", linewidth=1.2)
        eps_ax.axhline(self.min_epsilon, color="gray", linestyle="--", linewidth=1, label=f"ε_min = {self.min_epsilon}")
        eps_ax.set_title(f"{self.algo_label} — décroissance ε")
        eps_ax.set_xlabel("Épisode")
        eps_ax.set_ylabel("epsilon")
        eps_ax.set_ylim(0, 1.05)
        eps_ax.legend(fontsize=8)
        eps_ax.grid(True, alpha=0.3)

        q_im = q_ax.imshow(q_grid, cmap="Blues", aspect="equal")
        plt.colorbar(q_im, ax=q_ax, fraction=0.046, pad=0.04, label="max_a Q(s, a)" if q_table is not None else "Q-table absente")
        q_ax.set_title(f"{self.algo_label} — Q-table")
        q_ax.set_xlabel("Colonne")
        q_ax.set_ylabel("Ligne")
        q_ax.set_xticks(np.arange(self.n_cols))
        q_ax.set_yticks(np.arange(self.n_rows))

        visit_im = visit_ax.imshow(visit_grid, cmap="Blues", aspect="equal")
        plt.colorbar(visit_im, ax=visit_ax, fraction=0.046, pad=0.04, label="Nombre de visites")
        visit_ax.set_title(f"{self.algo_label} — visites")
        visit_ax.set_xlabel("Colonne")
        visit_ax.set_ylabel("Ligne")
        visit_ax.set_xticks(np.arange(self.n_cols))
        visit_ax.set_yticks(np.arange(self.n_rows))

        if q_table is not None and policy_grid is not None:
            h_min, h_max = q_grid.min(), q_grid.max()
            h_range = h_max - h_min if h_max != h_min else 1.0
            arrow_fontsize = max(10, min(22, 120 // max(self.n_rows, self.n_cols)))
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    q_value = q_grid[r, c]
                    norm_val = (q_value - h_min) / h_range
                    text_color = "white" if norm_val > 0.5 else "black"
                    cell_label = "" if np.isclose(q_value, 0.0) else self.action_arrows[int(policy_grid[r, c])] if int(policy_grid[r, c]) < len(self.action_arrows) else "?"
                    q_ax.text(c, r, cell_label, ha="center", va="center", fontsize=arrow_fontsize, color=text_color, fontweight="bold")

        v_min, v_max = visit_grid.min(), visit_grid.max()
        v_range = v_max - v_min if v_max != v_min else 1.0
        visit_fontsize = max(10, min(22, 120 // max(self.n_rows, self.n_cols)))
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                visit_value = visit_grid[r, c]
                norm_val = (visit_value - v_min) / v_range
                text_color = "white" if norm_val > 0.5 else "black"
                label = "" if np.isclose(visit_value, 0.0) else f"{int(visit_value)}"
                visit_ax.text(c, r, label, ha="center", va="center", fontsize=visit_fontsize, color=text_color, fontweight="bold")

        fig.suptitle(f"{self.algo_label} — Q-table, visites et environnement", fontsize=13)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.08, wspace=0.28, hspace=0.30)
        p = os.path.join(self.graph_dir, f"{prefix}_policy_visit_heatmap.png")
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(epsilon_values) + 1), epsilon_values, color="#9333ea", linewidth=1.2)
        ax.axhline(self.min_epsilon, color="gray", linestyle="--", linewidth=1, label=f"ε_min = {self.min_epsilon}")
        ax.set_title(f"{self.algo_label} — décroissance d'epsilon")
        ax.set_xlabel("Épisode")
        ax.set_ylabel("epsilon")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(self.graph_dir, f"{prefix}_epsilon_decay.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        gif_path = self.save_gif()
        if gif_path:
            paths.append(gif_path)

        if self.fig is not None:
            p = os.path.join(self.graph_dir, f"{prefix}_dashboard_live.png")
            self.fig.savefig(p, dpi=150, bbox_inches="tight")
            paths.append(p)

        for path in paths:
            if path != gif_path:
                print(f"  → {path}")

    def close(self) -> None:
        plt.ioff()
        if self.fig is not None:
            if os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes"):
                plt.close(self.fig)
            else:
                plt.show(block=True)
