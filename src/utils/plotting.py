"""Visualization utilities for chemical neutralization and agent performance.

Provides multi-panel plots for transient responses, chemical consumption,
and comparative controller benchmarks (PID vs PPO vs DQN).
"""

import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_episode_trajectory(
    df: pd.DataFrame,
    title: str = "Reactor Neutralization Trajectory",
    save_path: Optional[str] = None,
) -> None:
    """Plot multi-panel telemetry trajectory for an episode."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    steps = df["step"]

    # Panel 1: pH tracking
    axes[0].plot(steps, df["ph"], label="Reactor pH", color="#1f77b4", linewidth=2.0)
    axes[0].plot(
        steps,
        df["target_ph"],
        label="Target pH (7.0)",
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
    )
    axes[0].axhspan(6.5, 7.5, color="green", alpha=0.15, label="Safe Discharge Band")
    axes[0].set_ylabel("pH")
    axes[0].set_ylim(1.0, 13.0)
    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Panel 2: Reagent flows (Acid vs Base)
    axes[1].plot(
        steps,
        df["base_flow"],
        label="Base Dosing (NaOH L/min)",
        color="#0055ff",
        linewidth=1.5,
    )
    axes[1].plot(
        steps,
        df["acid_flow"],
        label="Acid Dosing (HCl L/min)",
        color="#d62728",
        linewidth=1.5,
    )
    axes[1].set_ylabel("Dosing Flow (L/min)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    # Panel 3: Cumulative reagent & reward
    ax3_twin = axes[2].twinx()
    p1 = axes[2].plot(
        steps,
        df["cumulative_reagent"],
        label="Cumulative Reagent (L)",
        color="#9467bd",
        linewidth=1.8,
    )
    p2 = ax3_twin.plot(
        steps,
        df["reward"],
        label="Step Reward",
        color="#ff7f0e",
        linestyle=":",
        alpha=0.8,
    )
    axes[2].set_xlabel("Time Step (dt = 0.1 min)")
    axes[2].set_ylabel("Total Reagent (L)", color="#9467bd")
    ax3_twin.set_ylabel("Reward", color="#ff7f0e")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    # Combine legends for twin axis
    plots = p1 + p2
    labels = [p.get_label() for p in plots]
    axes[2].legend(plots, labels, loc="upper right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_benchmark_comparison(
    benchmark_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparison across controllers for MAE, Reagent Used, and Total Reward."""
    agents = list(benchmark_metrics.keys())
    mae_values = [benchmark_metrics[a]["mean_abs_error"] for a in agents]
    reagent_values = [benchmark_metrics[a]["total_reagent"] for a in agents]
    reward_values = [benchmark_metrics[a]["mean_reward"] for a in agents]

    x = np.arange(len(agents))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(9, 5))

    rects1 = ax1.bar(x - width, mae_values, width, label="Mean Abs pH Error", color="#d62728")
    rects2 = ax1.bar(x, reagent_values, width, label="Total Reagent (L)", color="#1f77b4")

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, reward_values, width, label="Episode Return", color="#2ca02c")

    ax1.set_ylabel("Error / Reagent Volume")
    ax2.set_ylabel("Total Return")
    ax1.set_title("Controller Benchmark: PID vs PPO vs DQN", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
