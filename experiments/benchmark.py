"""Comparative benchmark runner: PID vs. PPO vs. DQN under identical disturbances."""

import argparse
import os
import sys
from typing import Dict

# Ensure repository root is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.envs.cstr_env import CSTRpHEnv
from src.agents.pid_controller import PIDController
from src.agents.ppo_agent import PPOAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.logger import TelemetryLogger
from src.utils.plotting import plot_benchmark_comparison, plot_episode_trajectory


def evaluate_agent(
    name: str,
    agent_type: str,
    agent_obj,
    num_episodes: int = 5,
    steps_per_episode: int = 200,
    seed: int = 100,
    log_dir: str = "logs",
) -> Dict[str, float]:
    """Evaluate an agent across multiple evaluation episodes."""
    is_discrete = (agent_type == "dqn")
    env = CSTRpHEnv(
        action_type="discrete" if is_discrete else "continuous",
        max_steps=steps_per_episode,
        stochastic_disturbance=True,
    )
    logger = TelemetryLogger(experiment_name=f"eval_{name.lower()}", log_dir=log_dir)

    all_errors = []
    all_reagents = []
    all_returns = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(agent_obj, "reset"):
            agent_obj.reset()

        ep_reward = 0.0
        for step in range(steps_per_episode):
            if agent_type == "pid":
                act = agent_obj.get_action(obs)
            elif agent_type == "ppo":
                act, _, _ = agent_obj.select_action(obs, deterministic=True)
            elif agent_type == "dqn":
                act = agent_obj.select_action(obs, eval_mode=True)
            else:
                act = 0.0

            next_obs, reward, terminated, truncated, step_info = env.step(act)
            logger.log_step(episode=ep, step=step, reward=reward, info=step_info)

            all_errors.append(abs(step_info["ph"] - step_info["target_ph"]))
            ep_reward += reward
            obs = next_obs

            if terminated or truncated:
                all_reagents.append(step_info["cumulative_reagent"])
                all_returns.append(ep_reward)
                break

    # Save trajectory plot for the final episode
    df = logger.to_dataframe()
    last_ep_df = df[df["episode"] == num_episodes]
    plot_episode_trajectory(
        last_ep_df,
        title=f"{name} Control Response (Episode {num_episodes})",
        save_path=os.path.join(log_dir, f"{name.lower()}_trajectory.png"),
    )

    return {
        "mean_abs_error": float(np.mean(all_errors)),
        "total_reagent": float(np.mean(all_reagents)),
        "mean_reward": float(np.mean(all_returns)),
    }


def run_benchmark(
    episodes: int = 5,
    steps: int = 200,
    ppo_weights: str = "checkpoints/ppo_best.pt",
    dqn_weights: str = "checkpoints/dqn_best.pt",
    results_dir: str = "results",
) -> None:
    """Run full comparative benchmark across PID, PPO, and DQN."""
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(" Running AquaDose-RL Controller Benchmark (PID vs PPO vs DQN)")
    print("=" * 60)

    # 1. PID Baseline
    pid = PIDController(kp=0.8, ki=0.15, kd=0.05)
    print("\nEvaluating PID Controller baseline...")
    pid_metrics = evaluate_agent("PID", "pid", pid, num_episodes=episodes, steps_per_episode=steps, log_dir=results_dir)

    # 2. PPO Agent
    ppo = PPOAgent()
    if os.path.exists(ppo_weights):
        print(f"\nLoading trained PPO weights from {ppo_weights}...")
        ppo.load(ppo_weights)
    else:
        print("\nNote: Trained PPO weights not found; evaluating baseline initialized PPO agent.")
    print("Evaluating PPO Controller...")
    ppo_metrics = evaluate_agent("PPO", "ppo", ppo, num_episodes=episodes, steps_per_episode=steps, log_dir=results_dir)

    # 3. DQN Agent
    dqn = DQNAgent()
    if os.path.exists(dqn_weights):
        print(f"\nLoading trained DQN weights from {dqn_weights}...")
        dqn.load(dqn_weights)
    else:
        print("\nNote: Trained DQN weights not found; evaluating baseline initialized DQN agent.")
    print("Evaluating DQN Controller...")
    dqn_metrics = evaluate_agent("DQN", "dqn", dqn, num_episodes=episodes, steps_per_episode=steps, log_dir=results_dir)

    results = {
        "PID": pid_metrics,
        "PPO": ppo_metrics,
        "DQN": dqn_metrics,
    }

    print("\n" + "=" * 60)
    print(f"{'Controller':<10} | {'MAE pH Error':<14} | {'Reagent (L)':<14} | {'Mean Return':<12}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<10} | {m['mean_abs_error']:<14.3f} | {m['total_reagent']:<14.2f} | {m['mean_reward']:<12.2f}")
    print("=" * 60)

    plot_benchmark_comparison(results, save_path=os.path.join(results_dir, "benchmark_comparison.png"))
    print(f"\nBenchmark plots saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaDose-RL Benchmark")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for benchmark results")
    args = parser.parse_args()

    run_benchmark(episodes=args.episodes, steps=args.steps, results_dir=args.results_dir)
