"""Training script for PPO continuous-action agent on CSTR pH neutralization environment."""

import argparse
import csv
import os
import sys
from typing import Dict, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from src.envs.cstr_ph_env import CSTRpHEnv
from src.agents.ppo import PPOAgent

# Alias to support both casing conventions
CSTRPhEnv = CSTRpHEnv


def evaluate(
    env: CSTRpHEnv,
    agent: PPOAgent,
    num_episodes: int = 5,
    seed: int = 1000,
) -> Dict[str, float]:
    """Perform deterministic evaluation rollouts on the environment.

    Tracks:
    - Average episodic reward
    - Mean absolute pH deviation from setpoint (7.0)
    - Total chemical reagent used (|a|)
    - Number of regulatory violation steps (pH < 6.5 or pH > 8.5)
    - Compliance rate percentage
    """
    returns = []
    reagents = []
    abs_deviations = []
    violation_steps = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_reagent = 0.0

        done = False
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            total_steps += 1

            ph = float(step_info.get("ph", obs[0]))
            abs_deviations.append(abs(ph - 7.0))

            if ph < 6.5 or ph > 8.5:
                violation_steps += 1

            step_reagent = float(
                step_info.get(
                    "step_reagent",
                    abs(float(action[0])) * getattr(env, "dt", 0.1),
                )
            )
            ep_reagent += step_reagent

        returns.append(ep_return)
        reagents.append(ep_reagent)

    compliance_rate = (
        ((total_steps - violation_steps) / total_steps * 100.0) if total_steps > 0 else 0.0
    )

    return {
        "mean_reward": float(np.mean(returns)),
        "std_reward": float(np.std(returns)),
        "mean_abs_deviation": float(np.mean(abs_deviations)),
        "total_chemical_used": float(np.sum(reagents)),
        "mean_chemical_used": float(np.mean(reagents)),
        "violation_steps": int(violation_steps),
        "total_eval_steps": int(total_steps),
        "compliance_rate": float(compliance_rate),
    }


def train_ppo(
    total_timesteps: int = 100000,
    seed: int = 42,
    lr: float = 3e-4,
    gamma: float = 0.99,
    eval_freq: int = 4096,
    buffer_size: int = 2048,
    batch_size: int = 64,
    update_epochs: int = 10,
    reward_scale: float = 0.05,
    log_dir: str = "experiments/logs",
    checkpoint_dir: str = "experiments/checkpoints",
) -> PPOAgent:
    """Train continuous PPO agent on CSTRPhEnv environment."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = CSTRPhEnv(max_steps=1000)
    eval_env = CSTRPhEnv(max_steps=1000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr=lr,
        gamma=gamma,
        gae_lambda=0.95,
        clip_ratio=0.2,
        buffer_size=buffer_size,
        batch_size=batch_size,
        update_epochs=update_epochs,
    )

    csv_log_path = os.path.join(log_dir, "training_metrics.csv")
    csv_headers = [
        "episode",
        "timestep",
        "actor_loss",
        "critic_loss",
        "mean_reward",
        "compliance_rate",
        "mean_abs_deviation",
        "total_chemical_used",
        "violation_steps",
    ]

    with open(csv_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    best_checkpoint_path = os.path.join(checkpoint_dir, "best_ppo_model.pt")
    best_eval_reward = -float("inf")

    print("=" * 75)
    print(" AquaDose-RL: PPO Training on CSTR pH Environment")
    print("=" * 75)
    print(f"Total Timesteps: {total_timesteps} | Seed: {seed} | LR: {lr} | Gamma: {gamma}")
    print(f"Eval Frequency:  {eval_freq} steps | Buffer Horizon: {buffer_size}")
    print(f"Telemetry Log:   {csv_log_path}")
    print(f"Best Checkpoint: {best_checkpoint_path}")
    print("=" * 75)

    global_step = 0
    episode_idx = 0
    last_eval_step = 0

    obs, _ = env.reset(seed=seed)
    actor_loss = 0.0
    critic_loss = 0.0

    while global_step < total_timesteps:
        # Collect rollout trajectory into buffer
        for _ in range(buffer_size):
            action, log_prob, value = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            global_step += 1

            agent.buffer.insert(
                state=obs,
                action=action,
                reward=float(reward) * reward_scale,
                done=done,
                log_prob=float(log_prob),
                value=float(value),
            )

            obs = next_obs

            if done:
                episode_idx += 1
                obs, _ = env.reset(seed=seed + episode_idx)

            if global_step >= total_timesteps:
                break

        # Compute GAE bootstrap value and update policy
        state_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            last_value = float(agent.network.get_value(state_t).squeeze().item())

        agent.buffer.compute_gae(
            last_value=last_value,
            last_done=done,
            gamma=gamma,
            gae_lambda=agent.gae_lambda,
        )

        update_metrics = agent.update()
        actor_loss = update_metrics.get("policy_loss", 0.0)
        critic_loss = update_metrics.get("value_loss", 0.0)

        # Periodic evaluation trigger
        if global_step - last_eval_step >= eval_freq or global_step >= total_timesteps:
            last_eval_step = global_step
            eval_metrics = evaluate(eval_env, agent, num_episodes=5, seed=seed + 5000)

            mean_rew = eval_metrics["mean_reward"]
            comp_rate = eval_metrics["compliance_rate"]
            mean_dev = eval_metrics["mean_abs_deviation"]
            chem_used = eval_metrics["total_chemical_used"]
            viol_steps = eval_metrics["violation_steps"]

            print(
                f"[Step {global_step:06d}/{total_timesteps:06d} | Ep {episode_idx:04d}] "
                f"Eval Reward: {mean_rew:7.2f} | "
                f"Compliance: {comp_rate:5.1f}% | "
                f"pH Dev: {mean_dev:4.2f} | "
                f"Reagent: {chem_used:5.2f}L | "
                f"Violations: {viol_steps:03d} | "
                f"Actor Loss: {actor_loss:6.4f} | "
                f"Critic Loss: {critic_loss:6.4f}"
            )

            # Telemetry exporter: log to CSV
            with open(csv_log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode_idx,
                    global_step,
                    f"{actor_loss:.6f}",
                    f"{critic_loss:.6f}",
                    f"{mean_rew:.4f}",
                    f"{comp_rate:.2f}",
                    f"{mean_dev:.4f}",
                    f"{chem_used:.4f}",
                    viol_steps,
                ])

            # Auto-save best model weights
            if mean_rew > best_eval_reward:
                best_eval_reward = mean_rew
                agent.save_checkpoint(best_checkpoint_path)
                print(f"  [*] New best model saved with Eval Reward: {best_eval_reward:.2f} -> {best_checkpoint_path}")

    # Final model checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_ppo_model.pt")
    agent.save_checkpoint(final_checkpoint_path)
    print("=" * 75)
    print(f"Training completed. Best Eval Reward: {best_eval_reward:.2f}")
    print(f"Model saved to {best_checkpoint_path}")
    print(f"Telemetry logs recorded at {csv_log_path}")
    print("=" * 75)

    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent on CSTR pH Environment")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam optimizer (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--eval-freq", type=int, default=4096, help="Evaluation frequency in timesteps (default: 4096)")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Rollout buffer horizon (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--log-dir", type=str, default="experiments/logs", help="Directory for CSV telemetry logs")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints", help="Directory for checkpoints")

    args = parser.parse_args()

    train_ppo(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        eval_freq=args.eval_freq,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
