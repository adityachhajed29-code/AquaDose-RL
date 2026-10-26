"""Training script for PPO continuous chemical dosing controller."""

import argparse
import os
import sys

# Ensure repository root is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.envs.cstr_env import CSTRpHEnv
from src.agents.ppo_agent import PPOAgent
from src.utils.logger import TelemetryLogger


def train_ppo(
    total_episodes: int = 150,
    steps_per_episode: int = 200,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    seed: int = 42,
) -> PPOAgent:
    """Train PPO Agent on CSTR pH neutralization environment."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = CSTRpHEnv(max_steps=steps_per_episode)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=64,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
    )

    logger = TelemetryLogger(experiment_name="train_ppo", log_dir=log_dir)

    print(f"[PPO Training] Starting training for {total_episodes} episodes...")

    best_return = -float("inf")

    for ep in range(1, total_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0

        for step in range(steps_per_episode):
            action, log_prob, val = agent.select_action(obs)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            agent.buffer.store(obs, action, log_prob, reward, done, val)
            logger.log_step(episode=ep, step=step, reward=reward, info=step_info)

            obs = next_obs
            ep_reward += reward

            if done:
                break

        # Calculate bootstrap value for GAE
        _, _, last_val = agent.select_action(obs)
        update_info = agent.update(last_value=last_val)

        if ep % 10 == 0 or ep == 1:
            actor_loss = update_info.get("actor_loss", 0.0)
            critic_loss = update_info.get("critic_loss", 0.0)
            print(
                f"Episode {ep:03d}/{total_episodes:03d} | "
                f"Return: {ep_reward:7.2f} | Final pH: {step_info['ph']:4.2f} | "
                f"Actor Loss: {actor_loss:6.4f} | Critic Loss: {critic_loss:6.4f}"
            )

        if ep_reward > best_return:
            best_return = ep_reward
            agent.save(os.path.join(checkpoint_dir, "ppo_best.pt"))

    agent.save(os.path.join(checkpoint_dir, "ppo_final.pt"))
    logger.save_csv("ppo_training_telemetry.csv")
    print(f"[PPO Training] Complete. Best Return: {best_return:.2f}. Model saved to {checkpoint_dir}/")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent for AquaDose-RL")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_ppo(total_episodes=args.episodes, steps_per_episode=args.steps, seed=args.seed)
