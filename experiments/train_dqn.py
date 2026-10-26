"""Training script for DQN discrete dosing baseline."""

import argparse
import os
import sys

# Ensure repository root is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.cstr_env import CSTRpHEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.logger import TelemetryLogger


def train_dqn(
    total_episodes: int = 150,
    steps_per_episode: int = 200,
    num_actions: int = 11,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    seed: int = 42,
) -> DQNAgent:
    """Train DQN Agent on discretized CSTR environment."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = CSTRpHEnv(
        action_type="discrete",
        num_discrete_actions=num_actions,
        max_steps=steps_per_episode,
    )

    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        num_actions=num_actions,
        hidden_dim=64,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
    )

    logger = TelemetryLogger(experiment_name="train_dqn", log_dir=log_dir)

    print(f"[DQN Training] Starting training for {total_episodes} episodes...")

    best_return = -float("inf")

    for ep in range(1, total_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0

        for step in range(steps_per_episode):
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()

            logger.log_step(episode=ep, step=step, reward=reward, info=step_info)

            obs = next_obs
            ep_reward += reward

            if done:
                break

        if ep % 10 == 0 or ep == 1:
            print(
                f"Episode {ep:03d}/{total_episodes:03d} | "
                f"Return: {ep_reward:7.2f} | Final pH: {step_info['ph']:4.2f} | "
                f"Epsilon: {agent.epsilon:4.3f}"
            )

        if ep_reward > best_return:
            best_return = ep_reward
            agent.save(os.path.join(checkpoint_dir, "dqn_best.pt"))

    agent.save(os.path.join(checkpoint_dir, "dqn_final.pt"))
    logger.save_csv("dqn_training_telemetry.csv")
    print(f"[DQN Training] Complete. Best Return: {best_return:.2f}. Model saved to {checkpoint_dir}/")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent for AquaDose-RL")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_dqn(total_episodes=args.episodes, steps_per_episode=args.steps, seed=args.seed)
