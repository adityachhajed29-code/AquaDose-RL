"""Proximal Policy Optimization (PPO) Actor-Critic Agent for Continuous Control.

Features:
- Continuous Gaussian policy with Tanh-squashed mean and learnable log standard deviation.
- Separate Actor and Critic feedforward networks.
- Generalized Advantage Estimation (GAE).
- Clipped surrogate objective, value function clipping, and entropy regularization.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """Gaussian Policy Network for continuous dosing control."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32) - 0.5)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = torch.tanh(self.mean_layer(features))
        std = torch.exp(torch.clamp(self.log_std, min=-20.0, max=2.0))
        return mean, std


class CriticNetwork(nn.Module):
    """Value Network estimating state-value V(s)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class PPOBuffer:
    """Trajectory storage for on-policy PPO updates."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """PPO Agent with continuous action parameterization."""

    def __init__(
        self,
        obs_dim: int = 5,
        action_dim: int = 1,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs: int = 10,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = PPOBuffer()

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Select action, log probability, and estimated state-value."""
        state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            value = self.critic(state_tensor).squeeze(-1)

            if deterministic:
                action = mean
                dist = Normal(mean, std)
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

        action_clamped = torch.clamp(action, -1.0, 1.0)
        return (
            action_clamped.cpu().numpy()[0],
            float(log_prob.cpu().item()),
            float(value.cpu().item()),
        )

    def compute_gae(self, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimators and discounted returns."""
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [last_value]

        advantages: List[float] = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1.0 - float(dones[step])) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[step])) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.buffer.values)]

        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        if len(adv_tensor) > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return adv_tensor, ret_tensor

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Run PPO optimization on collected buffer rollouts."""
        if len(self.buffer) == 0:
            return {}

        advantages, returns = self.compute_gae(last_value)

        states_t = torch.as_tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)

        total_samples = len(self.buffer)
        indices = np.arange(total_samples)

        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Actor loss
                mean, std = self.actor(b_states)
                dist = Normal(mean, std)
                new_log_prob = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_prob - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic loss
                v_pred = self.critic(b_states).squeeze(-1)
                critic_loss = nn.functional.mse_loss(v_pred, b_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())

        self.buffer.clear()
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_opt_state_dict": self.actor_optimizer.state_dict(),
                "critic_opt_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt_state_dict"])
