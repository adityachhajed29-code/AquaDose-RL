"""Complete PyTorch implementation of continuous-action Proximal Policy Optimization (PPO).

Architecture & Components:
1. Actor-Critic Network:
   - Actor Head: MLP mapping state (4D) to action mean mu(s) with trainable log_std parameter.
     Action sampled from N(mu, sigma^2) and bounded to [-1.0, 1.0].
   - Critic Head: MLP mapping state (4D) to scalar baseline state-value V(s).
2. Rollout Buffer:
   - Stores fixed horizon trajectories of (s, a, r, s', done, log_prob, value).
   - Computes Generalized Advantage Estimation (GAE-lambda) and discounted returns.
3. PPO Update Engine:
   - Clipped surrogate policy objective: L_CLIP(theta).
   - Value function MSE loss.
   - Entropy regularization bonus.
   - Mini-batch SGD/Adam with gradient norm clipping (max_grad_norm=0.5).
4. Core API:
   - select_action(state, deterministic=False)
   - evaluate_policy(env, episodes=5)
   - save_checkpoint(path) / load_checkpoint(path)
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal layer initialization with constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Actor-Critic Neural Network for continuous control."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Critic Network: V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Actor Network: mu(s)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        # Trainable log standard deviation parameter initialized for controlled continuous dosing
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 2.3)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Zero-center and scale observation components [pH, delta_pH, Q_in, pH_in]."""
        if state.shape[-1] == 4:
            mean = torch.tensor([7.0, 0.0, 10.0, 7.0], dtype=torch.float32, device=state.device)
            std = torch.tensor([3.5, 2.0, 5.0, 3.5], dtype=torch.float32, device=state.device)
            return (state - mean) / std
        return state

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        norm_state = self._normalize_state(state)
        return self.critic(norm_state)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action, log probability, entropy, and value estimate."""
        norm_state = self._normalize_state(state)
        mean = torch.tanh(self.actor_mean(norm_state))

        # Deadband squashing near neutrality: if tank pH and influent are nominal, idle cleanly
        if state.shape[-1] >= 1:
            ph = state[..., 0:1]
            in_neutral = (ph >= 6.6) & (ph <= 8.4)
            if state.shape[-1] >= 4:
                ph_in = state[..., 3:4]
                in_neutral = in_neutral & (ph_in >= 5.8) & (ph_in <= 8.2)
            mean = torch.where(in_neutral & (torch.abs(mean) < 0.25), torch.zeros_like(mean), mean)
        log_std = torch.clamp(self.actor_logstd.expand_as(mean), min=-20.0, max=2.0)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        if action is None:
            if deterministic:
                raw_action = mean
            else:
                raw_action = dist.sample()
            action = torch.clamp(raw_action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(norm_state).squeeze(-1)

        return action, log_prob, entropy, value


class RolloutBuffer:
    """Fixed-horizon trajectory storage with GAE calculation."""

    def __init__(self, buffer_size: int = 2048, obs_dim: int = 4, action_dim: int = 1):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        """Reset storage buffers."""
        self.states = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)

        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        self.step = 0
        self.full = False

    def insert(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store step transition in buffer."""
        idx = self.step
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.log_probs[idx] = log_prob
        self.values[idx] = value

        self.step += 1
        if self.step >= self.buffer_size:
            self.full = True

    def compute_gae(self, last_value: float, last_done: bool, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        last_gae_lam = 0.0
        for t in reversed(range(self.step)):
            if t == self.step - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns[: self.step] = self.advantages[: self.step] + self.values[: self.step]


class PPOAgent:
    """Continuous Proximal Policy Optimization (PPO) Agent."""

    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 1,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        clip_vloss: bool = False,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: Optional[str] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Actor-Critic network
        self.network = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Rollout storage
        self.buffer = RolloutBuffer(buffer_size=buffer_size, obs_dim=obs_dim, action_dim=action_dim)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Select action, log-prob, and value estimate for current state.

        Args:
            state: Environment observation array (obs_dim,).
            deterministic: If True, returns action mean mu(s) without sampling.

        Returns:
            Tuple of (action_array, log_probability, state_value).
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t, _, value_t = self.network.get_action_and_value(
                state_t, deterministic=deterministic
            )

        action = action_t.cpu().numpy()[0]
        log_prob = float(log_prob_t.cpu().item())
        value = float(value_t.cpu().item())
        return action, log_prob, value

    def update(self) -> Dict[str, float]:
        """Execute PPO policy and value updates over collected rollout buffer."""
        num_samples = self.buffer.step
        if num_samples == 0:
            return {}

        b_states = torch.as_tensor(self.buffer.states[:num_samples], dtype=torch.float32, device=self.device)
        b_actions = torch.as_tensor(self.buffer.actions[:num_samples], dtype=torch.float32, device=self.device)
        b_log_probs = torch.as_tensor(self.buffer.log_probs[:num_samples], dtype=torch.float32, device=self.device)
        b_advantages = torch.as_tensor(self.buffer.advantages[:num_samples], dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(self.buffer.returns[:num_samples], dtype=torch.float32, device=self.device)
        b_values = torch.as_tensor(self.buffer.values[:num_samples], dtype=torch.float32, device=self.device)

        # Advantage normalization
        if len(b_advantages) > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Mini-batch PPO optimization
        indices = np.arange(num_samples)
        pg_losses, v_losses, entropies, approx_kls = [], [], [], []

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    b_states[mb_idx], b_actions[mb_idx]
                )

                log_ratio = new_log_prob - b_log_probs[mb_idx]
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence for monitoring
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())

                # Clipped surrogate objective
                mb_adv = b_advantages[mb_idx]
                surr1 = -mb_adv * ratio
                surr2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = torch.max(surr1, surr2).mean()

                # Value function loss (with optional value clipping)
                if self.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        new_value - b_values[mb_idx],
                        -self.clip_ratio,
                        self.clip_ratio,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                actor_loss = policy_loss - self.entropy_coef * entropy_loss
                critic_loss = self.value_coef * v_loss

                self.optimizer.zero_grad()
                (actor_loss + critic_loss).backward()

                actor_params = list(self.network.actor_mean.parameters()) + [self.network.actor_logstd]
                critic_params = list(self.network.critic.parameters())
                nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(policy_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())

        self.buffer.reset()

        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
        }

    def evaluate_policy(self, env: Any, episodes: int = 5, seed: int = 100) -> Dict[str, float]:
        """Evaluate agent policy on environment over multiple deterministic episodes."""
        returns = []
        reagents = []
        violations = 0
        total_steps = 0
        abs_errors = []

        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            ep_return = 0.0
            ep_reagent = 0.0

            done = False
            while not done:
                action, _, _ = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = bool(terminated or truncated)

                ep_return += float(reward)
                total_steps += 1

                ph = float(step_info.get("ph", obs[0]))
                abs_errors.append(abs(ph - 7.0))
                if ph < 6.5 or ph > 8.5:
                    violations += 1

                step_reagent = float(
                    step_info.get("step_reagent", abs(float(action[0])) * getattr(env, "dt", 0.1))
                )
                ep_reagent += step_reagent

            returns.append(ep_return)
            reagents.append(ep_reagent)

        violation_pct = (violations / total_steps * 100.0) if total_steps > 0 else 0.0

        return {
            "mean_reward": float(np.mean(returns)),
            "std_reward": float(np.std(returns)),
            "mean_chemical_liters": float(np.mean(reagents)),
            "compliance_violation_pct": float(violation_pct),
            "mean_abs_error": float(np.mean(abs_errors)),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint and optimizer state."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "actor_logstd": self.network.actor_logstd.data,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
