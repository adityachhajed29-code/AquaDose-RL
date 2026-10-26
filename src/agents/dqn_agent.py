"""Deep Q-Network (DQN) Baseline for Discretized Chemical Dosing Control.

Features:
- Multi-layer perceptron Q-Network estimating discrete dosing action values.
- Experience Replay Buffer.
- Polyak soft target network update.
- Epsilon-greedy exploration with exponential decay.
"""

from collections import deque
import random
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Q-Network mapping states to action-values Q(s, a)."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    """Fixed-capacity experience replay buffer."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for discretized chemical dosing environments."""

    def __init__(
        self,
        obs_dim: int = 5,
        num_actions: int = 11,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_tau: float = 0.005,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_tau = target_tau
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_net = QNetwork(obs_dim, num_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, num_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Select action via epsilon-greedy strategy."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            return int(torch.argmax(q_values, dim=-1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """Sample batch from replay buffer and perform DQN gradient step."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q estimates
        current_q = self.q_net(s_t).gather(1, a_t)

        # Target Q calculation with target network
        with torch.no_grad():
            max_next_q = self.target_net(ns_t).max(1)[0].unsqueeze(1)
            target_q = r_t + (1.0 - d_t) * self.gamma * max_next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network weights
        for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.target_tau * param.data + (1.0 - self.target_tau) * target_param.data)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {"loss": float(loss.item()), "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
