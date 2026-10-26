"""Unit tests for continuous PPO implementation in src/agents/ppo.py."""

import os
import sys

# Ensure repository root is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch
from src.agents.ppo import PPOAgent, ActorCritic, RolloutBuffer


def test_actor_critic_shapes():
    """Verify ActorCritic outputs correct shapes and bounds."""
    model = ActorCritic(obs_dim=4, action_dim=1, hidden_dim=32)
    state = torch.randn(5, 4)

    value = model.get_value(state)
    assert value.shape == (5, 1)

    action, log_prob, entropy, val = model.get_action_and_value(state)
    assert action.shape == (5, 1)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert log_prob.shape == (5,)
    assert entropy.shape == (5,)
    assert val.shape == (5,)


def test_rollout_buffer_gae():
    """Verify GAE and return computation in RolloutBuffer."""
    buffer = RolloutBuffer(buffer_size=10, obs_dim=4, action_dim=1)

    for i in range(10):
        s = np.zeros(4, dtype=np.float32)
        a = np.zeros(1, dtype=np.float32)
        buffer.insert(state=s, action=a, reward=1.0, done=(i == 9), log_prob=-0.5, value=0.5)

    assert buffer.full is True
    buffer.compute_gae(last_value=0.5, last_done=True, gamma=0.99, gae_lambda=0.95)

    assert np.all(np.isfinite(buffer.advantages))
    assert np.all(np.isfinite(buffer.returns))


def test_ppo_agent_step_and_update(tmp_path):
    """Verify select_action, update loop, and checkpointing."""
    agent = PPOAgent(obs_dim=4, action_dim=1, hidden_dim=32, batch_size=4, update_epochs=2, buffer_size=16, device="cpu")

    obs = np.array([7.0, 0.0, 10.0, 4.0], dtype=np.float32)
    action, log_prob, val = agent.select_action(obs, deterministic=False)

    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0
    assert isinstance(log_prob, float)
    assert isinstance(val, float)

    # Fill small buffer
    for _ in range(16):
        agent.buffer.insert(obs, action, 1.0, False, log_prob, val)

    agent.buffer.compute_gae(last_value=val, last_done=False)
    metrics = agent.update()

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert np.isfinite(metrics["policy_loss"])

    # Test checkpoint save and load
    ckpt_path = str(tmp_path / "test_ppo.pt")
    agent.save_checkpoint(ckpt_path)
    assert os.path.exists(ckpt_path)
    agent.load_checkpoint(ckpt_path)
