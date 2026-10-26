"""Unit tests for PID, PPO, and DQN chemical dosing agents."""

import os
import sys

# Ensure repository root is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch
from src.agents.pid_controller import PIDController, run_pid_baseline
from src.agents.ppo_agent import PPOAgent
from src.agents.dqn_agent import DQNAgent
from src.envs.cstr_ph_env import CSTRpHEnv


def test_pid_controller_action():
    """Verify PID produces correct control sign for acidic pH (base dosing)."""
    pid = PIDController(kp=1.0, ki=0.1, kd=0.01)
    pid.reset()

    # pH = 4.0, target = 7.0 => error = +3.0 (needs base, action > 0)
    obs = np.array([4.0, 0.0, 10.0, 4.0], dtype=np.float32)
    act = pid.get_action(obs)

    assert isinstance(act, np.ndarray)
    assert act.shape == (1,)
    assert act[0] > 0.0, f"Expected positive action for base dosing, got {act[0]}"
    assert -1.0 <= act[0] <= 1.0


def test_pid_dual_split_alkaline_action():
    """Verify PID produces negative control sign for alkaline pH (acid dosing)."""
    pid = PIDController(kp=1.0, ki=0.1, kd=0.01)
    pid.reset()

    # pH = 10.5, target = 7.0 => error = -3.5 (needs acid, action < 0)
    obs = np.array([10.5, 0.0, 10.0, 10.5], dtype=np.float32)
    act = pid.get_action(obs)

    assert isinstance(act, np.ndarray)
    assert act.shape == (1,)
    assert act[0] < 0.0, f"Expected negative action for acid dosing, got {act[0]}"
    assert -1.0 <= act[0] <= 1.0


def test_pid_anti_windup():
    """Verify integrator does not exceed anti-windup bounds."""
    threshold = 2.0
    pid = PIDController(kp=1.0, ki=1.0, kd=0.0, anti_windup_threshold=threshold)
    pid.reset()

    # Repeatedly feed high error
    obs = np.array([2.0, 0.0, 10.0, 2.0], dtype=np.float32)
    for _ in range(50):
        pid.get_action(obs)

    assert abs(pid.integral) <= threshold


def test_pid_reset():
    """Verify reset() clears accumulated error and integrator history."""
    pid = PIDController(kp=1.0, ki=0.5, kd=0.1)
    obs = np.array([3.0, 0.0, 10.0, 3.0], dtype=np.float32)
    pid.get_action(obs)

    assert pid.integral != 0.0
    assert pid.initialized is True

    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error == 0.0
    assert pid.initialized is False


def test_run_pid_baseline_metrics():
    """Verify run_pid_baseline returns expected metrics and types."""
    env = CSTRpHEnv(max_steps=50)
    metrics = run_pid_baseline(env, episodes=2, seed=123)

    assert isinstance(metrics, dict)
    assert "mean_episodic_reward" in metrics
    assert "cumulative_chemical_liters" in metrics
    assert "compliance_violation_percentage" in metrics
    assert isinstance(metrics["mean_episodic_reward"], float)
    assert isinstance(metrics["cumulative_chemical_liters"], float)
    assert 0.0 <= metrics["compliance_violation_percentage"] <= 100.0


def test_ppo_agent_shapes():
    """Verify PPO action selection and buffer operations."""
    agent = PPOAgent(obs_dim=4, action_dim=1, hidden_dim=32, device="cpu")
    dummy_obs = np.array([6.5, 0.0, 10.0, 4.0], dtype=np.float32)

    action, log_prob, val = agent.select_action(dummy_obs, deterministic=False)
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0
    assert isinstance(log_prob, float)
    assert isinstance(val, float)

    # Store transition and check update
    agent.buffer.store(dummy_obs, action, log_prob, 1.0, False, val)
    assert len(agent.buffer) == 1


def test_dqn_agent_shapes():
    """Verify DQN discrete action selection and replay buffer."""
    num_actions = 7
    agent = DQNAgent(obs_dim=4, num_actions=num_actions, hidden_dim=32, device="cpu")
    dummy_obs = np.array([6.5, 0.0, 10.0, 4.0], dtype=np.float32)

    # Greedy action in eval mode
    action = agent.select_action(dummy_obs, eval_mode=True)
    assert isinstance(action, int)
    assert 0 <= action < num_actions

    # Transition push
    agent.store_transition(dummy_obs, action, 1.0, dummy_obs, False)
    assert len(agent.buffer) == 1
