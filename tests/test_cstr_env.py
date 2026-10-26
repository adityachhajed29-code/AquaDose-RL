"""Unit tests for the CSTR pH neutralization Gymnasium environment."""

import numpy as np
import pytest
from src.envs.cstr_env import CSTRpHEnv


def test_env_initialization():
    """Verify default spaces and initial attributes."""
    env = CSTRpHEnv()
    assert env.observation_space.shape == (5,)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_discrete_env_initialization():
    """Verify discrete mode spaces."""
    env = CSTRpHEnv(action_type="discrete", num_discrete_actions=7)
    assert env.action_space.n == 7


def test_reset_behavior():
    """Verify reset returns valid observation and info dict."""
    env = CSTRpHEnv()
    obs, info = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert 0.0 <= obs[0] <= 14.0  # pH within realistic limits
    assert "ph" in info
    assert "target_ph" in info
    assert "error" in info


def test_step_contract():
    """Verify step return structure and bounds."""
    env = CSTRpHEnv()
    obs, _ = env.reset(seed=42)

    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reagent_step" in info
    assert "cumulative_reagent" in info


def test_base_dosing_increases_ph():
    """Adding base to an acidic reactor must increase or buffer pH upward."""
    env = CSTRpHEnv(stochastic_disturbance=False)
    env.reset(seed=42)
    env.current_ph = 4.0
    env.wa = env._ph_to_wa(4.0)

    # Apply maximum base dosing (action = 1.0)
    for _ in range(5):
        obs, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))

    assert info["ph"] > 4.0


def test_acid_dosing_decreases_ph():
    """Adding acid to an alkaline reactor must decrease pH."""
    env = CSTRpHEnv(stochastic_disturbance=False)
    env.reset(seed=42)
    env.current_ph = 10.0
    env.wa = env._ph_to_wa(10.0)

    # Apply maximum acid dosing (action = -1.0)
    for _ in range(5):
        obs, _, _, _, info = env.step(np.array([-1.0], dtype=np.float32))

    assert info["ph"] < 10.0


def test_episode_truncation():
    """Verify environment truncates when reaching max_steps."""
    max_steps = 10
    env = CSTRpHEnv(max_steps=max_steps)
    env.reset(seed=42)

    truncated = False
    for _ in range(max_steps):
        _, _, _, truncated, _ = env.step(np.array([0.0], dtype=np.float32))

    assert truncated is True
