import os
import sys

# Ensure repository root is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.envs.cstr_ph_env import CSTRpHEnv


def test_env_spaces_and_reset():
    """Assert observation/action shapes and verify reset output."""
    env = CSTRpHEnv()

    # 1. Assert spaces
    assert env.observation_space.shape == (4,), f"Expected obs shape (4,), got {env.observation_space.shape}"
    assert env.action_space.shape == (1,), f"Expected action shape (1,), got {env.action_space.shape}"
    assert env.action_space.low[0] == -1.0, "Expected action low bound to be -1.0"
    assert env.action_space.high[0] == 1.0, "Expected action high bound to be 1.0"

    # 2. Reset check
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray), "Reset observation must be a numpy ndarray"
    assert obs.shape == (4,), f"Observation shape mismatch: {obs.shape}"
    assert 0.0 <= obs[0] <= 14.0, f"Initial pH out of bounds: {obs[0]}"
    assert 0.0 <= obs[3] <= 14.0, f"Initial influent pH out of bounds: {obs[3]}"
    assert isinstance(info, dict), "Reset info must be a dictionary"
    assert "c_net" in info

    env.close()


def test_100_random_steps():
    """Execute 100 random steps asserting shapes, types, and stability."""
    env = CSTRpHEnv(max_steps=200)
    obs, info = env.reset(seed=123)

    for step_idx in range(1, 101):
        # Sample random continuous action in [-1.0, 1.0]
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, step_info = env.step(action)

        # Assertions on step return contract
        assert isinstance(next_obs, np.ndarray), f"Step {step_idx}: next_obs is not ndarray"
        assert next_obs.shape == (4,), f"Step {step_idx}: obs shape mismatch: {next_obs.shape}"
        assert np.all(np.isfinite(next_obs)), f"Step {step_idx}: Non-finite values in observation: {next_obs}"
        assert np.isfinite(reward), f"Step {step_idx}: Non-finite reward encountered: {reward}"
        assert isinstance(terminated, bool), f"Step {step_idx}: terminated must be bool"
        assert isinstance(truncated, bool), f"Step {step_idx}: truncated must be bool"
        assert isinstance(step_info, dict), f"Step {step_idx}: info must be dict"
        assert "discharge_violation" in step_info

        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

    # Verify render and close
    env.render()
    env.close()


if __name__ == "__main__":
    print("Running environment verification tests...")
    test_env_spaces_and_reset()
    print("✓ test_env_spaces_and_reset passed.")
    test_100_random_steps()
    print("✓ test_100_random_steps (100 steps executed successfully) passed.")
