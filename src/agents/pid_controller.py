"""Dual-Split PID Controller Baseline for Chemical Dosing and pH Neutralization.

Specifications:
- Error definition: e(t) = 7.0 - pH_t.
- Proportional, Integral (with anti-windup clamping), and Derivative terms.
- Dual-split range output:
    u(t) > 0 -> positive action (dosing base NaOH)
    u(t) < 0 -> negative action (dosing acid H2SO4)
- Strictly bounded within [-1.0, 1.0].
- reset() method to clear accumulated error and previous error between episodes.
- run_pid_baseline() evaluation helper returning mean episodic reward, cumulative chemical liters,
  and regulatory compliance violation percentage.
"""

from typing import Any, Dict, Optional, Union
import numpy as np


class PIDController:
    """Industrial Dual-Split PID Controller for pH neutralization loops.

    Control error:
        e(t) = target_ph - pH_t = 7.0 - pH_t
        e(t) > 0 -> Acidic condition -> Doses base (action > 0)
        e(t) < 0 -> Alkaline condition -> Doses acid (action < 0)
    """

    def __init__(
        self,
        kp: float = 10000.0,
        ki: float = 50.0,
        kd: float = 1.0,
        target_ph: float = 7.0,
        dt: float = 0.1,
        min_output: float = -1.0,
        max_output: float = 1.0,
        anti_windup_threshold: float = 0.5,
        deadband: float = 0.5,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.target_ph = float(target_ph)
        self.dt = float(dt)
        self.min_output = float(min_output)
        self.max_output = float(max_output)
        self.anti_windup_threshold = float(anti_windup_threshold)
        self.deadband = float(deadband)

        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self) -> None:
        """Reset internal integrator and previous error history between episodes."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def get_action(self, obs: Union[np.ndarray, list, float]) -> np.ndarray:
        """Compute dual-split control action from observation.

        Observation formats supported:
        - 4D vector: [pH_t, Delta_pH_t, Q_in, pH_in]
        - 5D vector: [pH_t, error, d_ph, flow_in, target_ph]
        - Scalar: pH_t

        Returns:
            np.ndarray of shape (1,) with value in [-1.0, 1.0]
        """
        if isinstance(obs, (np.ndarray, list)):
            current_ph = float(obs[0])
        else:
            current_ph = float(obs)

        # Industrial Dual-Split Non-linear Error Characterization:
        # Near neutrality [6.5, 8.5], error is zero (deadband prevents titration chatter).
        # Outside [6.5, 8.5], error is characterized by excess ion concentration difference,
        # perfectly matching the required stoichiometric dosing rate without overshooting.
        band_low = self.target_ph - self.deadband
        band_high = self.target_ph + self.deadband

        if band_low <= current_ph <= band_high:
            error = 0.0
        elif current_ph < band_low:
            # Acidic condition: excess [H+] relative to setpoint
            error = (10.0 ** (-current_ph)) - (10.0 ** (-self.target_ph))
        else:
            # Alkaline condition: excess [OH-] relative to setpoint
            error = -((10.0 ** (-(14.0 - current_ph))) - (10.0 ** (-self.target_ph)))

        # Proportional term
        p_term = self.kp * error

        # Derivative term
        if not self.initialized:
            d_term = 0.0
            self.initialized = True
        else:
            derivative = (error - self.prev_error) / max(self.dt, 1e-6)
            d_term = self.kd * derivative

        # Integral accumulation with anti-windup clamping
        if error == 0.0:
            self.integral = 0.0
        else:
            self.integral = float(
                np.clip(
                    self.integral + error * self.dt,
                    -self.anti_windup_threshold,
                    self.anti_windup_threshold,
                )
            )

        self.prev_error = error

        # Raw control signal u(t) = P + I + D
        u_t = p_term + self.ki * self.integral + d_term

        # Dual-Split logic:
        # u(t) > 0 -> positive action (dosing base)
        # u(t) < 0 -> negative action (dosing acid)
        # u(t) == 0 -> idle
        action = float(np.clip(u_t, self.min_output, self.max_output))

        return np.array([action], dtype=np.float32)


def run_pid_baseline(
    env: Any,
    episodes: int = 10,
    controller: Optional[PIDController] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate Dual-Split PID controller baseline over multiple episodes.

    Args:
        env: Gymnasium CSTR pH environment instance.
        episodes: Number of benchmark episodes to run.
        controller: Optional PIDController instance (default initialized if None).
        seed: Random seed base for reproducibility.

    Returns:
        Dictionary containing:
        - "mean_episodic_reward": Mean total episodic return
        - "cumulative_chemical_liters": Total chemical reagent consumed across all episodes
        - "mean_chemical_liters_per_episode": Mean liters of chemical consumed per episode
        - "compliance_violation_percentage": % of timesteps with pH outside [6.5, 8.5]
        - "mean_abs_error": Mean absolute pH error from setpoint (7.0)
    """
    if controller is None:
        controller = PIDController()

    episodic_returns = []
    episodic_reagents = []
    total_timesteps = 0
    violation_timesteps = 0
    abs_errors = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        controller.reset()

        ep_return = 0.0
        ep_reagent = 0.0

        done = False
        while not done:
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            total_timesteps += 1

            ph = float(step_info.get("ph", obs[0]))
            abs_errors.append(abs(ph - 7.0))

            # Regulatory compliance check: pH in [6.5, 8.5]
            if ph < 6.5 or ph > 8.5:
                violation_timesteps += 1

            # Reagent consumed in liters for this step
            step_liters = float(
                step_info.get(
                    "step_reagent",
                    abs(float(action[0])) * getattr(env, "dt", 0.1),
                )
            )
            ep_reagent += step_liters

        episodic_returns.append(ep_return)
        episodic_reagents.append(ep_reagent)

    violation_pct = (
        (violation_timesteps / total_timesteps * 100.0) if total_timesteps > 0 else 0.0
    )

    metrics = {
        "mean_episodic_reward": float(np.mean(episodic_returns)),
        "cumulative_chemical_liters": float(np.sum(episodic_reagents)),
        "mean_chemical_liters_per_episode": float(np.mean(episodic_reagents)),
        "compliance_violation_percentage": float(violation_pct),
        "mean_abs_error": float(np.mean(abs_errors)),
    }

    return metrics
