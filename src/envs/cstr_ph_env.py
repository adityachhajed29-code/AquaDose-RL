"""Gymnasium Environment for CSTR Chemical Dosing and pH Neutralization.

Mathematical Simulation:
- Continuous Stirred-Tank Reactor (CSTR) mass balance on net ion concentration:
    C_net = [H+] - [OH-] (mol/L)
- ODE:
    d(C_net)/dt = (1 / V_tank) * (Q_in * C_in + Q_acid * C_acid - Q_base * C_base - Q_out * C_net)
- Water dissociation equilibrium:
    Kw = 1e-14
    [H+] = (C_net + sqrt(C_net^2 + 4 * Kw)) / 2
    pH = -log10([H+])
- Inflow disturbances & shock spikes (acid shock ~2.5, base shock ~11.5).
"""

from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CSTRpHEnv(gym.Env):
    """Continuous Stirred-Tank Reactor (CSTR) pH neutralization environment.

    Observation Space (Box(4,)):
        [0]: pH_t         - Current reactor pH [0.0, 14.0]
        [1]: Delta_pH_t   - Rate of change (pH_t - pH_{t-1})
        [2]: Q_in         - Influent flow rate (L/min)
        [3]: pH_in        - Influent stream pH [0.0, 14.0]

    Action Space (Box(1,)):
        action in [-1.0, 1.0]:
        - a > 0: Base dosing pump rate (NaOH) scaled by max_base_flow
        - a < 0: Acid dosing pump rate (H2SO4) scaled by max_acid_flow
        - a == 0: Idle
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        v_tank: float = 100.0,             # Reactor tank volume (L)
        nominal_q_in: float = 10.0,        # Nominal influent flow (L/min)
        q_in_std: float = 0.5,             # Stochastic noise on influent flow (L/min)
        nominal_ph_in: float = 7.0,        # Nominal neutral influent stream pH (6.0 - 8.0)
        max_acid_flow: float = 0.06,       # Max acid dosing rate Q_acid (L/min) calibrated for shock neutralization
        max_base_flow: float = 0.06,       # Max base dosing rate Q_base (L/min) calibrated for shock neutralization
        c_acid: float = 1.0,               # Concentration of acid stream (mol/L of H+)
        c_base: float = 1.0,               # Concentration of base stream (mol/L of OH-)
        dt: float = 0.1,                   # Simulation time step (min)
        max_steps: int = 200,              # Episode step limit
        w1: float = 1.0,                   # Target pH error weight
        w2: float = 0.02,                  # Control effort penalty weight (scaled to not discourage dosing)
        w3: float = 0.01,                  # Actuator slew rate penalty weight
        lambda_penalty: float = 25.0,      # Discharge compliance violation penalty
        compliance_reward: float = 2.0,    # Positive living reward inside [6.5, 8.5]
        shock_duration: int = 35,          # Steps per discrete shock disturbance event
        shock_interval: int = 200,         # Interval between disturbance shock events
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.v_tank = float(v_tank)
        self.nominal_q_in = float(nominal_q_in)
        self.q_in_std = float(q_in_std)
        self.nominal_ph_in = float(nominal_ph_in)
        self.max_acid_flow = float(max_acid_flow)
        self.max_base_flow = float(max_base_flow)
        self.c_acid = float(c_acid)
        self.c_base = float(c_base)
        self.dt = float(dt)
        self.max_steps = int(max_steps)

        # Reward weights
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.lambda_penalty = float(lambda_penalty)
        self.compliance_reward = float(compliance_reward)
        self.shock_duration = int(shock_duration)
        self.shock_interval = int(shock_interval)

        self.render_mode = render_mode
        self.kw = 1e-14  # Water dissociation constant at 25 °C

        # Observation space: [pH_t, Delta_pH_t, Q_in, pH_in]
        low_obs = np.array([0.0, -14.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([14.0, 14.0, 100.0, 14.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Continuous action space: [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # State variables
        self.current_step = 0
        self.c_net = 0.0
        self.current_ph = 7.0
        self.prev_ph = 7.0
        self.q_in = self.nominal_q_in
        self.ph_in = self.nominal_ph_in
        self.prev_action = 0.0
        self.cumulative_reagent = 0.0

    @staticmethod
    def ph_to_cnet(ph: float, kw: float = 1e-14) -> float:
        """Convert pH to net ion concentration C_net = [H+] - [OH-] in mol/L."""
        ph_clamped = max(0.0, min(14.0, ph))
        h_plus = 10.0 ** (-ph_clamped)
        oh_minus = kw / (h_plus + 1e-18)
        return float(h_plus - oh_minus)

    @staticmethod
    def cnet_to_ph(c_net: float, kw: float = 1e-14) -> float:
        """Calculate pH from net ion concentration C_net via water dissociation quadratic."""
        # [H+]^2 - C_net * [H+] - Kw = 0  =>  [H+] = (C_net + sqrt(C_net^2 + 4 * Kw)) / 2
        discriminant = max(0.0, c_net * c_net + 4.0 * kw)
        h_plus = 0.5 * (c_net + np.sqrt(discriminant))
        h_plus = max(1e-14, min(1.0, h_plus))
        return float(-np.log10(h_plus))

    def _get_obs(self) -> np.ndarray:
        delta_ph = self.current_ph - self.prev_ph
        return np.array(
            [self.current_ph, delta_ph, self.q_in, self.ph_in],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize influent state around nominal neutrality
        self.ph_in = float(self.nominal_ph_in + self.np_random.uniform(-0.3, 0.3))
        self.q_in = float(self.nominal_q_in + self.np_random.normal(0.0, self.q_in_std))
        self.q_in = max(1.0, self.q_in)

        # Initialize tank inside regulatory compliant operating range [6.8, 7.2]
        init_ph = float(self.np_random.uniform(6.8, 7.2))
        self.current_ph = init_ph
        self.prev_ph = init_ph
        self.c_net = self.ph_to_cnet(self.current_ph, self.kw)

        self.current_step = 0
        self.prev_action = 0.0
        self.cumulative_reagent = 0.0

        obs = self._get_obs()
        info = {
            "ph": self.current_ph,
            "q_in": self.q_in,
            "ph_in": self.ph_in,
            "c_net": self.c_net,
            "reagent_used": 0.0,
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: Union[np.ndarray, list, float]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # Extract continuous action scalar
        if isinstance(action, (np.ndarray, list)):
            act = float(action[0])
        else:
            act = float(action)
        act = float(np.clip(act, -1.0, 1.0))

        # Reagent pump dosing rates
        if act > 0.0:
            q_base = act * self.max_base_flow
            q_acid = 0.0
        elif act < 0.0:
            q_acid = (-act) * self.max_acid_flow
            q_base = 0.0
        else:
            q_acid = 0.0
            q_base = 0.0

        step_reagent = (q_acid + q_base) * self.dt
        self.cumulative_reagent += step_reagent

        # Upstream inflow fluctuation
        self.q_in = float(self.q_in + self.np_random.normal(0.0, 0.2))
        self.q_in = float(np.clip(self.q_in, self.nominal_q_in * 0.5, self.nominal_q_in * 1.5))

        # Discrete disturbance event scheduling: shocks lasting 30-50 steps every 200 steps
        cycle_pos = self.current_step % self.shock_interval
        shock_start = 30
        shock_end = shock_start + self.shock_duration

        if shock_start <= cycle_pos < shock_end:
            # Alternating shock events: acid shock (~2.5) or base shock (~11.5)
            cycle_idx = self.current_step // self.shock_interval
            if cycle_idx % 2 == 0:
                self.ph_in = float(self.np_random.normal(2.5, 0.1))
            else:
                self.ph_in = float(self.np_random.normal(11.5, 0.1))
        else:
            # Nominal inflow with gentle drift staying around pH 6.0 - 8.0
            drift = 0.2 * (self.nominal_ph_in - self.ph_in) + self.np_random.normal(0.0, 0.05)
            self.ph_in = float(np.clip(self.ph_in + drift, 6.2, 7.8))

        # Chemical mass balance on net ion concentration C_net:
        # C_in = [H+] - [OH-] of influent
        c_in = self.ph_to_cnet(self.ph_in, self.kw)

        # q_out by total volumetric balance
        q_out = self.q_in + q_acid + q_base

        # dC_net/dt = (1 / V_tank) * (Q_in * C_in + Q_acid * C_acid - Q_base * C_base - Q_out * C_net)
        dcnet_dt = (
            self.q_in * c_in
            + q_acid * self.c_acid
            - q_base * self.c_base
            - q_out * self.c_net
        ) / self.v_tank

        # Numerical integration (Euler)
        self.c_net += dcnet_dt * self.dt

        # Convert back to pH
        self.prev_ph = self.current_ph
        self.current_ph = self.cnet_to_ph(self.c_net, self.kw)

        # Reward Calculation:
        # Living reward when inside [6.5, 8.5] + setpoint tracking - effort - slew - violation penalty
        tracking_penalty = self.w1 * ((self.current_ph - 7.0) ** 2)
        effort_penalty = self.w2 * abs(act)
        slew_penalty = self.w3 * abs(act - self.prev_action)
        is_compliant = (6.5 <= self.current_ph <= 8.5)
        discharge_violation = 0.0 if is_compliant else 1.0
        compliance_penalty = self.lambda_penalty * discharge_violation
        living_reward = self.compliance_reward if is_compliant else 0.0

        reward = float(
            living_reward
            - tracking_penalty
            - effort_penalty
            - slew_penalty
            - compliance_penalty
        )

        self.prev_action = act

        # Episode termination / truncation
        terminated = bool(self.current_ph < 1.0 or self.current_ph > 13.5)
        truncated = bool(self.current_step >= self.max_steps)

        obs = self._get_obs()
        info = {
            "ph": self.current_ph,
            "target_ph": 7.0,
            "delta_ph": self.current_ph - self.prev_ph,
            "q_in": self.q_in,
            "ph_in": self.ph_in,
            "action": act,
            "q_acid": q_acid,
            "q_base": q_base,
            "step_reagent": step_reagent,
            "cumulative_reagent": self.cumulative_reagent,
            "discharge_violation": bool(discharge_violation > 0),
            "tracking_penalty": tracking_penalty,
            "effort_penalty": effort_penalty,
            "compliance_penalty": compliance_penalty,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render current state to console or return rgb array placeholder."""
        if self.render_mode == "human":
            status = "OK" if 6.5 <= self.current_ph <= 8.5 else "VIOLATION"
            print(
                f"[Step {self.current_step:03d}] "
                f"pH: {self.current_ph:5.2f} ({status}) | "
                f"Influent: {self.ph_in:5.2f} @ {self.q_in:4.1f} L/min | "
                f"Action: {self.prev_action:+4.2f} | "
                f"Reagent: {self.cumulative_reagent:5.2f} L"
            )
            return None
        return None

    def close(self) -> None:
        """Clean up environment resources."""
        pass
