"""Continuous Stirred-Tank Reactor (CSTR) pH Neutralization Environment.

Models a non-linear chemical neutralization process governed by conservation of
chemical species (proton net concentration Wa = [H+] - [OH-]). Features non-linear
titration curves with steep gradients near neutral pH (pH 7.0), stochastic influent
disturbances, and chemical reagent consumption costs.
"""

from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CSTRpHEnv(gym.Env):
    """Gymnasium continuous stirred-tank reactor environment for pH control.

    State:
        - pH: Current reactor pH [0, 14]
        - Error: pH - target_pH [-14, 14]
        - d(pH)/dt: Rate of change of pH
        - Influent flow rate (L/min)
        - Target pH (nominal 7.0)

    Actions:
        - Continuous mode (default):
            action in [-1.0, 1.0]
            action > 0 -> Base dosing rate (e.g., NaOH), scaled by max_base_flow
            action < 0 -> Acid dosing rate (e.g., HCl), scaled by max_acid_flow
            action == 0 -> No chemical added
        - Discrete mode:
            action in {0, ..., num_discrete_actions - 1} mapped uniformly to [-1.0, 1.0]
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        tank_volume: float = 100.0,           # Reactor liquid volume (L)
        nominal_flow_in: float = 10.0,        # Influent flow rate (L/min)
        influent_ph_range: Tuple[float, float] = (2.5, 4.5), # Acidic wastewater inflow
        target_ph: float = 7.0,               # Target neutral pH
        max_acid_flow: float = 1.0,           # Max acid reagent dosing rate (L/min)
        max_base_flow: float = 1.5,           # Max base reagent dosing rate (L/min)
        acid_normality: float = 0.1,          # Normality of HCl dosing stream (eq/L)
        base_normality: float = 0.1,          # Normality of NaOH dosing stream (eq/L)
        dt: float = 0.1,                      # Simulation step size (min)
        max_steps: int = 200,                 # Episode horizon
        action_type: str = "continuous",      # "continuous" or "discrete"
        num_discrete_actions: int = 11,       # Used when action_type is "discrete"
        stochastic_disturbance: bool = True,  # Inject flow and pH perturbations
    ):
        super().__init__()

        self.tank_volume = tank_volume
        self.nominal_flow_in = nominal_flow_in
        self.influent_ph_range = influent_ph_range
        self.target_ph = target_ph
        self.max_acid_flow = max_acid_flow
        self.max_base_flow = max_base_flow
        self.acid_normality = acid_normality
        self.base_normality = base_normality
        self.dt = dt
        self.max_steps = max_steps
        self.action_type = action_type.lower()
        self.num_discrete_actions = num_discrete_actions
        self.stochastic_disturbance = stochastic_disturbance

        self.kw = 1e-14  # Water dissociation constant at 25 deg C

        # Observation bounds: [pH, error, d(pH)/dt, flow_in, target_ph]
        low_obs = np.array([0.0, -14.0, -10.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([14.0, 14.0, 10.0, 50.0, 14.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action space setup
        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(self.num_discrete_actions)
            self._discrete_actions = np.linspace(-1.0, 1.0, self.num_discrete_actions, dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal state
        self.current_step = 0
        self.wa = 0.0  # Net proton concentration: [H+] - [OH-] (mol/L)
        self.current_ph = 7.0
        self.prev_ph = 7.0
        self.influent_ph = 3.0
        self.flow_in = self.nominal_flow_in
        self.prev_action = 0.0
        self.cumulative_reagent = 0.0

    @staticmethod
    def _ph_to_wa(ph: float, kw: float = 1e-14) -> float:
        """Convert pH to net proton concentration Wa = [H+] - [OH-]."""
        h_plus = 10.0 ** (-ph)
        oh_minus = kw / (h_plus + 1e-18)
        return float(h_plus - oh_minus)

    @staticmethod
    def _wa_to_ph(wa: float, kw: float = 1e-14) -> float:
        """Solve for pH from net proton concentration Wa via quadratic formula."""
        # [H+]^2 - Wa*[H+] - Kw = 0  => [H+] = (Wa + sqrt(Wa^2 + 4*Kw)) / 2
        discriminant = max(0.0, wa * wa + 4.0 * kw)
        h_plus = 0.5 * (wa + np.sqrt(discriminant))
        h_plus = max(1e-14, min(1.0, h_plus))
        return float(-np.log10(h_plus))

    def _get_obs(self) -> np.ndarray:
        error = self.current_ph - self.target_ph
        d_ph = (self.current_ph - self.prev_ph) / max(self.dt, 1e-5)
        return np.array(
            [self.current_ph, error, d_ph, self.flow_in, self.target_ph],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Randomize initial influent pH within defined acidic range
        self.influent_ph = float(
            self.np_random.uniform(self.influent_ph_range[0], self.influent_ph_range[1])
        )
        self.flow_in = float(self.nominal_flow_in)

        # Start reactor either perturbed or near influent condition
        init_ph = float(
            self.np_random.uniform(self.influent_ph_range[0], 6.0)
        )
        self.current_ph = init_ph
        self.prev_ph = init_ph
        self.wa = self._ph_to_wa(self.current_ph, self.kw)

        self.current_step = 0
        self.prev_action = 0.0
        self.cumulative_reagent = 0.0

        obs = self._get_obs()
        info = {
            "ph": self.current_ph,
            "target_ph": self.target_ph,
            "error": self.current_ph - self.target_ph,
            "influent_ph": self.influent_ph,
            "flow_in": self.flow_in,
        }
        return obs, info

    def step(self, action: Union[np.ndarray, int, float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # Normalize action scalar to [-1.0, 1.0]
        if self.action_type == "discrete":
            act_scalar = float(self._discrete_actions[int(action)])
        else:
            if isinstance(action, (np.ndarray, list)):
                act_scalar = float(action[0])
            else:
                act_scalar = float(action)
        act_scalar = float(np.clip(act_scalar, -1.0, 1.0))

        # Reagent flows
        if act_scalar >= 0.0:
            base_flow = act_scalar * self.max_base_flow
            acid_flow = 0.0
        else:
            acid_flow = (-act_scalar) * self.max_acid_flow
            base_flow = 0.0

        reagent_used = (acid_flow + base_flow) * self.dt
        self.cumulative_reagent += reagent_used

        # Disturbance evolution (slow random walk on flow and pH)
        if self.stochastic_disturbance:
            self.flow_in = float(
                np.clip(
                    self.flow_in + self.np_random.normal(0.0, 0.1),
                    self.nominal_flow_in * 0.7,
                    self.nominal_flow_in * 1.3,
                )
            )
            self.influent_ph = float(
                np.clip(
                    self.influent_ph + self.np_random.normal(0.0, 0.02),
                    self.influent_ph_range[0] - 0.5,
                    self.influent_ph_range[1] + 0.5,
                )
            )

        # Dynamic mass balance for CSTR
        # Inflow net proton concentrations
        wa_in = self._ph_to_wa(self.influent_ph, self.kw)
        wa_acid = self.acid_normality  # Strong acid: [H+] = Normality, [OH-] ~ 0
        wa_base = -self.base_normality  # Strong base: [OH-] = Normality, Wa = -Normality

        flow_out = self.flow_in + acid_flow + base_flow
        # dWa/dt = (Fin*Wa_in + Facid*Wa_acid + Fbase*Wa_base - Fout*Wa) / V
        dwa_dt = (
            self.flow_in * wa_in
            + acid_flow * wa_acid
            + base_flow * wa_base
            - flow_out * self.wa
        ) / self.tank_volume

        # Euler numerical integration
        self.wa += dwa_dt * self.dt
        self.prev_ph = self.current_ph
        self.current_ph = self._wa_to_ph(self.wa, self.kw)

        # Compute multi-objective reward
        error = abs(self.current_ph - self.target_ph)

        # 1. Tracking reward: Gaussian peak centered at target pH
        tracking_reward = np.exp(-0.5 * (error / 0.5) ** 2)

        # 2. Strict penalty for being outside allowable discharge band (6.5 - 8.5)
        band_penalty = 0.0
        if error > 1.5:
            band_penalty = -0.5 * (error - 1.5)

        # 3. Chemical consumption penalty (encourages dosing efficiency)
        reagent_penalty = -0.05 * (acid_flow / self.max_acid_flow + base_flow / self.max_base_flow)

        # 4. Action smoothness penalty (prevents valve chatter)
        smoothness_penalty = -0.02 * abs(act_scalar - self.prev_action)

        reward = float(tracking_reward + band_penalty + reagent_penalty + smoothness_penalty)

        self.prev_action = act_scalar

        # Termination / Truncation
        # Terminate if extreme pH (e.g., pH < 1 or pH > 13) causing catastrophic conditions
        terminated = bool(self.current_ph < 1.0 or self.current_ph > 13.5)
        truncated = bool(self.current_step >= self.max_steps)

        obs = self._get_obs()
        info = {
            "ph": self.current_ph,
            "target_ph": self.target_ph,
            "error": self.current_ph - self.target_ph,
            "action": act_scalar,
            "acid_flow": acid_flow,
            "base_flow": base_flow,
            "reagent_step": reagent_used,
            "cumulative_reagent": self.cumulative_reagent,
            "tracking_reward": float(tracking_reward),
            "flow_in": self.flow_in,
            "influent_ph": self.influent_ph,
        }

        return obs, reward, terminated, truncated, info
