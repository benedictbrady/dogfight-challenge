"""GPU-accelerated batch environments using NVIDIA Warp.

Drop-in replacements for the Rust SelfPlayBatchEnv / BatchEnv.
Returns numpy arrays (same interface) but runs physics on GPU.
"""

import numpy as np
import warp as wp
import torch
import random

from .constants import (
    OBS_SIZE, CONFIG_OBS_SIZE, ACTION_SIZE, BULLETS_PER_ENV, MAX_TICKS,
    GRAVITY, DRAG_COEFF, TURN_BLEED_COEFF, MAX_SPEED, MIN_SPEED,
    MAX_THRUST, BULLET_SPEED, GUN_COOLDOWN_TICKS, BULLET_LIFETIME_TICKS,
    MAX_HP, MAX_TURN_RATE, MIN_TURN_RATE, REAR_ASPECT_CONE,
    OPP_DO_NOTHING, OPP_CHASER, OPP_DOGFIGHTER, OPP_ACE, OPP_BRAWLER,
    DEFAULT_W_DAMAGE_DEALT, DEFAULT_W_DAMAGE_TAKEN, DEFAULT_W_WIN,
    DEFAULT_W_LOSE, DEFAULT_W_APPROACH, DEFAULT_W_ALIVE,
    DEFAULT_W_PROXIMITY, DEFAULT_W_FACING,
)
from . import kernels


# Map policy names to opponent type IDs
POLICY_NAME_TO_ID = {
    "do_nothing": OPP_DO_NOTHING,
    "chaser": OPP_CHASER,
    "dogfighter": OPP_DOGFIGHTER,
    "ace": OPP_ACE,
    "brawler": OPP_BRAWLER,
}


class GpuSelfPlayBatchEnv:
    """GPU-accelerated vectorized self-play environment.

    Drop-in replacement for the Rust SelfPlayBatchEnv. Both players are
    Python-controlled, with optional scripted P1 opponents for a fraction
    of envs (curriculum/anchoring).

    Returns numpy arrays with the same shapes and dtypes as the Rust version.
    """

    def __init__(
        self,
        n_envs: int = 256,
        randomize_spawns: bool = True,
        seed: int = 0,
        action_repeat: int = 10,
        include_config_obs: bool = False,
    ):
        self._n_envs = n_envs
        self._randomize = randomize_spawns
        self._action_repeat = action_repeat
        self._include_config_obs = include_config_obs
        self._rng = random.Random(seed)

        self._scripted_fraction = 0.0
        self._scripted_pool: list[str] = []
        self._config_ranges: dict = {}

        # Effective obs size
        self._effective_obs = OBS_SIZE + CONFIG_OBS_SIZE if include_config_obs else OBS_SIZE

        # Initialize Warp
        wp.init()
        self._device = "cuda:0"

        # Allocate state arrays
        n2 = n_envs * 2
        nb = n_envs * BULLETS_PER_ENV

        # Fighter state (N*2)
        self._pos_x = wp.zeros(n2, dtype=wp.float32, device=self._device)
        self._pos_y = wp.zeros(n2, dtype=wp.float32, device=self._device)
        self._yaw = wp.zeros(n2, dtype=wp.float32, device=self._device)
        self._speed = wp.zeros(n2, dtype=wp.float32, device=self._device)
        self._hp = wp.zeros(n2, dtype=wp.int32, device=self._device)
        self._cooldown = wp.zeros(n2, dtype=wp.int32, device=self._device)
        self._alive = wp.zeros(n2, dtype=wp.int32, device=self._device)
        self._stall_ticks = wp.zeros(n2, dtype=wp.int32, device=self._device)

        # Bullet state (N*24)
        self._bul_pos_x = wp.zeros(nb, dtype=wp.float32, device=self._device)
        self._bul_pos_y = wp.zeros(nb, dtype=wp.float32, device=self._device)
        self._bul_vel_x = wp.zeros(nb, dtype=wp.float32, device=self._device)
        self._bul_vel_y = wp.zeros(nb, dtype=wp.float32, device=self._device)
        self._bul_owner = wp.zeros(nb, dtype=wp.int32, device=self._device)
        self._bul_ticks = wp.zeros(nb, dtype=wp.int32, device=self._device)

        # Config arrays (N) — default values
        self._cfg_gravity = wp.full(n_envs, GRAVITY, dtype=wp.float32, device=self._device)
        self._cfg_drag = wp.full(n_envs, DRAG_COEFF, dtype=wp.float32, device=self._device)
        self._cfg_turn_bleed = wp.full(n_envs, TURN_BLEED_COEFF, dtype=wp.float32, device=self._device)
        self._cfg_max_speed = wp.full(n_envs, MAX_SPEED, dtype=wp.float32, device=self._device)
        self._cfg_min_speed = wp.full(n_envs, MIN_SPEED, dtype=wp.float32, device=self._device)
        self._cfg_max_thrust = wp.full(n_envs, MAX_THRUST, dtype=wp.float32, device=self._device)
        self._cfg_bullet_speed = wp.full(n_envs, BULLET_SPEED, dtype=wp.float32, device=self._device)
        self._cfg_gun_cooldown = wp.full(n_envs, int(GUN_COOLDOWN_TICKS), dtype=wp.int32, device=self._device)
        self._cfg_bullet_lifetime = wp.full(n_envs, int(BULLET_LIFETIME_TICKS), dtype=wp.int32, device=self._device)
        self._cfg_max_hp = wp.full(n_envs, int(MAX_HP), dtype=wp.int32, device=self._device)
        self._cfg_max_turn_rate = wp.full(n_envs, MAX_TURN_RATE, dtype=wp.float32, device=self._device)
        self._cfg_min_turn_rate = wp.full(n_envs, MIN_TURN_RATE, dtype=wp.float32, device=self._device)
        self._cfg_rear_aspect_cone = wp.full(n_envs, REAR_ASPECT_CONE, dtype=wp.float32, device=self._device)

        # Opponent type (-1 = neural, 0-4 = scripted)
        self._opponent_type = wp.full(n_envs, -1, dtype=wp.int32, device=self._device)

        # Opponent FSM state (N)
        self._opp_phase = wp.zeros(n_envs, dtype=wp.int32, device=self._device)
        self._opp_phase_timer = wp.zeros(n_envs, dtype=wp.int32, device=self._device)
        self._opp_evade_timer = wp.zeros(n_envs, dtype=wp.int32, device=self._device)
        self._opp_evade_dir = wp.full(n_envs, 1.0, dtype=wp.float32, device=self._device)
        self._opp_yo_yo_timer = wp.zeros(n_envs, dtype=wp.int32, device=self._device)
        self._opp_yo_yo_phase = wp.zeros(n_envs, dtype=wp.float32, device=self._device)
        self._opp_jink_timer = wp.zeros(n_envs, dtype=wp.int32, device=self._device)
        self._opp_jink_dir = wp.full(n_envs, 1.0, dtype=wp.float32, device=self._device)
        self._opp_overshoot_timer = wp.zeros(n_envs, dtype=wp.int32, device=self._device)
        self._opp_attack_patience = wp.zeros(n_envs, dtype=wp.int32, device=self._device)
        self._opp_last_distance = wp.full(n_envs, 400.0, dtype=wp.float32, device=self._device)

        # Previous state for reward computation (N)
        self._prev_hp_p0 = wp.zeros(n_envs, dtype=wp.float32, device=self._device)
        self._prev_hp_p1 = wp.zeros(n_envs, dtype=wp.float32, device=self._device)
        self._prev_opp_hp_p0 = wp.zeros(n_envs, dtype=wp.float32, device=self._device)
        self._prev_opp_hp_p1 = wp.zeros(n_envs, dtype=wp.float32, device=self._device)
        self._prev_dist = wp.zeros(n_envs, dtype=wp.float32, device=self._device)

        # Output buffers
        self._obs_p0 = wp.zeros((n_envs, OBS_SIZE), dtype=wp.float32, device=self._device)
        self._obs_p1 = wp.zeros((n_envs, OBS_SIZE), dtype=wp.float32, device=self._device)
        self._config_obs = wp.zeros((n_envs, CONFIG_OBS_SIZE), dtype=wp.float32, device=self._device)
        self._reward_p0 = wp.zeros(n_envs, dtype=wp.float32, device=self._device)
        self._reward_p1 = wp.zeros(n_envs, dtype=wp.float32, device=self._device)
        self._done = wp.zeros(n_envs, dtype=wp.int32, device=self._device)

        # Reward weights [N, 8]
        default_weights = np.array([
            DEFAULT_W_DAMAGE_DEALT, DEFAULT_W_DAMAGE_TAKEN,
            DEFAULT_W_WIN, DEFAULT_W_LOSE,
            DEFAULT_W_APPROACH, DEFAULT_W_ALIVE,
            DEFAULT_W_PROXIMITY, DEFAULT_W_FACING,
        ], dtype=np.float32)
        weights_np = np.tile(default_weights, (n_envs, 1))
        self._reward_weights = wp.array(weights_np, dtype=wp.float32, device=self._device)

        # Tick counter [N]
        self._tick_counter = wp.zeros(n_envs, dtype=wp.int32, device=self._device)

        # Reset mask [N]
        self._reset_mask = wp.zeros(n_envs, dtype=wp.int32, device=self._device)

        # Action buffers [N, 3]
        self._act_p0 = wp.zeros((n_envs, ACTION_SIZE), dtype=wp.float32, device=self._device)
        self._act_p1 = wp.zeros((n_envs, ACTION_SIZE), dtype=wp.float32, device=self._device)

        # Torch tensor views of warp arrays (zero-copy, shared CUDA memory)
        self._act_p0_torch = wp.to_torch(self._act_p0)
        self._act_p1_torch = wp.to_torch(self._act_p1)
        self._obs_p0_torch = wp.to_torch(self._obs_p0)
        self._obs_p1_torch = wp.to_torch(self._obs_p1)
        self._config_obs_torch = wp.to_torch(self._config_obs)
        self._reward_p0_torch = wp.to_torch(self._reward_p0)
        self._reward_p1_torch = wp.to_torch(self._reward_p1)
        self._done_torch = wp.to_torch(self._done)

        # Pre-allocated combined obs buffers for torch path (avoid per-step cat)
        if include_config_obs:
            self._combined_obs_p0 = torch.zeros(
                n_envs, self._effective_obs, dtype=torch.float32, device="cuda")
            self._combined_obs_p1 = torch.zeros(
                n_envs, self._effective_obs, dtype=torch.float32, device="cuda")

        # Scripted mask tracking (CPU)
        self._scripted_mask_np = np.zeros(n_envs, dtype=np.bool_)

        # Warmup: compile kernels
        self._warmup()

    def _warmup(self):
        """JIT compile kernels with a dummy launch."""
        mask = wp.full(self._n_envs, 1, dtype=wp.int32, device=self._device)
        wp.launch(kernels.reset_env, dim=self._n_envs, inputs=[
            self._pos_x, self._pos_y, self._yaw, self._speed,
            self._hp, self._cooldown, self._alive, self._stall_ticks,
            self._bul_pos_x, self._bul_pos_y, self._bul_vel_x, self._bul_vel_y,
            self._bul_owner, self._bul_ticks,
            self._cfg_max_hp, self._tick_counter, self._done, mask,
        ], device=self._device)
        wp.synchronize()

    def _assign_env_modes(self):
        """Assign scripted/neural modes for all envs. Called on reset."""
        opp_type_np = np.full(self._n_envs, -1, dtype=np.int32)
        self._scripted_mask_np = np.zeros(self._n_envs, dtype=np.bool_)

        for i in range(self._n_envs):
            if self._scripted_pool and self._scripted_fraction > 0:
                if self._rng.random() < self._scripted_fraction:
                    name = self._rng.choice(self._scripted_pool)
                    opp_type_np[i] = POLICY_NAME_TO_ID[name]
                    self._scripted_mask_np[i] = True

        wp.copy(self._opponent_type, wp.array(opp_type_np, dtype=wp.int32, device=self._device))

    def _sample_configs(self):
        """Sample domain randomization configs for all envs."""
        if not self._config_ranges:
            return

        n = self._n_envs
        configs = {
            "gravity": np.full(n, GRAVITY, dtype=np.float32),
            "drag_coeff": np.full(n, DRAG_COEFF, dtype=np.float32),
            "turn_bleed_coeff": np.full(n, TURN_BLEED_COEFF, dtype=np.float32),
            "max_speed": np.full(n, MAX_SPEED, dtype=np.float32),
            "min_speed": np.full(n, MIN_SPEED, dtype=np.float32),
            "max_thrust": np.full(n, MAX_THRUST, dtype=np.float32),
            "bullet_speed": np.full(n, BULLET_SPEED, dtype=np.float32),
            "gun_cooldown_ticks": np.full(n, GUN_COOLDOWN_TICKS, dtype=np.float32),
            "bullet_lifetime_ticks": np.full(n, BULLET_LIFETIME_TICKS, dtype=np.float32),
            "max_hp": np.full(n, MAX_HP, dtype=np.float32),
            "max_turn_rate": np.full(n, MAX_TURN_RATE, dtype=np.float32),
            "min_turn_rate": np.full(n, MIN_TURN_RATE, dtype=np.float32),
            "rear_aspect_cone": np.full(n, REAR_ASPECT_CONE, dtype=np.float32),
        }

        for param, (lo, hi) in self._config_ranges.items():
            configs[param] = np.random.uniform(lo, hi, size=n).astype(np.float32)

        # Upload to GPU
        wp.copy(self._cfg_gravity, wp.array(configs["gravity"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_drag, wp.array(configs["drag_coeff"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_turn_bleed, wp.array(configs["turn_bleed_coeff"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_max_speed, wp.array(configs["max_speed"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_min_speed, wp.array(configs["min_speed"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_max_thrust, wp.array(configs["max_thrust"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_bullet_speed, wp.array(configs["bullet_speed"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_gun_cooldown, wp.array(
            configs["gun_cooldown_ticks"].astype(np.int32), dtype=wp.int32, device=self._device))
        wp.copy(self._cfg_bullet_lifetime, wp.array(
            configs["bullet_lifetime_ticks"].astype(np.int32), dtype=wp.int32, device=self._device))
        wp.copy(self._cfg_max_hp, wp.array(
            configs["max_hp"].astype(np.int32), dtype=wp.int32, device=self._device))
        wp.copy(self._cfg_max_turn_rate, wp.array(configs["max_turn_rate"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_min_turn_rate, wp.array(configs["min_turn_rate"], dtype=wp.float32, device=self._device))
        wp.copy(self._cfg_rear_aspect_cone, wp.array(configs["rear_aspect_cone"], dtype=wp.float32, device=self._device))

    def _reset_opp_fsm(self):
        """Reset opponent FSM state for all envs."""
        n = self._n_envs
        self._opp_phase.zero_()
        self._opp_phase_timer.zero_()
        self._opp_evade_timer.zero_()
        wp.copy(self._opp_evade_dir, wp.full(n, 1.0, dtype=wp.float32, device=self._device))
        self._opp_yo_yo_timer.zero_()
        self._opp_yo_yo_phase.zero_()
        self._opp_jink_timer.zero_()
        wp.copy(self._opp_jink_dir, wp.full(n, 1.0, dtype=wp.float32, device=self._device))
        self._opp_overshoot_timer.zero_()
        self._opp_attack_patience.zero_()
        wp.copy(self._opp_last_distance, wp.full(n, 400.0, dtype=wp.float32, device=self._device))

    def reset(self):
        """Reset all envs. Returns (obs_p0, obs_p1) as numpy [n_envs, effective_obs]."""
        self._assign_env_modes()
        self._sample_configs()
        self._reset_opp_fsm()

        # Reset all envs
        mask = wp.full(self._n_envs, 1, dtype=wp.int32, device=self._device)
        wp.launch(kernels.reset_env, dim=self._n_envs, inputs=[
            self._pos_x, self._pos_y, self._yaw, self._speed,
            self._hp, self._cooldown, self._alive, self._stall_ticks,
            self._bul_pos_x, self._bul_pos_y, self._bul_vel_x, self._bul_vel_y,
            self._bul_owner, self._bul_ticks,
            self._cfg_max_hp, self._tick_counter, self._done, mask,
        ], device=self._device)

        # Compute initial observations
        wp.launch(kernels.compute_obs_and_config, dim=self._n_envs, inputs=[
            self._pos_x, self._pos_y, self._yaw, self._speed,
            self._hp, self._cooldown, self._alive,
            self._bul_pos_x, self._bul_pos_y, self._bul_vel_x, self._bul_vel_y,
            self._bul_owner, self._bul_ticks,
            self._cfg_gravity, self._cfg_drag, self._cfg_turn_bleed,
            self._cfg_max_speed, self._cfg_min_speed, self._cfg_max_thrust,
            self._cfg_bullet_speed, self._cfg_gun_cooldown, self._cfg_bullet_lifetime,
            self._cfg_max_hp, self._cfg_max_turn_rate, self._cfg_min_turn_rate,
            self._cfg_rear_aspect_cone,
            self._tick_counter,
            self._obs_p0, self._obs_p1, self._config_obs,
        ], device=self._device)
        wp.synchronize()

        return self._build_obs_output()

    def reset_torch(self):
        """Reset all envs. Returns (obs_p0, obs_p1) as torch.Tensor on CUDA."""
        self._assign_env_modes()
        self._sample_configs()
        self._reset_opp_fsm()

        mask = wp.full(self._n_envs, 1, dtype=wp.int32, device=self._device)
        wp.launch(kernels.reset_env, dim=self._n_envs, inputs=[
            self._pos_x, self._pos_y, self._yaw, self._speed,
            self._hp, self._cooldown, self._alive, self._stall_ticks,
            self._bul_pos_x, self._bul_pos_y, self._bul_vel_x, self._bul_vel_y,
            self._bul_owner, self._bul_ticks,
            self._cfg_max_hp, self._tick_counter, self._done, mask,
        ], device=self._device)

        wp.launch(kernels.compute_obs_and_config, dim=self._n_envs, inputs=[
            self._pos_x, self._pos_y, self._yaw, self._speed,
            self._hp, self._cooldown, self._alive,
            self._bul_pos_x, self._bul_pos_y, self._bul_vel_x, self._bul_vel_y,
            self._bul_owner, self._bul_ticks,
            self._cfg_gravity, self._cfg_drag, self._cfg_turn_bleed,
            self._cfg_max_speed, self._cfg_min_speed, self._cfg_max_thrust,
            self._cfg_bullet_speed, self._cfg_gun_cooldown, self._cfg_bullet_lifetime,
            self._cfg_max_hp, self._cfg_max_turn_rate, self._cfg_min_turn_rate,
            self._cfg_rear_aspect_cone,
            self._tick_counter,
            self._obs_p0, self._obs_p1, self._config_obs,
        ], device=self._device)
        wp.synchronize()

        return self._build_obs_torch()

    def _compute_config_obs_only(self):
        """Compute just the config observations (no base obs recomputation)."""
        wp.launch(kernels.compute_config_obs_kernel, dim=self._n_envs, inputs=[
            self._cfg_gravity, self._cfg_drag, self._cfg_turn_bleed,
            self._cfg_max_speed, self._cfg_min_speed, self._cfg_max_thrust,
            self._cfg_bullet_speed, self._cfg_gun_cooldown, self._cfg_bullet_lifetime,
            self._cfg_max_hp, self._cfg_max_turn_rate, self._cfg_min_turn_rate,
            self._cfg_rear_aspect_cone,
            self._config_obs,
        ], device=self._device)

    def _build_obs_output(self):
        """Build numpy observation arrays from GPU buffers."""
        obs_p0_np = self._obs_p0.numpy()  # [N, 46]
        obs_p1_np = self._obs_p1.numpy()  # [N, 46]

        if self._include_config_obs:
            config_np = self._config_obs.numpy()  # [N, 13]
            obs_p0_np = np.concatenate([obs_p0_np, config_np], axis=1)
            obs_p1_np = np.concatenate([obs_p1_np, config_np], axis=1)

        return obs_p0_np, obs_p1_np

    def _launch_step_kernel(self):
        """Launch the step_with_action_repeat kernel."""
        wp.launch(kernels.step_with_action_repeat, dim=self._n_envs, inputs=[
            self._pos_x, self._pos_y, self._yaw, self._speed,
            self._hp, self._cooldown, self._alive, self._stall_ticks,
            self._bul_pos_x, self._bul_pos_y, self._bul_vel_x, self._bul_vel_y,
            self._bul_owner, self._bul_ticks,
            self._cfg_gravity, self._cfg_drag, self._cfg_turn_bleed,
            self._cfg_max_speed, self._cfg_min_speed, self._cfg_max_thrust,
            self._cfg_bullet_speed, self._cfg_gun_cooldown, self._cfg_bullet_lifetime,
            self._cfg_max_hp, self._cfg_max_turn_rate, self._cfg_min_turn_rate,
            self._cfg_rear_aspect_cone,
            self._act_p0, self._act_p1,
            self._opponent_type,
            self._opp_phase, self._opp_phase_timer,
            self._opp_evade_timer, self._opp_evade_dir,
            self._opp_yo_yo_timer, self._opp_yo_yo_phase,
            self._opp_jink_timer, self._opp_jink_dir,
            self._opp_overshoot_timer, self._opp_attack_patience,
            self._opp_last_distance,
            self._prev_hp_p0, self._prev_hp_p1,
            self._prev_opp_hp_p0, self._prev_opp_hp_p1,
            self._prev_dist,
            self._obs_p0, self._obs_p1,
            self._reward_p0, self._reward_p1,
            self._done,
            self._reward_weights,
            self._tick_counter,
            self._action_repeat,
        ], device=self._device)

    def step_torch(self, actions_p0_t, actions_p1_t):
        """GPU-native step. Accepts/returns torch.Tensor on CUDA. No numpy round-trips.

        Args:
            actions_p0_t: torch.Tensor [n_envs, 3] on CUDA
            actions_p1_t: torch.Tensor [n_envs, 3] on CUDA

        Returns:
            obs_p0: torch.Tensor [n_envs, effective_obs] on CUDA
            obs_p1: torch.Tensor [n_envs, effective_obs] on CUDA
            rew_p0: torch.Tensor [n_envs] on CUDA
            rew_p1: torch.Tensor [n_envs] on CUDA
            dones: numpy bool [n_envs] (CPU, for episode tracking)
            infos: list of dicts (CPU)
        """
        # Zero-copy: write actions into warp arrays via shared torch views
        self._act_p0_torch.copy_(actions_p0_t)
        self._act_p1_torch.copy_(actions_p1_t)

        # Launch kernel
        self._launch_step_kernel()
        wp.synchronize()

        # Read dones to CPU (small: just N ints)
        dones_np = self._done_torch.cpu().numpy().astype(np.bool_)

        # Capture rewards before potential auto-reset
        rew_p0_t = self._reward_p0_torch.clone()
        rew_p1_t = self._reward_p1_torch.clone()

        # Build infos from GPU state (before reset)
        infos = self._build_infos(dones_np)

        # Auto-reset done envs
        if dones_np.any():
            self._auto_reset(dones_np)
            wp.synchronize()

        # Build combined obs on GPU
        obs_p0_t, obs_p1_t = self._build_obs_torch()
        return obs_p0_t, obs_p1_t, rew_p0_t, rew_p1_t, dones_np, infos

    def _build_obs_torch(self):
        """Build observation torch tensors on GPU (zero-copy + optional config cat)."""
        obs_p0_t = self._obs_p0_torch
        obs_p1_t = self._obs_p1_torch
        if self._include_config_obs:
            self._combined_obs_p0[:, :OBS_SIZE] = obs_p0_t
            self._combined_obs_p0[:, OBS_SIZE:] = self._config_obs_torch
            self._combined_obs_p1[:, :OBS_SIZE] = obs_p1_t
            self._combined_obs_p1[:, OBS_SIZE:] = self._config_obs_torch
            return self._combined_obs_p0, self._combined_obs_p1
        return obs_p0_t, obs_p1_t

    def step(self, actions_p0, actions_p1):
        """Step all envs. Returns (obs_p0, obs_p1, rew_p0, rew_p1, dones, infos).

        Args:
            actions_p0: numpy [n_envs, 3] float32
            actions_p1: numpy [n_envs, 3] float32
        """
        # Upload actions to GPU
        wp.copy(self._act_p0, wp.array(actions_p0, dtype=wp.float32, device=self._device))
        wp.copy(self._act_p1, wp.array(actions_p1, dtype=wp.float32, device=self._device))

        # Launch step kernel
        self._launch_step_kernel()

        wp.synchronize()

        # Read back outputs (before auto-reset modifies state)
        dones_np = self._done.numpy().astype(np.bool_)
        rew_p0_np = self._reward_p0.numpy().copy()
        rew_p1_np = self._reward_p1.numpy().copy()
        infos = self._build_infos(dones_np)

        # Auto-reset done envs (resets state, recomputes obs + config_obs)
        if dones_np.any():
            self._auto_reset(dones_np)
            wp.synchronize()
        obs_p0_np, obs_p1_np = self._build_obs_output()
        return obs_p0_np, obs_p1_np, rew_p0_np, rew_p1_np, dones_np, infos

    def _build_infos(self, dones_np):
        """Build info dicts for done envs."""
        infos = []
        if not dones_np.any():
            return [{} for _ in range(self._n_envs)]

        hp_np = self._hp.numpy()
        alive_np = self._alive.numpy()
        tick_np = self._tick_counter.numpy()

        for i in range(self._n_envs):
            info = {}
            if dones_np[i]:
                p0_hp = hp_np[i * 2]
                p1_hp = hp_np[i * 2 + 1]
                p0_alive = alive_np[i * 2]
                p1_alive = alive_np[i * 2 + 1]

                if p0_alive and not p1_alive:
                    info["outcome"] = "Player0Win"
                elif not p0_alive and p1_alive:
                    info["outcome"] = "Player1Win"
                elif not p0_alive and not p1_alive:
                    info["outcome"] = "Draw"
                elif tick_np[i] >= MAX_TICKS:
                    if p0_hp > p1_hp:
                        info["outcome"] = "Player0Win"
                    elif p1_hp > p0_hp:
                        info["outcome"] = "Player1Win"
                    else:
                        info["outcome"] = "Draw"
                else:
                    info["outcome"] = "Draw"

                info["my_hp"] = int(p0_hp)
                info["opp_hp"] = int(p1_hp)
            infos.append(info)
        return infos

    def _auto_reset(self, dones_np):
        """Auto-reset done envs with new config/opponent assignment."""
        # Build reset mask
        reset_mask_np = dones_np.astype(np.int32)

        # Re-assign scripted/neural for done envs
        opp_type_np = self._opponent_type.numpy()
        for i in range(self._n_envs):
            if dones_np[i]:
                if self._scripted_pool and self._scripted_fraction > 0:
                    if self._rng.random() < self._scripted_fraction:
                        name = self._rng.choice(self._scripted_pool)
                        opp_type_np[i] = POLICY_NAME_TO_ID[name]
                        self._scripted_mask_np[i] = True
                    else:
                        opp_type_np[i] = -1
                        self._scripted_mask_np[i] = False
                else:
                    opp_type_np[i] = -1
                    self._scripted_mask_np[i] = False

        wp.copy(self._opponent_type, wp.array(opp_type_np, dtype=wp.int32, device=self._device))

        # Re-sample configs for done envs if DR active
        if self._config_ranges:
            self._resample_configs_for(dones_np)

        # Reset opponent FSM for done envs (simple: zero out, will be overwritten)
        opp_phase_np = self._opp_phase.numpy()
        opp_phase_timer_np = self._opp_phase_timer.numpy()
        opp_evade_timer_np = self._opp_evade_timer.numpy()
        opp_evade_dir_np = self._opp_evade_dir.numpy()
        opp_yo_yo_timer_np = self._opp_yo_yo_timer.numpy()
        opp_yo_yo_phase_np = self._opp_yo_yo_phase.numpy()
        opp_jink_timer_np = self._opp_jink_timer.numpy()
        opp_jink_dir_np = self._opp_jink_dir.numpy()
        opp_overshoot_timer_np = self._opp_overshoot_timer.numpy()
        opp_attack_patience_np = self._opp_attack_patience.numpy()
        opp_last_distance_np = self._opp_last_distance.numpy()

        for i in range(self._n_envs):
            if dones_np[i]:
                opp_phase_np[i] = 0
                opp_phase_timer_np[i] = 0
                opp_evade_timer_np[i] = 0
                opp_evade_dir_np[i] = 1.0
                opp_yo_yo_timer_np[i] = 0
                opp_yo_yo_phase_np[i] = 0.0
                opp_jink_timer_np[i] = 0
                opp_jink_dir_np[i] = 1.0
                opp_overshoot_timer_np[i] = 0
                opp_attack_patience_np[i] = 0
                opp_last_distance_np[i] = 400.0

        wp.copy(self._opp_phase, wp.array(opp_phase_np, dtype=wp.int32, device=self._device))
        wp.copy(self._opp_phase_timer, wp.array(opp_phase_timer_np, dtype=wp.int32, device=self._device))
        wp.copy(self._opp_evade_timer, wp.array(opp_evade_timer_np, dtype=wp.int32, device=self._device))
        wp.copy(self._opp_evade_dir, wp.array(opp_evade_dir_np, dtype=wp.float32, device=self._device))
        wp.copy(self._opp_yo_yo_timer, wp.array(opp_yo_yo_timer_np, dtype=wp.int32, device=self._device))
        wp.copy(self._opp_yo_yo_phase, wp.array(opp_yo_yo_phase_np, dtype=wp.float32, device=self._device))
        wp.copy(self._opp_jink_timer, wp.array(opp_jink_timer_np, dtype=wp.int32, device=self._device))
        wp.copy(self._opp_jink_dir, wp.array(opp_jink_dir_np, dtype=wp.float32, device=self._device))
        wp.copy(self._opp_overshoot_timer, wp.array(opp_overshoot_timer_np, dtype=wp.int32, device=self._device))
        wp.copy(self._opp_attack_patience, wp.array(opp_attack_patience_np, dtype=wp.int32, device=self._device))
        wp.copy(self._opp_last_distance, wp.array(opp_last_distance_np, dtype=wp.float32, device=self._device))

        # Launch reset kernel
        wp.launch(kernels.reset_env, dim=self._n_envs, inputs=[
            self._pos_x, self._pos_y, self._yaw, self._speed,
            self._hp, self._cooldown, self._alive, self._stall_ticks,
            self._bul_pos_x, self._bul_pos_y, self._bul_vel_x, self._bul_vel_y,
            self._bul_owner, self._bul_ticks,
            self._cfg_max_hp, self._tick_counter, self._done,
            wp.array(reset_mask_np, dtype=wp.int32, device=self._device),
        ], device=self._device)

        # Recompute observations for reset envs
        wp.launch(kernels.compute_obs_and_config, dim=self._n_envs, inputs=[
            self._pos_x, self._pos_y, self._yaw, self._speed,
            self._hp, self._cooldown, self._alive,
            self._bul_pos_x, self._bul_pos_y, self._bul_vel_x, self._bul_vel_y,
            self._bul_owner, self._bul_ticks,
            self._cfg_gravity, self._cfg_drag, self._cfg_turn_bleed,
            self._cfg_max_speed, self._cfg_min_speed, self._cfg_max_thrust,
            self._cfg_bullet_speed, self._cfg_gun_cooldown, self._cfg_bullet_lifetime,
            self._cfg_max_hp, self._cfg_max_turn_rate, self._cfg_min_turn_rate,
            self._cfg_rear_aspect_cone,
            self._tick_counter,
            self._obs_p0, self._obs_p1, self._config_obs,
        ], device=self._device)
        wp.synchronize()

    def _resample_configs_for(self, dones_np):
        """Re-sample DR configs only for done envs."""
        cfg_arrays = {
            "gravity": (self._cfg_gravity, np.float32),
            "drag_coeff": (self._cfg_drag, np.float32),
            "turn_bleed_coeff": (self._cfg_turn_bleed, np.float32),
            "max_speed": (self._cfg_max_speed, np.float32),
            "min_speed": (self._cfg_min_speed, np.float32),
            "max_thrust": (self._cfg_max_thrust, np.float32),
            "bullet_speed": (self._cfg_bullet_speed, np.float32),
            "gun_cooldown_ticks": (self._cfg_gun_cooldown, np.int32),
            "bullet_lifetime_ticks": (self._cfg_bullet_lifetime, np.int32),
            "max_hp": (self._cfg_max_hp, np.int32),
            "max_turn_rate": (self._cfg_max_turn_rate, np.float32),
            "min_turn_rate": (self._cfg_min_turn_rate, np.float32),
            "rear_aspect_cone": (self._cfg_rear_aspect_cone, np.float32),
        }

        for param, (lo, hi) in self._config_ranges.items():
            wp_arr, np_dtype = cfg_arrays[param]
            current = wp_arr.numpy()
            for i in range(self._n_envs):
                if dones_np[i]:
                    current[i] = np.random.uniform(lo, hi)
            wp_dtype = wp.int32 if np_dtype == np.int32 else wp.float32
            if np_dtype == np.int32:
                current = current.astype(np.int32)
            wp.copy(wp_arr, wp.array(current, dtype=wp_dtype, device=self._device))

    # ── Properties matching Rust API ────────────────────────────────────────

    @property
    def n(self):
        return self._n_envs

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def obs_size(self):
        return self._effective_obs

    @property
    def action_size(self):
        return ACTION_SIZE

    @property
    def action_repeat(self):
        return self._action_repeat

    @action_repeat.setter
    def action_repeat(self, value):
        self._action_repeat = max(1, value)

    @property
    def include_config_obs(self):
        return self._include_config_obs

    @property
    def scripted_mask(self):
        return self._scripted_mask_np

    @property
    def n_scripted(self):
        return int(self._scripted_mask_np.sum())

    @property
    def scripted_fraction(self):
        return self._scripted_fraction

    # ── Setters matching Rust API ───────────────────────────────────────────

    def set_scripted_fraction(self, fraction: float):
        self._scripted_fraction = max(0.0, min(1.0, fraction))

    def set_scripted_pool(self, pool: list):
        for name in pool:
            if name not in POLICY_NAME_TO_ID:
                raise ValueError(f"Unknown policy: {name}")
        self._scripted_pool = list(pool)

    def set_domain_randomization(self, ranges: dict):
        self._config_ranges = dict(ranges)

    def set_rewards(self, damage_dealt=None, damage_taken=None, win=None,
                    lose=None, approach=None, alive=None, proximity=None,
                    facing=None):
        weights_np = self._reward_weights.numpy()
        if damage_dealt is not None:
            weights_np[:, 0] = damage_dealt
        if damage_taken is not None:
            weights_np[:, 1] = damage_taken
        if win is not None:
            weights_np[:, 2] = win
        if lose is not None:
            weights_np[:, 3] = lose
        if approach is not None:
            weights_np[:, 4] = approach
        if alive is not None:
            weights_np[:, 5] = alive
        if proximity is not None:
            weights_np[:, 6] = proximity
        if facing is not None:
            weights_np[:, 7] = facing
        wp.copy(self._reward_weights, wp.array(weights_np, dtype=wp.float32, device=self._device))
