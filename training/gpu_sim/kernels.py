"""
Core NVIDIA Warp GPU physics kernels for 2D dogfight sim.
Port of the Rust physics engine (crates/sim/src/physics.rs + observation.rs).

Architecture: Structure-of-arrays. All state in flat warp arrays indexed by
fighter_id = env_id * 2 + player. One kernel processes one env per thread,
doing 10 physics ticks + observation + reward computation.
"""

import warp as wp
import math

from .opponents import dispatch_opponent
from .constants import (
    OBS_SIZE,
    SINGLE_FRAME_OBS_SIZE,
    CONFIG_OBS_SIZE,
    BULLETS_PER_ENV,
    MAX_BULLET_SLOTS,
    MAX_TICKS,
    MAX_ENERGY,
    MAX_TURN_RATE,
    CONFIG_NORM_GRAVITY,
    CONFIG_NORM_DRAG_COEFF,
    CONFIG_NORM_TURN_BLEED_COEFF,
    CONFIG_NORM_MAX_SPEED,
    CONFIG_NORM_MIN_SPEED,
    CONFIG_NORM_MAX_THRUST,
    CONFIG_NORM_BULLET_SPEED,
    CONFIG_NORM_GUN_COOLDOWN_TICKS,
    CONFIG_NORM_BULLET_LIFETIME_TICKS,
    CONFIG_NORM_MAX_HP,
    CONFIG_NORM_MAX_TURN_RATE,
    CONFIG_NORM_MIN_TURN_RATE,
    CONFIG_NORM_REAR_ASPECT_CONE,
)

# ---------------------------------------------------------------------------
# Warp compile-time constants
# ---------------------------------------------------------------------------
PI = wp.constant(math.pi)
HALF_PI = wp.constant(math.pi / 2.0)
TWO_PI = wp.constant(2.0 * math.pi)

# Arena
W_ARENA_RADIUS = wp.constant(500.0)
W_BOUNDARY_START_HORIZ = wp.constant(450.0)
W_BOUNDARY_FORCE = wp.constant(300.0)
W_MAX_ALTITUDE = wp.constant(600.0)
W_ALT_BOUNDARY_LOW = wp.constant(50.0)
W_ALT_BOUNDARY_HIGH = wp.constant(550.0)
W_ARENA_DIAMETER = wp.constant(1000.0)

# Fighter
W_FIGHTER_RADIUS = wp.constant(8.0)
W_BULLET_RADIUS = wp.constant(3.0)
W_COLLISION_DIST_SQ = wp.constant((8.0 + 3.0) * (8.0 + 3.0))  # 121.0
W_SPAWN_OFFSET = wp.constant(8.0 + 3.0 + 1.0)  # fighter_radius + bullet_radius + 1

# Stall
W_STALL_SPEED = wp.constant(30.0)
W_STALL_RECOVERY_TICKS = wp.constant(36)
W_STALL_NOSE_DOWN_RATE = wp.constant(2.5)
W_STALL_EARLY_RECOVERY_MARGIN = wp.constant(10.0)

# Damage degradation
W_DAMAGE_SPEED_PENALTY = wp.constant(0.03)
W_DAMAGE_TURN_PENALTY = wp.constant(0.02)

# Match
W_MAX_TICKS = wp.constant(10800)

# Observation normalization
W_OBS_SIZE = wp.constant(OBS_SIZE)
W_CONFIG_OBS_SIZE = wp.constant(CONFIG_OBS_SIZE)
W_BULLETS_PER_ENV = wp.constant(BULLETS_PER_ENV)
W_MAX_BULLET_SLOTS = wp.constant(MAX_BULLET_SLOTS)
W_MAX_ENERGY = wp.constant(MAX_ENERGY)
W_MAX_TURN_RATE = wp.constant(MAX_TURN_RATE)

# Config normalization denominators
W_CONFIG_NORM_GRAVITY = wp.constant(CONFIG_NORM_GRAVITY)
W_CONFIG_NORM_DRAG_COEFF = wp.constant(CONFIG_NORM_DRAG_COEFF)
W_CONFIG_NORM_TURN_BLEED_COEFF = wp.constant(CONFIG_NORM_TURN_BLEED_COEFF)
W_CONFIG_NORM_MAX_SPEED = wp.constant(CONFIG_NORM_MAX_SPEED)
W_CONFIG_NORM_MIN_SPEED = wp.constant(CONFIG_NORM_MIN_SPEED)
W_CONFIG_NORM_MAX_THRUST = wp.constant(CONFIG_NORM_MAX_THRUST)
W_CONFIG_NORM_BULLET_SPEED = wp.constant(CONFIG_NORM_BULLET_SPEED)
W_CONFIG_NORM_GUN_COOLDOWN_TICKS = wp.constant(CONFIG_NORM_GUN_COOLDOWN_TICKS)
W_CONFIG_NORM_BULLET_LIFETIME_TICKS = wp.constant(CONFIG_NORM_BULLET_LIFETIME_TICKS)
W_CONFIG_NORM_MAX_HP = wp.constant(CONFIG_NORM_MAX_HP)
W_CONFIG_NORM_MAX_TURN_RATE_CFG = wp.constant(CONFIG_NORM_MAX_TURN_RATE)
W_CONFIG_NORM_MIN_TURN_RATE = wp.constant(CONFIG_NORM_MIN_TURN_RATE)
W_CONFIG_NORM_REAR_ASPECT_CONE = wp.constant(CONFIG_NORM_REAR_ASPECT_CONE)


# ---------------------------------------------------------------------------
# Helper functions (@wp.func)
# ---------------------------------------------------------------------------

@wp.func
def normalize_angle(a: float) -> float:
    """Wrap angle to [-PI, PI]."""
    result = a
    while result > PI:
        result = result - TWO_PI
    while result < -PI:
        result = result + TWO_PI
    return result


@wp.func
def apply_boundaries(
    pos_x: float,
    pos_y: float,
    dt: float,
) -> wp.vec2f:
    """Apply quadratic boundary forces and hard clamps. Returns (new_x, new_y)."""
    x = pos_x
    y = pos_y

    # Horizontal boundaries (x axis)
    abs_x = wp.abs(x)
    if abs_x > W_BOUNDARY_START_HORIZ:
        penetration = wp.clamp(
            (abs_x - W_BOUNDARY_START_HORIZ) / (W_ARENA_RADIUS - W_BOUNDARY_START_HORIZ),
            0.0,
            1.0,
        )
        force = penetration * penetration * W_BOUNDARY_FORCE
        sign_x = 1.0
        if x < 0.0:
            sign_x = -1.0
        x = x - sign_x * force * dt
    x = wp.clamp(x, -W_ARENA_RADIUS, W_ARENA_RADIUS)

    # Ground boundary (y near 0)
    if y < W_ALT_BOUNDARY_LOW:
        penetration = wp.clamp(
            (W_ALT_BOUNDARY_LOW - y) / W_ALT_BOUNDARY_LOW,
            0.0,
            1.0,
        )
        force = penetration * penetration * W_BOUNDARY_FORCE
        y = y + force * dt
    if y < 0.0:
        y = 0.0

    # Ceiling boundary (y near MAX_ALTITUDE)
    if y > W_ALT_BOUNDARY_HIGH:
        penetration = wp.clamp(
            (y - W_ALT_BOUNDARY_HIGH) / (W_MAX_ALTITUDE - W_ALT_BOUNDARY_HIGH),
            0.0,
            1.0,
        )
        force = penetration * penetration * W_BOUNDARY_FORCE
        y = y - force * dt
    if y > W_MAX_ALTITUDE:
        y = W_MAX_ALTITUDE

    return wp.vec2f(x, y)


@wp.func
def step_fighter(
    # Fighter state arrays (length N*2)
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    hp: wp.array(dtype=wp.int32),
    cooldown: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
    stall_ticks: wp.array(dtype=wp.int32),
    # Actions
    action_yaw: float,
    action_throttle: float,
    action_shoot: float,
    # Config arrays (length N, indexed by env_id)
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_drag: wp.array(dtype=wp.float32),
    cfg_turn_bleed: wp.array(dtype=wp.float32),
    cfg_max_speed: wp.array(dtype=wp.float32),
    cfg_min_speed: wp.array(dtype=wp.float32),
    cfg_max_thrust: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_gun_cooldown: wp.array(dtype=wp.int32),
    cfg_bullet_lifetime: wp.array(dtype=wp.int32),
    cfg_max_hp: wp.array(dtype=wp.int32),
    cfg_max_turn_rate: wp.array(dtype=wp.float32),
    cfg_min_turn_rate: wp.array(dtype=wp.float32),
    # Bullet arrays (length N * BULLETS_PER_ENV)
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    # Indices
    env_id: int,
    player: int,
):
    """Advance one fighter by one physics tick. Exact port of Rust step_fighter."""
    idx = env_id * 2 + player
    dt = 1.0 / 120.0

    # Skip dead fighters
    if alive[idx] == 0:
        return

    cur_speed = speed[idx]
    cur_yaw = yaw[idx]
    cur_px = pos_x[idx]
    cur_py = pos_y[idx]
    cur_hp = hp[idx]
    cur_stall = stall_ticks[idx]
    cur_cd = cooldown[idx]

    gravity = cfg_gravity[env_id]
    drag = cfg_drag[env_id]
    turn_bleed = cfg_turn_bleed[env_id]
    max_spd = cfg_max_speed[env_id]
    min_spd = cfg_min_speed[env_id]
    max_thr = cfg_max_thrust[env_id]
    bul_spd = cfg_bullet_speed[env_id]
    gun_cd = cfg_gun_cooldown[env_id]
    bul_life = cfg_bullet_lifetime[env_id]
    max_hp_val = cfg_max_hp[env_id]
    max_tr = cfg_max_turn_rate[env_id]
    min_tr = cfg_min_turn_rate[env_id]

    # Effective max speed (damage penalty)
    effective_max = max_spd * (1.0 - W_DAMAGE_SPEED_PENALTY * float(max_hp_val - cur_hp))

    # ---- STALL HANDLING ----
    if cur_stall > 0:
        cur_stall = cur_stall - 1

        # Rotate yaw toward -PI/2 (straight down)
        target = -HALF_PI
        diff = normalize_angle(target - cur_yaw)
        max_rot = W_STALL_NOSE_DOWN_RATE * dt
        clamped_diff = wp.clamp(diff, -max_rot, max_rot)
        cur_yaw = normalize_angle(cur_yaw + clamped_diff)

        # Gravity and drag only during stall
        cur_speed = cur_speed + (-gravity * wp.sin(cur_yaw)) * dt
        cur_speed = cur_speed - drag * cur_speed * dt
        cur_speed = wp.clamp(cur_speed, min_spd, effective_max)

        # Early recovery
        if cur_speed > W_STALL_SPEED + W_STALL_EARLY_RECOVERY_MARGIN:
            cur_stall = 0

        # Integrate position
        fwd_x = wp.cos(cur_yaw)
        fwd_y = wp.sin(cur_yaw)
        cur_px = cur_px + fwd_x * cur_speed * dt
        cur_py = cur_py + fwd_y * cur_speed * dt

        # Apply boundaries
        bounded = apply_boundaries(cur_px, cur_py, dt)
        cur_px = bounded[0]
        cur_py = bounded[1]

        # Tick cooldown
        if cur_cd > 0:
            cur_cd = cur_cd - 1

        # Write back state
        pos_x[idx] = cur_px
        pos_y[idx] = cur_py
        yaw[idx] = cur_yaw
        speed[idx] = cur_speed
        stall_ticks[idx] = cur_stall
        cooldown[idx] = cur_cd
        # No shooting during stall
        return

    # ---- STALL ENTRY ----
    if cur_speed < W_STALL_SPEED:
        stall_ticks[idx] = W_STALL_RECOVERY_TICKS
        return

    # ---- NORMAL FLIGHT ----
    # Effective turn rate: speed-dependent + damage penalty
    speed_range = max_spd - min_spd
    t = 0.0
    if speed_range > 0.0:
        t = wp.clamp((cur_speed - min_spd) / speed_range, 0.0, 1.0)
    base_turn_rate = max_tr + t * (min_tr - max_tr)
    effective_turn_rate = base_turn_rate * (1.0 - W_DAMAGE_TURN_PENALTY * float(max_hp_val - cur_hp))

    # Apply yaw input
    clamped_yaw_input = wp.clamp(action_yaw, -1.0, 1.0)
    yaw_delta = clamped_yaw_input * effective_turn_rate * dt
    cur_yaw = normalize_angle(cur_yaw + yaw_delta)

    # Energy bleed from turning
    cur_speed = cur_speed - turn_bleed * wp.abs(yaw_delta) * cur_speed

    # Thrust and drag
    throttle = wp.clamp(action_throttle, 0.0, 1.0)
    cur_speed = cur_speed + (throttle * max_thr - drag * cur_speed) * dt

    # Gravity: climbing costs speed, diving gains speed
    cur_speed = cur_speed + (-gravity * wp.sin(cur_yaw)) * dt

    # Clamp speed
    cur_speed = wp.clamp(cur_speed, min_spd, effective_max)

    # Integrate position
    fwd_x = wp.cos(cur_yaw)
    fwd_y = wp.sin(cur_yaw)
    cur_px = cur_px + fwd_x * cur_speed * dt
    cur_py = cur_py + fwd_y * cur_speed * dt

    # Apply boundaries
    bounded = apply_boundaries(cur_px, cur_py, dt)
    cur_px = bounded[0]
    cur_py = bounded[1]

    # Tick cooldown
    if cur_cd > 0:
        cur_cd = cur_cd - 1

    # Shoot: if action_shoot > 0 and cooldown == 0
    if action_shoot > 0.0 and cur_cd == 0:
        # Spawn bullet at fighter nose
        spawn_x = cur_px + fwd_x * W_SPAWN_OFFSET
        spawn_y = cur_py + fwd_y * W_SPAWN_OFFSET
        vel_x = fwd_x * bul_spd
        vel_y = fwd_y * bul_spd

        # Find an empty bullet slot for this env
        bul_base = env_id * W_BULLETS_PER_ENV
        found = int(0)
        for s in range(24):  # BULLETS_PER_ENV
            if found == 0:
                slot = bul_base + s
                if bul_ticks[slot] <= 0:
                    bul_pos_x[slot] = spawn_x
                    bul_pos_y[slot] = spawn_y
                    bul_vel_x[slot] = vel_x
                    bul_vel_y[slot] = vel_y
                    bul_owner[slot] = player
                    bul_ticks[slot] = bul_life
                    found = 1

        cur_cd = gun_cd

    # Write back state
    pos_x[idx] = cur_px
    pos_y[idx] = cur_py
    yaw[idx] = cur_yaw
    speed[idx] = cur_speed
    stall_ticks[idx] = cur_stall
    cooldown[idx] = cur_cd


@wp.func
def step_bullets(
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_ticks: wp.array(dtype=wp.int32),
    env_id: int,
):
    """Advance all bullets in one env by one tick."""
    dt = 1.0 / 120.0
    bul_base = env_id * W_BULLETS_PER_ENV
    for s in range(24):  # BULLETS_PER_ENV
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bul_pos_x[slot] = bul_pos_x[slot] + bul_vel_x[slot] * dt
            bul_pos_y[slot] = bul_pos_y[slot] + bul_vel_y[slot] * dt
            bul_ticks[slot] = bul_ticks[slot] - 1


@wp.func
def check_collisions(
    # Fighter arrays
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    hp: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
    # Bullet arrays
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    # Config
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    env_id: int,
):
    """Check bullet-fighter collisions for one env."""
    bul_base = env_id * W_BULLETS_PER_ENV
    rear_cos = wp.cos(cfg_rear_aspect_cone[env_id])

    for s in range(24):  # BULLETS_PER_ENV
        slot = bul_base + s
        if bul_ticks[slot] <= 0:
            continue

        b_px = bul_pos_x[slot]
        b_py = bul_pos_y[slot]
        b_vx = bul_vel_x[slot]
        b_vy = bul_vel_y[slot]
        b_owner = bul_owner[slot]

        # Check both fighters
        for p in range(2):
            f_idx = env_id * 2 + p
            if b_owner == p:
                continue
            if alive[f_idx] == 0:
                continue

            dx = pos_x[f_idx] - b_px
            dy = pos_y[f_idx] - b_py
            dist_sq = dx * dx + dy * dy

            if dist_sq <= W_COLLISION_DIST_SQ:
                # Rear-aspect armor check
                bv_len = wp.sqrt(b_vx * b_vx + b_vy * b_vy)
                bv_nx = 0.0
                bv_ny = 0.0
                if bv_len > 0.001:
                    bv_nx = b_vx / bv_len
                    bv_ny = b_vy / bv_len
                f_fwd_x = wp.cos(yaw[f_idx])
                f_fwd_y = wp.sin(yaw[f_idx])
                dot = bv_nx * f_fwd_x + bv_ny * f_fwd_y

                # Consume bullet regardless
                bul_ticks[slot] = 0

                # If bullet from behind (dot > cos(rear_aspect_cone)), glance off
                if dot > rear_cos:
                    # No damage, break inner loop
                    pass
                else:
                    # Apply damage
                    new_hp = hp[f_idx] - 1
                    if new_hp < 0:
                        new_hp = 0
                    hp[f_idx] = new_hp
                    if new_hp == 0:
                        alive[f_idx] = 0

                # Bullet consumed, stop checking fighters for this bullet
                break


# ---------------------------------------------------------------------------
# Observation: 56-float single frame + 4-frame stacking → 224 floats
# ---------------------------------------------------------------------------

@wp.func
def compute_single_frame(
    # Output: single-frame buffer [N*2, 56], indexed by fighter_id
    frame_buf: wp.array2d(dtype=wp.float32),
    # Fighter state arrays (length N*2)
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    hp: wp.array(dtype=wp.int32),
    cooldown: wp.array(dtype=wp.int32),
    # Previous-tick state for derivatives (length N*2)
    prev_pos_x: wp.array(dtype=wp.float32),
    prev_pos_y: wp.array(dtype=wp.float32),
    prev_yaw: wp.array(dtype=wp.float32),
    prev_speed: wp.array(dtype=wp.float32),
    # Bullet arrays (length N * BULLETS_PER_ENV)
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    # Config arrays (length N, indexed by env_id)
    cfg_max_speed: wp.array(dtype=wp.float32),
    cfg_max_hp: wp.array(dtype=wp.int32),
    cfg_gun_cooldown: wp.array(dtype=wp.int32),
    cfg_gravity: wp.array(dtype=wp.float32),
    # Indices
    env_id: int,
    player: int,
    tick: int,
):
    """Write 56-float enriched single-frame observation for one player.

    Layout: [0:8) self, [8:19) opponent, [19:51) bullets, [51:55) geometry, [55] meta
    """
    me = env_id * 2 + player
    opp = env_id * 2 + (1 - player)

    max_spd = cfg_max_speed[env_id]
    max_hp_val = float(cfg_max_hp[env_id])
    gun_cd = float(cfg_gun_cooldown[env_id])
    gravity = cfg_gravity[env_id]

    # --- Self state (8 floats) [0..8) ---
    frame_buf[me, 0] = speed[me] / max_spd
    frame_buf[me, 1] = wp.cos(yaw[me])
    frame_buf[me, 2] = wp.sin(yaw[me])
    frame_buf[me, 3] = float(hp[me]) / max_hp_val
    if gun_cd > 0.0:
        frame_buf[me, 4] = float(cooldown[me]) / gun_cd
    else:
        frame_buf[me, 4] = 0.0
    frame_buf[me, 5] = pos_y[me] / W_MAX_ALTITUDE
    frame_buf[me, 6] = pos_x[me] / W_ARENA_RADIUS
    my_energy = speed[me] * speed[me] + 2.0 * gravity * pos_y[me]
    frame_buf[me, 7] = my_energy / W_MAX_ENERGY

    # --- Opponent state (11 floats) [8..19) ---
    rel_x = pos_x[opp] - pos_x[me]
    rel_y = pos_y[opp] - pos_y[me]
    dist = wp.sqrt(rel_x * rel_x + rel_y * rel_y)

    frame_buf[me, 8] = rel_x / W_ARENA_DIAMETER
    frame_buf[me, 9] = rel_y / W_ARENA_DIAMETER
    frame_buf[me, 10] = speed[opp] / max_spd
    frame_buf[me, 11] = wp.cos(yaw[opp])
    frame_buf[me, 12] = wp.sin(yaw[opp])
    frame_buf[me, 13] = float(hp[opp]) / max_hp_val
    frame_buf[me, 14] = dist / W_ARENA_DIAMETER

    # Closure rate (positive = closing)
    closure_rate = 0.0
    if tick > 0:
        prev_rel_x = prev_pos_x[opp] - prev_pos_x[me]
        prev_rel_y = prev_pos_y[opp] - prev_pos_y[me]
        prev_dist = wp.sqrt(prev_rel_x * prev_rel_x + prev_rel_y * prev_rel_y)
        closure_rate = (prev_dist - dist) * 120.0  # / DT where DT = 1/120
    frame_buf[me, 15] = wp.clamp(closure_rate / max_spd, -1.0, 1.0)

    # Opponent angular velocity
    ang_vel = 0.0
    if tick > 0:
        ang_vel = normalize_angle(yaw[opp] - prev_yaw[opp]) * 120.0  # / DT
    frame_buf[me, 16] = wp.clamp(ang_vel / W_MAX_TURN_RATE, -1.0, 1.0)

    # Opponent energy
    opp_energy = speed[opp] * speed[opp] + 2.0 * gravity * pos_y[opp]
    frame_buf[me, 17] = opp_energy / W_MAX_ENERGY

    # Angle off tail: angle from opponent's heading to direction from opp to me
    angle_opp_to_me = wp.atan2(-rel_y, -rel_x)
    angle_off_tail = normalize_angle(angle_opp_to_me - yaw[opp])
    frame_buf[me, 18] = angle_off_tail / PI

    # --- Bullets (8 slots x 4 floats = 32 floats) [19..51) ---
    # Find 8 nearest bullets via 8 passes, each finding the closest unused one.
    sel0 = int(-1)
    sel1 = int(-1)
    sel2 = int(-1)
    sel3 = int(-1)
    sel4 = int(-1)
    sel5 = int(-1)
    sel6 = int(-1)
    sel7 = int(-1)

    bul_base = env_id * W_BULLETS_PER_ENV
    my_px = pos_x[me]
    my_py = pos_y[me]

    # Pass 0
    best_dist = float(1.0e10)
    best_slot = int(-1)
    for s in range(24):
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bx = bul_pos_x[slot] - my_px
            by = bul_pos_y[slot] - my_py
            d = wp.sqrt(bx * bx + by * by)
            if d < best_dist:
                best_dist = d
                best_slot = s
    sel0 = best_slot

    # Pass 1
    best_dist = float(1.0e10)
    best_slot = int(-1)
    for s in range(24):
        if s == sel0:
            continue
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bx = bul_pos_x[slot] - my_px
            by = bul_pos_y[slot] - my_py
            d = wp.sqrt(bx * bx + by * by)
            if d < best_dist:
                best_dist = d
                best_slot = s
    sel1 = best_slot

    # Pass 2
    best_dist = float(1.0e10)
    best_slot = int(-1)
    for s in range(24):
        if s == sel0 or s == sel1:
            continue
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bx = bul_pos_x[slot] - my_px
            by = bul_pos_y[slot] - my_py
            d = wp.sqrt(bx * bx + by * by)
            if d < best_dist:
                best_dist = d
                best_slot = s
    sel2 = best_slot

    # Pass 3
    best_dist = float(1.0e10)
    best_slot = int(-1)
    for s in range(24):
        if s == sel0 or s == sel1 or s == sel2:
            continue
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bx = bul_pos_x[slot] - my_px
            by = bul_pos_y[slot] - my_py
            d = wp.sqrt(bx * bx + by * by)
            if d < best_dist:
                best_dist = d
                best_slot = s
    sel3 = best_slot

    # Pass 4
    best_dist = float(1.0e10)
    best_slot = int(-1)
    for s in range(24):
        if s == sel0 or s == sel1 or s == sel2 or s == sel3:
            continue
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bx = bul_pos_x[slot] - my_px
            by = bul_pos_y[slot] - my_py
            d = wp.sqrt(bx * bx + by * by)
            if d < best_dist:
                best_dist = d
                best_slot = s
    sel4 = best_slot

    # Pass 5
    best_dist = float(1.0e10)
    best_slot = int(-1)
    for s in range(24):
        if s == sel0 or s == sel1 or s == sel2 or s == sel3 or s == sel4:
            continue
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bx = bul_pos_x[slot] - my_px
            by = bul_pos_y[slot] - my_py
            d = wp.sqrt(bx * bx + by * by)
            if d < best_dist:
                best_dist = d
                best_slot = s
    sel5 = best_slot

    # Pass 6
    best_dist = float(1.0e10)
    best_slot = int(-1)
    for s in range(24):
        if s == sel0 or s == sel1 or s == sel2 or s == sel3 or s == sel4 or s == sel5:
            continue
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bx = bul_pos_x[slot] - my_px
            by = bul_pos_y[slot] - my_py
            d = wp.sqrt(bx * bx + by * by)
            if d < best_dist:
                best_dist = d
                best_slot = s
    sel6 = best_slot

    # Pass 7
    best_dist = float(1.0e10)
    best_slot = int(-1)
    for s in range(24):
        if s == sel0 or s == sel1 or s == sel2 or s == sel3 or s == sel4 or s == sel5 or s == sel6:
            continue
        slot = bul_base + s
        if bul_ticks[slot] > 0:
            bx = bul_pos_x[slot] - my_px
            by = bul_pos_y[slot] - my_py
            d = wp.sqrt(bx * bx + by * by)
            if d < best_dist:
                best_dist = d
                best_slot = s
    sel7 = best_slot

    # Write bullet observation slots (base index 19, 4 floats per slot)
    # Slot 0
    base_idx = 19
    if sel0 >= 0:
        slot = bul_base + sel0
        brx = bul_pos_x[slot] - my_px
        bry = bul_pos_y[slot] - my_py
        frame_buf[me, base_idx + 0] = brx / W_ARENA_DIAMETER
        frame_buf[me, base_idx + 1] = bry / W_ARENA_DIAMETER
        if bul_owner[slot] == player:
            frame_buf[me, base_idx + 2] = 1.0
        else:
            frame_buf[me, base_idx + 2] = 0.0
        angle = wp.atan2(bry, brx)
        frame_buf[me, base_idx + 3] = angle / PI
    else:
        frame_buf[me, base_idx + 0] = 0.0
        frame_buf[me, base_idx + 1] = 0.0
        frame_buf[me, base_idx + 2] = 0.0
        frame_buf[me, base_idx + 3] = 0.0

    # Slot 1
    base_idx = 23
    if sel1 >= 0:
        slot = bul_base + sel1
        brx = bul_pos_x[slot] - my_px
        bry = bul_pos_y[slot] - my_py
        frame_buf[me, base_idx + 0] = brx / W_ARENA_DIAMETER
        frame_buf[me, base_idx + 1] = bry / W_ARENA_DIAMETER
        if bul_owner[slot] == player:
            frame_buf[me, base_idx + 2] = 1.0
        else:
            frame_buf[me, base_idx + 2] = 0.0
        angle = wp.atan2(bry, brx)
        frame_buf[me, base_idx + 3] = angle / PI
    else:
        frame_buf[me, base_idx + 0] = 0.0
        frame_buf[me, base_idx + 1] = 0.0
        frame_buf[me, base_idx + 2] = 0.0
        frame_buf[me, base_idx + 3] = 0.0

    # Slot 2
    base_idx = 27
    if sel2 >= 0:
        slot = bul_base + sel2
        brx = bul_pos_x[slot] - my_px
        bry = bul_pos_y[slot] - my_py
        frame_buf[me, base_idx + 0] = brx / W_ARENA_DIAMETER
        frame_buf[me, base_idx + 1] = bry / W_ARENA_DIAMETER
        if bul_owner[slot] == player:
            frame_buf[me, base_idx + 2] = 1.0
        else:
            frame_buf[me, base_idx + 2] = 0.0
        angle = wp.atan2(bry, brx)
        frame_buf[me, base_idx + 3] = angle / PI
    else:
        frame_buf[me, base_idx + 0] = 0.0
        frame_buf[me, base_idx + 1] = 0.0
        frame_buf[me, base_idx + 2] = 0.0
        frame_buf[me, base_idx + 3] = 0.0

    # Slot 3
    base_idx = 31
    if sel3 >= 0:
        slot = bul_base + sel3
        brx = bul_pos_x[slot] - my_px
        bry = bul_pos_y[slot] - my_py
        frame_buf[me, base_idx + 0] = brx / W_ARENA_DIAMETER
        frame_buf[me, base_idx + 1] = bry / W_ARENA_DIAMETER
        if bul_owner[slot] == player:
            frame_buf[me, base_idx + 2] = 1.0
        else:
            frame_buf[me, base_idx + 2] = 0.0
        angle = wp.atan2(bry, brx)
        frame_buf[me, base_idx + 3] = angle / PI
    else:
        frame_buf[me, base_idx + 0] = 0.0
        frame_buf[me, base_idx + 1] = 0.0
        frame_buf[me, base_idx + 2] = 0.0
        frame_buf[me, base_idx + 3] = 0.0

    # Slot 4
    base_idx = 35
    if sel4 >= 0:
        slot = bul_base + sel4
        brx = bul_pos_x[slot] - my_px
        bry = bul_pos_y[slot] - my_py
        frame_buf[me, base_idx + 0] = brx / W_ARENA_DIAMETER
        frame_buf[me, base_idx + 1] = bry / W_ARENA_DIAMETER
        if bul_owner[slot] == player:
            frame_buf[me, base_idx + 2] = 1.0
        else:
            frame_buf[me, base_idx + 2] = 0.0
        angle = wp.atan2(bry, brx)
        frame_buf[me, base_idx + 3] = angle / PI
    else:
        frame_buf[me, base_idx + 0] = 0.0
        frame_buf[me, base_idx + 1] = 0.0
        frame_buf[me, base_idx + 2] = 0.0
        frame_buf[me, base_idx + 3] = 0.0

    # Slot 5
    base_idx = 39
    if sel5 >= 0:
        slot = bul_base + sel5
        brx = bul_pos_x[slot] - my_px
        bry = bul_pos_y[slot] - my_py
        frame_buf[me, base_idx + 0] = brx / W_ARENA_DIAMETER
        frame_buf[me, base_idx + 1] = bry / W_ARENA_DIAMETER
        if bul_owner[slot] == player:
            frame_buf[me, base_idx + 2] = 1.0
        else:
            frame_buf[me, base_idx + 2] = 0.0
        angle = wp.atan2(bry, brx)
        frame_buf[me, base_idx + 3] = angle / PI
    else:
        frame_buf[me, base_idx + 0] = 0.0
        frame_buf[me, base_idx + 1] = 0.0
        frame_buf[me, base_idx + 2] = 0.0
        frame_buf[me, base_idx + 3] = 0.0

    # Slot 6
    base_idx = 43
    if sel6 >= 0:
        slot = bul_base + sel6
        brx = bul_pos_x[slot] - my_px
        bry = bul_pos_y[slot] - my_py
        frame_buf[me, base_idx + 0] = brx / W_ARENA_DIAMETER
        frame_buf[me, base_idx + 1] = bry / W_ARENA_DIAMETER
        if bul_owner[slot] == player:
            frame_buf[me, base_idx + 2] = 1.0
        else:
            frame_buf[me, base_idx + 2] = 0.0
        angle = wp.atan2(bry, brx)
        frame_buf[me, base_idx + 3] = angle / PI
    else:
        frame_buf[me, base_idx + 0] = 0.0
        frame_buf[me, base_idx + 1] = 0.0
        frame_buf[me, base_idx + 2] = 0.0
        frame_buf[me, base_idx + 3] = 0.0

    # Slot 7
    base_idx = 47
    if sel7 >= 0:
        slot = bul_base + sel7
        brx = bul_pos_x[slot] - my_px
        bry = bul_pos_y[slot] - my_py
        frame_buf[me, base_idx + 0] = brx / W_ARENA_DIAMETER
        frame_buf[me, base_idx + 1] = bry / W_ARENA_DIAMETER
        if bul_owner[slot] == player:
            frame_buf[me, base_idx + 2] = 1.0
        else:
            frame_buf[me, base_idx + 2] = 0.0
        angle = wp.atan2(bry, brx)
        frame_buf[me, base_idx + 3] = angle / PI
    else:
        frame_buf[me, base_idx + 0] = 0.0
        frame_buf[me, base_idx + 1] = 0.0
        frame_buf[me, base_idx + 2] = 0.0
        frame_buf[me, base_idx + 3] = 0.0

    # --- Relative geometry (4 floats) [51..55) ---
    angle_to_opp = wp.atan2(rel_y, rel_x)
    angle_off_nose = normalize_angle(angle_to_opp - yaw[me])
    frame_buf[me, 51] = angle_off_nose / PI

    opp_angle_to_me = wp.atan2(-rel_y, -rel_x)
    opp_angle_off_nose = normalize_angle(opp_angle_to_me - yaw[opp])
    frame_buf[me, 52] = opp_angle_off_nose / PI

    my_vel_x = speed[me] * wp.cos(yaw[me])
    my_vel_y = speed[me] * wp.sin(yaw[me])
    opp_vel_x = speed[opp] * wp.cos(yaw[opp])
    opp_vel_y = speed[opp] * wp.sin(yaw[opp])
    frame_buf[me, 53] = (my_vel_x - opp_vel_x) / max_spd
    frame_buf[me, 54] = (my_vel_y - opp_vel_y) / max_spd

    # --- Meta (1 float) [55] ---
    ticks_remaining = W_MAX_TICKS - tick
    if ticks_remaining < 0:
        ticks_remaining = 0
    frame_buf[me, 55] = float(ticks_remaining) / float(W_MAX_TICKS)


@wp.func
def stack_frames(
    # Output: stacked observation [N, 224], indexed by env_id
    obs: wp.array2d(dtype=wp.float32),
    # Input: current single frame [N*2, 56], indexed by fighter_id
    frame_buf: wp.array2d(dtype=wp.float32),
    # History ring buffer [N*2, 168] (3 prev frames × 56 floats), indexed by fighter_id
    obs_history: wp.array2d(dtype=wp.float32),
    # History count per fighter [N*2]
    obs_history_count: wp.array(dtype=wp.int32),
    # Indices
    env_id: int,
    player: int,
):
    """Stack current frame + up to 3 history frames into 224-float obs.

    Updates history ring buffer: shifts [2]←[1], [1]←[0], [0]←current.
    """
    fid = env_id * 2 + player
    count = obs_history_count[fid]

    # Copy current frame → obs[env_id, 0:56]
    for i in range(56):
        obs[env_id, i] = frame_buf[fid, i]

    # Frame 1: obs[env_id, 56:112] from obs_history[fid, 0:56]
    if count >= 1:
        for i in range(56):
            obs[env_id, 56 + i] = obs_history[fid, i]
    else:
        for i in range(56):
            obs[env_id, 56 + i] = 0.0

    # Frame 2: obs[env_id, 112:168] from obs_history[fid, 56:112]
    if count >= 2:
        for i in range(56):
            obs[env_id, 112 + i] = obs_history[fid, 56 + i]
    else:
        for i in range(56):
            obs[env_id, 112 + i] = 0.0

    # Frame 3: obs[env_id, 168:224] from obs_history[fid, 112:168]
    if count >= 3:
        for i in range(56):
            obs[env_id, 168 + i] = obs_history[fid, 112 + i]
    else:
        for i in range(56):
            obs[env_id, 168 + i] = 0.0

    # Shift history: [2] ← [1], [1] ← [0], [0] ← current
    for i in range(56):
        obs_history[fid, 112 + i] = obs_history[fid, 56 + i]
    for i in range(56):
        obs_history[fid, 56 + i] = obs_history[fid, i]
    for i in range(56):
        obs_history[fid, i] = frame_buf[fid, i]

    # Increment count (cap at 3)
    if count < 3:
        obs_history_count[fid] = count + 1


@wp.func
def compute_config_obs(
    config_obs: wp.array2d(dtype=wp.float32),
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_drag: wp.array(dtype=wp.float32),
    cfg_turn_bleed: wp.array(dtype=wp.float32),
    cfg_max_speed: wp.array(dtype=wp.float32),
    cfg_min_speed: wp.array(dtype=wp.float32),
    cfg_max_thrust: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_gun_cooldown: wp.array(dtype=wp.int32),
    cfg_bullet_lifetime: wp.array(dtype=wp.int32),
    cfg_max_hp: wp.array(dtype=wp.int32),
    cfg_max_turn_rate: wp.array(dtype=wp.float32),
    cfg_min_turn_rate: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    env_id: int,
):
    """Write 13 normalized config floats."""
    config_obs[env_id, 0] = cfg_gravity[env_id] / W_CONFIG_NORM_GRAVITY
    config_obs[env_id, 1] = cfg_drag[env_id] / W_CONFIG_NORM_DRAG_COEFF
    config_obs[env_id, 2] = cfg_turn_bleed[env_id] / W_CONFIG_NORM_TURN_BLEED_COEFF
    config_obs[env_id, 3] = cfg_max_speed[env_id] / W_CONFIG_NORM_MAX_SPEED
    config_obs[env_id, 4] = cfg_min_speed[env_id] / W_CONFIG_NORM_MIN_SPEED
    config_obs[env_id, 5] = cfg_max_thrust[env_id] / W_CONFIG_NORM_MAX_THRUST
    config_obs[env_id, 6] = cfg_bullet_speed[env_id] / W_CONFIG_NORM_BULLET_SPEED
    config_obs[env_id, 7] = float(cfg_gun_cooldown[env_id]) / W_CONFIG_NORM_GUN_COOLDOWN_TICKS
    config_obs[env_id, 8] = float(cfg_bullet_lifetime[env_id]) / W_CONFIG_NORM_BULLET_LIFETIME_TICKS
    config_obs[env_id, 9] = float(cfg_max_hp[env_id]) / W_CONFIG_NORM_MAX_HP
    config_obs[env_id, 10] = cfg_max_turn_rate[env_id] / W_CONFIG_NORM_MAX_TURN_RATE_CFG
    config_obs[env_id, 11] = cfg_min_turn_rate[env_id] / W_CONFIG_NORM_MIN_TURN_RATE
    config_obs[env_id, 12] = cfg_rear_aspect_cone[env_id] / W_CONFIG_NORM_REAR_ASPECT_CONE


@wp.func
def compute_reward(
    # Output
    reward: wp.array(dtype=wp.float32),
    # Fighter arrays
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    hp: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
    # Previous state for deltas
    prev_hp: wp.array(dtype=wp.float32),
    prev_opp_hp: wp.array(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    # Reward weights [N, 8]
    reward_weights: wp.array2d(dtype=wp.float32),
    # Indices
    env_id: int,
    player: int,
    done: int,
    tick: int,
):
    """Compute delta-based reward for one player."""
    me = env_id * 2 + player
    opp = env_id * 2 + (1 - player)

    w_damage = reward_weights[env_id, 0]       # damage dealt
    w_taken = reward_weights[env_id, 1]         # damage taken (negative weight)
    w_win = reward_weights[env_id, 2]           # win terminal
    w_lose = reward_weights[env_id, 3]          # lose terminal
    w_approach = reward_weights[env_id, 4]      # approach
    w_alive = reward_weights[env_id, 5]         # alive bonus (unused currently)
    w_proximity = reward_weights[env_id, 6]     # proximity
    w_facing = reward_weights[env_id, 7]        # facing

    my_hp = float(hp[me])
    opp_hp = float(hp[opp])
    my_alive = alive[me]
    opp_alive = alive[opp]

    r = float(0.0)

    # Damage dealt: (prev_opp_hp - current_opp_hp) * w_damage
    damage_dealt = prev_opp_hp[env_id] - opp_hp
    if damage_dealt > 0.0:
        r = r + damage_dealt * w_damage

    # Damage taken: (prev_my_hp - current_my_hp) * w_taken (w_taken is negative)
    damage_taken = prev_hp[env_id] - my_hp
    if damage_taken > 0.0:
        r = r + damage_taken * w_taken

    # Distance
    dx = pos_x[opp] - pos_x[me]
    dy = pos_y[opp] - pos_y[me]
    dist = wp.sqrt(dx * dx + dy * dy)

    # Approach reward: (prev_dist - dist) * w_approach
    approach_delta = prev_dist[env_id] - dist
    r = r + approach_delta * w_approach

    # Proximity: if alive and dist < 250
    if my_alive == 1 and dist < 250.0:
        r = r + w_proximity

    # Facing: if alive and 1 < dist < 300, and dot(forward, rel_normalized) > 0.866
    if my_alive == 1 and dist > 1.0 and dist < 300.0:
        fwd_x = wp.cos(yaw[me])
        fwd_y = wp.sin(yaw[me])
        rel_nx = dx / dist
        rel_ny = dy / dist
        facing_dot = fwd_x * rel_nx + fwd_y * rel_ny
        if facing_dot > 0.866:
            r = r + w_facing

    # Terminal reward
    if done == 1:
        # Determine winner
        if my_alive == 1 and opp_alive == 0:
            # I won by elimination
            r = r + w_win
        elif my_alive == 0 and opp_alive == 1:
            # I lost by elimination
            r = r + w_lose
        elif my_alive == 0 and opp_alive == 0:
            # Both dead - draw (no bonus)
            pass
        elif tick >= W_MAX_TICKS:
            # Timeout: compare HP
            if my_hp > opp_hp:
                r = r + w_win
            elif opp_hp > my_hp:
                r = r + w_lose
            # Equal HP at timeout: draw, no bonus

    reward[env_id] = r


# ---------------------------------------------------------------------------
# Main stepping kernel
# ---------------------------------------------------------------------------

@wp.kernel
def step_with_action_repeat(
    # Fighter state arrays (length N*2)
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    hp: wp.array(dtype=wp.int32),
    cooldown: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
    stall_ticks: wp.array(dtype=wp.int32),
    # Bullet arrays (length N * BULLETS_PER_ENV)
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    # Config arrays (length N)
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_drag: wp.array(dtype=wp.float32),
    cfg_turn_bleed: wp.array(dtype=wp.float32),
    cfg_max_speed: wp.array(dtype=wp.float32),
    cfg_min_speed: wp.array(dtype=wp.float32),
    cfg_max_thrust: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_gun_cooldown: wp.array(dtype=wp.int32),
    cfg_bullet_lifetime: wp.array(dtype=wp.int32),
    cfg_max_hp: wp.array(dtype=wp.int32),
    cfg_max_turn_rate: wp.array(dtype=wp.float32),
    cfg_min_turn_rate: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    # Actions P0 [N, 3]
    act_p0: wp.array2d(dtype=wp.float32),
    # Actions P1 [N, 3]
    act_p1: wp.array2d(dtype=wp.float32),
    # Opponent type (length N): -1 = neural, 0-4 = scripted type
    opponent_type: wp.array(dtype=wp.int32),
    # Opponent FSM state (length N each)
    opp_phase: wp.array(dtype=wp.int32),
    opp_phase_timer: wp.array(dtype=wp.int32),
    opp_evade_timer: wp.array(dtype=wp.int32),
    opp_evade_dir: wp.array(dtype=wp.float32),
    opp_yo_yo_timer: wp.array(dtype=wp.int32),
    opp_yo_yo_phase: wp.array(dtype=wp.float32),
    opp_jink_timer: wp.array(dtype=wp.int32),
    opp_jink_dir: wp.array(dtype=wp.float32),
    opp_overshoot_timer: wp.array(dtype=wp.int32),
    opp_attack_patience: wp.array(dtype=wp.int32),
    opp_last_distance: wp.array(dtype=wp.float32),
    # Previous state for reward computation (length N each)
    prev_hp_p0: wp.array(dtype=wp.float32),
    prev_hp_p1: wp.array(dtype=wp.float32),
    prev_opp_hp_p0: wp.array(dtype=wp.float32),
    prev_opp_hp_p1: wp.array(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    # Outputs: observations [N, OBS_SIZE]
    obs_p0: wp.array2d(dtype=wp.float32),
    obs_p1: wp.array2d(dtype=wp.float32),
    # Rewards [N]
    reward_p0: wp.array(dtype=wp.float32),
    reward_p1: wp.array(dtype=wp.float32),
    # Done flags [N]
    done: wp.array(dtype=wp.int32),
    # Reward weights [N, 8]
    reward_weights: wp.array2d(dtype=wp.float32),
    # Tick counter [N]
    tick_counter: wp.array(dtype=wp.int32),
    # Action repeat count
    action_repeat: int,
    # Previous-tick state for observation derivatives [N*2]
    prev_pos_x: wp.array(dtype=wp.float32),
    prev_pos_y: wp.array(dtype=wp.float32),
    prev_yaw: wp.array(dtype=wp.float32),
    prev_speed: wp.array(dtype=wp.float32),
    # Frame buffer for single-frame computation [N*2, 56]
    frame_buf: wp.array2d(dtype=wp.float32),
    # Observation history ring buffer [N*2, 168]
    obs_history: wp.array2d(dtype=wp.float32),
    obs_history_count: wp.array(dtype=wp.int32),
):
    """Main simulation kernel. Each thread processes one env for action_repeat ticks."""
    env_id = wp.tid()

    p0_idx = env_id * 2
    p1_idx = env_id * 2 + 1

    # Save prev state for reward computation
    prev_hp_p0[env_id] = float(hp[p0_idx])
    prev_hp_p1[env_id] = float(hp[p1_idx])
    prev_opp_hp_p0[env_id] = float(hp[p1_idx])
    prev_opp_hp_p1[env_id] = float(hp[p0_idx])

    dx0 = pos_x[p1_idx] - pos_x[p0_idx]
    dy0 = pos_y[p1_idx] - pos_y[p0_idx]
    prev_dist[env_id] = wp.sqrt(dx0 * dx0 + dy0 * dy0)

    # Read P0 actions (the RL agent)
    a0_yaw = act_p0[env_id, 0]
    a0_throttle = act_p0[env_id, 1]
    a0_shoot = act_p0[env_id, 2]

    # Read P1 actions (neural opponent default)
    a1_yaw = act_p1[env_id, 0]
    a1_throttle = act_p1[env_id, 1]
    a1_shoot = act_p1[env_id, 2]

    opp_type = opponent_type[env_id]
    env_done = int(0)

    # Action repeat loop
    for repeat in range(10):  # max action_repeat = 10
        if repeat >= action_repeat:
            break

        cur_tick = tick_counter[env_id]

        # Check if already terminal
        if cur_tick >= W_MAX_TICKS:
            env_done = 1
            break
        if alive[p0_idx] == 0 or alive[p1_idx] == 0:
            env_done = 1
            break

        # Save prev state for observation derivatives (one tick before current)
        prev_pos_x[p0_idx] = pos_x[p0_idx]
        prev_pos_y[p0_idx] = pos_y[p0_idx]
        prev_yaw[p0_idx] = yaw[p0_idx]
        prev_speed[p0_idx] = speed[p0_idx]
        prev_pos_x[p1_idx] = pos_x[p1_idx]
        prev_pos_y[p1_idx] = pos_y[p1_idx]
        prev_yaw[p1_idx] = yaw[p1_idx]
        prev_speed[p1_idx] = speed[p1_idx]

        # For scripted opponents, compute action each physics tick
        if opp_type >= 0:
            opp_act = dispatch_opponent(
                opp_type,
                pos_x, pos_y, yaw, speed, hp, cooldown, alive,
                bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
                cfg_gravity, cfg_bullet_speed, cfg_rear_aspect_cone,
                opp_phase, opp_phase_timer, opp_evade_timer, opp_evade_dir,
                opp_yo_yo_timer, opp_yo_yo_phase, opp_jink_timer, opp_jink_dir,
                opp_overshoot_timer, opp_attack_patience, opp_last_distance,
                env_id,
            )
            a1_yaw = opp_act[0]
            a1_throttle = opp_act[1]
            a1_shoot = opp_act[2]

        # Step both fighters
        step_fighter(
            pos_x, pos_y, yaw, speed, hp, cooldown, alive, stall_ticks,
            a0_yaw, a0_throttle, a0_shoot,
            cfg_gravity, cfg_drag, cfg_turn_bleed, cfg_max_speed, cfg_min_speed,
            cfg_max_thrust, cfg_bullet_speed, cfg_gun_cooldown, cfg_bullet_lifetime,
            cfg_max_hp, cfg_max_turn_rate, cfg_min_turn_rate,
            bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
            env_id, 0,
        )
        step_fighter(
            pos_x, pos_y, yaw, speed, hp, cooldown, alive, stall_ticks,
            a1_yaw, a1_throttle, a1_shoot,
            cfg_gravity, cfg_drag, cfg_turn_bleed, cfg_max_speed, cfg_min_speed,
            cfg_max_thrust, cfg_bullet_speed, cfg_gun_cooldown, cfg_bullet_lifetime,
            cfg_max_hp, cfg_max_turn_rate, cfg_min_turn_rate,
            bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
            env_id, 1,
        )

        # Step bullets
        step_bullets(bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_ticks, env_id)

        # Check collisions
        check_collisions(
            pos_x, pos_y, yaw, hp, alive,
            bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
            cfg_rear_aspect_cone, env_id,
        )

        # Increment tick
        tick_counter[env_id] = cur_tick + 1

        # Check terminal after tick
        new_tick = cur_tick + 1
        if new_tick >= W_MAX_TICKS:
            env_done = 1
            break
        if alive[p0_idx] == 0 or alive[p1_idx] == 0:
            env_done = 1
            break

    # Compute observations for both players (single frame + stack)
    final_tick = tick_counter[env_id]

    compute_single_frame(
        frame_buf, pos_x, pos_y, yaw, speed, hp, cooldown,
        prev_pos_x, prev_pos_y, prev_yaw, prev_speed,
        bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
        cfg_max_speed, cfg_max_hp, cfg_gun_cooldown, cfg_gravity,
        env_id, 0, final_tick,
    )
    stack_frames(obs_p0, frame_buf, obs_history, obs_history_count, env_id, 0)

    compute_single_frame(
        frame_buf, pos_x, pos_y, yaw, speed, hp, cooldown,
        prev_pos_x, prev_pos_y, prev_yaw, prev_speed,
        bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
        cfg_max_speed, cfg_max_hp, cfg_gun_cooldown, cfg_gravity,
        env_id, 1, final_tick,
    )
    stack_frames(obs_p1, frame_buf, obs_history, obs_history_count, env_id, 1)

    # Compute rewards
    compute_reward(
        reward_p0, pos_x, pos_y, yaw, speed, hp, alive,
        prev_hp_p0, prev_opp_hp_p0, prev_dist,
        reward_weights, env_id, 0, env_done, final_tick,
    )
    compute_reward(
        reward_p1, pos_x, pos_y, yaw, speed, hp, alive,
        prev_hp_p1, prev_opp_hp_p1, prev_dist,
        reward_weights, env_id, 1, env_done, final_tick,
    )

    # Set done flag
    done[env_id] = env_done


# ---------------------------------------------------------------------------
# Reset kernel
# ---------------------------------------------------------------------------

@wp.kernel
def reset_env(
    # Fighter state arrays (length N*2)
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    hp: wp.array(dtype=wp.int32),
    cooldown: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
    stall_ticks: wp.array(dtype=wp.int32),
    # Bullet arrays (length N * BULLETS_PER_ENV)
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    # Config arrays (length N) for max_hp
    cfg_max_hp: wp.array(dtype=wp.int32),
    # Tick counter [N]
    tick_counter: wp.array(dtype=wp.int32),
    # Done flags [N]
    done: wp.array(dtype=wp.int32),
    # Mask: which envs to reset [N], 1 = reset, 0 = skip
    reset_mask: wp.array(dtype=wp.int32),
    # Previous-tick state for observation derivatives [N*2]
    prev_pos_x: wp.array(dtype=wp.float32),
    prev_pos_y: wp.array(dtype=wp.float32),
    prev_yaw: wp.array(dtype=wp.float32),
    prev_speed: wp.array(dtype=wp.float32),
    # Observation history [N*2, 168]
    obs_history: wp.array2d(dtype=wp.float32),
    obs_history_count: wp.array(dtype=wp.int32),
):
    """Reset specified envs to initial state. P0 at (-200, 300), P1 at (200, 300)."""
    env_id = wp.tid()

    if reset_mask[env_id] == 0:
        return

    p0 = env_id * 2
    p1 = env_id * 2 + 1
    max_hp_val = cfg_max_hp[env_id]

    # P0: left side, facing right
    pos_x[p0] = -200.0
    pos_y[p0] = 300.0
    yaw[p0] = 0.0
    speed[p0] = 50.0  # SPAWN_SPEED (above stall threshold)
    hp[p0] = max_hp_val
    cooldown[p0] = 0
    alive[p0] = 1
    stall_ticks[p0] = 0

    # P1: right side, facing left
    pos_x[p1] = 200.0
    pos_y[p1] = 300.0
    yaw[p1] = PI
    speed[p1] = 50.0  # SPAWN_SPEED (above stall threshold)
    hp[p1] = max_hp_val
    cooldown[p1] = 0
    alive[p1] = 1
    stall_ticks[p1] = 0

    # Clear all bullet slots for this env
    bul_base = env_id * W_BULLETS_PER_ENV
    for s in range(24):  # BULLETS_PER_ENV
        slot = bul_base + s
        bul_pos_x[slot] = 0.0
        bul_pos_y[slot] = 0.0
        bul_vel_x[slot] = 0.0
        bul_vel_y[slot] = 0.0
        bul_owner[slot] = 0
        bul_ticks[slot] = 0

    # Reset tick counter and done flag
    tick_counter[env_id] = 0
    done[env_id] = 0

    # Set prev state to spawn values (so tick-0 derivatives = 0)
    prev_pos_x[p0] = -200.0
    prev_pos_y[p0] = 300.0
    prev_yaw[p0] = 0.0
    prev_speed[p0] = 50.0
    prev_pos_x[p1] = 200.0
    prev_pos_y[p1] = 300.0
    prev_yaw[p1] = PI
    prev_speed[p1] = 50.0

    # Zero out observation history for both fighters
    for i in range(168):  # 3 frames * 56 floats
        obs_history[p0, i] = 0.0
        obs_history[p1, i] = 0.0
    obs_history_count[p0] = 0
    obs_history_count[p1] = 0


# ---------------------------------------------------------------------------
# Combined observation + config observation kernel
# ---------------------------------------------------------------------------

@wp.kernel
def compute_obs_and_config(
    # Fighter state arrays (length N*2)
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    hp: wp.array(dtype=wp.int32),
    cooldown: wp.array(dtype=wp.int32),
    # Bullet arrays
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    # Config arrays (length N)
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_drag: wp.array(dtype=wp.float32),
    cfg_turn_bleed: wp.array(dtype=wp.float32),
    cfg_max_speed: wp.array(dtype=wp.float32),
    cfg_min_speed: wp.array(dtype=wp.float32),
    cfg_max_thrust: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_gun_cooldown: wp.array(dtype=wp.int32),
    cfg_bullet_lifetime: wp.array(dtype=wp.int32),
    cfg_max_hp: wp.array(dtype=wp.int32),
    cfg_max_turn_rate: wp.array(dtype=wp.float32),
    cfg_min_turn_rate: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    # Tick counter [N]
    tick_counter: wp.array(dtype=wp.int32),
    # Outputs
    obs_p0: wp.array2d(dtype=wp.float32),
    obs_p1: wp.array2d(dtype=wp.float32),
    config_obs: wp.array2d(dtype=wp.float32),
    # Previous-tick state for observation derivatives [N*2]
    prev_pos_x: wp.array(dtype=wp.float32),
    prev_pos_y: wp.array(dtype=wp.float32),
    prev_yaw: wp.array(dtype=wp.float32),
    prev_speed: wp.array(dtype=wp.float32),
    # Frame buffer [N*2, 56]
    frame_buf: wp.array2d(dtype=wp.float32),
    # Observation history [N*2, 168]
    obs_history: wp.array2d(dtype=wp.float32),
    obs_history_count: wp.array(dtype=wp.int32),
):
    """Compute observations and config observations for both players in one pass."""
    env_id = wp.tid()
    tick = tick_counter[env_id]

    compute_single_frame(
        frame_buf, pos_x, pos_y, yaw, speed, hp, cooldown,
        prev_pos_x, prev_pos_y, prev_yaw, prev_speed,
        bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
        cfg_max_speed, cfg_max_hp, cfg_gun_cooldown, cfg_gravity,
        env_id, 0, tick,
    )
    stack_frames(obs_p0, frame_buf, obs_history, obs_history_count, env_id, 0)

    compute_single_frame(
        frame_buf, pos_x, pos_y, yaw, speed, hp, cooldown,
        prev_pos_x, prev_pos_y, prev_yaw, prev_speed,
        bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
        cfg_max_speed, cfg_max_hp, cfg_gun_cooldown, cfg_gravity,
        env_id, 1, tick,
    )
    stack_frames(obs_p1, frame_buf, obs_history, obs_history_count, env_id, 1)

    compute_config_obs(
        config_obs,
        cfg_gravity, cfg_drag, cfg_turn_bleed, cfg_max_speed, cfg_min_speed,
        cfg_max_thrust, cfg_bullet_speed, cfg_gun_cooldown, cfg_bullet_lifetime,
        cfg_max_hp, cfg_max_turn_rate, cfg_min_turn_rate, cfg_rear_aspect_cone,
        env_id,
    )


# ---------------------------------------------------------------------------
# Config-only observation kernel (no base obs recomputation)
# ---------------------------------------------------------------------------

@wp.kernel
def compute_config_obs_kernel(
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_drag: wp.array(dtype=wp.float32),
    cfg_turn_bleed: wp.array(dtype=wp.float32),
    cfg_max_speed: wp.array(dtype=wp.float32),
    cfg_min_speed: wp.array(dtype=wp.float32),
    cfg_max_thrust: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_gun_cooldown: wp.array(dtype=wp.int32),
    cfg_bullet_lifetime: wp.array(dtype=wp.int32),
    cfg_max_hp: wp.array(dtype=wp.int32),
    cfg_max_turn_rate: wp.array(dtype=wp.float32),
    cfg_min_turn_rate: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    config_obs: wp.array2d(dtype=wp.float32),
):
    """Compute just the 13-float config observation (no base obs recomputation)."""
    env_id = wp.tid()
    compute_config_obs(
        config_obs,
        cfg_gravity, cfg_drag, cfg_turn_bleed, cfg_max_speed, cfg_min_speed,
        cfg_max_thrust, cfg_bullet_speed, cfg_gun_cooldown, cfg_bullet_lifetime,
        cfg_max_hp, cfg_max_turn_rate, cfg_min_turn_rate, cfg_rear_aspect_cone,
        env_id,
    )
