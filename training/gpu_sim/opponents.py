"""GPU opponent policies — Warp @wp.func device functions ported from Rust.

Each opponent reads fighter/bullet state from arrays and returns (yaw_input, throttle, shoot)
packed in a wp.vec3. The dispatch_opponent() function is the main entry point called from kernels.

Opponent FSM state is stored in per-env arrays passed by the caller.
"""

import warp as wp
import math

# ── Constants ────────────────────────────────────────────────────────────────

PI = wp.constant(math.pi)
HALF_PI = wp.constant(math.pi / 2.0)
TWO_PI = wp.constant(2.0 * math.pi)
STALL_SPEED = wp.constant(30.0)
BULLETS_PER_ENV = wp.constant(24)

# Opponent type IDs (must match constants.py)
OPP_DO_NOTHING = wp.constant(0)
OPP_CHASER = wp.constant(1)
OPP_DOGFIGHTER = wp.constant(2)
OPP_ACE = wp.constant(3)
OPP_BRAWLER = wp.constant(4)

# Dogfighter modes
MODE_ATTACK = wp.constant(0)
MODE_DEFEND = wp.constant(1)
MODE_ENERGY = wp.constant(2)
MODE_DISENGAGE = wp.constant(3)

# Brawler phases
PHASE_CLOSE = wp.constant(0)
PHASE_BRAWL = wp.constant(1)
PHASE_OVERSHOOT_BAIT = wp.constant(2)
PHASE_OVERSHOOT_PUNISH = wp.constant(3)
PHASE_RETREAT = wp.constant(4)


# ── Shared tactical helpers ──────────────────────────────────────────────────


@wp.func
def angle_diff(target: float, current: float) -> float:
    """Shortest angular difference in [-PI, PI]."""
    d = target - current
    # Wrap into [-PI, PI] without a while loop
    d = d - wp.round(d / TWO_PI) * TWO_PI
    return d


@wp.func
def yaw_toward(desired: float, current: float, gain: float) -> float:
    """Returns clamped yaw_input [-1,1] to steer toward desired yaw."""
    d = angle_diff(desired, current)
    return wp.clamp(d * gain, -1.0, 1.0)


@wp.func
def altitude_safety(altitude: float, yaw: float) -> wp.vec2:
    """Returns (has_override, yaw_input). has_override > 0.5 means use the override."""
    has_override = 0.0
    yaw_input = 0.0
    # Emergency ground avoidance
    if altitude < 70.0 and wp.sin(yaw) < -0.3:
        has_override = 1.0
        if wp.cos(yaw) > 0.0:
            yaw_input = 1.0
        else:
            yaw_input = -1.0
    # Emergency ceiling avoidance
    if altitude > 570.0 and wp.sin(yaw) > 0.3:
        has_override = 1.0
        if wp.cos(yaw) > 0.0:
            yaw_input = -1.0
        else:
            yaw_input = 1.0
    return wp.vec2(has_override, yaw_input)


@wp.func
def stall_avoidance(speed: float, yaw_input: float) -> wp.vec2:
    """Returns (adjusted_yaw, min_throttle)."""
    adj_yaw = yaw_input
    min_thr = 0.0
    if speed < STALL_SPEED + 15.0:
        urgency = wp.clamp((STALL_SPEED + 15.0 - speed) / 15.0, 0.0, 1.0)
        max_yaw = 1.0 - urgency * 0.7
        min_thr = urgency * 0.8
        adj_yaw = wp.clamp(yaw_input, -max_yaw, max_yaw)
    return wp.vec2(adj_yaw, min_thr)


@wp.func
def lead_aim(
    rel_x: float,
    rel_y: float,
    opp_speed: float,
    opp_yaw: float,
    distance: float,
    lead_factor: float,
    bullet_speed: float,
) -> float:
    """Lead pursuit aim — returns desired yaw angle."""
    time_to_target = distance / bullet_speed
    lead_x = rel_x + wp.cos(opp_yaw) * opp_speed * time_to_target * lead_factor
    lead_y = rel_y + wp.sin(opp_yaw) * opp_speed * time_to_target * lead_factor
    return wp.atan2(lead_y, lead_x)


@wp.func
def crossing_aim(
    rel_x: float,
    rel_y: float,
    opp_speed: float,
    opp_yaw: float,
    distance: float,
    lead_factor: float,
    bullet_speed: float,
) -> float:
    """Crossing aim — lead aim with ~60m perpendicular offset."""
    time_to_target = distance / bullet_speed
    opp_fwd_x = wp.cos(opp_yaw)
    opp_fwd_y = wp.sin(opp_yaw)
    lead_x = rel_x + opp_fwd_x * opp_speed * time_to_target * lead_factor
    lead_y = rel_y + opp_fwd_y * opp_speed * time_to_target * lead_factor
    perp_x = -opp_fwd_y
    perp_y = opp_fwd_x
    side = 1.0
    if (rel_x * perp_x + rel_y * perp_y) < 0.0:
        side = -1.0
    offset = 60.0
    return wp.atan2(lead_y + perp_y * offset * side, lead_x + perp_x * offset * side)


@wp.func
def smart_aim(
    rel_x: float,
    rel_y: float,
    opp_speed: float,
    opp_yaw: float,
    am_behind_opponent: int,
    distance: float,
    lead_factor: float,
    bullet_speed: float,
) -> float:
    """Use crossing_aim when behind, lead_aim otherwise."""
    if am_behind_opponent == 1:
        return crossing_aim(rel_x, rel_y, opp_speed, opp_yaw, distance, lead_factor, bullet_speed)
    else:
        return lead_aim(rel_x, rel_y, opp_speed, opp_yaw, distance, lead_factor, bullet_speed)


@wp.func
def is_rear_aspect_shot(
    rel_x: float,
    rel_y: float,
    opp_yaw: float,
    my_yaw: float,
    rear_aspect_cone: float,
) -> int:
    """Returns 1 if a shot now would be rear-aspect (wasted)."""
    bullet_dir_x = wp.cos(my_yaw)
    bullet_dir_y = wp.sin(my_yaw)
    opp_fwd_x = wp.cos(opp_yaw)
    opp_fwd_y = wp.sin(opp_yaw)
    angle_to_opp = wp.atan2(rel_y, rel_x)
    angle_off = wp.abs(angle_diff(angle_to_opp, my_yaw))
    if angle_off > 0.5:
        return 0
    dot = bullet_dir_x * opp_fwd_x + bullet_dir_y * opp_fwd_y
    if dot > wp.cos(rear_aspect_cone):
        return 1
    return 0


@wp.func
def can_shoot(
    angle_off_nose: float,
    distance: float,
    gun_cooldown_ticks: int,
    would_be_rear_aspect: int,
    angle_thresh: float,
    dist_thresh: float,
) -> float:
    """Returns 1.0 if should fire, 0.0 otherwise."""
    if wp.abs(angle_off_nose) < angle_thresh and distance < dist_thresh and gun_cooldown_ticks == 0 and would_be_rear_aspect == 0:
        return 1.0
    return 0.0


@wp.func
def compute_bullet_threat(
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    my_x: float,
    my_y: float,
    my_player: int,
    env_id: int,
) -> wp.vec3:
    """Returns (nearest_dist, nearest_angle, threat_count)."""
    nearest_dist = float(99999.0)
    nearest_angle = float(0.0)
    threat_count = float(0.0)
    base = env_id * 24
    for i in range(24):
        idx = base + i
        if bul_ticks[idx] > 0 and bul_owner[idx] != my_player:
            dx = bul_pos_x[idx] - my_x
            dy = bul_pos_y[idx] - my_y
            dist = wp.sqrt(dx * dx + dy * dy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_angle = wp.atan2(dy, dx)
            if dist < 150.0:
                threat_count = threat_count + 1.0
    return wp.vec3(nearest_dist, nearest_angle, threat_count)


@wp.func
def altitude_bias_dogfighter(altitude: float, yaw: float, speed: float) -> float:
    """Altitude nudge for dogfighter attack mode."""
    bias = 0.0
    # ALT_BOUNDARY_HIGH = 550
    if altitude > 550.0:
        urgency = wp.min((altitude - 550.0) / 50.0, 1.0)
        if wp.sin(yaw) > 0.0:
            bias = bias - urgency * 0.3
    elif altitude > 450.0 and wp.sin(yaw) > 0.3:
        bias = bias - 0.1
    if altitude < 100.0:
        urgency = wp.min((100.0 - altitude) / 60.0, 1.0)
        if wp.sin(yaw) < 0.1:
            bias = bias + urgency * 0.4
    elif altitude < 180.0 and wp.sin(yaw) < -0.2:
        bias = bias + 0.15
    if speed < 100.0 and altitude > 250.0 and wp.sin(yaw) > -0.2:
        bias = bias - 0.08
    return bias


# ── Per-opponent act functions ───────────────────────────────────────────────


@wp.func
def act_do_nothing() -> wp.vec3:
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def act_chaser(
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw_arr: wp.array(dtype=wp.float32),
    speed_arr: wp.array(dtype=wp.float32),
    hp_arr: wp.array(dtype=wp.int32),
    cooldown_arr: wp.array(dtype=wp.int32),
    alive_arr: wp.array(dtype=wp.int32),
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    opp_evade_timer: wp.array(dtype=wp.int32),
    opp_evade_dir: wp.array(dtype=wp.float32),
    opp_yo_yo_timer: wp.array(dtype=wp.int32),
    opp_yo_yo_phase: wp.array(dtype=wp.float32),
    env_id: int,
) -> wp.vec3:
    my_idx = env_id * 2 + 1
    opp_idx = env_id * 2

    my_speed = speed_arr[my_idx]
    my_yaw = yaw_arr[my_idx]
    altitude = pos_y[my_idx]

    rel_x = pos_x[opp_idx] - pos_x[my_idx]
    rel_y = pos_y[opp_idx] - pos_y[my_idx]
    opp_speed = speed_arr[opp_idx]
    opp_yaw = yaw_arr[opp_idx]
    distance = wp.sqrt(rel_x * rel_x + rel_y * rel_y)
    distance_safe = wp.max(distance, 1.0)

    angle_to_opp = wp.atan2(rel_y, rel_x)
    angle_off_nose = angle_diff(angle_to_opp, my_yaw)
    angle_opp_to_me = wp.atan2(-rel_y, -rel_x)
    opp_angle_to_me = angle_diff(angle_opp_to_me, opp_yaw)

    am_behind_opponent = 0
    if wp.abs(opp_angle_to_me) > 2.0:
        am_behind_opponent = 1

    # Closing rate
    my_vx = my_speed * wp.cos(my_yaw)
    my_vy = my_speed * wp.sin(my_yaw)
    opp_vx = opp_speed * wp.cos(opp_yaw)
    opp_vy = opp_speed * wp.sin(opp_yaw)
    closing_rate = (rel_x * (my_vx - opp_vx) + rel_y * (my_vy - opp_vy)) / distance_safe

    rear_cone = cfg_rear_aspect_cone[env_id]
    bullet_spd = cfg_bullet_speed[env_id]
    gun_cd = cooldown_arr[my_idx]

    rear_asp = is_rear_aspect_shot(rel_x, rel_y, opp_yaw, my_yaw, rear_cone)

    # Altitude safety
    alt_safe = altitude_safety(altitude, my_yaw)
    if alt_safe[0] > 0.5:
        return wp.vec3(alt_safe[1], 1.0, 0.0)

    # Tick timers
    ev_timer = opp_evade_timer[env_id]
    ev_dir = opp_evade_dir[env_id]
    yy_timer = opp_yo_yo_timer[env_id]
    yy_phase = opp_yo_yo_phase[env_id]

    if ev_timer > 0:
        ev_timer = ev_timer - 1
    if yy_timer > 0:
        yy_timer = yy_timer - 1

    # Bullet threat
    threat = compute_bullet_threat(bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y,
                                   bul_owner, bul_ticks, pos_x[my_idx], pos_y[my_idx], 1, env_id)
    nearest_bul_dist = threat[0]

    # Bullet evasion trigger
    if nearest_bul_dist < 80.0 and ev_timer == 0:
        ev_timer = 15
        ev_dir = -ev_dir

    # If evading
    if ev_timer > 0:
        opp_evade_timer[env_id] = ev_timer
        opp_evade_dir[env_id] = ev_dir
        opp_yo_yo_timer[env_id] = yy_timer
        opp_yo_yo_phase[env_id] = yy_phase
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.3, 320.0)
        return wp.vec3(ev_dir, 0.8, shoot_val)

    # Yo-yo maneuvers
    if yy_timer == 0:
        if distance < 120.0 and closing_rate > 50.0 and altitude < 500.0:
            yy_timer = 40
            yy_phase = 1.0
        elif distance > 300.0 and closing_rate < -20.0 and altitude > 150.0:
            yy_timer = 30
            yy_phase = -1.0

    # Smart aim
    desired_yaw = smart_aim(rel_x, rel_y, opp_speed, opp_yaw, am_behind_opponent, distance_safe, 1.0, bullet_spd)
    yaw_diff = angle_diff(desired_yaw, my_yaw)

    # Yo-yo bias
    if yy_timer > 0:
        yaw_diff = yaw_diff + yy_phase * 0.4

    # Altitude management
    if altitude < 120.0 and wp.sin(my_yaw) < 0.1:
        yaw_diff = yaw_diff + 0.25
    elif altitude > 520.0 and wp.sin(my_yaw) > -0.1:
        yaw_diff = yaw_diff - 0.25

    yaw_input = wp.clamp(yaw_diff * 2.5, -1.0, 1.0)

    # Throttle
    throttle = 1.0
    if my_speed > 120.0 and wp.abs(yaw_input) > 0.7:
        throttle = 0.7

    shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.22, 320.0)

    # Stall avoidance
    sa = stall_avoidance(my_speed, yaw_input)
    yaw_input = sa[0]
    throttle = wp.max(throttle, sa[1])

    # Write back FSM state
    opp_evade_timer[env_id] = ev_timer
    opp_evade_dir[env_id] = ev_dir
    opp_yo_yo_timer[env_id] = yy_timer
    opp_yo_yo_phase[env_id] = yy_phase

    return wp.vec3(yaw_input, throttle, shoot_val)


@wp.func
def act_dogfighter(
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw_arr: wp.array(dtype=wp.float32),
    speed_arr: wp.array(dtype=wp.float32),
    hp_arr: wp.array(dtype=wp.int32),
    cooldown_arr: wp.array(dtype=wp.int32),
    alive_arr: wp.array(dtype=wp.int32),
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    opp_phase: wp.array(dtype=wp.int32),
    opp_phase_timer: wp.array(dtype=wp.int32),
    opp_evade_timer: wp.array(dtype=wp.int32),
    opp_evade_dir: wp.array(dtype=wp.float32),
    opp_attack_patience: wp.array(dtype=wp.int32),
    opp_last_distance: wp.array(dtype=wp.float32),
    env_id: int,
) -> wp.vec3:
    my_idx = env_id * 2 + 1
    opp_idx = env_id * 2

    my_speed = speed_arr[my_idx]
    my_yaw = yaw_arr[my_idx]
    altitude = pos_y[my_idx]

    rel_x = pos_x[opp_idx] - pos_x[my_idx]
    rel_y = pos_y[opp_idx] - pos_y[my_idx]
    opp_speed = speed_arr[opp_idx]
    opp_yaw = yaw_arr[opp_idx]
    distance = wp.sqrt(rel_x * rel_x + rel_y * rel_y)
    distance_safe = wp.max(distance, 1.0)

    angle_to_opp = wp.atan2(rel_y, rel_x)
    angle_off_nose = angle_diff(angle_to_opp, my_yaw)
    angle_opp_to_me = wp.atan2(-rel_y, -rel_x)
    opp_angle_to_me = angle_diff(angle_opp_to_me, opp_yaw)

    am_behind_opponent = 0
    if wp.abs(opp_angle_to_me) > 2.0:
        am_behind_opponent = 1
    opponent_behind_me = 0
    if wp.abs(angle_off_nose) > 2.0:
        opponent_behind_me = 1

    # Energy
    gravity = cfg_gravity[env_id]
    opp_alt = altitude + rel_y
    my_energy = my_speed * my_speed + 2.0 * gravity * altitude
    opp_energy = opp_speed * opp_speed + 2.0 * gravity * opp_alt
    energy_advantage = my_energy / wp.max(opp_energy, 1.0)

    rear_cone = cfg_rear_aspect_cone[env_id]
    bullet_spd = cfg_bullet_speed[env_id]
    gun_cd = cooldown_arr[my_idx]
    rear_asp = is_rear_aspect_shot(rel_x, rel_y, opp_yaw, my_yaw, rear_cone)

    # Altitude safety
    alt_safe = altitude_safety(altitude, my_yaw)
    if alt_safe[0] > 0.5:
        return wp.vec3(alt_safe[1], 1.0, 0.0)

    # Load FSM state
    mode = opp_phase[env_id]
    mode_timer = opp_phase_timer[env_id]
    ev_timer = opp_evade_timer[env_id]
    ev_dir = opp_evade_dir[env_id]
    atk_patience = opp_attack_patience[env_id]

    # Tick timers
    if mode_timer > 0:
        mode_timer = mode_timer - 1
    if ev_timer > 0:
        ev_timer = ev_timer - 1

    # Bullet evasion
    threat = compute_bullet_threat(bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y,
                                   bul_owner, bul_ticks, pos_x[my_idx], pos_y[my_idx], 1, env_id)
    nearest_bul_dist = threat[0]

    if nearest_bul_dist < 90.0 and ev_timer == 0:
        ev_timer = 18
        ev_dir = -ev_dir

    if ev_timer > 0:
        # Evade action
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.3, 350.0)
        evade_yaw = ev_dir
        if altitude < 80.0 and ev_dir < 0.0 and wp.sin(my_yaw) < 0.0:
            evade_yaw = 1.0
        sa = stall_avoidance(my_speed, evade_yaw)
        evade_yaw = sa[0]
        throttle = wp.max(0.8, sa[1])
        # Write back
        opp_phase[env_id] = mode
        opp_phase_timer[env_id] = mode_timer
        opp_evade_timer[env_id] = ev_timer
        opp_evade_dir[env_id] = ev_dir
        opp_attack_patience[env_id] = atk_patience
        opp_last_distance[env_id] = distance
        return wp.vec3(evade_yaw, throttle, shoot_val)

    # Mode switching (with hysteresis)
    if mode_timer == 0:
        new_mode = -1
        new_timer = 0
        if mode == MODE_ATTACK:
            if opponent_behind_me == 1 and distance < 200.0:
                new_mode = MODE_DEFEND
                new_timer = 45
            elif energy_advantage < 0.7 and altitude > 100.0:
                new_mode = MODE_ENERGY
                new_timer = 60
            elif atk_patience > 360 and distance < 250.0:
                new_mode = MODE_DISENGAGE
                new_timer = 90
        elif mode == MODE_DEFEND:
            if opponent_behind_me == 0 or distance > 300.0:
                new_mode = MODE_ATTACK
                new_timer = 30
            elif energy_advantage < 0.6:
                new_mode = MODE_ENERGY
                new_timer = 60
        elif mode == MODE_ENERGY:
            if energy_advantage > 0.9:
                new_mode = MODE_ATTACK
                new_timer = 30
            elif opponent_behind_me == 1 and distance < 150.0:
                new_mode = MODE_DEFEND
                new_timer = 45
        elif mode == MODE_DISENGAGE:
            if distance > 350.0 or am_behind_opponent == 1:
                new_mode = MODE_ATTACK
                new_timer = 30
            elif opponent_behind_me == 1 and distance < 150.0:
                new_mode = MODE_DEFEND
                new_timer = 45
        if new_mode >= 0:
            mode = new_mode
            mode_timer = new_timer

    # Track attack patience
    if mode == MODE_ATTACK:
        atk_patience = atk_patience + 1
    else:
        atk_patience = 0

    # Execute mode
    yaw_input = 0.0
    throttle = 1.0
    shoot_val = 0.0

    if mode == MODE_ATTACK:
        desired_yaw = smart_aim(rel_x, rel_y, opp_speed, opp_yaw, am_behind_opponent, distance_safe, 1.0, bullet_spd)
        yaw_diff = angle_diff(desired_yaw, my_yaw)
        yaw_diff = yaw_diff + altitude_bias_dogfighter(altitude, my_yaw, my_speed)
        yaw_input = wp.clamp(yaw_diff * 3.0, -1.0, 1.0)
        if my_speed < 80.0:
            throttle = 1.0
        elif wp.abs(yaw_input) > 0.7:
            throttle = 0.5
        elif distance > 250.0:
            throttle = 1.0
        else:
            throttle = 0.7
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.22, 350.0)
    elif mode == MODE_DEFEND:
        break_dir = 1.0
        if angle_off_nose < 0.0:
            break_dir = -1.0
        perp_yaw = angle_to_opp + HALF_PI * break_dir
        yaw_input = yaw_toward(perp_yaw, my_yaw, 3.5)
        if my_speed > 100.0:
            throttle = 0.3
        else:
            throttle = 0.6
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.3, 250.0)
    elif mode == MODE_ENERGY:
        if altitude < 400.0:
            away_x = -rel_x
            if away_x > 0.0:
                climb_angle = wp.atan2(0.5, 1.0)
            else:
                climb_angle = PI - wp.atan2(0.5, 1.0)
            desired_yaw_e = climb_angle
        else:
            if wp.cos(my_yaw) > 0.0:
                desired_yaw_e = 0.0
            else:
                desired_yaw_e = PI
        yaw_input = yaw_toward(desired_yaw_e, my_yaw, 2.0)
        throttle = 1.0
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.25, 300.0)
    elif mode == MODE_DISENGAGE:
        away_yaw = wp.atan2(-rel_y, -rel_x)
        yaw_input = yaw_toward(away_yaw, my_yaw, 2.5)
        throttle = 1.0
        shoot_val = 0.0

    # Stall avoidance
    sa = stall_avoidance(my_speed, yaw_input)
    yaw_input = sa[0]
    throttle = wp.max(throttle, sa[1])

    # Write back FSM state
    opp_phase[env_id] = mode
    opp_phase_timer[env_id] = mode_timer
    opp_evade_timer[env_id] = ev_timer
    opp_evade_dir[env_id] = ev_dir
    opp_attack_patience[env_id] = atk_patience
    opp_last_distance[env_id] = distance

    return wp.vec3(yaw_input, throttle, shoot_val)


@wp.func
def act_ace(
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw_arr: wp.array(dtype=wp.float32),
    speed_arr: wp.array(dtype=wp.float32),
    hp_arr: wp.array(dtype=wp.int32),
    cooldown_arr: wp.array(dtype=wp.int32),
    alive_arr: wp.array(dtype=wp.int32),
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    opp_evade_timer: wp.array(dtype=wp.int32),
    opp_evade_dir: wp.array(dtype=wp.float32),
    env_id: int,
) -> wp.vec3:
    my_idx = env_id * 2 + 1
    opp_idx = env_id * 2

    my_speed = speed_arr[my_idx]
    my_yaw = yaw_arr[my_idx]
    altitude = pos_y[my_idx]

    rel_x = pos_x[opp_idx] - pos_x[my_idx]
    rel_y = pos_y[opp_idx] - pos_y[my_idx]
    opp_speed = speed_arr[opp_idx]
    opp_yaw = yaw_arr[opp_idx]
    distance = wp.sqrt(rel_x * rel_x + rel_y * rel_y)
    distance_safe = wp.max(distance, 1.0)

    angle_to_opp = wp.atan2(rel_y, rel_x)
    angle_off_nose = angle_diff(angle_to_opp, my_yaw)
    angle_opp_to_me = wp.atan2(-rel_y, -rel_x)
    opp_angle_to_me = angle_diff(angle_opp_to_me, opp_yaw)

    am_behind_opponent = 0
    if wp.abs(opp_angle_to_me) > 2.0:
        am_behind_opponent = 1
    opponent_behind_me = 0
    if wp.abs(angle_off_nose) > 2.0:
        opponent_behind_me = 1

    rear_cone = cfg_rear_aspect_cone[env_id]
    bullet_spd = cfg_bullet_speed[env_id]
    gun_cd = cooldown_arr[my_idx]
    rear_asp = is_rear_aspect_shot(rel_x, rel_y, opp_yaw, my_yaw, rear_cone)

    # Altitude safety
    alt_safe = altitude_safety(altitude, my_yaw)
    if alt_safe[0] > 0.5:
        return wp.vec3(alt_safe[1], 1.0, 0.0)

    # Load FSM
    ev_timer = opp_evade_timer[env_id]
    ev_dir = opp_evade_dir[env_id]

    # Tick timer
    if ev_timer > 0:
        ev_timer = ev_timer - 1

    # Strong bullet evasion at 130m, 18-tick dodge
    threat = compute_bullet_threat(bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y,
                                   bul_owner, bul_ticks, pos_x[my_idx], pos_y[my_idx], 1, env_id)
    nearest_bul_dist = threat[0]

    if nearest_bul_dist < 130.0 and ev_timer == 0:
        ev_timer = 18
        ev_dir = -ev_dir

    # Evasion mode
    if ev_timer > 0:
        evade_yaw = ev_dir
        # Don't evade into ground
        if altitude < 90.0 and evade_yaw < 0.0 and wp.sin(my_yaw) < 0.0:
            evade_yaw = 1.0
        # Don't evade into ceiling
        if altitude > 540.0 and evade_yaw > 0.0 and wp.sin(my_yaw) > 0.0:
            evade_yaw = -1.0
        sa = stall_avoidance(my_speed, evade_yaw)
        evade_yaw = sa[0]
        throttle = wp.max(0.7, sa[1])
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.30, 380.0)
        opp_evade_timer[env_id] = ev_timer
        opp_evade_dir[env_id] = ev_dir
        return wp.vec3(evade_yaw, throttle, shoot_val)

    # Defensive mode: opponent behind me at close range
    if opponent_behind_me == 1 and distance < 250.0:
        break_dir = 1.0
        if angle_off_nose < 0.0:
            break_dir = -1.0
        perp_yaw = angle_to_opp + HALF_PI * break_dir
        # Slight upward bias
        target_yaw = perp_yaw + 0.15
        yaw_input = yaw_toward(target_yaw, my_yaw, 3.5)
        if my_speed > 100.0:
            throttle = 0.3
        else:
            throttle = 0.5
        sa = stall_avoidance(my_speed, yaw_input)
        yaw_input = sa[0]
        throttle = wp.max(throttle, sa[1])
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.35, 350.0)
        opp_evade_timer[env_id] = ev_timer
        opp_evade_dir[env_id] = ev_dir
        return wp.vec3(yaw_input, throttle, shoot_val)

    # Standard pursuit
    desired_yaw = smart_aim(rel_x, rel_y, opp_speed, opp_yaw, am_behind_opponent, distance_safe, 1.0, bullet_spd)
    yaw_diff = angle_diff(desired_yaw, my_yaw)

    # Altitude management: prefer 250-480m band
    if altitude < 250.0:
        urgency = wp.min((250.0 - altitude) / 100.0, 1.0)
        if wp.sin(my_yaw) < 0.2:
            yaw_diff = yaw_diff + urgency * 0.35
    elif altitude > 480.0:
        urgency = wp.min((altitude - 480.0) / 80.0, 1.0)
        if wp.sin(my_yaw) > -0.2:
            yaw_diff = yaw_diff - urgency * 0.3

    # When slow, dive for speed above 300m
    if my_speed < 90.0 and altitude > 300.0 and wp.sin(my_yaw) > -0.1:
        yaw_diff = yaw_diff - 0.08

    yaw_input = wp.clamp(yaw_diff * 3.0, -1.0, 1.0)

    # Throttle
    if my_speed < 80.0:
        throttle = 1.0
    elif wp.abs(yaw_input) > 0.7:
        throttle = 0.5
    elif distance > 250.0:
        throttle = 1.0
    else:
        throttle = 0.7

    sa = stall_avoidance(my_speed, yaw_input)
    yaw_input = sa[0]
    throttle = wp.max(throttle, sa[1])

    shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.20, 380.0)

    # Write back
    opp_evade_timer[env_id] = ev_timer
    opp_evade_dir[env_id] = ev_dir

    return wp.vec3(yaw_input, throttle, shoot_val)


@wp.func
def act_brawler(
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw_arr: wp.array(dtype=wp.float32),
    speed_arr: wp.array(dtype=wp.float32),
    hp_arr: wp.array(dtype=wp.int32),
    cooldown_arr: wp.array(dtype=wp.int32),
    alive_arr: wp.array(dtype=wp.int32),
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
    opp_phase: wp.array(dtype=wp.int32),
    opp_phase_timer: wp.array(dtype=wp.int32),
    opp_jink_timer: wp.array(dtype=wp.int32),
    opp_jink_dir: wp.array(dtype=wp.float32),
    opp_overshoot_timer: wp.array(dtype=wp.int32),
    env_id: int,
) -> wp.vec3:
    my_idx = env_id * 2 + 1
    opp_idx = env_id * 2

    my_speed = speed_arr[my_idx]
    my_yaw = yaw_arr[my_idx]
    altitude = pos_y[my_idx]

    rel_x = pos_x[opp_idx] - pos_x[my_idx]
    rel_y = pos_y[opp_idx] - pos_y[my_idx]
    opp_speed = speed_arr[opp_idx]
    opp_yaw = yaw_arr[opp_idx]
    distance = wp.sqrt(rel_x * rel_x + rel_y * rel_y)
    distance_safe = wp.max(distance, 1.0)

    angle_to_opp = wp.atan2(rel_y, rel_x)
    angle_off_nose = angle_diff(angle_to_opp, my_yaw)
    angle_opp_to_me = wp.atan2(-rel_y, -rel_x)
    opp_angle_to_me = angle_diff(angle_opp_to_me, opp_yaw)

    am_behind_opponent = 0
    if wp.abs(opp_angle_to_me) > 2.0:
        am_behind_opponent = 1
    opponent_behind_me = 0
    if wp.abs(angle_off_nose) > 2.0:
        opponent_behind_me = 1

    # Closing rate
    my_vx = my_speed * wp.cos(my_yaw)
    my_vy = my_speed * wp.sin(my_yaw)
    opp_vx = opp_speed * wp.cos(opp_yaw)
    opp_vy = opp_speed * wp.sin(opp_yaw)
    closing_rate = (rel_x * (my_vx - opp_vx) + rel_y * (my_vy - opp_vy)) / distance_safe

    rear_cone = cfg_rear_aspect_cone[env_id]
    bullet_spd = cfg_bullet_speed[env_id]
    gun_cd = cooldown_arr[my_idx]
    rear_asp = is_rear_aspect_shot(rel_x, rel_y, opp_yaw, my_yaw, rear_cone)

    # Altitude safety
    alt_safe = altitude_safety(altitude, my_yaw)
    if alt_safe[0] > 0.5:
        return wp.vec3(alt_safe[1], 1.0, 0.0)

    # Load FSM
    phase = opp_phase[env_id]
    phase_timer = opp_phase_timer[env_id]
    jnk_timer = opp_jink_timer[env_id]
    jnk_dir = opp_jink_dir[env_id]
    os_timer = opp_overshoot_timer[env_id]

    # Tick timers
    if phase_timer > 0:
        phase_timer = phase_timer - 1
    if jnk_timer > 0:
        jnk_timer = jnk_timer - 1
    if os_timer > 0:
        os_timer = os_timer - 1

    # Bullet threat for jinks
    threat = compute_bullet_threat(bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y,
                                   bul_owner, bul_ticks, pos_x[my_idx], pos_y[my_idx], 1, env_id)
    nearest_bul_dist = threat[0]

    if nearest_bul_dist < 100.0 and jnk_timer == 0:
        jnk_timer = 8
        jnk_dir = -jnk_dir

    # Phase transitions (with hysteresis)
    if phase_timer == 0:
        new_phase = -1
        if phase == PHASE_CLOSE:
            if distance < 200.0:
                new_phase = PHASE_BRAWL
        elif phase == PHASE_BRAWL:
            if distance > 300.0:
                new_phase = PHASE_CLOSE
            elif opponent_behind_me == 1 and distance < 180.0 and closing_rate > 30.0:
                new_phase = PHASE_OVERSHOOT_BAIT
            elif altitude < 80.0 and my_speed < 60.0:
                new_phase = PHASE_RETREAT
        elif phase == PHASE_OVERSHOOT_BAIT:
            if am_behind_opponent == 1 or (wp.abs(angle_off_nose) < 1.0 and distance < 200.0):
                new_phase = PHASE_OVERSHOOT_PUNISH
            elif distance > 250.0:
                new_phase = PHASE_CLOSE
            elif opponent_behind_me == 0:
                new_phase = PHASE_BRAWL
        elif phase == PHASE_OVERSHOOT_PUNISH:
            if os_timer == 0:
                new_phase = PHASE_BRAWL
        elif phase == PHASE_RETREAT:
            if altitude > 180.0 and my_speed > 80.0:
                new_phase = PHASE_CLOSE

        if new_phase >= 0:
            phase = new_phase
            phase_timer = 20
            if new_phase == PHASE_OVERSHOOT_PUNISH:
                os_timer = 60

    # Execute phase
    yaw_input = 0.0
    throttle = 1.0
    shoot_val = 0.0

    if phase == PHASE_CLOSE:
        desired_yaw = smart_aim(rel_x, rel_y, opp_speed, opp_yaw, am_behind_opponent, distance_safe, 1.0, bullet_spd)
        yaw_input = yaw_toward(desired_yaw, my_yaw, 3.0)
        if altitude < 120.0 and wp.sin(my_yaw) < 0.0:
            yaw_input = wp.clamp(yaw_input + 0.3, -1.0, 1.0)
        throttle = 1.0
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.25, 300.0)
    elif phase == PHASE_BRAWL:
        desired_yaw = smart_aim(rel_x, rel_y, opp_speed, opp_yaw, am_behind_opponent, distance_safe, 0.7, bullet_spd)
        yaw_input = yaw_toward(desired_yaw, my_yaw, 4.0)
        if jnk_timer > 0:
            yaw_input = jnk_dir
        if my_speed > 120.0:
            throttle = 0.0
        elif my_speed < 70.0:
            throttle = 0.6
        else:
            throttle = 0.2
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.30, 250.0)
    elif phase == PHASE_OVERSHOOT_BAIT:
        perp_yaw = angle_to_opp + HALF_PI * jnk_dir
        yaw_input = yaw_toward(perp_yaw, my_yaw, 2.0)
        throttle = 0.0
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.35, 200.0)
    elif phase == PHASE_OVERSHOOT_PUNISH:
        desired_yaw = smart_aim(rel_x, rel_y, opp_speed, opp_yaw, am_behind_opponent, distance_safe, 0.8, bullet_spd)
        yaw_input = yaw_toward(desired_yaw, my_yaw, 4.0)
        throttle = 0.5
        shoot_val = can_shoot(angle_off_nose, distance, gun_cd, rear_asp, 0.35, 300.0)
    elif phase == PHASE_RETREAT:
        climb_yaw = 0.5
        if wp.cos(my_yaw) < 0.0:
            climb_yaw = PI - 0.5
        yaw_input = yaw_toward(climb_yaw, my_yaw, 2.0)
        throttle = 1.0
        shoot_val = 0.0

    # Stall avoidance
    sa = stall_avoidance(my_speed, yaw_input)
    yaw_input = sa[0]
    # For overshoot_bait, min_throttle comes from stall avoidance only
    if phase == PHASE_OVERSHOOT_BAIT:
        throttle = sa[1]
    else:
        throttle = wp.max(throttle, sa[1])

    # Write back FSM
    opp_phase[env_id] = phase
    opp_phase_timer[env_id] = phase_timer
    opp_jink_timer[env_id] = jnk_timer
    opp_jink_dir[env_id] = jnk_dir
    opp_overshoot_timer[env_id] = os_timer

    return wp.vec3(yaw_input, throttle, shoot_val)


# ── Dispatch ─────────────────────────────────────────────────────────────────


@wp.func
def dispatch_opponent(
    opp_type_val: int,
    pos_x: wp.array(dtype=wp.float32),
    pos_y: wp.array(dtype=wp.float32),
    yaw_arr: wp.array(dtype=wp.float32),
    speed_arr: wp.array(dtype=wp.float32),
    hp_arr: wp.array(dtype=wp.int32),
    cooldown_arr: wp.array(dtype=wp.int32),
    alive_arr: wp.array(dtype=wp.int32),
    bul_pos_x: wp.array(dtype=wp.float32),
    bul_pos_y: wp.array(dtype=wp.float32),
    bul_vel_x: wp.array(dtype=wp.float32),
    bul_vel_y: wp.array(dtype=wp.float32),
    bul_owner: wp.array(dtype=wp.int32),
    bul_ticks: wp.array(dtype=wp.int32),
    cfg_gravity: wp.array(dtype=wp.float32),
    cfg_bullet_speed: wp.array(dtype=wp.float32),
    cfg_rear_aspect_cone: wp.array(dtype=wp.float32),
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
    env_id: int,
) -> wp.vec3:
    """Main entry point: dispatch to the correct opponent act function."""
    if opp_type_val == OPP_CHASER:
        return act_chaser(
            pos_x, pos_y, yaw_arr, speed_arr, hp_arr, cooldown_arr, alive_arr,
            bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
            cfg_gravity, cfg_bullet_speed, cfg_rear_aspect_cone,
            opp_evade_timer, opp_evade_dir, opp_yo_yo_timer, opp_yo_yo_phase,
            env_id,
        )
    elif opp_type_val == OPP_DOGFIGHTER:
        return act_dogfighter(
            pos_x, pos_y, yaw_arr, speed_arr, hp_arr, cooldown_arr, alive_arr,
            bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
            cfg_gravity, cfg_bullet_speed, cfg_rear_aspect_cone,
            opp_phase, opp_phase_timer, opp_evade_timer, opp_evade_dir,
            opp_attack_patience, opp_last_distance,
            env_id,
        )
    elif opp_type_val == OPP_ACE:
        return act_ace(
            pos_x, pos_y, yaw_arr, speed_arr, hp_arr, cooldown_arr, alive_arr,
            bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
            cfg_gravity, cfg_bullet_speed, cfg_rear_aspect_cone,
            opp_evade_timer, opp_evade_dir,
            env_id,
        )
    elif opp_type_val == OPP_BRAWLER:
        return act_brawler(
            pos_x, pos_y, yaw_arr, speed_arr, hp_arr, cooldown_arr, alive_arr,
            bul_pos_x, bul_pos_y, bul_vel_x, bul_vel_y, bul_owner, bul_ticks,
            cfg_gravity, cfg_bullet_speed, cfg_rear_aspect_cone,
            opp_phase, opp_phase_timer, opp_jink_timer, opp_jink_dir,
            opp_overshoot_timer,
            env_id,
        )
    else:
        return act_do_nothing()
