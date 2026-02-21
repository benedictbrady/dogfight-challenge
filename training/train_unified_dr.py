"""Config-aware domain-randomized unified training pipeline.

Three-phase training with domain randomization and config observations:
  Phase 1 — Curriculum + DR:  All envs use scripted opponents, DR widens progressively
  Phase 2 — Transition:       Ramp scripted_fraction from 1.0 → target, seed self-play pool
  Phase 3 — Self-Play + DR:   Neural self-play + scripted anchoring + full DR

The model receives 59-float observations (46 base + 13 config parameters), enabling it
to adapt its strategy based on the active physics parameters.

Usage:
    # Config-driven (recommended)
    python train_unified_dr.py --config experiments/configs/unified_dr_v1.json

    # Quick local smoke test
    python train_unified_dr.py --n-envs 4 --n-steps 64 \
        --curriculum-updates 3 --transition-updates 2 --selfplay-updates 3 \
        --hidden 64 --n-blocks 1

    # Full production run (via Modal)
    modal run --detach modal_train.py --unified-dr
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import ActorCritic, OBS_SIZE, CONFIG_OBS_SIZE, ACTION_SIZE
from naming import make_run_name
from opponent_pool import OpponentPool
from slack import slack_notify
from utils import save_checkpoint, export_onnx_checkpoint, compute_gae, EVAL_OPPONENTS

try:
    from dogfight_pyenv import SelfPlayBatchEnv, BatchEnv
except ImportError:
    SelfPlayBatchEnv = None
    BatchEnv = None


# ---------------------------------------------------------------------------
# Domain Randomization Ranges
# ---------------------------------------------------------------------------

DR_NONE = {
    "gravity": (80.0, 80.0),
    "drag_coeff": (0.9, 0.9),
    "turn_bleed_coeff": (0.25, 0.25),
    "max_speed": (250.0, 250.0),
    "min_speed": (20.0, 20.0),
    "max_thrust": (180.0, 180.0),
    "bullet_speed": (400.0, 400.0),
    "gun_cooldown_ticks": (90, 90),
    "bullet_lifetime_ticks": (60, 60),
    "max_hp": (5, 5),
    "max_turn_rate": (4.0, 4.0),
    "min_turn_rate": (0.8, 0.8),
    "rear_aspect_cone": (0.785, 0.785),
}

DR_NARROW = {
    "gravity": (60.0, 100.0),
    "drag_coeff": (0.7, 1.1),
    "turn_bleed_coeff": (0.15, 0.35),
    "max_speed": (220.0, 280.0),
    "min_speed": (15.0, 30.0),
    "max_thrust": (140.0, 220.0),
    "bullet_speed": (350.0, 450.0),
    "gun_cooldown_ticks": (70, 110),
    "bullet_lifetime_ticks": (45, 75),
    "max_hp": (4, 7),
    "max_turn_rate": (3.2, 4.8),
    "min_turn_rate": (0.6, 1.0),
    "rear_aspect_cone": (0.5, 1.0),
}

DR_FULL = {
    "gravity": (40.0, 120.0),
    "drag_coeff": (0.5, 1.3),
    "turn_bleed_coeff": (0.05, 0.45),
    "max_speed": (180.0, 320.0),
    "min_speed": (10.0, 40.0),
    "max_thrust": (100.0, 260.0),
    "bullet_speed": (300.0, 500.0),
    "gun_cooldown_ticks": (45, 135),
    "bullet_lifetime_ticks": (30, 90),
    "max_hp": (3, 8),
    "max_turn_rate": (2.5, 5.5),
    "min_turn_rate": (0.4, 1.2),
    "rear_aspect_cone": (0.3, 1.2),
}

# Named evaluation regimes
EVAL_REGIMES = {
    "default": {},
    "high_gravity": {"gravity": (120.0, 120.0)},
    "low_gravity": {"gravity": (40.0, 40.0)},
    "glass_cannon": {"max_hp": (3, 3), "bullet_lifetime_ticks": (75, 75)},
    "tank_fight": {"max_hp": (8, 8), "gun_cooldown_ticks": (60, 60)},
    "knife_fight": {
        "bullet_speed": (300.0, 300.0),
        "bullet_lifetime_ticks": (30, 30),
        "max_turn_rate": (5.5, 5.5),
    },
    "energy_fighter": {
        "turn_bleed_coeff": (0.45, 0.45),
        "drag_coeff": (1.3, 1.3),
        "max_thrust": (100.0, 100.0),
    },
    "sniper": {
        "bullet_speed": (500.0, 500.0),
        "bullet_lifetime_ticks": (90, 90),
        "rear_aspect_cone": (0.3, 0.3),
    },
}


EFFECTIVE_OBS_SIZE = OBS_SIZE + CONFIG_OBS_SIZE  # 59


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_curriculum_pool(update: int, schedule: dict[int, list[str]]) -> list[str]:
    """Return opponent pool based on update count and curriculum schedule."""
    pool = ["do_nothing"]
    for threshold in sorted(int(k) for k in schedule):
        if update >= threshold:
            pool = schedule[str(threshold)]
    return pool


def get_dr_ranges(update: int, dr_none_until: int, dr_narrow_until: int) -> tuple[dict, str]:
    """Return (DR ranges, mode name) based on 3-phase schedule: none → narrow → full."""
    if update < dr_none_until:
        return DR_NONE, "none"
    if update < dr_narrow_until:
        return DR_NARROW, "narrow"
    return DR_FULL, "full"


def _default_config_obs() -> np.ndarray:
    """Return the 13-float normalized config obs for default physics parameters."""
    return np.array([
        80.0 / 200.0,     # gravity / norm
        0.9 / 2.0,        # drag_coeff
        0.25 / 1.0,       # turn_bleed_coeff
        250.0 / 500.0,    # max_speed
        20.0 / 100.0,     # min_speed
        180.0 / 400.0,    # max_thrust
        400.0 / 800.0,    # bullet_speed
        90.0 / 240.0,     # gun_cooldown_ticks
        60.0 / 180.0,     # bullet_lifetime_ticks
        5.0 / 10.0,       # max_hp
        4.0 / 8.0,        # max_turn_rate
        0.8 / 4.0,        # min_turn_rate
        0.785 / 3.14159,  # rear_aspect_cone
    ], dtype=np.float32)


def scripted_eval_dr(model, opponent: str, n_matches: int, action_repeat: int,
                     device: torch.device, regime_params: dict = None) -> float:
    """Quick eval against a scripted opponent with config-aware model. Returns win rate."""
    if BatchEnv is None:
        return 0.0  # Rust pyenv not available (e.g., --gpu-sim without build)
    # BatchEnv doesn't support config obs — append default config manually
    config_obs = _default_config_obs().reshape(1, -1)  # [1, 13]
    wins = 0
    for i in range(n_matches):
        env = BatchEnv(1, [opponent], True, seed=i * 7 + 1, action_repeat=1)
        if regime_params:
            env.set_domain_randomization(regime_params)
        obs_np = env.reset()
        done = False
        while not done:
            obs_with_config = np.concatenate([obs_np, config_obs], axis=1)
            obs_t = torch.tensor(obs_with_config, dtype=torch.float32, device=device)
            action = model.get_deterministic_action(obs_t)
            for _ in range(action_repeat):
                obs_np, rewards, dones, infos = env.step(
                    np.array(action, dtype=np.float32))
                done = dones[0]
                if done:
                    break
        if infos[0].get("outcome") == "Player0Win":
            wins += 1
    return wins / n_matches if n_matches > 0 else 0.0


def regime_eval(model, n_matches: int, action_repeat: int, device: torch.device) -> dict:
    """Evaluate config-aware model across all named regimes vs ace+brawler. Returns {regime: {opp: wr}}."""
    results = {}
    for regime_name, regime_params in EVAL_REGIMES.items():
        regime_wins = {}
        for opp in ["ace", "brawler"]:
            wr = scripted_eval_dr(model, opp, n_matches, action_repeat, device, regime_params or None)
            regime_wins[opp] = wr
        results[regime_name] = regime_wins
    return results


# ---------------------------------------------------------------------------
# PPO update (identical to train_unified.py)
# ---------------------------------------------------------------------------

def ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, rew_buf, done_buf,
               val_buf, next_obs_t, next_done_t, args, device, scaler=None, target_kl=None):
    """Run PPO update with optional fp16 mixed precision. Returns dict of metrics."""
    use_amp = scaler is not None
    kl_exceeded = False
    epochs_run = 0

    with torch.inference_mode():
        if use_amp:
            with torch.amp.autocast("cuda"):
                next_val = model.get_value(next_obs_t)
        else:
            next_val = model.get_value(next_obs_t)

    advantages, returns = compute_gae(
        rew_buf, val_buf, done_buf, next_done_t, next_val,
        args.n_steps, args.n_envs, args.gamma, args.gae_lambda, device,
    )

    b_obs = obs_buf.reshape(-1, EFFECTIVE_OBS_SIZE)
    b_act = act_buf.reshape(-1, ACTION_SIZE)
    b_logp = logp_buf.reshape(-1)
    b_adv = advantages.reshape(-1)
    b_ret = returns.reshape(-1)
    b_val = val_buf.reshape(-1)

    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    batch_size = args.n_envs * args.n_steps

    clipfracs = []
    for epoch in range(args.n_epochs):
        b_inds = torch.randperm(batch_size, device="cpu")
        for start in range(0, batch_size, args.minibatch_size):
            mb = b_inds[start:start + args.minibatch_size]

            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, new_logp, entropy, new_val = model.get_action_and_value(
                        b_obs[mb], b_act[mb]
                    )

                    logratio = new_logp - b_logp[mb]
                    logratio = logratio.clamp(-20.0, 20.0)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > args.clip_eps).float().mean().item()
                        )

                    if target_kl is not None and approx_kl > target_kl:
                        kl_exceeded = True
                        break

                    mb_adv = b_adv[mb]

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1 - args.clip_eps, 1 + args.clip_eps
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if args.clip_vloss:
                        v_clipped = b_val[mb] + torch.clamp(
                            new_val - b_val[mb], -args.clip_eps, args.clip_eps
                        )
                        v_loss1 = (new_val - b_ret[mb]) ** 2
                        v_loss2 = (v_clipped - b_ret[mb]) ** 2
                        v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                    else:
                        v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()

                    ent_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, new_logp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb], b_act[mb]
                )

                logratio = new_logp - b_logp[mb]
                logratio = logratio.clamp(-20.0, 20.0)
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_eps).float().mean().item()
                    )

                if target_kl is not None and approx_kl > target_kl:
                    kl_exceeded = True
                    break

                mb_adv = b_adv[mb]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_eps, 1 + args.clip_eps
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if args.clip_vloss:
                    v_clipped = b_val[mb] + torch.clamp(
                        new_val - b_val[mb], -args.clip_eps, args.clip_eps
                    )
                    v_loss1 = (new_val - b_ret[mb]) ** 2
                    v_loss2 = (v_clipped - b_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        epochs_run += 1
        if kl_exceeded:
            break

    with torch.no_grad():
        var_y = b_ret.var().item()
        explained_var = float("nan") if var_y == 0 else 1 - (b_ret - b_val).var().item() / var_y

    return {
        "pg_loss": pg_loss.item(),
        "v_loss": v_loss.item(),
        "ent_loss": ent_loss.item(),
        "approx_kl": approx_kl,
        "clipfrac": np.mean(clipfracs),
        "explained_var": explained_var,
        "kl_exceeded": kl_exceeded,
        "epochs_run": epochs_run,
    }


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(model, opp_model, vec_env, obs_buf, act_buf, logp_buf,
                    rew_buf, done_buf, val_buf, next_obs_p0, next_obs_p1,
                    next_done_t, ep_state, args, device, use_cuda,
                    _pin_obs_p0=None, _pin_obs_p1=None, _pin_rew=None, _pin_done=None):
    """Collect one rollout of n_steps across all envs with config-aware observations."""
    t_infer_total = 0.0
    t_env_total = 0.0
    scripted_mask = np.asarray(vec_env.scripted_mask, dtype=np.bool_)
    neural_mask = ~scripted_mask

    use_amp = getattr(args, '_use_amp', False)
    gpu_sim = getattr(args, 'gpu_sim', False) and use_cuda

    # GPU sim zero-copy path: obs/rewards stay as torch tensors on CUDA
    if gpu_sim:
        _zero_opp = torch.zeros(args.n_envs, ACTION_SIZE, dtype=torch.float32, device=device)

    for step in range(args.n_steps):
        # --- Store obs into rollout buffer ---
        if gpu_sim:
            obs_buf[step].copy_(next_obs_p0)  # both already on CUDA
        elif use_cuda:
            _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
            obs_buf[step].copy_(_pin_obs_p0, non_blocking=True)
        else:
            obs_buf[step].copy_(torch.from_numpy(next_obs_p0))
        done_buf[step] = next_done_t

        # --- Model inference ---
        t0 = time.time()
        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    action, logp, _, value = model.get_action_and_value(obs_buf[step])
            else:
                action, logp, _, value = model.get_action_and_value(obs_buf[step])

            if opp_model is not None and neural_mask.any():
                if gpu_sim:
                    obs_p1_gpu = next_obs_p1  # already on CUDA
                elif use_cuda:
                    _pin_obs_p1.copy_(torch.from_numpy(next_obs_p1))
                    obs_p1_gpu = _pin_obs_p1.to(device, non_blocking=True)
                else:
                    obs_p1_gpu = torch.from_numpy(next_obs_p1)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        opp_action_raw = opp_model.get_deterministic_action(obs_p1_gpu)
                else:
                    opp_action_raw = opp_model.get_deterministic_action(obs_p1_gpu)
            else:
                if gpu_sim:
                    opp_action_raw = _zero_opp
                else:
                    opp_action_raw = np.zeros((args.n_envs, ACTION_SIZE), dtype=np.float32)

        act_buf[step] = action
        logp_buf[step] = logp
        val_buf[step] = value

        clamped = action.detach().clone()
        clamped[:, 0].clamp_(-1.0, 1.0)
        clamped[:, 1].clamp_(0.0, 1.0)
        t_infer_total += time.time() - t0

        # --- Env step ---
        t0 = time.time()
        if gpu_sim:
            # Zero-copy: actions stay on GPU, obs/rewards return as GPU tensors
            opp_actions_t = opp_action_raw if isinstance(opp_action_raw, torch.Tensor) else _zero_opp
            next_obs_p0, next_obs_p1, rew_p0_t, _, dones_np, infos = vec_env.step_torch(
                clamped, opp_actions_t,
            )
            rew_buf[step] = rew_p0_t  # already on CUDA
            next_done = torch.from_numpy(dones_np.astype(np.float32)).to(device)
            next_done_t = next_done
            rewards_p0_np = rew_p0_t.cpu().numpy()
        else:
            clamped_p0 = np.ascontiguousarray(clamped.cpu().numpy(), dtype=np.float32)
            opp_actions = np.ascontiguousarray(opp_action_raw, dtype=np.float32)
            next_obs_p0, next_obs_p1, rewards_p0, _, dones, infos = vec_env.step(
                clamped_p0, opp_actions,
            )
            next_obs_p0 = np.asarray(next_obs_p0, dtype=np.float32)
            next_obs_p1 = np.asarray(next_obs_p1, dtype=np.float32)
            rewards_p0_np = np.asarray(rewards_p0, dtype=np.float32)
            dones_np = np.asarray(dones, dtype=np.bool_)
            next_done = dones_np.astype(np.float32)
            if use_cuda:
                _pin_rew.copy_(torch.from_numpy(rewards_p0_np))
                rew_buf[step].copy_(_pin_rew, non_blocking=True)
                _pin_done.copy_(torch.from_numpy(next_done))
                next_done_t = _pin_done.to(device, non_blocking=True)
            else:
                rew_buf[step] = torch.from_numpy(rewards_p0_np.copy())
                next_done_t = torch.from_numpy(next_done.copy())
        t_env_total += time.time() - t0

        ep_state["return_running"] += rewards_p0_np
        ep_state["len_running"] += 1
        for i, d in enumerate(dones_np):
            if d:
                ep_state["returns"].append(ep_state["return_running"][i])
                ep_state["lengths"].append(ep_state["len_running"][i])
                outcome = infos[i].get("outcome", "Draw")
                ep_state["wins"].append(1.0 if outcome == "Player0Win" else 0.0)
                ep_state["is_scripted"].append(bool(scripted_mask[i]))
                ep_state["return_running"][i] = 0.0
                ep_state["len_running"][i] = 0

        if dones_np.any():
            scripted_mask = np.asarray(vec_env.scripted_mask, dtype=np.bool_)
            neural_mask = ~scripted_mask

    return next_obs_p0, next_obs_p1, next_done_t, t_infer_total, t_env_total


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_unified_dr(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")
    print(f"Model: hidden={args.hidden}, n_blocks={args.n_blocks}, obs_dim={EFFECTIVE_OBS_SIZE}")
    print(f"Action repeat: {args.action_repeat} (episode ~{10800 // args.action_repeat} RL steps)")
    print(f"Domain randomization: none until {args.dr_none_until}, narrow until {args.dr_narrow_until}, then full")

    total_updates_planned = args.curriculum_updates + args.transition_updates + args.selfplay_updates
    print(f"Pipeline: curriculum={args.curriculum_updates} → transition={args.transition_updates} → selfplay={args.selfplay_updates} ({total_updates_planned} total)")

    run_name = args.run_name or make_run_name("unified_dr")
    log_dir = Path(args.log_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir)
    pool_dir = Path(args.pool_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # Initialize model with config-aware obs dim
    model = ActorCritic(obs_dim=EFFECTIVE_OBS_SIZE, hidden=args.hidden, n_blocks=args.n_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # fp16 mixed precision with optional warmup
    scaler = None
    fp16_warmup_remaining = getattr(args, 'fp16_warmup', 0) if (use_cuda and args.fp16) else 0
    if use_cuda and args.fp16:
        scaler = torch.amp.GradScaler("cuda")
        if fp16_warmup_remaining > 0:
            args._use_amp = False
            print(f"  fp16 mixed precision enabled (warmup: {fp16_warmup_remaining} updates in fp32 first)")
        else:
            args._use_amp = True
            print(f"  fp16 mixed precision enabled")
    else:
        args._use_amp = False

    global_step = 0
    total_updates = 0

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        total_updates = ckpt.get("total_updates", 0)
        # Skip fp16 warmup on resume — model is already warmed up
        fp16_warmup_remaining = 0
        if use_cuda and args.fp16:
            args._use_amp = True
        print(f"  Resumed at global_step={global_step}, total_updates={total_updates}")

    # torch.compile
    model_unwrapped = model
    if use_cuda and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception:
            print("  torch.compile not available")

    # Initialize self-play env WITH config obs
    if args.gpu_sim:
        from gpu_sim import GpuSelfPlayBatchEnv
        vec_env = GpuSelfPlayBatchEnv(
            n_envs=args.n_envs,
            randomize_spawns=args.randomize,
            action_repeat=args.action_repeat,
            include_config_obs=True,
        )
        print("  GPU sim enabled (NVIDIA Warp)")
    else:
        if SelfPlayBatchEnv is None:
            raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")
        vec_env = SelfPlayBatchEnv(
            n_envs=args.n_envs,
            randomize_spawns=args.randomize,
            action_repeat=args.action_repeat,
            include_config_obs=True,
        )

    # Set reward weights
    vec_env.set_rewards(
        damage_dealt=args.w_damage_dealt,
        damage_taken=args.w_damage_taken,
        win=args.w_win,
        lose=args.w_lose,
        approach=args.w_approach,
        alive=args.w_alive,
        proximity=args.w_proximity,
        facing=args.w_facing,
    )

    # Set initial DR ranges
    dr_ranges, current_dr_mode = get_dr_ranges(0, args.dr_none_until, args.dr_narrow_until)
    vec_env.set_domain_randomization(dr_ranges)
    print(f"  Initial DR mode: {current_dr_mode}")

    # Allocate PPO rollout buffers — sized for 59-float obs
    obs_buf = torch.zeros((args.n_steps, args.n_envs, EFFECTIVE_OBS_SIZE), dtype=torch.float32, device=device)
    act_buf = torch.zeros((args.n_steps, args.n_envs, ACTION_SIZE), dtype=torch.float32, device=device)
    logp_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    done_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    val_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)

    # Pinned CPU staging buffers
    _pin_obs_p0 = _pin_obs_p1 = _pin_rew = _pin_done = None
    if use_cuda:
        _pin_obs_p0 = torch.zeros((args.n_envs, EFFECTIVE_OBS_SIZE), dtype=torch.float32, pin_memory=True)
        _pin_obs_p1 = torch.zeros((args.n_envs, EFFECTIVE_OBS_SIZE), dtype=torch.float32, pin_memory=True)
        _pin_rew = torch.zeros(args.n_envs, dtype=torch.float32, pin_memory=True)
        _pin_done = torch.zeros(args.n_envs, dtype=torch.float32, pin_memory=True)

    ep_state = {
        "returns": [],
        "lengths": [],
        "wins": [],
        "is_scripted": [],
        "return_running": np.zeros(args.n_envs),
        "len_running": np.zeros(args.n_envs, dtype=np.int32),
    }

    t_start = time.time()

    slack_notify(
        f":rocket: *{run_name}* started (DR) — {args.hidden}h/{args.n_blocks}b, "
        f"obs_dim={EFFECTIVE_OBS_SIZE}, {args.n_envs} envs, {total_updates_planned} updates "
        f"(curriculum {args.curriculum_updates} → transition {args.transition_updates} → selfplay {args.selfplay_updates})"
    )

    # =====================================================================
    # PHASE 1: Curriculum + DR (scripted_fraction=1.0)
    # =====================================================================
    skip_curriculum = total_updates >= args.curriculum_updates
    skip_transition = total_updates >= args.curriculum_updates + args.transition_updates

    if skip_curriculum:
        print(f"\n>>> Skipping Phase 1 (curriculum) — already at update {total_updates}")
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Curriculum + DR ({args.curriculum_updates} updates, scripted_fraction=1.0)")
        print(f"{'='*60}")

    vec_env.set_scripted_fraction(1.0)
    vec_env.set_scripted_pool(["do_nothing"])

    if args.gpu_sim and use_cuda:
        next_obs_p0, next_obs_p1 = vec_env.reset_torch()  # torch on CUDA
    else:
        next_obs_p0, next_obs_p1 = vec_env.reset()
        next_obs_p0 = np.asarray(next_obs_p0, dtype=np.float32)
        next_obs_p1 = np.asarray(next_obs_p1, dtype=np.float32)
    next_done_t = torch.zeros(args.n_envs, dtype=torch.float32, device=device)

    current_pool = ["do_nothing"]
    curriculum_eval = {}

    if not skip_curriculum:
        for update in range(args.curriculum_updates):
            # Curriculum schedule
            new_pool = get_curriculum_pool(total_updates, args.curriculum_schedule)
            if new_pool != current_pool:
                current_pool = new_pool
                vec_env.set_scripted_pool(current_pool)
                print(f"\n>>> Curriculum shift at update {total_updates}: pool = {current_pool}")

            # DR schedule: none → narrow → full
            new_dr_ranges, new_dr_mode = get_dr_ranges(total_updates, args.dr_none_until, args.dr_narrow_until)
            if new_dr_mode != current_dr_mode:
                vec_env.set_domain_randomization(new_dr_ranges)
                print(f"\n>>> DR changed: {current_dr_mode} → {new_dr_mode} at update {total_updates}")
                current_dr_mode = new_dr_mode

            # LR annealing
            if args.anneal_lr:
                frac = 1.0 - total_updates / total_updates_planned
                lr = args.lr * max(frac, 0.1)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            next_obs_p0, next_obs_p1, next_done_t, t_infer, t_env = collect_rollout(
                model, None, vec_env, obs_buf, act_buf, logp_buf, rew_buf, done_buf,
                val_buf, next_obs_p0, next_obs_p1, next_done_t, ep_state, args, device,
                use_cuda, _pin_obs_p0, _pin_obs_p1, _pin_rew, _pin_done,
            )

            if args.gpu_sim and use_cuda:
                next_obs_t = next_obs_p0  # already torch on CUDA
            elif use_cuda:
                _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
                next_obs_t = _pin_obs_p0.to(device, non_blocking=True)
            else:
                next_obs_t = torch.from_numpy(next_obs_p0)

            t_ppo = time.time()
            _scaler = scaler if args._use_amp else None
            _target_kl = getattr(args, 'target_kl', None)
            metrics = ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, rew_buf,
                                 done_buf, val_buf, next_obs_t, next_done_t, args, device,
                                 scaler=_scaler, target_kl=_target_kl)
            ppo_time = time.time() - t_ppo

            # fp16 warmup: transition from fp32 to fp16 after warmup updates
            if fp16_warmup_remaining > 0:
                fp16_warmup_remaining -= 1
                if fp16_warmup_remaining == 0:
                    args._use_amp = True
                    print(f"  fp16 warmup complete — enabling mixed precision")

            total_updates += 1
            global_step += args.n_envs * args.n_steps

            # Logging
            writer.add_scalar("losses/policy_loss", metrics["pg_loss"], global_step)
            writer.add_scalar("losses/value_loss", metrics["v_loss"], global_step)
            writer.add_scalar("losses/entropy", metrics["ent_loss"], global_step)
            writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
            writer.add_scalar("losses/clipfrac", metrics["clipfrac"], global_step)
            writer.add_scalar("charts/explained_variance", metrics["explained_var"], global_step)
            writer.add_scalar("charts/log_std_0", model.log_std[0].item(), global_step)
            writer.add_scalar("charts/log_std_1", model.log_std[1].item(), global_step)
            writer.add_scalar("phase/phase", 1, global_step)
            writer.add_scalar("phase/scripted_fraction", 1.0, global_step)
            writer.add_scalar("dr/mode", {"none": 0, "narrow": 1, "full": 2}.get(current_dr_mode, 2), global_step)

            if ep_state["returns"]:
                writer.add_scalar("charts/ep_return_mean", np.mean(ep_state["returns"][-100:]), global_step)
                writer.add_scalar("charts/ep_length_mean", np.mean(ep_state["lengths"][-100:]), global_step)
                writer.add_scalar("charts/win_rate", np.mean(ep_state["wins"][-100:]), global_step)

            if (update + 1) % 5 == 0 or update == 0:
                wr = np.mean(ep_state["wins"][-100:]) if ep_state["wins"] else 0.0
                ret = np.mean(ep_state["returns"][-100:]) if ep_state["returns"] else 0.0
                print(
                    f"  [P1 curriculum+DR] {total_updates}/{total_updates_planned} | "
                    f"pool {current_pool} | DR {current_dr_mode} | "
                    f"return {ret:.2f} | wr {wr:.2%} | "
                    f"pg {metrics['pg_loss']:.4f} | vl {metrics['v_loss']:.4f} | "
                    f"std [{model.log_std[0].item():.2f},{model.log_std[1].item():.2f}] | "
                    f"infer {t_infer:.2f}s sim {t_env:.2f}s ppo {ppo_time:.2f}s"
                )
                ep_state["returns"].clear()
                ep_state["lengths"].clear()
                ep_state["wins"].clear()
                ep_state["is_scripted"].clear()

            if (update + 1) % 50 == 0:
                elapsed = time.time() - t_start
                slack_notify(
                    f":books: *{run_name}* [P1 curriculum+DR] {total_updates}/{total_updates_planned} | "
                    f"pool {current_pool} | DR {current_dr_mode} | {elapsed / 60:.0f}m"
                )

            if (update + 1) % args.save_every == 0:
                save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, f"step_{global_step}")
                export_onnx_checkpoint(model_unwrapped, ckpt_dir, f"step_{global_step}", args.hidden, args.n_blocks)

        save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, "curriculum_final")
        export_onnx_checkpoint(model_unwrapped, ckpt_dir, "curriculum_final", args.hidden, args.n_blocks)

        # Quick eval at end of curriculum (default regime)
        model.eval()
        for opp in ["ace", "brawler"]:
            wr = scripted_eval_dr(model, opp, 20, args.action_repeat, device)
            curriculum_eval[opp] = wr
            print(f"  [curriculum eval] vs {opp}: {wr:.0%}")
        model.train()

        slack_notify(
            f":mortar_board: *{run_name}* curriculum done — "
            + " | ".join(f"{k}: {v:.0%}" for k, v in curriculum_eval.items())
        )

    # =====================================================================
    # PHASE 2: Transition (ramp scripted_fraction from 1.0 → target)
    # =====================================================================

    # Update DR based on schedule
    dr_ranges, current_dr_mode = get_dr_ranges(total_updates, args.dr_none_until, args.dr_narrow_until)
    vec_env.set_domain_randomization(dr_ranges)

    # Initialize opponent pool (needed by both transition and self-play)
    pool = OpponentPool(
        str(pool_dir), device,
        hidden=args.hidden, n_blocks=args.n_blocks,
        max_size=args.pool_max_size,
        obs_dim=EFFECTIVE_OBS_SIZE,
    )
    if pool.size == 0:
        pool.add_checkpoint(model_unwrapped, "curriculum_final", total_updates)
        print(f"Seeded pool with curriculum_final checkpoint")

    vec_env.set_scripted_pool(args.scripted_pool)

    learner_elo = 1000.0
    last_snapshot_elo = learner_elo
    elo_milestones_hit = set()
    best_scripted_wr = max(curriculum_eval.values()) if curriculum_eval else 0.0
    last_scripted_wr = best_scripted_wr

    if skip_transition:
        print(f"\n>>> Skipping Phase 2 (transition) — already at update {total_updates}")
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Transition ({args.transition_updates} updates, "
              f"fraction {args.transition_start_fraction} → {args.selfplay_scripted_fraction})")
        print(f"{'='*60}")
        print(f"  DR mode at transition start: {current_dr_mode}")

        ep_state["returns"].clear()
        ep_state["lengths"].clear()
        ep_state["wins"].clear()
        ep_state["is_scripted"].clear()

        # Reset exploration
        if args.transition_reset_std is not None:
            with torch.no_grad():
                model.log_std.fill_(args.transition_reset_std)
            print(f"Reset log_std to {args.transition_reset_std}")

    for update in range(0 if skip_transition else args.transition_updates):
        progress = update / max(args.transition_updates - 1, 1)
        fraction = args.transition_start_fraction + progress * (args.selfplay_scripted_fraction - args.transition_start_fraction)
        vec_env.set_scripted_fraction(fraction)

        # DR schedule continues through transition
        new_dr_ranges, new_dr_mode = get_dr_ranges(total_updates, args.dr_none_until, args.dr_narrow_until)
        if new_dr_mode != current_dr_mode:
            vec_env.set_domain_randomization(new_dr_ranges)
            print(f"\n>>> DR changed: {current_dr_mode} → {new_dr_mode} at update {total_updates}")
            current_dr_mode = new_dr_mode

        opp_entry = pool.sample_opponent(method=args.sampling)
        opp_model = pool.load_opponent(opp_entry)

        if args.anneal_lr:
            frac = 1.0 - total_updates / total_updates_planned
            lr = args.lr * max(frac, 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        next_obs_p0, next_obs_p1, next_done_t, t_infer, t_env = collect_rollout(
            model, opp_model, vec_env, obs_buf, act_buf, logp_buf, rew_buf, done_buf,
            val_buf, next_obs_p0, next_obs_p1, next_done_t, ep_state, args, device,
            use_cuda, _pin_obs_p0, _pin_obs_p1, _pin_rew, _pin_done,
        )

        if args.gpu_sim and use_cuda:
            next_obs_t = next_obs_p0  # already torch on CUDA
        elif use_cuda:
            _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
            next_obs_t = _pin_obs_p0.to(device, non_blocking=True)
        else:
            next_obs_t = torch.from_numpy(next_obs_p0)

        t_ppo = time.time()
        _scaler = scaler if args._use_amp else None
        _target_kl = getattr(args, 'target_kl', None)
        metrics = ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, rew_buf,
                             done_buf, val_buf, next_obs_t, next_done_t, args, device,
                             scaler=_scaler, target_kl=_target_kl)
        ppo_time = time.time() - t_ppo

        total_updates += 1
        global_step += args.n_envs * args.n_steps

        writer.add_scalar("losses/policy_loss", metrics["pg_loss"], global_step)
        writer.add_scalar("losses/value_loss", metrics["v_loss"], global_step)
        writer.add_scalar("losses/entropy", metrics["ent_loss"], global_step)
        writer.add_scalar("charts/log_std_0", model.log_std[0].item(), global_step)
        writer.add_scalar("charts/log_std_1", model.log_std[1].item(), global_step)
        writer.add_scalar("phase/phase", 2, global_step)
        writer.add_scalar("phase/scripted_fraction", fraction, global_step)
        writer.add_scalar("selfplay/pool_size", pool.size, global_step)

        if ep_state["returns"]:
            writer.add_scalar("charts/ep_return_mean", np.mean(ep_state["returns"][-100:]), global_step)
            writer.add_scalar("charts/win_rate", np.mean(ep_state["wins"][-100:]), global_step)

        if (update + 1) % 5 == 0 or update == 0:
            wr = np.mean(ep_state["wins"][-100:]) if ep_state["wins"] else 0.0
            ret = np.mean(ep_state["returns"][-100:]) if ep_state["returns"] else 0.0
            n_s = vec_env.n_scripted
            print(
                f"  [P2 transition+DR] {total_updates}/{total_updates_planned} | "
                f"frac {fraction:.2f} ({n_s}/{args.n_envs} scripted) | "
                f"vs {opp_entry.name} | "
                f"return {ret:.2f} | wr {wr:.2%} | "
                f"pg {metrics['pg_loss']:.4f} | "
                f"std [{model.log_std[0].item():.2f},{model.log_std[1].item():.2f}] | "
                f"ppo {ppo_time:.2f}s"
            )
            ep_state["returns"].clear()
            ep_state["lengths"].clear()
            ep_state["wins"].clear()
            ep_state["is_scripted"].clear()

        if pool.should_snapshot(update + 1, args.pool_snapshot_every, 0):
            name = f"trans_{total_updates:04d}"
            pool.add_checkpoint(model_unwrapped, name, total_updates)

        if (update + 1) % args.save_every == 0:
            save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, f"step_{global_step}")
            export_onnx_checkpoint(model_unwrapped, ckpt_dir, f"step_{global_step}", args.hidden, args.n_blocks)

    save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, "transition_final")
    export_onnx_checkpoint(model_unwrapped, ckpt_dir, "transition_final", args.hidden, args.n_blocks)

    slack_notify(
        f":bridge_at_night: *{run_name}* transition done — entering self-play+DR "
        f"(pool {pool.size}, frac {args.selfplay_scripted_fraction})"
    )

    # =====================================================================
    # PHASE 3: Self-Play + DR
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 3: Self-Play + DR ({args.selfplay_updates} updates, "
          f"scripted_fraction={args.selfplay_scripted_fraction})")
    print(f"{'='*60}")

    vec_env.set_scripted_fraction(args.selfplay_scripted_fraction)

    ep_state["returns"].clear()
    ep_state["lengths"].clear()
    ep_state["wins"].clear()
    ep_state["is_scripted"].clear()

    for update in range(args.selfplay_updates):
        opp_entry = pool.sample_opponent(method=args.sampling)
        opp_model = pool.load_opponent(opp_entry)

        if args.anneal_lr:
            frac = 1.0 - total_updates / total_updates_planned
            lr = args.lr * max(frac, 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        next_obs_p0, next_obs_p1, next_done_t, t_infer, t_env = collect_rollout(
            model, opp_model, vec_env, obs_buf, act_buf, logp_buf, rew_buf, done_buf,
            val_buf, next_obs_p0, next_obs_p1, next_done_t, ep_state, args, device,
            use_cuda, _pin_obs_p0, _pin_obs_p1, _pin_rew, _pin_done,
        )

        if args.gpu_sim and use_cuda:
            next_obs_t = next_obs_p0  # already torch on CUDA
        elif use_cuda:
            _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
            next_obs_t = _pin_obs_p0.to(device, non_blocking=True)
        else:
            next_obs_t = torch.from_numpy(next_obs_p0)

        t_ppo = time.time()
        _scaler = scaler if args._use_amp else None
        _target_kl = getattr(args, 'target_kl', None)
        metrics = ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, rew_buf,
                             done_buf, val_buf, next_obs_t, next_done_t, args, device,
                             scaler=_scaler, target_kl=_target_kl)
        ppo_time = time.time() - t_ppo

        total_updates += 1
        global_step += args.n_envs * args.n_steps

        # ELO tracking
        for win_val, is_scr in zip(ep_state["wins"], ep_state["is_scripted"]):
            if is_scr:
                continue
            won = win_val > 0.5
            drawn = abs(win_val) < 0.01
            learner_elo = pool.update_learner_elo(learner_elo, opp_entry, won=won, drawn=drawn)
            pool.update_elo(opp_entry, learner_elo, won=won, drawn=drawn)

        # Logging
        writer.add_scalar("losses/policy_loss", metrics["pg_loss"], global_step)
        writer.add_scalar("losses/value_loss", metrics["v_loss"], global_step)
        writer.add_scalar("losses/entropy", metrics["ent_loss"], global_step)
        writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
        writer.add_scalar("losses/clipfrac", metrics["clipfrac"], global_step)
        writer.add_scalar("charts/explained_variance", metrics["explained_var"], global_step)
        writer.add_scalar("charts/log_std_0", model.log_std[0].item(), global_step)
        writer.add_scalar("charts/log_std_1", model.log_std[1].item(), global_step)
        writer.add_scalar("selfplay/learner_elo", learner_elo, global_step)
        writer.add_scalar("selfplay/pool_size", pool.size, global_step)
        writer.add_scalar("selfplay/opponent_elo", opp_entry.elo, global_step)
        writer.add_scalar("phase/phase", 3, global_step)
        writer.add_scalar("phase/scripted_fraction", args.selfplay_scripted_fraction, global_step)

        if ep_state["returns"]:
            writer.add_scalar("charts/ep_return_mean", np.mean(ep_state["returns"][-100:]), global_step)
            writer.add_scalar("charts/ep_length_mean", np.mean(ep_state["lengths"][-100:]), global_step)
            writer.add_scalar("charts/win_rate", np.mean(ep_state["wins"][-100:]), global_step)

        if (update + 1) % 5 == 0 or update == 0:
            wr = np.mean(ep_state["wins"][-100:]) if ep_state["wins"] else 0.0
            ret = np.mean(ep_state["returns"][-100:]) if ep_state["returns"] else 0.0
            n_s = vec_env.n_scripted
            print(
                f"  [P3 selfplay+DR] {total_updates}/{total_updates_planned} | "
                f"vs {opp_entry.name} (elo {opp_entry.elo:.0f}) | "
                f"elo {learner_elo:.0f} | "
                f"return {ret:.2f} | wr {wr:.2%} | "
                f"pool {pool.size} | scripted {n_s}/{args.n_envs} | "
                f"pg {metrics['pg_loss']:.4f} | "
                f"std [{model.log_std[0].item():.2f},{model.log_std[1].item():.2f}] | "
                f"ppo {ppo_time:.2f}s"
            )
            ep_state["returns"].clear()
            ep_state["lengths"].clear()
            ep_state["wins"].clear()
            ep_state["is_scripted"].clear()

        if (update + 1) % 50 == 0:
            elapsed = time.time() - t_start
            slack_notify(
                f":crossed_swords: *{run_name}* [P3 selfplay+DR] {total_updates}/{total_updates_planned} | "
                f"ELO {learner_elo:.0f} | pool {pool.size} | {elapsed / 60:.0f}m"
            )

        # ELO milestones
        elo_bucket = int(learner_elo // 100) * 100
        if elo_bucket >= 1100 and elo_bucket not in elo_milestones_hit:
            elo_milestones_hit.add(elo_bucket)
            slack_notify(f":trophy: *{run_name}* ELO {elo_bucket}+ (actual: {learner_elo:.0f}) | update {total_updates}")

        # Pool snapshot
        elo_jump = learner_elo - last_snapshot_elo
        if pool.should_snapshot(update + 1, args.pool_snapshot_every, elo_jump):
            name = f"sp_{total_updates:04d}"
            pool.add_checkpoint(model_unwrapped, name, total_updates)
            last_snapshot_elo = learner_elo
            print(f"  [pool] Snapshot {name} (elo {learner_elo:.0f}, pool size {pool.size})")

        # Periodic scripted eval (default regime)
        if args.scripted_eval_every > 0 and (update + 1) % args.scripted_eval_every == 0:
            model.eval()
            brawler_wr = scripted_eval_dr(
                model, args.scripted_eval_opponent,
                args.scripted_eval_matches, args.action_repeat, device,
            )
            model.train()
            writer.add_scalar("eval/brawler_win_rate", brawler_wr, global_step)
            print(f"  [eval] vs {args.scripted_eval_opponent}: {brawler_wr:.0%} (best: {best_scripted_wr:.0%})")

            if best_scripted_wr >= args.regression_threshold:
                drop_from_last = last_scripted_wr - brawler_wr
                if brawler_wr < args.regression_threshold or drop_from_last >= 0.2:
                    slack_notify(
                        f":warning: *{run_name}* regression — {args.scripted_eval_opponent} WR "
                        f"{brawler_wr:.0%} (was {best_scripted_wr:.0%}) at update {total_updates}"
                    )
            best_scripted_wr = max(best_scripted_wr, brawler_wr)
            last_scripted_wr = brawler_wr

        # Periodic regime eval (comprehensive)
        if args.regime_eval_every > 0 and (update + 1) % args.regime_eval_every == 0:
            model.eval()
            regime_results = regime_eval(model, 10, args.action_repeat, device)
            model.train()
            print(f"  [regime eval] at update {total_updates}:")
            for rname, rwins in regime_results.items():
                for opp, wr in rwins.items():
                    writer.add_scalar(f"regime/{rname}_{opp}", wr, global_step)
                print(f"    {rname}: ace={rwins['ace']:.0%} brawler={rwins['brawler']:.0%}")

        if (update + 1) % args.save_every == 0:
            save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, f"step_{global_step}")
            export_onnx_checkpoint(model_unwrapped, ckpt_dir, f"step_{global_step}", args.hidden, args.n_blocks)

    # =====================================================================
    # Final
    # =====================================================================
    save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, "final")
    export_onnx_checkpoint(model_unwrapped, ckpt_dir, "final", args.hidden, args.n_blocks)
    pool.save_metadata()
    writer.close()

    # Final eval across all regimes
    model.eval()
    print("\n=== Final Evaluation (default regime) ===")
    final_results = {}
    for opp in EVAL_OPPONENTS:
        wr = scripted_eval_dr(model, opp, 50, args.action_repeat, device)
        final_results[opp] = wr
        print(f"  vs {opp}: {wr:.0%}")

    print("\n=== Final Regime Evaluation ===")
    regime_results = regime_eval(model, 20, args.action_repeat, device)
    for rname, rwins in regime_results.items():
        print(f"  {rname}: ace={rwins['ace']:.0%} brawler={rwins['brawler']:.0%}")

    elapsed = time.time() - t_start

    eval_summary = " | ".join(f"{k}: {v:.0%}" for k, v in final_results.items())
    slack_notify(
        f":white_check_mark: *{run_name}* done (DR) — ELO {learner_elo:.0f} | "
        f"pool {pool.size} | {elapsed / 60:.0f}m\n{eval_summary}"
    )

    print(f"\nTraining complete in {elapsed / 60:.1f}m. ELO: {learner_elo:.0f}, Pool: {pool.size}")
    print(f"Checkpoints: {ckpt_dir}, Pool: {pool_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def apply_config(config: dict, args: argparse.Namespace) -> argparse.Namespace:
    """Override argparse defaults with values from a unified config dict."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    curr_cfg = config.get("curriculum", {})
    trans_cfg = config.get("transition", {})
    sp_cfg = config.get("selfplay", {})
    reward_cfg = config.get("rewards", {})
    dr_cfg = config.get("domain_randomization", {})

    # Model
    if "hidden" in model_cfg:
        args.hidden = model_cfg["hidden"]
    if "n_blocks" in model_cfg:
        args.n_blocks = model_cfg["n_blocks"]

    # Training
    for key in ["n_envs", "n_steps", "lr", "gamma", "gae_lambda", "clip_eps",
                "vf_coef", "ent_coef", "action_repeat", "n_epochs",
                "minibatch_size", "max_grad_norm", "save_every"]:
        if key in train_cfg:
            setattr(args, key.replace("-", "_"), train_cfg[key])
    if "anneal_lr" in train_cfg:
        args.anneal_lr = train_cfg["anneal_lr"]
    if "clip_vloss" in train_cfg:
        args.clip_vloss = train_cfg["clip_vloss"]
    if "randomize" in train_cfg:
        args.randomize = train_cfg["randomize"]
    if "fp16" in train_cfg:
        args.fp16 = train_cfg["fp16"]
    if "fp16_warmup" in train_cfg:
        args.fp16_warmup = train_cfg["fp16_warmup"]
    if "target_kl" in train_cfg:
        args.target_kl = train_cfg["target_kl"]

    # Curriculum
    if "updates" in curr_cfg:
        args.curriculum_updates = curr_cfg["updates"]
    if "schedule" in curr_cfg:
        args.curriculum_schedule = curr_cfg["schedule"]

    # Transition
    if "updates" in trans_cfg:
        args.transition_updates = trans_cfg["updates"]
    if "start_scripted_fraction" in trans_cfg:
        args.transition_start_fraction = trans_cfg["start_scripted_fraction"]
    if "end_scripted_fraction" in trans_cfg:
        args.selfplay_scripted_fraction = trans_cfg["end_scripted_fraction"]
    if "reset_std" in trans_cfg:
        args.transition_reset_std = trans_cfg["reset_std"]

    # Self-play
    if "updates" in sp_cfg:
        args.selfplay_updates = sp_cfg["updates"]
    if "scripted_fraction" in sp_cfg:
        args.selfplay_scripted_fraction = sp_cfg["scripted_fraction"]
    if "scripted_pool" in sp_cfg:
        args.scripted_pool = sp_cfg["scripted_pool"]
    if "sampling" in sp_cfg:
        args.sampling = sp_cfg["sampling"]
    if "pool_snapshot_every" in sp_cfg:
        args.pool_snapshot_every = sp_cfg["pool_snapshot_every"]
    if "pool_max_size" in sp_cfg:
        args.pool_max_size = sp_cfg["pool_max_size"]
    if "scripted_eval_every" in sp_cfg:
        args.scripted_eval_every = sp_cfg["scripted_eval_every"]
    if "scripted_eval_opponent" in sp_cfg:
        args.scripted_eval_opponent = sp_cfg["scripted_eval_opponent"]
    if "scripted_eval_matches" in sp_cfg:
        args.scripted_eval_matches = sp_cfg["scripted_eval_matches"]
    if "regression_threshold" in sp_cfg:
        args.regression_threshold = sp_cfg["regression_threshold"]
    if "regime_eval_every" in sp_cfg:
        args.regime_eval_every = sp_cfg["regime_eval_every"]

    # Domain Randomization
    if "none_until" in dr_cfg:
        args.dr_none_until = dr_cfg["none_until"]
    if "narrow_until" in dr_cfg:
        args.dr_narrow_until = dr_cfg["narrow_until"]

    # Rewards
    reward_map = {
        "damage_dealt": "w_damage_dealt", "damage_taken": "w_damage_taken",
        "win": "w_win", "lose": "w_lose", "approach": "w_approach",
        "alive": "w_alive", "proximity": "w_proximity", "facing": "w_facing",
    }
    for cfg_key, arg_key in reward_map.items():
        if cfg_key in reward_cfg:
            setattr(args, arg_key, reward_cfg[cfg_key])

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-aware DR unified training")

    # Config
    parser.add_argument("--config", type=str, default=None)

    # Model
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n-blocks", type=int, default=4)

    # Training
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--clip-vloss", action="store_true", default=True)
    parser.add_argument("--gpu-sim", action="store_true", default=False,
                        help="Use NVIDIA Warp GPU physics sim instead of Rust CPU sim")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use fp16 mixed precision for inference and PPO (GPU only)")
    parser.add_argument("--no-fp16", action="store_true", default=False)
    parser.add_argument("--fp16-warmup", type=int, default=10,
                        help="Number of updates to run in fp32 before enabling fp16")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="KL early stopping threshold (None=disabled, recommended: 0.05)")
    parser.add_argument("--randomize", action="store_true", default=True)
    parser.add_argument("--save-every", type=int, default=50)

    # Phase durations
    parser.add_argument("--curriculum-updates", type=int, default=600)
    parser.add_argument("--transition-updates", type=int, default=200)
    parser.add_argument("--selfplay-updates", type=int, default=1500)

    # Domain Randomization
    parser.add_argument("--dr-none-until", type=int, default=0,
                        help="Use fixed (no DR) physics until this update")
    parser.add_argument("--dr-narrow-until", type=int, default=200,
                        help="Use narrow DR ranges until this update, then switch to full")

    # Curriculum
    parser.add_argument("--curriculum-schedule", type=json.loads, default=json.dumps({
        "0": ["do_nothing"],
        "50": ["do_nothing", "dogfighter"],
        "150": ["dogfighter", "chaser"],
        "300": ["chaser", "ace"],
        "500": ["ace", "brawler"],
    }))

    # Transition
    parser.add_argument("--transition-start-fraction", type=float, default=1.0)
    parser.add_argument("--transition-reset-std", type=float, default=-1.0)

    # Self-play
    parser.add_argument("--selfplay-scripted-fraction", type=float, default=0.2)
    parser.add_argument("--scripted-pool", nargs="+", default=["ace", "brawler"])
    parser.add_argument("--sampling", type=str, default="pfsp",
                        choices=["pfsp", "uniform", "latest", "mixed"])
    parser.add_argument("--pool-snapshot-every", type=int, default=50)
    parser.add_argument("--pool-max-size", type=int, default=30)
    parser.add_argument("--scripted-eval-every", type=int, default=20)
    parser.add_argument("--scripted-eval-opponent", type=str, default="brawler")
    parser.add_argument("--scripted-eval-matches", type=int, default=20)
    parser.add_argument("--regression-threshold", type=float, default=0.8)
    parser.add_argument("--regime-eval-every", type=int, default=100,
                        help="Run comprehensive regime eval every N self-play updates (0=disabled)")

    # Reward weights
    parser.add_argument("--w-damage-dealt", type=float, default=3.0)
    parser.add_argument("--w-damage-taken", type=float, default=-1.0)
    parser.add_argument("--w-win", type=float, default=5.0)
    parser.add_argument("--w-lose", type=float, default=-5.0)
    parser.add_argument("--w-approach", type=float, default=0.0001)
    parser.add_argument("--w-alive", type=float, default=0.0)
    parser.add_argument("--w-proximity", type=float, default=0.001)
    parser.add_argument("--w-facing", type=float, default=0.0005)

    # Paths
    parser.add_argument("--log-dir", default="training/runs")
    parser.add_argument("--checkpoint-dir", default="training/checkpoints")
    parser.add_argument("--pool-dir", default="training/pool")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Mnemonic run name (auto-generated if not provided)")

    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False

    if args.config:
        config = load_config(args.config)
        args = apply_config(config, args)

    train_unified_dr(args)
