"""Unified curriculum → self-play PPO training pipeline.

Three-phase training in a single script:
  Phase 1 — Curriculum:   All envs use scripted opponents (scripted_fraction=1.0)
  Phase 2 — Transition:   Ramp scripted_fraction from 1.0 → target, seed self-play pool
  Phase 3 — Self-Play:    Neural self-play + scripted anchoring (scripted_fraction=0.2)

Usage:
    # Config-driven (recommended)
    python train_unified.py --config experiments/configs/unified_v1.json

    # Quick local smoke test
    python train_unified.py --n-envs 4 --n-steps 64 \
        --curriculum-updates 3 --transition-updates 2 --selfplay-updates 3

    # Full production run (via Modal)
    modal run --detach modal_train.py --unified
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import ActorCritic, OBS_SIZE, ACTION_SIZE
from naming import make_run_name
from opponent_pool import OpponentPool
from slack import slack_notify
from utils import save_checkpoint, scripted_eval, compute_gae, EVAL_OPPONENTS

try:
    from dogfight_pyenv import SelfPlayBatchEnv, DogfightEnv
except ImportError:
    raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_curriculum_pool(update: int, schedule: dict[int, list[str]]) -> list[str]:
    """Return opponent pool based on update count and curriculum schedule."""
    pool = ["do_nothing"]  # default
    for threshold in sorted(int(k) for k in schedule):
        if update >= threshold:
            pool = schedule[str(threshold)]
    return pool


# ---------------------------------------------------------------------------
# PPO update (shared across all phases)
# ---------------------------------------------------------------------------

def ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, rew_buf, done_buf,
               val_buf, next_obs_t, next_done_t, args, device):
    """Run PPO update. Returns dict of metrics."""
    with torch.inference_mode():
        next_val = model.get_value(next_obs_t)

    # GAE on GPU
    advantages, returns = compute_gae(
        rew_buf, val_buf, done_buf, next_done_t, next_val,
        args.n_steps, args.n_envs, args.gamma, args.gae_lambda, device,
    )

    b_obs = obs_buf.reshape(-1, OBS_SIZE)
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
    }


# ---------------------------------------------------------------------------
# Rollout collection (shared across all phases)
# ---------------------------------------------------------------------------

def collect_rollout(model, opp_model, vec_env, obs_buf, act_buf, logp_buf,
                    rew_buf, done_buf, val_buf, next_obs_p0, next_obs_p1,
                    next_done_t, ep_state, args, device, use_cuda,
                    _pin_obs_p0=None, _pin_obs_p1=None, _pin_rew=None, _pin_done=None):
    """Collect one rollout of n_steps across all envs.

    Uses P0 as the learner. For neural envs, opp_model provides P1 actions.
    For scripted envs, P1 actions are computed inside Rust (ignored by Python).

    Returns updated (next_obs_p0, next_obs_p1, next_done_t, t_infer, t_env, rollout_stats).
    """
    t_infer_total = 0.0
    t_env_total = 0.0
    scripted_mask = np.asarray(vec_env.scripted_mask, dtype=np.bool_)
    neural_mask = ~scripted_mask

    for step in range(args.n_steps):
        # Copy obs/done to GPU buffers
        if use_cuda:
            _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
            obs_buf[step].copy_(_pin_obs_p0, non_blocking=True)
        else:
            obs_buf[step].copy_(torch.from_numpy(next_obs_p0))
        done_buf[step] = next_done_t

        t0 = time.time()
        with torch.inference_mode():
            # Learner (P0) forward pass
            action, logp, _, value = model.get_action_and_value(obs_buf[step])

            # Opponent (P1) forward pass — only for neural envs
            if opp_model is not None and neural_mask.any():
                if use_cuda:
                    _pin_obs_p1.copy_(torch.from_numpy(next_obs_p1))
                    obs_p1_gpu = _pin_obs_p1.to(device, non_blocking=True)
                else:
                    obs_p1_gpu = torch.from_numpy(next_obs_p1)
                opp_action_raw = opp_model.get_deterministic_action(obs_p1_gpu)
            else:
                # All scripted or no opp model — dummy P1 actions (ignored by Rust)
                opp_action_raw = np.zeros((args.n_envs, ACTION_SIZE), dtype=np.float32)

        act_buf[step] = action
        logp_buf[step] = logp
        val_buf[step] = value

        # Clamp P0 actions
        clamped = action.detach().clone()
        clamped[:, 0].clamp_(-1.0, 1.0)
        clamped[:, 1].clamp_(0.0, 1.0)
        clamped_p0 = np.ascontiguousarray(clamped.cpu().numpy(), dtype=np.float32)
        opp_actions = np.ascontiguousarray(opp_action_raw, dtype=np.float32)
        t_infer_total += time.time() - t0

        t0 = time.time()
        next_obs_p0, next_obs_p1, rewards_p0, _, dones, infos = vec_env.step(
            clamped_p0, opp_actions,
        )
        next_obs_p0 = np.asarray(next_obs_p0, dtype=np.float32)
        next_obs_p1 = np.asarray(next_obs_p1, dtype=np.float32)
        rewards_p0_np = np.asarray(rewards_p0, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.bool_)
        t_env_total += time.time() - t0

        # Copy rewards/dones to GPU
        next_done = dones_np.astype(np.float32)
        if use_cuda:
            _pin_rew.copy_(torch.from_numpy(rewards_p0_np))
            rew_buf[step].copy_(_pin_rew, non_blocking=True)
            _pin_done.copy_(torch.from_numpy(next_done))
            next_done_t = _pin_done.to(device, non_blocking=True)
        else:
            rew_buf[step] = torch.from_numpy(rewards_p0_np.copy())
            next_done_t = torch.from_numpy(next_done.copy())

        # Track episodes
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

        # Update scripted mask after auto-resets (some envs may have changed mode)
        if dones_np.any():
            scripted_mask = np.asarray(vec_env.scripted_mask, dtype=np.bool_)
            neural_mask = ~scripted_mask

    return next_obs_p0, next_obs_p1, next_done_t, t_infer_total, t_env_total


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_unified(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")
    print(f"Model: hidden={args.hidden}, n_blocks={args.n_blocks}")
    print(f"Action repeat: {args.action_repeat} (episode ~{10800 // args.action_repeat} RL steps)")

    total_updates_planned = args.curriculum_updates + args.transition_updates + args.selfplay_updates
    print(f"Pipeline: curriculum={args.curriculum_updates} → transition={args.transition_updates} → selfplay={args.selfplay_updates} ({total_updates_planned} total)")

    run_name = args.run_name or make_run_name("unified")
    log_dir = Path(args.log_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir)
    pool_dir = Path(args.pool_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # Initialize model
    model = ActorCritic(hidden=args.hidden, n_blocks=args.n_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

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

    # torch.compile for faster forward passes
    model_unwrapped = model
    if use_cuda and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception:
            print("  torch.compile not available")

    # Initialize self-play env (used for ALL phases — scripted_fraction controls mode)
    vec_env = SelfPlayBatchEnv(
        n_envs=args.n_envs,
        randomize_spawns=args.randomize,
        action_repeat=args.action_repeat,
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

    # Allocate PPO rollout buffers on GPU
    obs_buf = torch.zeros((args.n_steps, args.n_envs, OBS_SIZE), dtype=torch.float32, device=device)
    act_buf = torch.zeros((args.n_steps, args.n_envs, ACTION_SIZE), dtype=torch.float32, device=device)
    logp_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    done_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    val_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)

    # Pinned CPU staging buffers
    _pin_obs_p0 = _pin_obs_p1 = _pin_rew = _pin_done = None
    if use_cuda:
        _pin_obs_p0 = torch.zeros((args.n_envs, OBS_SIZE), dtype=torch.float32, pin_memory=True)
        _pin_obs_p1 = torch.zeros((args.n_envs, OBS_SIZE), dtype=torch.float32, pin_memory=True)
        _pin_rew = torch.zeros(args.n_envs, dtype=torch.float32, pin_memory=True)
        _pin_done = torch.zeros(args.n_envs, dtype=torch.float32, pin_memory=True)

    # Episode tracking
    ep_state = {
        "returns": [],
        "lengths": [],
        "wins": [],
        "is_scripted": [],  # whether each completed episode came from a scripted env
        "return_running": np.zeros(args.n_envs),
        "len_running": np.zeros(args.n_envs, dtype=np.int32),
    }

    t_start = time.time()

    slack_notify(
        f":rocket: *{run_name}* started — {args.hidden}h/{args.n_blocks}b, "
        f"{args.n_envs} envs, {total_updates_planned} updates "
        f"(curriculum {args.curriculum_updates} → transition {args.transition_updates} → selfplay {args.selfplay_updates})"
    )

    # =====================================================================
    # PHASE 1: Curriculum (scripted_fraction=1.0)
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Curriculum ({args.curriculum_updates} updates, scripted_fraction=1.0)")
    print(f"{'='*60}")

    # All envs use scripted opponents
    vec_env.set_scripted_fraction(1.0)
    vec_env.set_scripted_pool(["do_nothing"])  # initial pool

    next_obs_p0, next_obs_p1 = vec_env.reset()
    next_obs_p0 = np.asarray(next_obs_p0, dtype=np.float32)
    next_obs_p1 = np.asarray(next_obs_p1, dtype=np.float32)
    next_done_t = torch.zeros(args.n_envs, dtype=torch.float32, device=device)

    current_pool = ["do_nothing"]

    for update in range(args.curriculum_updates):
        # Curriculum schedule
        new_pool = get_curriculum_pool(total_updates, args.curriculum_schedule)
        if new_pool != current_pool:
            current_pool = new_pool
            vec_env.set_scripted_pool(current_pool)
            print(f"\n>>> Curriculum shift at update {total_updates}: pool = {current_pool}")

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

        # Bootstrap value for GAE
        if use_cuda:
            _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
            next_obs_t = _pin_obs_p0.to(device, non_blocking=True)
        else:
            next_obs_t = torch.from_numpy(next_obs_p0)

        t_ppo = time.time()
        metrics = ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, rew_buf,
                             done_buf, val_buf, next_obs_t, next_done_t, args, device)
        ppo_time = time.time() - t_ppo

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

        if ep_state["returns"]:
            writer.add_scalar("charts/ep_return_mean", np.mean(ep_state["returns"][-100:]), global_step)
            writer.add_scalar("charts/ep_length_mean", np.mean(ep_state["lengths"][-100:]), global_step)
            writer.add_scalar("charts/win_rate", np.mean(ep_state["wins"][-100:]), global_step)

        if (update + 1) % 5 == 0 or update == 0:
            wr = np.mean(ep_state["wins"][-100:]) if ep_state["wins"] else 0.0
            ret = np.mean(ep_state["returns"][-100:]) if ep_state["returns"] else 0.0
            print(
                f"  [P1 curriculum] {total_updates}/{total_updates_planned} | "
                f"pool {current_pool} | "
                f"return {ret:.2f} | wr {wr:.2%} | "
                f"pg {metrics['pg_loss']:.4f} | vl {metrics['v_loss']:.4f} | "
                f"std [{model.log_std[0].item():.2f},{model.log_std[1].item():.2f}] | "
                f"infer {t_infer:.2f}s sim {t_env:.2f}s ppo {ppo_time:.2f}s"
            )
            ep_state["returns"].clear()
            ep_state["lengths"].clear()
            ep_state["wins"].clear()
            ep_state["is_scripted"].clear()

        # Slack progress
        if (update + 1) % 50 == 0:
            elapsed = time.time() - t_start
            slack_notify(
                f":books: *{run_name}* [P1 curriculum] {total_updates}/{total_updates_planned} | "
                f"pool {current_pool} | {elapsed / 60:.0f}m"
            )

        # Checkpoint
        if (update + 1) % args.save_every == 0:
            save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, f"step_{global_step}")

    # Save curriculum-final checkpoint
    save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, "curriculum_final")

    # Quick eval at end of curriculum
    model.eval()
    curriculum_eval = {}
    for opp in ["ace", "brawler"]:
        wr = scripted_eval(model, opp, 20, args.action_repeat, device)
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
    print(f"\n{'='*60}")
    print(f"PHASE 2: Transition ({args.transition_updates} updates, "
          f"fraction {args.transition_start_fraction} → {args.selfplay_scripted_fraction})")
    print(f"{'='*60}")

    # Clear episode tracking lists (keep return_running/len_running for in-progress episodes)
    ep_state["returns"].clear()
    ep_state["lengths"].clear()
    ep_state["wins"].clear()
    ep_state["is_scripted"].clear()

    # Initialize opponent pool with curriculum-final checkpoint
    pool = OpponentPool(
        str(pool_dir), device,
        hidden=args.hidden, n_blocks=args.n_blocks,
        max_size=args.pool_max_size,
    )
    if pool.size == 0:
        pool.add_checkpoint(model_unwrapped, "curriculum_final", total_updates)
        print(f"Seeded pool with curriculum_final checkpoint")

    # Reset exploration
    if args.transition_reset_std is not None:
        with torch.no_grad():
            model.log_std.fill_(args.transition_reset_std)
        print(f"Reset log_std to {args.transition_reset_std}")

    # Set scripted pool for anchoring
    vec_env.set_scripted_pool(args.scripted_pool)

    learner_elo = 1000.0
    last_snapshot_elo = learner_elo
    elo_milestones_hit = set()
    best_scripted_wr = max(curriculum_eval.values()) if curriculum_eval else 0.0
    last_scripted_wr = best_scripted_wr

    for update in range(args.transition_updates):
        # Linearly ramp scripted fraction
        progress = update / max(args.transition_updates - 1, 1)
        fraction = args.transition_start_fraction + progress * (args.selfplay_scripted_fraction - args.transition_start_fraction)
        vec_env.set_scripted_fraction(fraction)

        # Sample opponent from pool for neural envs
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

        if use_cuda:
            _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
            next_obs_t = _pin_obs_p0.to(device, non_blocking=True)
        else:
            next_obs_t = torch.from_numpy(next_obs_p0)

        t_ppo = time.time()
        metrics = ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, rew_buf,
                             done_buf, val_buf, next_obs_t, next_done_t, args, device)
        ppo_time = time.time() - t_ppo

        total_updates += 1
        global_step += args.n_envs * args.n_steps

        # Logging
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
                f"  [P2 transition] {total_updates}/{total_updates_planned} | "
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

        # Pool snapshot during transition
        if pool.should_snapshot(update + 1, args.pool_snapshot_every, 0):
            name = f"trans_{total_updates:04d}"
            pool.add_checkpoint(model_unwrapped, name, total_updates)

        if (update + 1) % args.save_every == 0:
            save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, f"step_{global_step}")

    save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, "transition_final")

    slack_notify(
        f":bridge_at_night: *{run_name}* transition done — entering self-play "
        f"(pool {pool.size}, frac {args.selfplay_scripted_fraction})"
    )

    # =====================================================================
    # PHASE 3: Self-Play + Anchoring
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 3: Self-Play ({args.selfplay_updates} updates, "
          f"scripted_fraction={args.selfplay_scripted_fraction})")
    print(f"{'='*60}")

    vec_env.set_scripted_fraction(args.selfplay_scripted_fraction)

    # Clear episode tracking for Phase 3
    ep_state["returns"].clear()
    ep_state["lengths"].clear()
    ep_state["wins"].clear()
    ep_state["is_scripted"].clear()

    for update in range(args.selfplay_updates):
        # Sample opponent from pool
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

        if use_cuda:
            _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
            next_obs_t = _pin_obs_p0.to(device, non_blocking=True)
        else:
            next_obs_t = torch.from_numpy(next_obs_p0)

        t_ppo = time.time()
        metrics = ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, rew_buf,
                             done_buf, val_buf, next_obs_t, next_done_t, args, device)
        ppo_time = time.time() - t_ppo

        total_updates += 1
        global_step += args.n_envs * args.n_steps

        # ELO tracking — only for neural env episodes (skip scripted anchoring results)
        for win_val, is_scr in zip(ep_state["wins"], ep_state["is_scripted"]):
            if is_scr:
                continue  # don't attribute scripted outcomes to the neural opponent
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
                f"  [P3 selfplay] {total_updates}/{total_updates_planned} | "
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

        # Slack progress
        if (update + 1) % 50 == 0:
            elapsed = time.time() - t_start
            slack_notify(
                f":crossed_swords: *{run_name}* [P3 selfplay] {total_updates}/{total_updates_planned} | "
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

        # Periodic scripted eval
        if args.scripted_eval_every > 0 and (update + 1) % args.scripted_eval_every == 0:
            model.eval()
            brawler_wr = scripted_eval(
                model, args.scripted_eval_opponent,
                args.scripted_eval_matches, args.action_repeat, device,
            )
            model.train()
            writer.add_scalar("eval/brawler_win_rate", brawler_wr, global_step)
            print(f"  [eval] vs {args.scripted_eval_opponent}: {brawler_wr:.0%} (best: {best_scripted_wr:.0%})")

            # Regression alert
            if best_scripted_wr >= args.regression_threshold:
                drop_from_last = last_scripted_wr - brawler_wr
                if brawler_wr < args.regression_threshold or drop_from_last >= 0.2:
                    slack_notify(
                        f":warning: *{run_name}* regression — {args.scripted_eval_opponent} WR "
                        f"{brawler_wr:.0%} (was {best_scripted_wr:.0%}) at update {total_updates}"
                    )
            best_scripted_wr = max(best_scripted_wr, brawler_wr)
            last_scripted_wr = brawler_wr

        # Checkpoint
        if (update + 1) % args.save_every == 0:
            save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, f"step_{global_step}")

    # =====================================================================
    # Final
    # =====================================================================
    save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, "final")
    pool.save_metadata()
    writer.close()

    # Final eval
    model.eval()
    print("\n=== Final Evaluation ===")
    final_results = {}
    for opp in EVAL_OPPONENTS:
        wr = scripted_eval(model, opp, 50, args.action_repeat, device)
        final_results[opp] = wr
        print(f"  vs {opp}: {wr:.0%}")

    elapsed = time.time() - t_start

    eval_summary = " | ".join(f"{k}: {v:.0%}" for k, v in final_results.items())
    slack_notify(
        f":white_check_mark: *{run_name}* done — ELO {learner_elo:.0f} | "
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
    parser = argparse.ArgumentParser(description="Unified curriculum → self-play training")

    # Config
    parser.add_argument("--config", type=str, default=None)

    # Model
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=3)

    # Training
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--clip-vloss", action="store_true", default=True)
    parser.add_argument("--randomize", action="store_true", default=True)
    parser.add_argument("--save-every", type=int, default=50)

    # Phase durations
    parser.add_argument("--curriculum-updates", type=int, default=500)
    parser.add_argument("--transition-updates", type=int, default=200)
    parser.add_argument("--selfplay-updates", type=int, default=1500)

    # Curriculum
    parser.add_argument("--curriculum-schedule", type=json.loads, default=json.dumps({
        "0": ["do_nothing"],
        "50": ["do_nothing", "dogfighter"],
        "150": ["dogfighter", "chaser"],
        "300": ["chaser", "ace"],
        "450": ["ace", "brawler"],
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

    if args.config:
        config = load_config(args.config)
        args = apply_config(config, args)

    train_unified(args)
