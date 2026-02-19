"""Self-play PPO training loop for dogfight agent.

Trains against past versions of itself using an opponent pool with
ELO-based sampling (PFSP). Bootstraps from a curriculum-trained checkpoint.

Usage:
    # From config file
    python train_selfplay.py --config experiments/configs/selfplay_v1.json

    # CLI args
    python train_selfplay.py --hidden 384 --n-blocks 3 --bootstrap training/checkpoints/final.pt

    # Quick local test
    python train_selfplay.py --hidden 384 --n-blocks 3 --n-envs 4 --n-steps 64 --num-updates 5
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

try:
    from dogfight_pyenv import SelfPlayBatchEnv, DogfightEnv
except ImportError:
    raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")


def scripted_eval(model, opponent: str, n_matches: int, action_repeat: int, device: torch.device) -> float:
    """Quick eval against a scripted opponent. Returns win rate."""
    wins = 0
    for i in range(n_matches):
        env = DogfightEnv(opponent, seed=i * 7 + 1, randomize_spawns=True)
        obs = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor([obs], dtype=torch.float32, device=device)
            action = model.get_deterministic_action(obs_t)[0]
            for _ in range(action_repeat):
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    break
        if info.get("outcome") == "Player0Win":
            wins += 1
    return wins / n_matches if n_matches > 0 else 0.0


def train_selfplay(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: hidden={args.hidden}, n_blocks={args.n_blocks}")
    print(f"Action repeat: {args.action_repeat} (episode ~{10800 // args.action_repeat} RL steps)")
    print(f"Self-play sampling: {args.sampling}")

    run_name = args.run_name or make_run_name("selfplay")
    log_dir = Path(args.log_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir)
    pool_dir = Path(args.pool_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # Initialize model
    model = ActorCritic(hidden=args.hidden, n_blocks=args.n_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    global_step = 0
    start_update = 0

    # Bootstrap from curriculum checkpoint
    if args.bootstrap:
        print(f"Bootstrapping from {args.bootstrap}")
        ckpt = torch.load(args.bootstrap, map_location=device, weights_only=True)

        # Try to load state dict — may fail if architecture changed
        try:
            model.load_state_dict(ckpt["model"])
            print("  Loaded full state dict (same architecture)")
        except RuntimeError:
            print("  Architecture mismatch — starting with fresh weights")
            # If the bootstrap checkpoint is a different architecture,
            # we start fresh but still seed the pool with it
            pass

        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except (RuntimeError, ValueError):
                print("  Optimizer state incompatible — using fresh optimizer")

        global_step = ckpt.get("global_step", 0)
        start_update = ckpt.get("total_updates", 0)

    # Resume from self-play checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        start_update = ckpt.get("total_updates", 0)

    # Reset exploration if requested
    if args.reset_std is not None:
        with torch.no_grad():
            model.log_std.fill_(args.reset_std)
        print(f"Reset log_std to {args.reset_std}")

    use_cuda = device.type == "cuda"

    # torch.compile for faster forward passes (PyTorch 2+)
    # model_unwrapped always points to the raw nn.Module (for save/load)
    model_unwrapped = model
    if use_cuda and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  torch.compile enabled for learner")
        except Exception:
            print("  torch.compile not available, continuing without")

    # Initialize opponent pool
    pool = OpponentPool(
        str(pool_dir), device,
        hidden=args.hidden, n_blocks=args.n_blocks,
        max_size=args.pool_max_size,
    )

    # Seed pool with initial checkpoint
    if pool.size == 0:
        pool.add_checkpoint(model_unwrapped, "sp_0000", update_num=0)
        print(f"Seeded pool with initial checkpoint")

    print(f"Pool: {pool.size} opponents")

    # Initialize self-play environment
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

    # Allocate PPO rollout buffers on GPU — eliminates bulk CPU→GPU copy at PPO time
    obs_buf = torch.zeros((args.n_steps, args.n_envs, OBS_SIZE), dtype=torch.float32, device=device)
    act_buf = torch.zeros((args.n_steps, args.n_envs, ACTION_SIZE), dtype=torch.float32, device=device)
    logp_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    done_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    val_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)

    # Pinned CPU staging buffers for fast async CPU→GPU transfer
    if use_cuda:
        _pin_obs_p0 = torch.zeros((args.n_envs, OBS_SIZE), dtype=torch.float32, pin_memory=True)
        _pin_obs_p1 = torch.zeros((args.n_envs, OBS_SIZE), dtype=torch.float32, pin_memory=True)
        _pin_rew = torch.zeros(args.n_envs, dtype=torch.float32, pin_memory=True)
        _pin_done = torch.zeros(args.n_envs, dtype=torch.float32, pin_memory=True)

    next_obs_p0, next_obs_p1 = vec_env.reset()
    next_obs_p0 = np.asarray(next_obs_p0, dtype=np.float32)
    next_obs_p1 = np.asarray(next_obs_p1, dtype=np.float32)
    next_done = np.zeros(args.n_envs, dtype=np.float32)
    next_done_t = torch.zeros(args.n_envs, dtype=torch.float32, device=device)

    # Episode tracking
    ep_returns = []
    ep_lengths = []
    ep_wins = []
    ep_return_running = np.zeros(args.n_envs)
    ep_len_running = np.zeros(args.n_envs, dtype=np.int32)

    # ELO tracking
    learner_elo = 1000.0
    last_snapshot_elo = learner_elo
    elo_milestones_hit = set()

    # Scripted eval tracking — only alert after crossing threshold once
    best_scripted_wr = 0.0
    last_scripted_wr = 0.0

    total_updates = start_update
    t_start = time.time()

    # Slack start notification
    slack_notify(
        f":rocket: *{run_name}* started — {args.hidden}h/{args.n_blocks}b, {args.sampling}, {args.n_envs} envs, {args.num_updates} updates"
    )

    for update in range(args.num_updates):
        # Sample opponent for this rollout
        opp_entry = pool.sample_opponent(method=args.sampling)
        opp_model = pool.load_opponent(opp_entry)

        if args.anneal_lr:
            frac = 1.0 - update / args.num_updates
            lr = args.lr * max(frac, 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        t_rollout = time.time()
        t_infer_total = 0.0
        t_env_total = 0.0
        rollout_wins = 0
        rollout_losses = 0
        rollout_draws = 0

        for step in range(args.n_steps):
            # Copy obs/done to GPU buffers via pinned staging
            if use_cuda:
                _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
                obs_buf[step].copy_(_pin_obs_p0, non_blocking=True)
            else:
                obs_buf[step].copy_(torch.from_numpy(next_obs_p0))
            done_buf[step] = next_done_t

            t0 = time.time()
            with torch.inference_mode():
                # Learner (P0) forward pass — obs already on GPU
                action, logp, _, value = model.get_action_and_value(obs_buf[step])

                # Opponent (P1) forward pass
                if use_cuda:
                    _pin_obs_p1.copy_(torch.from_numpy(next_obs_p1))
                    obs_p1_gpu = _pin_obs_p1.to(device, non_blocking=True)
                else:
                    obs_p1_gpu = torch.from_numpy(next_obs_p1)
                opp_action_raw = opp_model.get_deterministic_action(obs_p1_gpu)

            # Store on GPU buffers directly — no CPU round-trip for logp/value
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value

            # Clamp on GPU, single copy to CPU for env
            clamped = action.detach().clone()
            clamped[:, 0].clamp_(-1.0, 1.0)
            clamped[:, 1].clamp_(0.0, 1.0)
            clamped_p0 = np.ascontiguousarray(clamped.cpu().numpy(), dtype=np.float32)

            # Opponent actions already clamped by get_deterministic_action
            opp_actions = np.ascontiguousarray(opp_action_raw, dtype=np.float32)
            t_infer_total += time.time() - t0

            t0 = time.time()
            next_obs_p0, next_obs_p1, rewards_p0, rewards_p1, dones, infos = vec_env.step(
                clamped_p0, opp_actions,
            )
            next_obs_p0 = np.asarray(next_obs_p0, dtype=np.float32)
            next_obs_p1 = np.asarray(next_obs_p1, dtype=np.float32)
            rewards_p0_np = np.asarray(rewards_p0, dtype=np.float32)
            dones_np = np.asarray(dones, dtype=np.bool_)
            t_env_total += time.time() - t0

            # Copy rewards/dones to GPU via pinned staging
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
            ep_return_running += rewards_p0_np
            ep_len_running += 1
            for i, d in enumerate(dones_np):
                if d:
                    ep_returns.append(ep_return_running[i])
                    ep_lengths.append(ep_len_running[i])
                    outcome = infos[i].get("outcome", "Draw")
                    won = outcome == "Player0Win"
                    lost = outcome == "Player1Win"
                    drawn = outcome == "Draw"
                    ep_wins.append(1.0 if won else 0.0)

                    # Update ELO (symmetric — both learner and opponent adjust)
                    learner_elo = pool.update_learner_elo(
                        learner_elo, opp_entry, won=won, drawn=drawn
                    )
                    pool.update_elo(opp_entry, learner_elo, won=won, drawn=drawn)

                    if won:
                        rollout_wins += 1
                    elif lost:
                        rollout_losses += 1
                    else:
                        rollout_draws += 1

                    ep_return_running[i] = 0.0
                    ep_len_running[i] = 0

            global_step += args.n_envs

        rollout_time = time.time() - t_rollout

        # GAE — computed entirely on GPU (no CPU round-trip)
        with torch.inference_mode():
            if use_cuda:
                _pin_obs_p0.copy_(torch.from_numpy(next_obs_p0))
                next_obs_t = _pin_obs_p0.to(device, non_blocking=True)
            else:
                next_obs_t = torch.from_numpy(next_obs_p0)
            next_val = model.get_value(next_obs_t)

        t_ppo = time.time()

        with torch.no_grad():
            advantages = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(args.n_envs, device=device)
            for t in reversed(range(args.n_steps)):
                if t == args.n_steps - 1:
                    next_nonterminal = 1.0 - next_done_t
                    next_values = next_val
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_values = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * next_values * next_nonterminal - val_buf[t]
                advantages[t] = last_gae = (
                    delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                )
            returns = advantages + val_buf

        # PPO update — buffers already on GPU, just reshape (zero-copy views)
        b_obs = obs_buf.reshape(-1, OBS_SIZE)
        b_act = act_buf.reshape(-1, ACTION_SIZE)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = args.n_envs * args.n_steps
        b_inds = torch.randperm(batch_size, device="cpu")

        clipfracs = []
        for epoch in range(args.n_epochs):
            b_inds = torch.randperm(batch_size, device="cpu")
            for start in range(0, batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]

                _, new_logp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb], b_act[mb]
                )

                logratio = new_logp - b_logp[mb]
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

        ppo_time = time.time() - t_ppo

        total_updates += 1

        # Logging — compute explained variance on GPU
        with torch.no_grad():
            var_y = b_ret.var().item()
            explained_var = float("nan") if var_y == 0 else 1 - (b_ret - b_val).var().item() / var_y

        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", ent_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/log_std_0", model.log_std[0].item(), global_step)
        writer.add_scalar("charts/log_std_1", model.log_std[1].item(), global_step)
        writer.add_scalar("selfplay/learner_elo", learner_elo, global_step)
        writer.add_scalar("selfplay/pool_size", pool.size, global_step)
        writer.add_scalar("selfplay/opponent_elo", opp_entry.elo, global_step)

        if ep_returns:
            writer.add_scalar("charts/ep_return_mean", np.mean(ep_returns[-100:]), global_step)
            writer.add_scalar("charts/ep_length_mean", np.mean(ep_lengths[-100:]), global_step)
            writer.add_scalar("charts/win_rate", np.mean(ep_wins[-100:]), global_step)

        # Console output
        if (update + 1) % 5 == 0 or update == 0:
            wr = np.mean(ep_wins[-100:]) if ep_wins else 0.0
            ret = np.mean(ep_returns[-100:]) if ep_returns else 0.0
            print(
                f"  update {total_updates}/{start_update + args.num_updates} | "
                f"step {global_step:,} | "
                f"vs {opp_entry.name} (elo {opp_entry.elo:.0f}) | "
                f"elo {learner_elo:.0f} | "
                f"return {ret:.2f} | "
                f"win_rate {wr:.2%} | "
                f"pool {pool.size} | "
                f"pg {pg_loss.item():.4f} | "
                f"vl {v_loss.item():.4f} | "
                f"std [{model.log_std[0].item():.2f},{model.log_std[1].item():.2f}] | "
                f"infer {t_infer_total:.2f}s sim {t_env_total:.2f}s ppo {ppo_time:.2f}s"
            )
            ep_returns.clear()
            ep_lengths.clear()
            ep_wins.clear()

        # Slack progress (every 50 updates)
        if (update + 1) % 50 == 0:
            elapsed = time.time() - t_start
            slack_notify(
                f":bar_chart: *{run_name}* — {total_updates}/{start_update + args.num_updates} | ELO {learner_elo:.0f} | pool {pool.size} | {elapsed / 60:.0f}m"
            )

        # ELO milestone notifications
        elo_bucket = int(learner_elo // 100) * 100
        if elo_bucket >= 1100 and elo_bucket not in elo_milestones_hit:
            elo_milestones_hit.add(elo_bucket)
            slack_notify(f":trophy: *{run_name}* ELO {elo_bucket}+ (actual: {learner_elo:.0f}) | update {total_updates} | pool {pool.size}")

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

            # Alert on regression: only after we've crossed the threshold once,
            # and only if WR dropped significantly from peak or from last eval
            drop_from_best = best_scripted_wr - brawler_wr
            drop_from_last = last_scripted_wr - brawler_wr
            if best_scripted_wr >= args.regression_threshold:
                if brawler_wr < args.regression_threshold or drop_from_last >= 0.2:
                    slack_notify(
                        f":warning: *{run_name}* regression — {args.scripted_eval_opponent} WR {brawler_wr:.0%} (was {best_scripted_wr:.0%}) at update {total_updates}"
                    )
            best_scripted_wr = max(best_scripted_wr, brawler_wr)
            last_scripted_wr = brawler_wr

        # Save periodic checkpoint
        if (update + 1) % args.save_every == 0:
            save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, f"step_{global_step}")

    # Final checkpoint
    save_checkpoint(model_unwrapped, optimizer, global_step, total_updates, ckpt_dir, "final")
    pool.save_metadata()
    writer.close()

    # Final scripted eval
    model.eval()
    print("\n=== Final Evaluation ===")
    final_results = {}
    for opp in ["do_nothing", "dogfighter", "chaser", "ace", "brawler"]:
        wr = scripted_eval(model, opp, 50, args.action_repeat, device)
        final_results[opp] = wr
        print(f"  vs {opp}: {wr:.0%}")

    elapsed = time.time() - t_start

    # Completion notification
    eval_summary = " | ".join(f"{k}: {v:.0%}" for k, v in final_results.items())
    slack_notify(
        f":white_check_mark: *{run_name}* done — ELO {learner_elo:.0f} | pool {pool.size} | {elapsed / 60:.0f}m\n{eval_summary}"
    )

    print(f"\nTraining complete. ELO: {learner_elo:.0f}, Pool: {pool.size}")
    print(f"Checkpoints: {ckpt_dir}, Pool: {pool_dir}")


def save_checkpoint(model, optimizer, global_step, total_updates, ckpt_dir, name):
    # Unwrap torch.compile wrapper if present
    raw_model = getattr(model, "_orig_mod", model)
    path = Path(ckpt_dir) / f"{name}.pt"
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "total_updates": total_updates,
        },
        path,
    )
    print(f"  [checkpoint] {path}")


def load_config(config_path: str) -> dict:
    """Load a JSON config file and return as dict."""
    with open(config_path) as f:
        return json.load(f)


def args_from_config(config: dict, args: argparse.Namespace) -> argparse.Namespace:
    """Override argparse defaults with values from a config dict."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    sp_cfg = config.get("selfplay", {})
    reward_cfg = config.get("rewards", {})

    # Model
    if "hidden" in model_cfg:
        args.hidden = model_cfg["hidden"]
    if "n_blocks" in model_cfg:
        args.n_blocks = model_cfg["n_blocks"]

    # Training
    for key in ["n_envs", "n_steps", "num_updates", "lr", "gamma", "gae_lambda",
                "clip_eps", "vf_coef", "ent_coef", "action_repeat", "n_epochs",
                "minibatch_size", "max_grad_norm", "save_every"]:
        if key in train_cfg:
            setattr(args, key.replace("-", "_"), train_cfg[key])
    if "anneal_lr" in train_cfg:
        args.anneal_lr = train_cfg["anneal_lr"]
    if "clip_vloss" in train_cfg:
        args.clip_vloss = train_cfg["clip_vloss"]
    if "randomize" in train_cfg:
        args.randomize = train_cfg["randomize"]

    # Self-play
    if "sampling" in sp_cfg:
        args.sampling = sp_cfg["sampling"]
    if "pool_snapshot_every" in sp_cfg:
        args.pool_snapshot_every = sp_cfg["pool_snapshot_every"]
    if "pool_max_size" in sp_cfg:
        args.pool_max_size = sp_cfg["pool_max_size"]
    if "bootstrap" in sp_cfg and sp_cfg["bootstrap"] and not args.bootstrap:
        args.bootstrap = sp_cfg["bootstrap"]
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
        "damage_dealt": "w_damage_dealt",
        "damage_taken": "w_damage_taken",
        "win": "w_win",
        "lose": "w_lose",
        "approach": "w_approach",
        "alive": "w_alive",
        "proximity": "w_proximity",
        "facing": "w_facing",
    }
    for cfg_key, arg_key in reward_map.items():
        if cfg_key in reward_cfg:
            setattr(args, arg_key, reward_cfg[cfg_key])

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play PPO training for dogfight")

    # Config file (overrides defaults)
    parser.add_argument("--config", type=str, default=None, help="JSON config file path")

    # Model architecture
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=3)

    # Training hyperparameters
    parser.add_argument("--num-updates", type=int, default=2000)
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

    # Self-play specific
    parser.add_argument("--sampling", type=str, default="pfsp",
                        choices=["pfsp", "uniform", "latest", "mixed"])
    parser.add_argument("--pool-snapshot-every", type=int, default=50)
    parser.add_argument("--pool-max-size", type=int, default=30)
    parser.add_argument("--bootstrap", type=str, default=None,
                        help="Path to curriculum checkpoint to bootstrap from")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from self-play checkpoint")
    parser.add_argument("--reset-std", type=float, default=None)

    # Scripted eval
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
    parser.add_argument("--run-name", type=str, default=None,
                        help="Mnemonic run name (auto-generated if not provided)")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        args = args_from_config(config, args)

    train_selfplay(args)
