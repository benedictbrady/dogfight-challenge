"""PPO training loop for dogfight agent."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import ActorCritic, OBS_SIZE, ACTION_SIZE
from naming import make_run_name

# Try to import the Rust env; helpful error if not built yet.
try:
    from dogfight_pyenv import BatchEnv
except ImportError:
    raise ImportError(
        "dogfight_pyenv not found. Build it first: make pyenv"
    )


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def get_opponent_pool(total_updates: int) -> list[str]:
    """Return opponent pool based on total update count."""
    if total_updates < 50:
        return ["do_nothing"]
    elif total_updates < 150:
        return ["do_nothing", "dogfighter"]
    elif total_updates < 300:
        return ["dogfighter", "chaser"]
    elif total_updates < 450:
        return ["chaser", "ace"]
    else:
        return ["ace", "brawler"]


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Action repeat: {args.action_repeat} (episode ~{10800 // args.action_repeat} RL steps)")

    run_name = args.run_name or make_run_name("curriculum")
    log_dir = Path(args.log_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    model = ActorCritic(hidden=args.hidden, n_blocks=args.n_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    global_step = 0
    start_update = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        start_update = ckpt.get("total_updates", 0)
        print(f"Resumed from {args.resume} at step {global_step}, update {start_update}")

    # Optionally reset exploration (useful when fine-tuning against new opponents)
    if args.reset_std is not None:
        with torch.no_grad():
            model.log_std.fill_(args.reset_std)
        print(f"Reset log_std to {args.reset_std}")

    pool = get_opponent_pool(start_update)
    print(f"Starting opponent pool: {pool}")
    vec_env = BatchEnv(args.n_envs, pool, randomize_spawns=args.randomize, action_repeat=args.action_repeat)

    obs_buf = np.zeros((args.n_steps, args.n_envs, OBS_SIZE), dtype=np.float32)
    act_buf = np.zeros((args.n_steps, args.n_envs, ACTION_SIZE), dtype=np.float32)
    logp_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.float32)
    rew_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.float32)
    done_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.float32)
    val_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.float32)

    next_obs = np.asarray(vec_env.reset(), dtype=np.float32)
    next_done = np.zeros(args.n_envs, dtype=np.float32)

    ep_returns = []
    ep_lengths = []
    ep_wins = []
    ep_return_running = np.zeros(args.n_envs)
    ep_len_running = np.zeros(args.n_envs, dtype=np.int32)

    total_updates = start_update

    for update in range(args.num_updates):
        new_pool = get_opponent_pool(total_updates)
        if new_pool != pool:
            pool = new_pool
            vec_env.set_opponent_pool(pool)
            print(f"\n>>> Curriculum shift at update {total_updates}: pool = {pool}")

        if args.anneal_lr:
            frac = 1.0 - update / args.num_updates
            lr = args.lr * max(frac, 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        t_rollout = time.time()
        t_infer_total = 0.0
        t_env_total = 0.0

        for step in range(args.n_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            t0 = time.time()
            with torch.no_grad():
                obs_t = torch.tensor(next_obs, device=device)
                action, logp, _, value = model.get_action_and_value(obs_t)

            raw_actions = action.cpu().numpy()
            act_buf[step] = raw_actions
            logp_buf[step] = logp.cpu().numpy()
            val_buf[step] = value.cpu().numpy()

            clamped_actions = raw_actions.copy()
            clamped_actions[:, 0] = np.clip(clamped_actions[:, 0], -1.0, 1.0)
            clamped_actions[:, 1] = np.clip(clamped_actions[:, 1], 0.0, 1.0)
            t_infer_total += time.time() - t0

            t0 = time.time()
            next_obs, rewards, dones, infos = vec_env.step(
                np.ascontiguousarray(clamped_actions, dtype=np.float32)
            )
            next_obs = np.asarray(next_obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=np.bool_)
            t_env_total += time.time() - t0

            rew_buf[step] = rewards
            next_done = dones.astype(np.float32)

            ep_return_running += rewards
            ep_len_running += 1
            for i, d in enumerate(dones):
                if d:
                    ep_returns.append(ep_return_running[i])
                    ep_lengths.append(ep_len_running[i])
                    outcome = infos[i].get("outcome", "Draw")
                    ep_wins.append(1.0 if outcome == "Player0Win" else 0.0)
                    ep_return_running[i] = 0.0
                    ep_len_running[i] = 0

            global_step += args.n_envs

        rollout_time = time.time() - t_rollout

        with torch.no_grad():
            next_val = model.get_value(torch.tensor(next_obs, device=device)).cpu().numpy()

        t_ppo = time.time()

        advantages = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(args.n_steps)):
            if t == args.n_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_val
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_values = val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * next_values * next_nonterminal - val_buf[t]
            advantages[t] = last_gae = (
                delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            )
        returns = advantages + val_buf

        b_obs = torch.tensor(obs_buf.reshape(-1, OBS_SIZE), device=device)
        b_act = torch.tensor(act_buf.reshape(-1, ACTION_SIZE), device=device)
        b_logp = torch.tensor(logp_buf.reshape(-1), device=device)
        b_adv = torch.tensor(advantages.reshape(-1), device=device)
        b_ret = torch.tensor(returns.reshape(-1), device=device)
        b_val = torch.tensor(val_buf.reshape(-1), device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = args.n_envs * args.n_steps
        b_inds = np.arange(batch_size)

        clipfracs = []
        for epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb = b_inds[start:end]

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
        y_pred = b_val.cpu().numpy()
        y_true = b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", ent_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/log_std_0", model.log_std[0].item(), global_step)
        writer.add_scalar("charts/log_std_1", model.log_std[1].item(), global_step)

        if ep_returns:
            writer.add_scalar("charts/ep_return_mean", np.mean(ep_returns[-100:]), global_step)
            writer.add_scalar("charts/ep_length_mean", np.mean(ep_lengths[-100:]), global_step)
            writer.add_scalar("charts/win_rate", np.mean(ep_wins[-100:]), global_step)

        if (update + 1) % 5 == 0 or update == 0:
            wr = np.mean(ep_wins[-100:]) if ep_wins else 0.0
            ret = np.mean(ep_returns[-100:]) if ep_returns else 0.0
            print(
                f"  update {total_updates}/{start_update + args.num_updates} | "
                f"step {global_step:,} | "
                f"pool {pool} | "
                f"return {ret:.2f} | "
                f"win_rate {wr:.2%} | "
                f"pg {pg_loss.item():.4f} | "
                f"vl {v_loss.item():.4f} | "
                f"ent {ent_loss.item():.3f} | "
                f"std [{model.log_std[0].item():.2f},{model.log_std[1].item():.2f}] | "
                f"infer {t_infer_total:.2f}s sim {t_env_total:.2f}s ppo {ppo_time:.2f}s"
            )
            ep_returns.clear()
            ep_lengths.clear()
            ep_wins.clear()

        if (update + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, global_step, total_updates, ckpt_dir, f"step_{global_step}")

    save_checkpoint(model, optimizer, global_step, total_updates, ckpt_dir, "final")
    writer.close()
    print(f"\nTraining complete. Final checkpoint saved to {ckpt_dir}/final.pt")


def save_checkpoint(model, optimizer, global_step, total_updates, ckpt_dir, name):
    path = Path(ckpt_dir) / f"{name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "total_updates": total_updates,
        },
        path,
    )
    print(f"  [checkpoint] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for dogfight")
    parser.add_argument("--num-updates", type=int, default=500, help="Total number of PPO updates")
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=2048, help="RL steps per env per rollout")
    parser.add_argument("--action-repeat", type=int, default=10, help="Physics ticks per RL step")
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
    parser.add_argument("--log-dir", default="training/runs")
    parser.add_argument("--checkpoint-dir", default="training/checkpoints")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-blocks", type=int, default=0, help="Number of residual blocks (0=legacy MLP)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-std", type=float, default=None, help="Reset log_std to this value on resume")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Mnemonic run name (auto-generated if not provided)")

    args = parser.parse_args()
    train(args)
