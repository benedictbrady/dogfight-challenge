"""Shared utilities for dogfight training scripts.

Extracted from train_unified.py, train_selfplay.py, and train.py to
eliminate duplication.
"""

from pathlib import Path

import torch

# Eval opponents used across training and eval scripts
EVAL_OPPONENTS = ["do_nothing", "dogfighter", "chaser", "ace", "brawler"]


def save_checkpoint(model, optimizer, global_step, total_updates, ckpt_dir, name):
    """Save a training checkpoint.

    Handles torch.compile wrappers automatically by unwrapping _orig_mod.
    """
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


def scripted_eval(model, opponent: str, n_matches: int, action_repeat: int,
                  device: torch.device) -> float:
    """Quick eval against a scripted opponent. Returns win rate."""
    try:
        from dogfight_pyenv import DogfightEnv
    except ImportError:
        raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")

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


def compute_gae(rew_buf, val_buf, done_buf, next_done_t, next_val,
                n_steps, n_envs, gamma, gae_lambda, device):
    """Compute Generalized Advantage Estimation on GPU tensors.

    Args:
        rew_buf: Reward buffer [n_steps, n_envs]
        val_buf: Value buffer [n_steps, n_envs]
        done_buf: Done buffer [n_steps, n_envs]
        next_done_t: Done flags for the step after the buffer [n_envs]
        next_val: Bootstrap value for the step after the buffer [n_envs]
        n_steps: Number of rollout steps
        n_envs: Number of environments
        gamma: Discount factor
        gae_lambda: GAE lambda
        device: Torch device

    Returns:
        (advantages, returns) — both [n_steps, n_envs] tensors
    """
    with torch.no_grad():
        advantages = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(n_envs, device=device)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_nonterminal = 1.0 - next_done_t
                next_values = next_val
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_values = val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_values * next_nonterminal - val_buf[t]
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_nonterminal * last_gae
            )
        returns = advantages + val_buf
    return advantages, returns
