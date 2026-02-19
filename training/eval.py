"""Evaluate a trained model against all built-in opponents."""

import argparse

import numpy as np
import torch

from model import ActorCritic, OBS_SIZE
from utils import EVAL_OPPONENTS

try:
    from dogfight_pyenv import DogfightEnv
except ImportError:
    raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")


def evaluate(model_path: str, n_matches: int = 50, randomize: bool = True,
             action_repeat: int = 10, hidden: int = 256, n_blocks: int = 0):
    device = torch.device("cpu")
    model = ActorCritic(hidden=hidden, n_blocks=n_blocks).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Evaluating {model_path} over {n_matches} matches per opponent")
    print(f"Action repeat: {action_repeat}, hidden: {hidden}, n_blocks: {n_blocks}\n")
    print(f"{'Opponent':<14} {'Wins':>6} {'Draws':>6} {'Losses':>6} {'Win%':>8} {'Avg HP':>8}")
    print("-" * 56)

    for opp_name in EVAL_OPPONENTS:
        wins, draws, losses = 0, 0, 0
        total_hp = 0

        for match_idx in range(n_matches):
            env = DogfightEnv(opp_name, seed=match_idx * 7 + 1, randomize_spawns=randomize)
            obs = env.reset()
            done = False

            while not done:
                obs_t = torch.tensor([obs], dtype=torch.float32, device=device)
                action = model.get_deterministic_action(obs_t)[0]

                # Action repeat: apply same action for multiple physics ticks
                for _ in range(action_repeat):
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        break

            outcome = info.get("outcome", "Draw")
            if outcome == "Player0Win":
                wins += 1
            elif outcome == "Player1Win":
                losses += 1
            else:
                draws += 1
            total_hp += info.get("my_hp", 0)

        total = wins + draws + losses
        win_pct = wins / total * 100 if total > 0 else 0
        avg_hp = total_hp / total if total > 0 else 0
        print(f"{opp_name:<14} {wins:>6} {draws:>6} {losses:>6} {win_pct:>7.1f}% {avg_hp:>8.2f}")

    print()


def evaluate_pool(model_path: str, pool_dir: str, n_matches: int = 20,
                  randomize: bool = True, action_repeat: int = 10,
                  hidden: int = 256, n_blocks: int = 0):
    """Evaluate a model against all opponents in a self-play pool."""
    from opponent_pool import OpponentPool

    device = torch.device("cpu")

    # Load the model to evaluate
    model = ActorCritic(hidden=hidden, n_blocks=n_blocks).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load pool
    pool = OpponentPool(pool_dir, device, hidden=hidden, n_blocks=n_blocks)
    if pool.size == 0:
        print(f"Pool at {pool_dir} is empty.")
        return

    print(f"Evaluating {model_path} against pool ({pool.size} opponents)")
    print(f"Action repeat: {action_repeat}\n")
    print(f"{'Opponent':<20} {'ELO':>6} {'Wins':>6} {'Draws':>6} {'Losses':>6} {'Win%':>8}")
    print("-" * 60)

    try:
        from dogfight_pyenv import SelfPlayBatchEnv
        has_selfplay_env = True
    except ImportError:
        has_selfplay_env = False

    for entry in pool.entries:
        opp_model = pool.load_opponent(entry)
        wins, draws, losses = 0, 0, 0

        for match_idx in range(n_matches):
            # Use single DogfightEnv-style manual loop for simplicity
            env = DogfightEnv("do_nothing", seed=match_idx * 7 + 1, randomize_spawns=randomize)
            # We need SelfPlayBatchEnv for true self-play eval, but
            # for now use a simpler approach: run both models step by step
            # using the raw sim. Since DogfightEnv only supports scripted opponents,
            # we'll use SelfPlayBatchEnv with n_envs=1 if available.
            if has_selfplay_env:
                sp_env = SelfPlayBatchEnv(n_envs=1, randomize_spawns=randomize,
                                          seed=match_idx * 7 + 1, action_repeat=action_repeat)
                obs_p0, obs_p1 = sp_env.reset()
                done = False
                while not done:
                    obs_p0_t = torch.tensor(obs_p0, dtype=torch.float32, device=device)
                    obs_p1_t = torch.tensor(obs_p1, dtype=torch.float32, device=device)
                    act_p0 = model.get_deterministic_action(obs_p0_t)
                    act_p1 = opp_model.get_deterministic_action(obs_p1_t)
                    obs_p0, obs_p1, _, _, dones, infos = sp_env.step(
                        np.array(act_p0, dtype=np.float32),
                        np.array(act_p1, dtype=np.float32),
                    )
                    done = dones[0]
                    if done:
                        outcome = infos[0].get("outcome", "Draw")
            else:
                # Fallback: skip pool eval if SelfPlayBatchEnv not available
                print("  (SelfPlayBatchEnv not available, skipping pool eval)")
                return

            if outcome == "Player0Win":
                wins += 1
            elif outcome == "Player1Win":
                losses += 1
            else:
                draws += 1

        total = wins + draws + losses
        win_pct = wins / total * 100 if total > 0 else 0
        print(f"{entry.name:<20} {entry.elo:>6.0f} {wins:>6} {draws:>6} {losses:>6} {win_pct:>7.1f}%")

    print()
    print(pool.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .pt checkpoint")
    parser.add_argument("--matches", type=int, default=50)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-blocks", type=int, default=0, help="Number of residual blocks (0=legacy MLP)")
    parser.add_argument("--pool-dir", type=str, default=None, help="Path to self-play opponent pool directory")
    args = parser.parse_args()

    if args.pool_dir:
        evaluate_pool(args.model, args.pool_dir, args.matches,
                      not args.no_randomize, args.action_repeat,
                      args.hidden, args.n_blocks)
    else:
        evaluate(args.model, args.matches, not args.no_randomize,
                 args.action_repeat, args.hidden, args.n_blocks)
