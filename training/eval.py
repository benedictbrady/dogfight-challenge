"""Evaluate a trained model against all built-in opponents."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from model import ActorCritic, OBS_SIZE

try:
    from dogfight_pyenv import DogfightEnv
except ImportError:
    raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")


OPPONENTS = ["do_nothing", "dogfighter", "chaser", "ace", "brawler"]


def evaluate(model_path: str, n_matches: int = 50, randomize: bool = True, action_repeat: int = 10):
    device = torch.device("cpu")
    model = ActorCritic().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Evaluating {model_path} over {n_matches} matches per opponent")
    print(f"Action repeat: {action_repeat}\n")
    print(f"{'Opponent':<14} {'Wins':>6} {'Draws':>6} {'Losses':>6} {'Win%':>8} {'Avg HP':>8}")
    print("-" * 56)

    for opp_name in OPPONENTS:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .pt checkpoint")
    parser.add_argument("--matches", type=int, default=50)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--no-randomize", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, args.matches, not args.no_randomize, args.action_repeat)
