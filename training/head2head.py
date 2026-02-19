"""Head-to-head evaluation between two neural models."""
import argparse
import numpy as np
import torch
from model import ActorCritic

try:
    from dogfight_pyenv import SelfPlayBatchEnv
except ImportError:
    raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")


def head2head(p0_path, p1_path, n_matches=50, action_repeat=10,
              p0_hidden=256, p0_blocks=0, p1_hidden=256, p1_blocks=0):
    device = torch.device("cpu")

    p0 = ActorCritic(hidden=p0_hidden, n_blocks=p0_blocks).to(device)
    p0.load_state_dict(torch.load(p0_path, map_location=device, weights_only=True)["model"])
    p0.eval()

    p1 = ActorCritic(hidden=p1_hidden, n_blocks=p1_blocks).to(device)
    p1.load_state_dict(torch.load(p1_path, map_location=device, weights_only=True)["model"])
    p1.eval()

    wins, draws, losses = 0, 0, 0
    p0_hp_total, p1_hp_total = 0, 0

    for i in range(n_matches):
        env = SelfPlayBatchEnv(n_envs=1, randomize_spawns=True, seed=i * 7 + 1,
                               action_repeat=action_repeat)
        obs_p0, obs_p1 = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                act_p0 = p0.get_deterministic_action(
                    torch.tensor(obs_p0, dtype=torch.float32, device=device))
                act_p1 = p1.get_deterministic_action(
                    torch.tensor(obs_p1, dtype=torch.float32, device=device))

            obs_p0, obs_p1, _, _, dones, infos = env.step(
                np.array(act_p0, dtype=np.float32),
                np.array(act_p1, dtype=np.float32),
            )
            done = dones[0]

        outcome = infos[0].get("outcome", "Draw")
        if outcome == "Player0Win":
            wins += 1
        elif outcome == "Player1Win":
            losses += 1
        else:
            draws += 1
        p0_hp_total += infos[0].get("p0_hp", 0)
        p1_hp_total += infos[0].get("p1_hp", 0)

    total = wins + draws + losses
    print(f"P0: {p0_path}")
    print(f"    ({p0_hidden}h/{p0_blocks}b)")
    print(f"P1: {p1_path}")
    print(f"    ({p1_hidden}h/{p1_blocks}b)")
    print(f"\n{'':>10} {'Wins':>6} {'Draws':>6} {'Losses':>6} {'Win%':>8}")
    print("-" * 42)
    print(f"{'P0':>10} {wins:>6} {draws:>6} {losses:>6} {wins/total*100:>7.1f}%")
    print(f"{'P1':>10} {losses:>6} {draws:>6} {wins:>6} {losses/total*100:>7.1f}%")
    print(f"\nAvg HP — P0: {p0_hp_total/total:.2f}, P1: {p1_hp_total/total:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("p0", help="Path to P0 checkpoint")
    parser.add_argument("p1", help="Path to P1 checkpoint")
    parser.add_argument("--matches", type=int, default=50)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--p0-hidden", type=int, default=256)
    parser.add_argument("--p0-blocks", type=int, default=0)
    parser.add_argument("--p1-hidden", type=int, default=256)
    parser.add_argument("--p1-blocks", type=int, default=0)
    args = parser.parse_args()

    head2head(args.p0, args.p1, args.matches, args.action_repeat,
              args.p0_hidden, args.p0_blocks, args.p1_hidden, args.p1_blocks)
