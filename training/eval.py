"""Evaluate a trained model against all built-in opponents."""

import argparse

import numpy as np
import torch

from model import ActorCritic, OBS_SIZE, CONFIG_OBS_SIZE
from utils import EVAL_OPPONENTS

try:
    from dogfight_pyenv import DogfightEnv, BatchEnv
except ImportError:
    raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")


# Named evaluation regimes: each overrides specific SimConfig parameters
EVAL_REGIMES = {
    "default": {},
    "high_gravity": {"gravity": (120.0, 120.0)},
    "low_gravity": {"gravity": (40.0, 40.0)},
    "glass_cannon": {
        "max_hp": (3, 3),
        "bullet_lifetime_ticks": (75, 75),
    },
    "tank_fight": {
        "max_hp": (8, 8),
        "gun_cooldown_ticks": (60, 60),
    },
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


def evaluate(model_path: str, n_matches: int = 50, randomize: bool = True,
             action_repeat: int = 10, hidden: int = 256, n_blocks: int = 0,
             obs_dim: int = OBS_SIZE, regime: str = "default"):
    device = torch.device("cpu")
    model = ActorCritic(obs_dim=obs_dim, hidden=hidden, n_blocks=n_blocks).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    use_config_obs = obs_dim > OBS_SIZE
    regime_params = EVAL_REGIMES.get(regime, {})

    print(f"Evaluating {model_path} over {n_matches} matches per opponent")
    print(f"Action repeat: {action_repeat}, hidden: {hidden}, n_blocks: {n_blocks}, obs_dim: {obs_dim}")
    if regime != "default":
        print(f"Regime: {regime} — {regime_params}")
    print(f"\n{'Opponent':<14} {'Wins':>6} {'Draws':>6} {'Losses':>6} {'Win%':>8} {'Avg HP':>8}")
    print("-" * 56)

    for opp_name in EVAL_OPPONENTS:
        wins, draws, losses = 0, 0, 0
        total_hp = 0

        for match_idx in range(n_matches):
            # Use BatchEnv for config-aware models to get the config obs appended
            if use_config_obs:
                env = BatchEnv(1, [opp_name], randomize, seed=match_idx * 7 + 1,
                               action_repeat=1, include_config_obs=True)
                if regime_params:
                    env.set_domain_randomization(regime_params)
                obs_np = env.reset()
                obs = obs_np[0].tolist()
                done = False
                while not done:
                    obs_t = torch.tensor([obs], dtype=torch.float32, device=device)
                    action = model.get_deterministic_action(obs_t)[0]
                    for _ in range(action_repeat):
                        obs_np, rewards, dones, infos = env.step(
                            np.array([action], dtype=np.float32))
                        obs = obs_np[0].tolist()
                        done = dones[0]
                        if done:
                            info = infos[0]
                            break
                    if done:
                        break
            else:
                env = DogfightEnv(opp_name, seed=match_idx * 7 + 1, randomize_spawns=randomize)
                obs = env.reset()
                done = False
                while not done:
                    obs_t = torch.tensor([obs], dtype=torch.float32, device=device)
                    action = model.get_deterministic_action(obs_t)[0]
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


def evaluate_all_regimes(model_path: str, n_matches: int = 20, randomize: bool = True,
                         action_repeat: int = 10, hidden: int = 256, n_blocks: int = 0,
                         obs_dim: int = OBS_SIZE):
    """Evaluate a config-aware model across all named regimes."""
    device = torch.device("cpu")
    model = ActorCritic(obs_dim=obs_dim, hidden=hidden, n_blocks=n_blocks).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = {}
    for regime_name in EVAL_REGIMES:
        regime_params = EVAL_REGIMES[regime_name]
        regime_wins = {}

        for opp_name in ["ace", "brawler"]:
            wins = 0
            for match_idx in range(n_matches):
                env = BatchEnv(1, [opp_name], randomize, seed=match_idx * 7 + 1,
                               action_repeat=1, include_config_obs=True)
                if regime_params:
                    env.set_domain_randomization(regime_params)
                obs_np = env.reset()
                done = False
                while not done:
                    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
                    action = model.get_deterministic_action(obs_t)
                    for _ in range(action_repeat):
                        obs_np, rewards, dones, infos = env.step(
                            np.array(action, dtype=np.float32))
                        done = dones[0]
                        if done:
                            break
                if infos[0].get("outcome") == "Player0Win":
                    wins += 1
            regime_wins[opp_name] = wins / n_matches
        results[regime_name] = regime_wins

    # Print summary table
    print(f"\n{'Regime':<16}", end="")
    for opp in ["ace", "brawler"]:
        print(f" {opp:>10}", end="")
    print()
    print("-" * 38)
    for regime_name, regime_wins in results.items():
        print(f"{regime_name:<16}", end="")
        for opp in ["ace", "brawler"]:
            wr = regime_wins.get(opp, 0.0)
            print(f" {wr:>9.0%}", end="")
        print()
    print()

    return results


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
    parser.add_argument("--config-obs", action="store_true",
                        help="Use 59-dim obs (46 base + 13 config) for config-aware models")
    parser.add_argument("--regime", type=str, default="default",
                        choices=list(EVAL_REGIMES.keys()),
                        help="Named physics regime for evaluation")
    parser.add_argument("--all-regimes", action="store_true",
                        help="Evaluate across all named regimes (summary table)")
    args = parser.parse_args()

    obs_dim = OBS_SIZE + CONFIG_OBS_SIZE if args.config_obs else OBS_SIZE

    if args.all_regimes:
        evaluate_all_regimes(args.model, args.matches, not args.no_randomize,
                             args.action_repeat, args.hidden, args.n_blocks, obs_dim)
    elif args.pool_dir:
        evaluate_pool(args.model, args.pool_dir, args.matches,
                      not args.no_randomize, args.action_repeat,
                      args.hidden, args.n_blocks)
    else:
        evaluate(args.model, args.matches, not args.no_randomize,
                 args.action_repeat, args.hidden, args.n_blocks,
                 obs_dim, args.regime)
