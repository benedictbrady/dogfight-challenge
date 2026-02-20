"""Full round-robin tournament between all downloaded model checkpoints."""
import argparse
import sys
import time
import numpy as np
import torch
from model import ActorCritic
from utils import EVAL_OPPONENTS

try:
    from dogfight_pyenv import DogfightEnv, SelfPlayBatchEnv
except ImportError:
    raise ImportError("dogfight_pyenv not found. Build it first: make pyenv")

MODELS = [
    ("mean-knight-production", "arena/mean-knight-production.pt", 256, 0),
    ("sharp-talon-production", "arena/sharp-talon-production.pt", 256, 0),
    ("wild-lancer-unified",    "arena/wild-lancer-unified.pt",    384, 3),
    ("vivid-icarus-selfplay",  "arena/vivid-icarus-selfplay.pt",  384, 3),
    ("loud-ghost-selfplay_v2", "arena/loud-ghost-selfplay_v2.pt", 384, 3),
]

N_MATCHES = 50
ACTION_REPEAT = 10


def load_model(path, hidden, n_blocks):
    device = torch.device("cpu")
    model = ActorCritic(hidden=hidden, n_blocks=n_blocks).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def eval_vs_scripted(name, model, n_matches=N_MATCHES):
    """Evaluate model against all scripted opponents."""
    results = {}
    for opp_name in EVAL_OPPONENTS:
        wins, draws, losses, total_hp = 0, 0, 0, 0
        for i in range(n_matches):
            env = DogfightEnv(opp_name, seed=i * 7 + 1, randomize_spawns=True)
            obs = env.reset()
            done = False
            while not done:
                obs_t = torch.tensor([obs], dtype=torch.float32)
                action = model.get_deterministic_action(obs_t)[0]
                for _ in range(ACTION_REPEAT):
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
        results[opp_name] = {
            "wins": wins, "draws": draws, "losses": losses,
            "win_pct": wins / n_matches * 100,
            "avg_hp": total_hp / n_matches,
        }
    return results


def head2head(model_a, model_b, n_matches=N_MATCHES):
    """Run head-to-head between two neural models."""
    wins, draws, losses = 0, 0, 0
    hp_a_total, hp_b_total = 0, 0
    for i in range(n_matches):
        env = SelfPlayBatchEnv(n_envs=1, randomize_spawns=True,
                               seed=i * 7 + 1, action_repeat=ACTION_REPEAT)
        obs_a, obs_b = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                act_a = model_a.get_deterministic_action(
                    torch.tensor(obs_a, dtype=torch.float32))
                act_b = model_b.get_deterministic_action(
                    torch.tensor(obs_b, dtype=torch.float32))
            obs_a, obs_b, _, _, dones, infos = env.step(
                np.array(act_a, dtype=np.float32),
                np.array(act_b, dtype=np.float32),
            )
            done = dones[0]
        outcome = infos[0].get("outcome", "Draw")
        if outcome == "Player0Win":
            wins += 1
        elif outcome == "Player1Win":
            losses += 1
        else:
            draws += 1
        hp_a_total += infos[0].get("p0_hp", 0)
        hp_b_total += infos[0].get("p1_hp", 0)
    return wins, draws, losses, hp_a_total / n_matches, hp_b_total / n_matches


def main():
    t0 = time.time()

    # Load all models
    print("=" * 70)
    print("ARENA TOURNAMENT — 5 Models, 50 Matches Each")
    print("=" * 70)
    print("\nLoading models...")
    models = {}
    for name, path, hidden, blocks in MODELS:
        models[name] = load_model(path, hidden, blocks)
        print(f"  Loaded {name} ({hidden}h/{blocks}b)")

    # Phase 1: Eval vs scripted opponents
    print("\n" + "=" * 70)
    print("PHASE 1: Each Model vs Scripted Opponents (50 matches each)")
    print("=" * 70)

    scripted_results = {}
    for name, _, _, _ in MODELS:
        print(f"\n--- {name} ---")
        results = eval_vs_scripted(name, models[name])
        scripted_results[name] = results
        total_wins = sum(r["wins"] for r in results.values())
        total_matches = len(results) * N_MATCHES
        for opp, r in results.items():
            print(f"  vs {opp:<12} {r['wins']:>3}W {r['draws']:>3}D {r['losses']:>3}L  "
                  f"{r['win_pct']:>5.1f}%  HP:{r['avg_hp']:.2f}")
        print(f"  TOTAL: {total_wins}/{total_matches} wins ({total_wins/total_matches*100:.1f}%)")

    # Phase 2: Full round-robin head-to-head
    print("\n" + "=" * 70)
    print("PHASE 2: Head-to-Head Round Robin (50 matches each pairing)")
    print("=" * 70)

    names = [m[0] for m in MODELS]
    n = len(names)
    h2h_matrix = {}  # (a, b) -> (wins_a, draws, wins_b)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = names[i], names[j]
            print(f"\n  {a} vs {b}...", end=" ", flush=True)
            wa, d, wb, hp_a, hp_b = head2head(models[a], models[b])
            h2h_matrix[(a, b)] = (wa, d, wb, hp_a, hp_b)
            print(f"{wa}W-{d}D-{wb}L (HP: {hp_a:.1f} vs {hp_b:.1f})")

    # Summary tables
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Scripted summary table
    print("\n--- vs Scripted Opponents (win%) ---")
    print(f"{'Model':<28}", end="")
    for opp in EVAL_OPPONENTS:
        print(f" {opp:>10}", end="")
    print(f" {'TOTAL':>8}")
    print("-" * (28 + 11 * len(EVAL_OPPONENTS) + 9))
    for name, _, _, _ in MODELS:
        print(f"{name:<28}", end="")
        total_w = 0
        for opp in EVAL_OPPONENTS:
            r = scripted_results[name][opp]
            print(f" {r['win_pct']:>9.0f}%", end="")
            total_w += r["wins"]
        total_m = len(EVAL_OPPONENTS) * N_MATCHES
        print(f" {total_w/total_m*100:>7.1f}%")

    # H2H summary matrix
    print(f"\n--- Head-to-Head Matrix (wins for row model) ---")
    short = [n[:12] for n in names]
    print(f"{'':>28}", end="")
    for s in short:
        print(f" {s:>12}", end="")
    print(f" {'H2H Score':>10}")
    print("-" * (28 + 13 * n + 11))

    h2h_scores = {name: 0 for name in names}
    for i, a in enumerate(names):
        print(f"{a:<28}", end="")
        for j, b in enumerate(names):
            if i == j:
                print(f" {'---':>12}", end="")
            elif i < j:
                wa, d, wb, _, _ = h2h_matrix[(a, b)]
                print(f" {f'{wa}W-{d}D-{wb}L':>12}", end="")
                h2h_scores[a] += wa * 3 + d
                h2h_scores[b] += wb * 3 + d
            else:
                wb, d, wa, _, _ = h2h_matrix[(b, a)]
                print(f" {f'{wa}W-{d}D-{wb}L':>12}", end="")
        print(f" {h2h_scores[a]:>10}")

    # Final ranking
    print(f"\n{'=' * 70}")
    print("FINAL RANKING (by H2H points: 3 per win, 1 per draw)")
    print(f"{'=' * 70}")
    ranked = sorted(h2h_scores.items(), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(ranked, 1):
        total_w = sum(scripted_results[name][o]["wins"] for o in EVAL_OPPONENTS)
        total_m = len(EVAL_OPPONENTS) * N_MATCHES
        print(f"  #{rank} {name:<28} H2H: {score:>4} pts | "
              f"vs Scripted: {total_w/total_m*100:.0f}%")

    elapsed = time.time() - t0
    print(f"\nTournament completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
