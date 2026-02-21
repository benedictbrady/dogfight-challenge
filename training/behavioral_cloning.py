"""Behavioral cloning: train small MLPs to mimic scripted policies.

Generates ONNX baseline models from scripted policies using the Rust sim
for data collection and PyTorch for training.

Usage:
    python behavioral_cloning.py --policy chaser --episodes 2000
    python behavioral_cloning.py --all
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import dogfight_pyenv

OBS_SIZE = dogfight_pyenv.OBS_SIZE
ACTION_SIZE = dogfight_pyenv.ACTION_SIZE

ALL_POLICIES = ["do_nothing", "chaser", "dogfighter", "ace", "brawler"]
OPPONENTS = ["do_nothing", "chaser", "dogfighter", "ace", "brawler"]

BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"


class BCPolicy(nn.Module):
    """Small MLP for behavioral cloning.

    Output: [yaw, throttle, shoot_logit] — matches competition spec.
    """

    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.cont_head = nn.Linear(hidden, 2)   # yaw, throttle
        self.shoot_head = nn.Linear(hidden, 1)  # shoot logit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(obs)
        cont = self.cont_head(h)
        yaw = cont[:, 0:1].clamp(-1.0, 1.0)
        throttle = cont[:, 1:2].clamp(0.0, 1.0)
        shoot = self.shoot_head(h)
        return torch.cat([yaw, throttle, shoot], dim=-1)


def collect_data(policy_name: str, n_episodes: int = 2000) -> tuple:
    """Collect (obs, action) pairs using the Rust BCDataCollector."""
    collector = dogfight_pyenv.BCDataCollector(policy_name, control_period=10)
    opponents = [p for p in OPPONENTS if p != policy_name]
    if not opponents:
        opponents = OPPONENTS

    print(f"  Collecting {n_episodes} episodes of '{policy_name}' vs {opponents}...")
    obs, actions = collector.collect(n_episodes, opponents)
    print(f"  Collected {obs.shape[0]} decision points")
    return obs, actions


def train_bc(obs: np.ndarray, actions: np.ndarray, policy_name: str,
             epochs: int = 50, batch_size: int = 2048, lr: float = 3e-4) -> BCPolicy:
    """Train a BCPolicy to mimic the collected data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_t = torch.from_numpy(obs).float().to(device)
    act_t = torch.from_numpy(actions).float().to(device)

    model = BCPolicy(obs_dim=OBS_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = obs_t.shape[0]
    print(f"  Training BC model for '{policy_name}' ({n} samples, {epochs} epochs)...")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        obs_t = obs_t[perm]
        act_t = act_t[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch_obs = obs_t[i:i + batch_size]
            batch_act = act_t[i:i + batch_size]

            pred = model(batch_obs)

            # MSE loss on yaw + throttle
            cont_loss = nn.functional.mse_loss(pred[:, :2], batch_act[:, :2])

            # BCE loss on shoot (convert raw action to binary: >0 means shoot)
            shoot_target = (batch_act[:, 2] > 0).float()
            shoot_loss = nn.functional.binary_cross_entropy_with_logits(
                pred[:, 2], shoot_target
            )

            loss = cont_loss + shoot_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    return model


def export_onnx(model: BCPolicy, output_path: Path):
    """Export BCPolicy to ONNX."""
    model.eval().cpu()
    dummy = torch.randn(1, OBS_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    test_obs = np.random.randn(1, OBS_SIZE).astype(np.float32)
    result = sess.run(None, {"obs": test_obs})
    assert result[0].shape == (1, ACTION_SIZE), f"Bad output shape: {result[0].shape}"

    size_kb = output_path.stat().st_size / 1024
    print(f"  Exported to {output_path} ({size_kb:.1f} KB)")


def train_and_export(policy_name: str, n_episodes: int = 2000, epochs: int = 50):
    """Full pipeline: collect data, train, export ONNX."""
    print(f"\n{'='*50}")
    print(f"Behavioral cloning: {policy_name}")
    print(f"{'='*50}")

    if policy_name == "do_nothing":
        model = BCPolicy(obs_dim=OBS_SIZE)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        print("  do_nothing: using zero-weight model (always outputs ~0)")
    else:
        obs, actions = collect_data(policy_name, n_episodes)
        model = train_bc(obs, actions, policy_name, epochs=epochs)

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BASELINES_DIR / f"{policy_name}.onnx"
    export_onnx(model, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Behavioral cloning for dogfight baselines")
    parser.add_argument("--policy", type=str, help="Policy to clone (e.g., chaser)")
    parser.add_argument("--all", action="store_true", help="Clone all policies")
    parser.add_argument("--episodes", type=int, default=2000, help="Episodes per policy")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    if args.all:
        for policy in ALL_POLICIES:
            train_and_export(policy, args.episodes, args.epochs)
    elif args.policy:
        train_and_export(args.policy, args.episodes, args.epochs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
