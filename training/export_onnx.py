"""Export trained actor to ONNX and optionally validate with the Rust validator."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import numpy as np

from model import ActorCritic, ActorOnly, OBS_SIZE, CONFIG_OBS_SIZE, ACTION_SIZE


def export(model_path: str, output_path: str, validate: bool = True,
           hidden: int = 256, n_blocks: int = 0, obs_dim: int = OBS_SIZE):
    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    ac = ActorCritic(obs_dim=obs_dim, hidden=hidden, n_blocks=n_blocks).to(device)
    ac.load_state_dict(ckpt["model"])
    ac.eval()

    # Extract actor-only network
    actor = ActorOnly.from_actor_critic(ac)
    actor.eval()

    # Dummy input
    dummy = torch.randn(1, obs_dim)

    # Export
    torch.onnx.export(
        actor,
        dummy,
        output_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

    # Verify output shape
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path)
    test_obs = np.random.randn(1, obs_dim).astype(np.float32)
    result = sess.run(None, {"obs": test_obs})
    assert result[0].shape == (1, ACTION_SIZE), f"Bad output shape: {result[0].shape}"

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"Exported ONNX model to {output_path} ({size_kb:.1f} KB)")
    print(f"  Input:  obs [{obs_dim}]")
    print(f"  Output: action [{ACTION_SIZE}]")

    if validate:
        if obs_dim > OBS_SIZE:
            print("\nNote: Rust validator expects 46-input models. Skipping validation for config-aware model.")
        else:
            print("\nRunning Rust validator...")
            try:
                result = subprocess.run(
                    ["cargo", "run", "-p", "dogfight", "--release", "--", "validate", output_path],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).resolve().parent.parent,
                )
                print(result.stdout)
                if result.returncode != 0:
                    print(f"Validator stderr:\n{result.stderr}")
                    sys.exit(1)
                print("Validation passed!")
            except FileNotFoundError:
                print("Warning: cargo not found, skipping Rust validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="training/checkpoints/final.pt",
                        help="Path to .pt checkpoint")
    parser.add_argument("-o", "--output", default="policy.onnx",
                        help="Output ONNX path")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-blocks", type=int, default=0, help="Number of residual blocks (0=legacy MLP)")
    parser.add_argument("--obs-dim", type=int, default=OBS_SIZE,
                        help=f"Observation dimension (default {OBS_SIZE}, use {OBS_SIZE + CONFIG_OBS_SIZE} for config-aware)")
    args = parser.parse_args()
    export(args.model, args.output, not args.no_validate, args.hidden, args.n_blocks, args.obs_dim)
