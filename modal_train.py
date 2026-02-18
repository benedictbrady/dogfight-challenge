"""Modal deployment for dogfight PPO training.

Usage:
    # Single run with defaults
    modal run modal_train.py

    # Custom hyperparameters
    modal run modal_train.py --n-envs 256 --n-steps 1024 --num-updates 500

    # Parallel sweep (runs all configs simultaneously)
    modal run modal_train.py --sweep

Setup:
    pip install modal
    modal token set
"""

import modal
import os

app = modal.App("dogfight-train")

# ---------------------------------------------------------------------------
# Image: Rust toolchain + maturin + PyTorch + compiled pyenv crate
# ---------------------------------------------------------------------------

# Minimal workspace Cargo.toml with only the crates needed for pyenv
WORKSPACE_TOML = """\
[workspace]
members = [
    "crates/shared",
    "crates/sim",
    "crates/pyenv",
]
resolver = "2"

[workspace.dependencies]
glam = { version = "0.29", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = "0.8"
rand_pcg = "0.3"
"""

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("torch", "numpy", "tensorboard", "maturin")
    # Copy only the crates needed for pyenv
    .copy_local_dir("crates/shared", "/app/crates/shared")
    .copy_local_dir("crates/sim", "/app/crates/sim")
    .copy_local_dir("crates/pyenv", "/app/crates/pyenv")
    # Write minimal workspace Cargo.toml (avoids needing server/cli/validator stubs)
    .run_commands(
        f"cat > /app/Cargo.toml << 'TOML'\n{WORKSPACE_TOML}TOML",
    )
    # Build the pyenv crate (pre-compiled in the image, zero startup cost)
    .run_commands(
        "cd /app && maturin build --release -m crates/pyenv/Cargo.toml",
        "pip install /app/target/wheels/*.whl",
    )
    # Copy training scripts last (changes most often, minimizes image rebuild)
    .copy_local_dir("training", "/app/training")
)

# Persistent volume for checkpoints and TensorBoard logs
vol = modal.Volume.from_name("dogfight-training", create_if_missing=True)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=16,
    memory=8192,
    timeout=3600,
    volumes={"/results": vol},
)
def train(
    n_envs: int = 128,
    n_steps: int = 2048,
    num_updates: int = 500,
    n_epochs: int = 4,
    minibatch_size: int = 4096,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    save_every: int = 100,
    tag: str = "",
    resume: str = "",
) -> dict:
    """Run a single PPO training experiment on Modal."""
    import subprocess
    import time
    import json

    exp_name = f"exp_{tag}_{int(time.time())}" if tag else f"exp_{int(time.time())}"
    exp_dir = f"/results/{exp_name}"
    ckpt_dir = f"{exp_dir}/checkpoints"
    log_dir = f"{exp_dir}/runs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "n_envs": n_envs, "n_steps": n_steps, "num_updates": num_updates,
        "n_epochs": n_epochs, "minibatch_size": minibatch_size,
        "lr": lr, "ent_coef": ent_coef, "gamma": gamma,
        "gae_lambda": gae_lambda, "clip_eps": clip_eps, "vf_coef": vf_coef,
    }
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    cmd = [
        "python", "/app/training/train.py",
        "--n-envs", str(n_envs),
        "--n-steps", str(n_steps),
        "--num-updates", str(num_updates),
        "--n-epochs", str(n_epochs),
        "--minibatch-size", str(minibatch_size),
        "--lr", str(lr),
        "--ent-coef", str(ent_coef),
        "--gamma", str(gamma),
        "--gae-lambda", str(gae_lambda),
        "--clip-eps", str(clip_eps),
        "--vf-coef", str(vf_coef),
        "--save-every", str(save_every),
        "--checkpoint-dir", ckpt_dir,
        "--log-dir", log_dir,
    ]
    if resume:
        cmd.extend(["--resume", resume])

    print(f"=== {exp_name} ===")
    print(f"Config: {json.dumps(config)}")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/app")
    elapsed = time.time() - t0

    # Run evaluation against all opponents
    eval_output = ""
    final_ckpt = f"{ckpt_dir}/final.pt"
    if os.path.exists(final_ckpt):
        eval_proc = subprocess.run(
            ["python", "/app/training/eval.py", final_ckpt, "--matches", "50"],
            cwd="/app", capture_output=True, text=True,
        )
        eval_output = eval_proc.stdout
        print(eval_output)
        with open(f"{exp_dir}/eval.txt", "w") as f:
            f.write(eval_output)

    vol.commit()

    return {
        "experiment": exp_name,
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": proc.returncode,
        "config": config,
        "eval": eval_output,
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    n_envs: int = 128,
    n_steps: int = 2048,
    num_updates: int = 500,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    sweep: bool = False,
    tag: str = "",
):
    if sweep:
        # Parallel hyperparameter sweep — all run simultaneously on Modal
        configs = [
            {"n_envs": 128, "n_steps": 2048, "ent_coef": 0.001, "tag": "ent0.001"},
            {"n_envs": 128, "n_steps": 2048, "ent_coef": 0.005, "tag": "ent0.005"},
            {"n_envs": 128, "n_steps": 2048, "ent_coef": 0.01,  "tag": "ent0.01"},
            {"n_envs": 128, "n_steps": 2048, "ent_coef": 0.02,  "tag": "ent0.02"},
            {"n_envs": 128, "n_steps": 1024, "ent_coef": 0.01,  "tag": "steps1024"},
            {"n_envs": 128, "n_steps": 4096, "ent_coef": 0.01,  "tag": "steps4096"},
            {"n_envs": 64,  "n_steps": 2048, "ent_coef": 0.01,  "tag": "envs64"},
            {"n_envs": 256, "n_steps": 2048, "ent_coef": 0.01,  "tag": "envs256"},
        ]

        print(f"Launching {len(configs)} experiments in parallel on Modal...")
        handles = []
        for cfg in configs:
            tag_val = cfg.pop("tag")
            handles.append(train.spawn(tag=tag_val, **cfg))

        print(f"Waiting for {len(handles)} experiments to complete...")
        print()
        for h in handles:
            r = h.get()
            status = "OK" if r["exit_code"] == 0 else "FAIL"
            print(f"  [{status}] {r['experiment']}: {r['elapsed_seconds']}s")
            if r.get("eval"):
                # Print just the summary lines
                for line in r["eval"].strip().split("\n"):
                    if line.strip() and not line.startswith("-"):
                        print(f"         {line.strip()}")
            print()

    else:
        result = train.remote(
            n_envs=n_envs,
            n_steps=n_steps,
            num_updates=num_updates,
            lr=lr,
            ent_coef=ent_coef,
            tag=tag,
        )
        print(f"\nDone: {result['experiment']} in {result['elapsed_seconds']}s")
        if result.get("eval"):
            print(result["eval"])
