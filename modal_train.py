"""Modal deployment for dogfight PPO training.

Usage:
    # Single run with defaults
    modal run modal_train.py

    # Custom hyperparameters
    modal run modal_train.py --n-envs 256 --n-steps 1024 --num-updates 500

    # Smoke test (verifies full pipeline in <2 min)
    modal run modal_train.py --smoke-test

    # Production run (scaled-up proven config)
    modal run modal_train.py --production

    # Parallel sweep (runs all configs simultaneously)
    modal run modal_train.py --sweep

    # Self-play training
    modal run modal_train.py --selfplay
    modal run modal_train.py --selfplay --smoke-test
    modal run modal_train.py --selfplay --bootstrap training/checkpoints/final.pt

    # Config-driven (any experiment)
    modal run modal_train.py --config experiments/configs/selfplay_v1.json

    # Download results
    modal run modal_train.py --download list
    modal run modal_train.py --download <exp_name>

Setup:
    pip install modal
    modal token set
"""

import modal
import os
import hashlib

SLACK_WEBHOOK = os.environ.get(
    "SLACK_WEBHOOK_URL",
    "https://hooks.slack.com/services/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",
)

# Mnemonic name generator — deterministic from timestamp
_ADJECTIVES = [
    "angry", "bold", "calm", "dark", "eager", "fierce", "grim", "hot",
    "iron", "keen", "loud", "mean", "nova", "odd", "prime", "quick",
    "raw", "sharp", "tight", "ultra", "vivid", "wild", "xray", "zero",
]
_CALLSIGNS = [
    "baron", "cobra", "dagger", "eagle", "falcon", "ghost", "hawk", "icarus",
    "jester", "knight", "lancer", "maverick", "nomad", "outlaw", "phantom",
    "raptor", "saber", "talon", "viper", "wolf",
]

def _make_name(tag: str, ts: int) -> str:
    """Generate a mnemonic experiment name like 'fierce-hawk-production'."""
    h = int(hashlib.md5(str(ts).encode()).hexdigest(), 16)
    adj = _ADJECTIVES[h % len(_ADJECTIVES)]
    call = _CALLSIGNS[(h // len(_ADJECTIVES)) % len(_CALLSIGNS)]
    suffix = f"-{tag}" if tag else ""
    return f"{adj}-{call}{suffix}"

def _slack_notify(msg: str):
    """Send a Slack notification. Used inside Modal functions."""
    import json
    import urllib.request

    url = SLACK_WEBHOOK
    if not url:
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": msg}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[slack] failed: {e}")

app = modal.App("dogfight-train")

# ---------------------------------------------------------------------------
# Image: Rust toolchain + maturin + PyTorch + compiled pyenv crate
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("numpy", "tensorboard", "maturin", "warp-lang")
    .run_commands("pip install torch --index-url https://download.pytorch.org/whl/cu121")
    # Copy crates into image (copy=True bakes into image so maturin build works)
    .add_local_dir("crates/shared", "/app/crates/shared", copy=True)
    .add_local_dir("crates/sim", "/app/crates/sim", copy=True)
    .add_local_dir("crates/pyenv", "/app/crates/pyenv", copy=True)
    # Minimal workspace Cargo.toml (only shared+sim+pyenv, no server/cli/validator)
    .add_local_file("crates/pyenv/Cargo.workspace.toml", "/app/Cargo.toml", copy=True)
    # Build the pyenv crate (pre-compiled in the image, zero startup cost)
    .run_commands(
        "cd /app && maturin build --release -m crates/pyenv/Cargo.toml",
        "pip install /app/target/wheels/*.whl",
    )
    # Training scripts added at startup (no build step needed, fast iteration)
    .add_local_dir("training", "/app/training")
    # Experiment configs
    .add_local_dir("experiments/configs", "/app/experiments/configs")
)

# Persistent volume for checkpoints and TensorBoard logs
vol = modal.Volume.from_name("dogfight-training", create_if_missing=True)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=16384,
    timeout=14400,
    volumes={"/results": vol},
)
def train(
    n_envs: int = 128,
    n_steps: int = 2048,
    num_updates: int = 500,
    n_epochs: int = 4,
    minibatch_size: int = 2048,
    lr: float = 3e-4,
    ent_coef: float = 0.0,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    action_repeat: int = 10,
    save_every: int = 100,
    tag: str = "",
    resume: str = "",
) -> dict:
    """Run a single PPO training experiment on Modal."""
    import subprocess
    import time
    import json
    import re

    ts = int(time.time())
    exp_name = _make_name(tag, ts)
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
        "action_repeat": action_repeat,
    }
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    cmd = [
        "python", "-u", "/app/training/train.py",
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
        "--action-repeat", str(action_repeat),
        "--save-every", str(save_every),
        "--checkpoint-dir", ckpt_dir,
        "--log-dir", log_dir,
        "--run-name", exp_name,
    ]
    if resume:
        cmd.extend(["--resume", resume])

    print(f"=== {exp_name} ===")
    print(f"Config: {json.dumps(config)}")
    # Note: training script sends its own Slack notifications using --run-name

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd="/app", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")

    proc.wait()
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

    # Final Slack notification
    status = ":white_check_mark:" if proc.returncode == 0 else ":x:"
    mins = elapsed / 60
    eval_summary = ""
    if eval_output:
        # Grab win-rate lines from eval
        for eline in eval_output.strip().split("\n"):
            if "%" in eline and "vs" in eline.lower():
                eval_summary += f"\n> {eline.strip()}"
    _slack_notify(f"{status} *{exp_name}* finished in {mins:.0f}m (exit {proc.returncode}){eval_summary}")

    return {
        "experiment": exp_name,
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": proc.returncode,
        "config": config,
        "eval": eval_output,
    }


# ---------------------------------------------------------------------------
# Self-play training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    cpu=32,
    memory=16384,
    timeout=28800,
    volumes={"/results": vol},
)
def train_selfplay(config: dict) -> dict:
    """Run a self-play PPO training experiment on Modal.

    Args:
        config: Full experiment config dict (model, training, selfplay, rewards, etc.)
    """
    import subprocess
    import time
    import json

    ts = int(time.time())
    tag = config.get("_tag", "selfplay")
    exp_name = _make_name(tag, ts)
    exp_dir = f"/results/selfplay/{exp_name}"
    ckpt_dir = f"{exp_dir}/checkpoints"
    log_dir = f"{exp_dir}/runs"
    pool_dir = f"{exp_dir}/pool"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)

    # Save config
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Set SLACK_WEBHOOK_URL env var for train_selfplay.py's slack helper
    env = os.environ.copy()
    env["SLACK_WEBHOOK_URL"] = SLACK_WEBHOOK

    # Build command from config
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    sp_cfg = config.get("selfplay", {})
    reward_cfg = config.get("rewards", {})

    cmd = [
        "python", "-u", "/app/training/train_selfplay.py",
        "--hidden", str(model_cfg.get("hidden", 384)),
        "--n-blocks", str(model_cfg.get("n_blocks", 3)),
        "--n-envs", str(train_cfg.get("n_envs", 256)),
        "--n-steps", str(train_cfg.get("n_steps", 2048)),
        "--num-updates", str(train_cfg.get("num_updates", 2000)),
        "--n-epochs", str(train_cfg.get("n_epochs", 4)),
        "--minibatch-size", str(train_cfg.get("minibatch_size", 2048)),
        "--lr", str(train_cfg.get("lr", 3e-4)),
        "--gamma", str(train_cfg.get("gamma", 0.999)),
        "--gae-lambda", str(train_cfg.get("gae_lambda", 0.95)),
        "--clip-eps", str(train_cfg.get("clip_eps", 0.2)),
        "--vf-coef", str(train_cfg.get("vf_coef", 0.5)),
        "--ent-coef", str(train_cfg.get("ent_coef", 0.0)),
        "--action-repeat", str(train_cfg.get("action_repeat", 10)),
        "--save-every", str(train_cfg.get("save_every", 50)),
        "--max-grad-norm", str(train_cfg.get("max_grad_norm", 0.5)),
        "--sampling", sp_cfg.get("sampling", "pfsp"),
        "--pool-snapshot-every", str(sp_cfg.get("pool_snapshot_every", 50)),
        "--pool-max-size", str(sp_cfg.get("pool_max_size", 30)),
        "--checkpoint-dir", ckpt_dir,
        "--log-dir", log_dir,
        "--pool-dir", pool_dir,
        "--run-name", exp_name,
    ]

    # Scripted eval config
    if "scripted_eval_every" in sp_cfg:
        cmd.extend(["--scripted-eval-every", str(sp_cfg["scripted_eval_every"])])
    if "scripted_eval_opponent" in sp_cfg:
        cmd.extend(["--scripted-eval-opponent", sp_cfg["scripted_eval_opponent"]])
    if "scripted_eval_matches" in sp_cfg:
        cmd.extend(["--scripted-eval-matches", str(sp_cfg["scripted_eval_matches"])])
    if "regression_threshold" in sp_cfg:
        cmd.extend(["--regression-threshold", str(sp_cfg["regression_threshold"])])

    # Reward weights
    for key, flag in [
        ("damage_dealt", "--w-damage-dealt"), ("damage_taken", "--w-damage-taken"),
        ("win", "--w-win"), ("lose", "--w-lose"),
        ("approach", "--w-approach"), ("alive", "--w-alive"),
        ("proximity", "--w-proximity"), ("facing", "--w-facing"),
    ]:
        if key in reward_cfg:
            cmd.extend([flag, str(reward_cfg[key])])

    # Bootstrap
    bootstrap = config.get("_bootstrap", sp_cfg.get("bootstrap", ""))
    if bootstrap and bootstrap != "curriculum_final":
        cmd.extend(["--bootstrap", bootstrap])

    print(f"=== {exp_name} (self-play) ===")
    print(f"Model: {model_cfg.get('hidden', 384)}h / {model_cfg.get('n_blocks', 3)}b")
    # Note: training script sends its own Slack notifications using --run-name

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd="/app", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        print(line, end="")

    proc.wait()
    elapsed = time.time() - t0

    # Capture eval output
    eval_output = ""
    final_ckpt = f"{ckpt_dir}/final.pt"
    if os.path.exists(final_ckpt):
        eval_proc = subprocess.run(
            [
                "python", "/app/training/eval.py", final_ckpt,
                "--matches", "50",
                "--hidden", str(model_cfg.get("hidden", 384)),
                "--n-blocks", str(model_cfg.get("n_blocks", 3)),
            ],
            cwd="/app", capture_output=True, text=True, env=env,
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
# Unified curriculum → self-play training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    cpu=32,
    memory=16384,
    timeout=36000,  # 10 hours for full pipeline
    volumes={"/results": vol},
)
def train_unified_modal(config: dict) -> dict:
    """Run a unified curriculum → self-play training pipeline on Modal.

    Args:
        config: Full experiment config dict with curriculum, transition, selfplay sections.
    """
    import subprocess
    import time
    import json

    ts = int(time.time())
    tag = config.get("_tag", "unified")
    exp_name = _make_name(tag, ts)
    exp_dir = f"/results/unified/{exp_name}"
    ckpt_dir = f"{exp_dir}/checkpoints"
    log_dir = f"{exp_dir}/runs"
    pool_dir = f"{exp_dir}/pool"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)

    # Save config
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write config to temp file for train_unified.py --config
    config_path = f"{exp_dir}/run_config.json"
    run_config = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    env = os.environ.copy()
    env["SLACK_WEBHOOK_URL"] = SLACK_WEBHOOK

    cmd = [
        "python", "-u", "/app/training/train_unified.py",
        "--config", config_path,
        "--checkpoint-dir", ckpt_dir,
        "--log-dir", log_dir,
        "--pool-dir", pool_dir,
        "--run-name", exp_name,
    ]

    # Resume if specified
    resume = config.get("_resume", "")
    if resume:
        cmd.extend(["--resume", resume])

    # GPU sim flag
    if config.get("_gpu_sim", False):
        cmd.append("--gpu-sim")

    model_cfg = config.get("model", {})
    print(f"=== {exp_name} (unified) ===")
    print(f"Model: {model_cfg.get('hidden', 384)}h / {model_cfg.get('n_blocks', 3)}b")
    # Note: training script sends its own Slack notifications using --run-name

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd="/app", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        print(line, end="")

    proc.wait()
    elapsed = time.time() - t0

    # Capture eval output
    eval_output = ""
    final_ckpt = f"{ckpt_dir}/final.pt"
    if os.path.exists(final_ckpt):
        eval_proc = subprocess.run(
            [
                "python", "/app/training/eval.py", final_ckpt,
                "--matches", "50",
                "--hidden", str(model_cfg.get("hidden", 384)),
                "--n-blocks", str(model_cfg.get("n_blocks", 3)),
            ],
            cwd="/app", capture_output=True, text=True, env=env,
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
# GPU Sim parity test + benchmark
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=16384,
    timeout=600,
)
def gpu_sim_test() -> dict:
    """Run GPU sim parity tests and benchmark on Modal."""
    import subprocess
    import time

    t0 = time.time()
    proc = subprocess.run(
        ["python", "-m", "gpu_sim.test_parity"],
        cwd="/app/training",
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - t0

    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Result download helpers
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/results": vol})
def list_results() -> list[str]:
    """List all experiment directories on the Modal volume."""
    vol.reload()
    results = []
    if os.path.exists("/results"):
        for d in sorted(os.listdir("/results")):
            path = f"/results/{d}"
            if os.path.isdir(path) and d != "selfplay":
                results.append(d)
    # Also list self-play experiments
    sp_dir = "/results/selfplay"
    if os.path.exists(sp_dir):
        for d in sorted(os.listdir(sp_dir)):
            if os.path.isdir(f"{sp_dir}/{d}"):
                results.append(f"selfplay/{d}")
    # Also list unified experiments
    uni_dir = "/results/unified"
    if os.path.exists(uni_dir):
        for d in sorted(os.listdir(uni_dir)):
            if os.path.isdir(f"{uni_dir}/{d}"):
                results.append(f"unified/{d}")
    return results


@app.function(image=image, volumes={"/results": vol})
def get_result_files(exp_name: str) -> dict[str, bytes]:
    """Download key files from an experiment directory."""
    vol.reload()
    exp_dir = f"/results/{exp_name}"
    if not os.path.exists(exp_dir):
        return {}

    files = {}
    # Grab config, eval, and the final checkpoint
    for name in ["config.json", "eval.txt"]:
        path = f"{exp_dir}/{name}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                files[name] = f.read()

    ckpt_dir = f"{exp_dir}/checkpoints"
    if os.path.isdir(ckpt_dir):
        for fname in os.listdir(ckpt_dir):
            path = f"{ckpt_dir}/{fname}"
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    files[f"checkpoints/{fname}"] = f.read()

    # Include pool metadata for self-play experiments
    pool_meta = f"{exp_dir}/pool/pool.json"
    if os.path.exists(pool_meta):
        with open(pool_meta, "rb") as f:
            files["pool/pool.json"] = f.read()

    return files


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    n_envs: int = 128,
    n_steps: int = 2048,
    num_updates: int = 500,
    lr: float = 3e-4,
    ent_coef: float = 0.0,
    sweep: bool = False,
    smoke_test: bool = False,
    production: bool = False,
    selfplay: bool = False,
    unified: bool = False,
    gpu_sim_test_flag: bool = False,
    config: str = "",
    bootstrap: str = "",
    download: str = "",
    tag: str = "",
):
    import json as _json

    # ----- GPU Sim test -----
    if gpu_sim_test_flag:
        print("Running GPU sim parity tests on Modal...")
        result = gpu_sim_test.remote()
        print(result["stdout"])
        if result["stderr"]:
            print("STDERR:", result["stderr"])
        status = "PASS" if result["exit_code"] == 0 else "FAIL"
        print(f"\nGPU sim test: {status} ({result['elapsed_seconds']}s)")
        return

    # ----- Download results -----
    if download:
        if download == "list":
            experiments = list_results.remote()
            if not experiments:
                print("No experiments found on Modal volume.")
                return
            print(f"Found {len(experiments)} experiments:")
            for name in experiments:
                print(f"  {name}")
            return

        # Download a specific experiment
        print(f"Downloading {download}...")
        files = get_result_files.remote(download)
        if not files:
            print(f"Experiment '{download}' not found.")
            return

        out_dir = f"training/modal_results/{download}"
        os.makedirs(out_dir, exist_ok=True)
        for name, data in files.items():
            path = f"{out_dir}/{name}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            size = len(data)
            print(f"  {name} ({size:,} bytes)")
        print(f"\nSaved to {out_dir}/")
        return

    # ----- Config-driven run -----
    if config:
        with open(config) as f:
            cfg = _json.load(f)

        script = cfg.get("script", "train")
        if not tag:
            tag = os.path.splitext(os.path.basename(config))[0]
        cfg["_tag"] = tag
        if bootstrap:
            cfg["_bootstrap"] = bootstrap

        if script == "train_unified":
            print(f"Launching unified pipeline from config: {config}")
            result = train_unified_modal.remote(cfg)
        elif script == "train_selfplay":
            print(f"Launching self-play from config: {config}")
            result = train_selfplay.remote(cfg)
        else:
            train_cfg = cfg.get("training", {})
            print(f"Launching curriculum training from config: {config}")
            result = train.remote(
                n_envs=train_cfg.get("n_envs", 128),
                n_steps=train_cfg.get("n_steps", 2048),
                num_updates=train_cfg.get("num_updates", 500),
                lr=train_cfg.get("lr", 3e-4),
                ent_coef=train_cfg.get("ent_coef", 0.0),
                tag=tag,
            )
        print(f"\nDone: {result['experiment']} in {result['elapsed_seconds']}s")
        if result.get("eval"):
            print(result["eval"])
        return

    # ----- Unified curriculum → self-play -----
    if unified:
        default_uni_config = {
            "script": "train_unified",
            "model": {"hidden": 384, "n_blocks": 3},
            "training": {
                "n_envs": n_envs if n_envs != 128 else 256,
                "n_steps": n_steps,
                "lr": lr, "gamma": 0.999, "gae_lambda": 0.95,
                "clip_eps": 0.2, "vf_coef": 0.5, "ent_coef": 0.0,
                "action_repeat": 10, "n_epochs": 4, "minibatch_size": 2048,
                "max_grad_norm": 0.5, "save_every": 50,
            },
            "curriculum": {
                "updates": 500,
                "schedule": {
                    "0": ["do_nothing"],
                    "50": ["do_nothing", "dogfighter"],
                    "150": ["dogfighter", "chaser"],
                    "300": ["chaser", "ace"],
                    "450": ["ace", "brawler"],
                },
            },
            "transition": {
                "updates": 200,
                "start_scripted_fraction": 1.0,
                "end_scripted_fraction": 0.2,
                "reset_std": -1.0,
            },
            "selfplay": {
                "updates": 1500,
                "scripted_fraction": 0.2,
                "scripted_pool": ["ace", "brawler"],
                "sampling": "pfsp",
                "pool_snapshot_every": 50,
                "pool_max_size": 30,
                "scripted_eval_every": 20,
                "scripted_eval_opponent": "brawler",
                "scripted_eval_matches": 20,
                "regression_threshold": 0.8,
            },
            "rewards": {
                "damage_dealt": 3.0, "damage_taken": -1.0,
                "win": 5.0, "lose": -5.0,
                "approach": 0.0001, "alive": 0.0,
                "proximity": 0.001, "facing": 0.0005,
            },
        }

        if smoke_test:
            default_uni_config["training"]["n_envs"] = 4
            default_uni_config["training"]["n_steps"] = 64
            default_uni_config["curriculum"]["updates"] = 3
            default_uni_config["transition"]["updates"] = 2
            default_uni_config["selfplay"]["updates"] = 3
            default_uni_config["selfplay"]["scripted_eval_every"] = 2
            default_uni_config["selfplay"]["pool_snapshot_every"] = 2
            default_uni_config["_tag"] = tag or "unified-smoke"
            print("Running unified smoke test...")
        else:
            default_uni_config["_tag"] = tag or "unified"
            total = (default_uni_config["curriculum"]["updates"]
                     + default_uni_config["transition"]["updates"]
                     + default_uni_config["selfplay"]["updates"])
            print(f"Launching unified pipeline ({total} total updates)...")

        result = train_unified_modal.remote(default_uni_config)
        status_str = "PASS" if result["exit_code"] == 0 else "FAIL"
        print(f"\n{'Smoke test' if smoke_test else 'Unified'}: {status_str} — {result['experiment']} in {result['elapsed_seconds']}s")
        if result.get("eval"):
            print(result["eval"])
        return

    # ----- Self-play -----
    if selfplay:
        default_sp_config = {
            "script": "train_selfplay",
            "model": {"hidden": 384, "n_blocks": 3},
            "training": {
                "n_envs": n_envs if n_envs != 128 else 256,
                "n_steps": n_steps,
                "num_updates": num_updates if num_updates != 500 else 2000,
                "lr": lr, "gamma": 0.999, "gae_lambda": 0.95,
                "clip_eps": 0.2, "vf_coef": 0.5, "ent_coef": 0.0,
                "action_repeat": 10, "n_epochs": 4, "minibatch_size": 2048,
                "max_grad_norm": 0.5, "save_every": 50,
            },
            "selfplay": {
                "sampling": "pfsp",
                "pool_snapshot_every": 50,
                "pool_max_size": 30,
                "scripted_eval_every": 20,
                "scripted_eval_opponent": "brawler",
                "scripted_eval_matches": 20,
                "regression_threshold": 0.8,
            },
            "rewards": {
                "damage_dealt": 3.0, "damage_taken": -1.0,
                "win": 5.0, "lose": -5.0,
                "approach": 0.0001, "alive": 0.0,
                "proximity": 0.001, "facing": 0.0005,
            },
        }

        if smoke_test:
            default_sp_config["training"]["n_envs"] = 4
            default_sp_config["training"]["n_steps"] = 64
            default_sp_config["training"]["num_updates"] = 5
            default_sp_config["selfplay"]["scripted_eval_every"] = 2
            default_sp_config["selfplay"]["pool_snapshot_every"] = 2
            default_sp_config["_tag"] = tag or "selfplay-smoke"
            print("Running self-play smoke test (5 updates, 4 envs)...")
        else:
            default_sp_config["_tag"] = tag or "selfplay"
            print(f"Launching self-play training ({default_sp_config['training']['n_envs']} envs, {default_sp_config['training']['num_updates']} updates)...")

        if bootstrap:
            default_sp_config["_bootstrap"] = bootstrap

        result = train_selfplay.remote(default_sp_config)
        status_str = "PASS" if result["exit_code"] == 0 else "FAIL"
        print(f"\n{'Smoke test' if smoke_test else 'Self-play'}: {status_str} — {result['experiment']} in {result['elapsed_seconds']}s")
        if result.get("eval"):
            print(result["eval"])
        return

    # ----- Smoke test (curriculum) -----
    if smoke_test:
        print("Running smoke test (3 updates, 4 envs, 64 steps)...")
        result = train.remote(
            n_envs=4, n_steps=64, num_updates=3,
            save_every=1, tag="smoke",
        )
        status = "PASS" if result["exit_code"] == 0 else "FAIL"
        print(f"\nSmoke test: {status} ({result['elapsed_seconds']}s)")
        if result.get("eval"):
            print(result["eval"])
        return

    # ----- Production run -----
    if production:
        print("Launching production run (128 envs, 2048 steps, 750 updates)...")
        result = train.remote(
            n_envs=128, n_steps=2048, num_updates=750,
            save_every=50, tag="production",
        )
        print(f"\nDone: {result['experiment']} in {result['elapsed_seconds']}s")
        if result.get("eval"):
            print(result["eval"])
        return

    # ----- Sweep -----
    if sweep:
        # Scaling sweep — all use ent_coef=0.0 (proven best)
        configs = [
            {"n_envs": 64,  "n_steps": 2048, "num_updates": 750, "tag": "scale-64env"},
            {"n_envs": 128, "n_steps": 2048, "num_updates": 750, "tag": "scale-128env"},
            {"n_envs": 256, "n_steps": 2048, "num_updates": 750, "tag": "scale-256env"},
            {"n_envs": 128, "n_steps": 1024, "num_updates": 1000, "tag": "scale-1024steps"},
            {"n_envs": 128, "n_steps": 4096, "num_updates": 500, "tag": "scale-4096steps"},
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
        # ----- Single run -----
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
