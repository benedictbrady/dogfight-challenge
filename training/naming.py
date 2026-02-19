"""Mnemonic run name generator for training experiments.

Generates deterministic human-readable names like 'fierce-hawk-production'
from a tag and timestamp. Used by all training scripts and the Modal wrapper.

Usage:
    from naming import make_run_name

    name = make_run_name("selfplay")       # e.g. "bold-falcon-selfplay"
    name = make_run_name("production")     # e.g. "mean-knight-production"
    name = make_run_name("")               # e.g. "sharp-talon"
"""

import hashlib
import time

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


def make_run_name(tag: str = "", ts: int | None = None) -> str:
    """Generate a mnemonic experiment name like 'fierce-hawk-production'.

    Args:
        tag: Suffix appended after the mnemonic (e.g. "production", "selfplay").
        ts: Unix timestamp for deterministic naming. Defaults to current time.

    Returns:
        Human-readable name like "bold-falcon-selfplay".
    """
    if ts is None:
        ts = int(time.time())
    h = int(hashlib.md5(str(ts).encode()).hexdigest(), 16)
    adj = _ADJECTIVES[h % len(_ADJECTIVES)]
    call = _CALLSIGNS[(h // len(_ADJECTIVES)) % len(_CALLSIGNS)]
    suffix = f"-{tag}" if tag else ""
    return f"{adj}-{call}{suffix}"
