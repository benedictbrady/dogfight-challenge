"""Opponent pool manager for self-play training.

Manages a pool of past model checkpoints with ELO ratings and various
sampling strategies (PFSP, uniform, latest, mixed).
"""

import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch


@dataclass
class PoolEntry:
    """A single opponent in the pool."""
    name: str
    path: str  # relative path within pool_dir
    update_num: int
    elo: float = 1000.0
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.5


class OpponentPool:
    """Manages a pool of opponent checkpoints for self-play training.

    Usage:
        pool = OpponentPool("training/pool", device, hidden=384, n_blocks=3)
        pool.add_checkpoint(model, "sp_0000", update_num=0)

        entry = pool.sample_opponent(method="pfsp")
        opponent_model = pool.load_opponent(entry)

        pool.update_elo(entry, learner_elo=1200.0, won=True)
        pool.save_metadata()
    """

    ELO_K = 32  # Standard ELO K-factor
    METADATA_FILE = "pool.json"

    def __init__(
        self,
        pool_dir: str,
        device: torch.device,
        hidden: int = 384,
        n_blocks: int = 3,
        max_size: int = 30,
        cache_size: int = 5,
    ):
        self.pool_dir = Path(pool_dir)
        self.device = device
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.max_size = max_size
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._cache_size = cache_size
        self.entries: list[PoolEntry] = []

        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata if present
        meta_path = self.pool_dir / self.METADATA_FILE
        if meta_path.exists():
            self.load_metadata()

    def add_checkpoint(self, model, name: str, update_num: int):
        """Save a model snapshot to the pool.

        Args:
            model: ActorCritic model (will save state_dict)
            name: Checkpoint name (e.g., "sp_0050")
            update_num: Training update number when snapshot was taken
        """
        # Check if name already exists
        for entry in self.entries:
            if entry.name == name:
                return  # Already in pool

        # Evict least-played if at capacity
        if len(self.entries) >= self.max_size:
            self._evict_least_played()

        # Unwrap torch.compile wrapper if present
        raw_model = getattr(model, "_orig_mod", model)
        filename = f"{name}.pt"
        filepath = self.pool_dir / filename
        torch.save({"model": raw_model.state_dict()}, filepath)

        entry = PoolEntry(
            name=name,
            path=filename,
            update_num=update_num,
        )
        self.entries.append(entry)
        self.save_metadata()

    def sample_opponent(self, method: str = "pfsp") -> PoolEntry:
        """Sample an opponent from the pool.

        Methods:
            pfsp: Prioritized Fictitious Self-Play — weight proportional to
                  (1 - win_rate + eps). Play more against opponents we lose to.
            uniform: Equal probability for all entries.
            latest: Always return the most recent checkpoint.
            mixed: 50% latest, 30% PFSP, 20% random scripted anchor.
        """
        import random

        if not self.entries:
            raise ValueError("Pool is empty — add at least one checkpoint first")

        if method == "latest":
            return self.entries[-1]

        if method == "uniform":
            return random.choice(self.entries)

        if method == "pfsp":
            return self._sample_pfsp()

        if method == "mixed":
            r = random.random()
            if r < 0.5:
                return self.entries[-1]  # latest
            elif r < 0.8:
                return self._sample_pfsp()  # PFSP
            else:
                return random.choice(self.entries)  # uniform

        raise ValueError(f"Unknown sampling method: {method}")

    def _sample_pfsp(self) -> PoolEntry:
        """PFSP sampling: weight ∝ (1 - win_rate + eps)."""
        import random

        eps = 0.1
        weights = [1.0 - e.win_rate + eps for e in self.entries]
        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(self.entries, weights=weights, k=1)[0]

    def load_opponent(self, entry: PoolEntry):
        """Load an opponent model from the pool. Uses LRU cache.

        Returns an ActorCritic model in eval mode.
        """
        # Check cache
        if entry.name in self._cache:
            self._cache.move_to_end(entry.name)
            return self._cache[entry.name]

        # Import here to avoid circular imports at module level
        from model import ActorCritic

        model = ActorCritic(hidden=self.hidden, n_blocks=self.n_blocks).to(self.device)
        ckpt = torch.load(
            self.pool_dir / entry.path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(ckpt["model"])
        model.eval()

        # LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[entry.name] = model

        return model

    def update_elo(self, entry: PoolEntry, learner_elo: float, won: bool, drawn: bool = False):
        """Update ELO rating for an opponent after a match.

        Updates both game counts AND the opponent's ELO symmetrically.
        If the learner won, the opponent's ELO decreases (and vice versa).

        Args:
            entry: The opponent pool entry
            learner_elo: Current learner ELO (needed for symmetric update)
            won: Whether the learner won (from learner's perspective)
            drawn: Whether the match was a draw
        """
        entry.games += 1
        if drawn:
            entry.draws += 1
        elif won:
            entry.wins += 1  # learner won = opponent lost
        else:
            entry.losses += 1  # learner lost = opponent won

        # Symmetric ELO update for the opponent (inverse of learner's outcome)
        expected_opp = 1.0 / (1.0 + 10.0 ** ((learner_elo - entry.elo) / 400.0))
        if drawn:
            actual_opp = 0.5
        elif won:
            actual_opp = 0.0  # opponent lost
        else:
            actual_opp = 1.0  # opponent won
        entry.elo += self.ELO_K * (actual_opp - expected_opp)

    def update_learner_elo(self, learner_elo: float, opponent_entry: PoolEntry, won: bool, drawn: bool = False) -> float:
        """Compute new learner ELO after a match against a pool opponent.

        Args:
            learner_elo: Current learner ELO
            opponent_entry: The opponent played against
            won: Whether the learner won
            drawn: Whether it was a draw

        Returns:
            Updated learner ELO
        """
        expected = 1.0 / (1.0 + 10.0 ** ((opponent_entry.elo - learner_elo) / 400.0))
        if drawn:
            actual = 0.5
        elif won:
            actual = 1.0
        else:
            actual = 0.0

        new_elo = learner_elo + self.ELO_K * (actual - expected)
        return new_elo

    def save_metadata(self):
        """Save pool metadata to pool.json."""
        data = {
            "hidden": self.hidden,
            "n_blocks": self.n_blocks,
            "max_size": self.max_size,
            "entries": [asdict(e) for e in self.entries],
        }
        with open(self.pool_dir / self.METADATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load_metadata(self):
        """Load pool metadata from pool.json."""
        with open(self.pool_dir / self.METADATA_FILE) as f:
            data = json.load(f)

        self.hidden = data.get("hidden", self.hidden)
        self.n_blocks = data.get("n_blocks", self.n_blocks)
        self.max_size = data.get("max_size", self.max_size)
        self.entries = [PoolEntry(**e) for e in data.get("entries", [])]

    def _evict_least_played(self):
        """Remove the least-played entry from the pool."""
        if not self.entries:
            return
        # Never evict the first (bootstrap) or last (most recent) entry
        if len(self.entries) <= 2:
            return
        candidates = self.entries[1:-1]
        least = min(candidates, key=lambda e: e.games)
        # Remove from cache
        self._cache.pop(least.name, None)
        # Remove checkpoint file
        filepath = self.pool_dir / least.path
        if filepath.exists():
            filepath.unlink()
        self.entries.remove(least)

    def should_snapshot(self, update_num: int, snapshot_every: int, elo_jump: float = 0.0, elo_threshold: float = 50.0) -> bool:
        """Decide whether to snapshot the current model to the pool.

        Snapshots on:
        1. Every `snapshot_every` updates
        2. ELO jump of `elo_threshold` or more since last snapshot
        """
        if update_num % snapshot_every == 0:
            return True
        if elo_jump >= elo_threshold:
            return True
        return False

    @property
    def size(self) -> int:
        return len(self.entries)

    def summary(self) -> str:
        """Return a human-readable summary of the pool."""
        lines = [f"Opponent Pool ({self.size}/{self.max_size} entries):"]
        lines.append(f"  {'Name':<16} {'ELO':>6} {'Games':>6} {'WR':>6}")
        lines.append("  " + "-" * 40)
        for e in self.entries:
            wr = f"{e.win_rate:.0%}" if e.games > 0 else "N/A"
            lines.append(f"  {e.name:<16} {e.elo:>6.0f} {e.games:>6} {wr:>6}")
        return "\n".join(lines)
