"""Behavioral cloning: train small MLPs to mimic scripted policies.

Requires the dogfight_pyenv Rust module (crates/pyenv) to be built via maturin.

Usage:
    python behavioral_cloning.py --policy chaser --episodes 2000
    python behavioral_cloning.py --all
"""

# This script requires dogfight_pyenv which is built separately.
# See crates/pyenv/ for the Rust PyO3 environment.
# Full implementation available on the training branch.

print("behavioral_cloning.py requires dogfight_pyenv. Build with: cd crates/pyenv && maturin develop")
