# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""One entry point for the matched screen and isolated follow-up variants."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from benchmarks.swarm_mean_shift import run
from benchmarks.swarm_mean_shift.observation_followups import (
    make_followup_environment as make_augmented_environment,
)
from benchmarks.swarm_mean_shift.reward_followups import (
    FOLLOWUP_NAMES,
    make_followup_environment,
)
from benchmarks.swarm_mean_shift.variants import make_environment


def make_any_environment(variant: str, **kwargs: Any) -> Any:
    if variant in FOLLOWUP_NAMES:
        return make_followup_environment(variant, **kwargs)
    if variant == "paper_augmented":
        return make_augmented_environment(variant, **kwargs)
    return make_environment(variant, **kwargs)


def main() -> None:
    original_hashes = run.source_hashes

    def hashes(source_experiment: Path) -> dict[str, str | None]:
        values = original_hashes(source_experiment)
        for name in ("compare.py", "reward_followups.py", "observation_followups.py"):
            path = Path(__file__).with_name(name).resolve()
            values[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return values

    run.make_environment = make_any_environment
    run.VARIANT_NAMES = (*run.VARIANT_NAMES, *FOLLOWUP_NAMES, "paper_augmented")
    run.source_hashes = hashes
    run.main()


if __name__ == "__main__":
    main()
