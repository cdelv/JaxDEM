# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Experimental reinforcement-learning models, environments, and trainers.

RL has a separate regression matrix and a narrower compatibility promise than
the DEM core. See the RL support contract in the user guide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .environments import Environment
from .env_wrappers import (
    PufferEnvWrapper,
    clip_action_env,
    is_wrapped,
    unwrap,
    vectorise_env,
)
from .models import Model
from .trainers import Trainer, TrajectoryData
from .action_spaces import ActionSpace

if TYPE_CHECKING:
    from .trainers import (
        ParetoPoint,
        ProteinOptimizer,
        ProteinSweep,
        PufferTrainer,
        SuggestionInfo,
        SweepParam,
        SweepResult,
        SweepTrial,
        default_sweep_params,
        make_ppo_eval_fn,
    )

_LAZY_TRAINER_EXPORTS = frozenset(
    {
        "ParetoPoint",
        "ProteinOptimizer",
        "ProteinSweep",
        "PufferTrainer",
        "SuggestionInfo",
        "SweepParam",
        "SweepResult",
        "SweepTrial",
        "default_sweep_params",
        "make_ppo_eval_fn",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_TRAINER_EXPORTS:
        from . import trainers

        return getattr(trainers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionSpace",
    "Environment",
    "Model",
    "ParetoPoint",
    "ProteinOptimizer",
    "ProteinSweep",
    "PufferEnvWrapper",
    "PufferTrainer",
    "SuggestionInfo",
    "SweepParam",
    "SweepResult",
    "SweepTrial",
    "Trainer",
    "TrajectoryData",
    "clip_action_env",
    "default_sweep_params",
    "is_wrapped",
    "make_ppo_eval_fn",
    "unwrap",
    "vectorise_env",
]
