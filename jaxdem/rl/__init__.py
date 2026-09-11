# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Experimental reinforcement-learning models, environments, and trainers.

RL has a separate regression matrix and a narrower compatibility promise than
the DEM core. See the RL support contract in the user guide.
"""

from __future__ import annotations

from .environments import Environment
from .env_wrappers import vectorise_env, clip_action_env, is_wrapped, unwrap
from .models import Model
from .trainers import Trainer, TrajectoryData
from .action_spaces import ActionSpace

__all__ = [
    "ActionSpace",
    "Environment",
    "Model",
    "Trainer",
    "TrajectoryData",
    "clip_action_env",
    "is_wrapped",
    "unwrap",
    "vectorise_env",
]
