# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""JaxDEM module."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax

if not hasattr(jax.tree, "static"):

    def _static(*args: Any, **kwargs: Any) -> Any:
        metadata = dict(kwargs.get("metadata", {}))
        metadata["static"] = True
        kwargs["metadata"] = metadata
        return dataclasses.field(*args, **kwargs)

    jax.tree.static = _static

# import os
# os.environ[
#     "XLA_FLAGS"
# ] = """
#     --xla_gpu_enable_latency_hiding_scheduler=true
#     --xla_gpu_enable_command_buffer=''
#     --xla_disable_hlo_passes=collective-permute-motion
#     --xla_gpu_experimental_pipeline_parallelism_opt_level=PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE
# """
from . import utils
from .bonded_forces import BondedForceModel
from .colliders import Collider
from .domains import Domain
from .factory import Factory
from .forces import (
    ForceManager,
    ForceModel,
    ForceRouter,
    LawCombiner,
)
from .integrators import Integrator, LinearIntegrator, RotationIntegrator
from .material_matchmakers import MaterialMatchmaker
from .materials import Material, MaterialTable
from .minimizers import damped_newtonian, fire, minimize
from .state import State
from .system import System
from .writers import (
    CheckpointLoader,
    CheckpointModelLoader,
    CheckpointModelWriter,
    CheckpointWriter,
    VTKBaseWriter,
    VTKWriter,
)

__all__ = [
    "BondedForceModel",
    "CheckpointLoader",
    "CheckpointModelLoader",
    "CheckpointModelWriter",
    "CheckpointWriter",
    "Collider",
    "Domain",
    "Factory",
    "ForceManager",
    "ForceModel",
    "ForceRouter",
    "Integrator",
    "LawCombiner",
    "LinearIntegrator",
    "Material",
    "MaterialMatchmaker",
    "MaterialTable",
    "minimize",
    "fire",
    "damped_newtonian",
    "RotationIntegrator",
    "State",
    "System",
    "VTKBaseWriter",
    "VTKWriter",
    "utils",
]
