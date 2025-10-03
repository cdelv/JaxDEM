# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
JaxDEM module
"""

from __future__ import annotations

# import os

# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_latency_hiding_scheduler=true "
#     "--xla_gpu_enable_command_buffer='' "
#     "--xla_disable_hlo_passes=collective-permute-motion "
#     "--xla_gpu_experimental_pipeline_parallelism_opt_level=PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE "
# )

from .state import State
from .system import System
from .writers import (
    VTKWriter,
    VTKBaseWriter,
    CheckpointWriter,
    CheckpointLoader,
    CheckpointModelWriter,
    CheckpointModelLoader,
)
from .materials import Material, MaterialTable
from .material_matchmakers import MaterialMatchmaker
from .forces import ForceModel, ForceRouter, LawCombiner, ForceManager
from .integrators import Integrator
from .colliders import Collider
from .domains import Domain
from .factory import Factory

__all__ = [
    "State",
    "System",
    "VTKWriter",
    "VTKBaseWriter",
    "CheckpointWriter",
    "CheckpointLoader",
    "CheckpointModelWriter",
    "CheckpointModelLoader",
    "Material",
    "MaterialTable",
    "MaterialMatchmaker",
    "ForceModel",
    "Integrator",
    "Collider",
    "Domain",
    "Factory",
    "ForceRouter",
    "LawCombiner",
    "ForceManager",
]
