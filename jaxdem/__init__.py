# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""JaxDEM module."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _package_version
from typing import TYPE_CHECKING, Any

try:
    __version__ = _package_version("JaxDEM")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

from . import utils
from .bonded_forces import BondedForceModel
from .colliders import (
    Collider,
    is_pallas_enabled,
    pallas_enabled,
    set_pallas_enabled,
)
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
from .state import BodyTopology, State, body_topology
from .system import System

if TYPE_CHECKING:
    from .writers import (
        CheckpointLoader as CheckpointLoader,
        CheckpointModelLoader as CheckpointModelLoader,
        CheckpointModelWriter as CheckpointModelWriter,
        CheckpointWriter as CheckpointWriter,
        H5Writer as H5Writer,
        VTKBaseWriter,
        VTKWriter,
    )


__all__ = [
    "__version__",
    "BondedForceModel",
    "BodyTopology",
    "Collider",
    "Domain",
    "Factory",
    "ForceManager",
    "ForceModel",
    "ForceRouter",
    "H5Writer",
    "Integrator",
    "LawCombiner",
    "LinearIntegrator",
    "Material",
    "MaterialMatchmaker",
    "MaterialTable",
    "is_pallas_enabled",
    "pallas_enabled",
    "set_pallas_enabled",
    "minimize",
    "fire",
    "damped_newtonian",
    "RotationIntegrator",
    "State",
    "System",
    "VTKBaseWriter",
    "VTKWriter",
    "utils",
    "body_topology",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CheckpointLoader",
        "CheckpointModelLoader",
        "CheckpointModelWriter",
        "CheckpointWriter",
        "H5Writer",
        "VTKBaseWriter",
        "VTKWriter",
    }:
        from . import writers

        value = getattr(writers, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
