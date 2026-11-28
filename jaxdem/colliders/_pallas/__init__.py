# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Optional Pallas Triton GPU kernels for variable-length collider traversals.

All kernels dispatch through ``jax.lax.platform_dependent`` so CUDA and ROCm
GPUs execute fused variable-length Triton kernels—compiling *any* ``ForceModel``
directly from its JAX ``force`` jaxpr—while CPU, TPU, Metal, unsupported
primitives, and automatic differentiation seamlessly fall back to Pure-JAX.

Control activation via :func:`set_pallas_enabled`, :func:`pallas_enabled`, or
the ``JAXDEM_DISABLE_PALLAS=1`` environment variable.
"""

from __future__ import annotations

from .cell import extract_grid_topology, try_pallas_build_pairs, try_pallas_cell_forces
from .config import (
    has_manual_axes,
    is_pallas_enabled,
    pallas_enabled,
    set_pallas_enabled,
)
from .csr import (
    can_use_pallas_csr_forces,
    pallas_csr_forces,
    pallas_remap_csr_history,
)
from .triton_jaxpr import compile_displacement_spec, compile_force_spec

__all__ = [
    "can_use_pallas_csr_forces",
    "compile_displacement_spec",
    "compile_force_spec",
    "extract_grid_topology",
    "has_manual_axes",
    "is_pallas_enabled",
    "pallas_csr_forces",
    "pallas_enabled",
    "pallas_remap_csr_history",
    "set_pallas_enabled",
    "try_pallas_build_pairs",
    "try_pallas_cell_forces",
]
