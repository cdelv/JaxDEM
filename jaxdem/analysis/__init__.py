# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Post-processing and analysis utilities.

This subpackage provides a binned-accumulation engine for JAX:

- Bin specifications (time bins, lag bins) in :mod:`jaxdem.analysis.bins`
- Flattening bins to index-pairs in :mod:`jaxdem.analysis.pairs`
- A JAX engine (vmap + segment_sum) in :mod:`jaxdem.analysis.engine`
- Example kernels in :mod:`jaxdem.analysis.kernels`
"""

from __future__ import annotations

from .bins import (
    BinSpec,
    TimeBins,
    LagBinsExact,
    LagBinsLinear,
    LagBinsLog,
    LagBinsPseudoLog,
)
from .engine import Binned, evaluate_binned
from .kernels import (
    KernelFn,
    msd_kernel,
    isf_self_isotropic_kernel,
    unwrap_angles_2d,
    msad_kernel_2d,
    isf_angular_kernel_2d,
)
from .pairs import Pairs, build_pairs

__all__ = [
    "BinSpec",
    "Binned",
    "KernelFn",
    "LagBinsExact",
    "LagBinsLinear",
    "LagBinsLog",
    "LagBinsPseudoLog",
    "Pairs",
    "TimeBins",
    "build_pairs",
    "evaluate_binned",
    "msd_kernel",
    "isf_self_isotropic_kernel",
    "unwrap_angles_2d",
    "msad_kernel_2d",
    "isf_angular_kernel_2d",
]
