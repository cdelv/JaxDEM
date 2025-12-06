# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Jamming routines.
https://doi.org/10.1103/PhysRevE.68.011306
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import TYPE_CHECKING, Sequence, Optional
from functools import partial

if TYPE_CHECKING:
    from ..state import State

