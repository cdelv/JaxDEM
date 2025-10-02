# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Linear averaging material matchmaker."""

from __future__ import annotations

import jax

from dataclasses import dataclass
from functools import partial

from . import MaterialMatchmaker


@MaterialMatchmaker.register("linear")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class LinearMaterialMatchmaker(MaterialMatchmaker):
    r"""
    A `MaterialMatchmaker` implementation that computes the effective property
    as the arithmetic mean (linear average) of two properties:

    .. math::
        P_{eff} = \frac{P_1 + P_2}{2}

    where :math:`P_1` and :math:`P_2` are the property values from the two materials.
    """

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LinearMaterialMatchmaker.get_effective_property")
    @jax.profiler.annotate_function
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return (prop1 + prop2) / 2


__all__ = ["LinearMaterialMatchmaker"]
