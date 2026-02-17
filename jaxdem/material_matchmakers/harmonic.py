# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Harmonic averaging material matchmaker."""

from __future__ import annotations

import jax

from dataclasses import dataclass
from functools import partial

from . import MaterialMatchmaker


@MaterialMatchmaker.register("harmonic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class HarmonicMaterialMatchmaker(MaterialMatchmaker):
    r"""
    A `MaterialMatchmaker` implementation that computes the effective property
    as the harmonic mean of two properties:

    .. math::
        P_{eff} = \frac{2}{\frac{1}{P_1} + \frac{1}{P_2}}

    where :math:`P_1` and :math:`P_2` are the property values from the two materials.
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="HarmonicMaterialMatchmaker.get_effective_property")
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return 2.0 / (1.0 / prop1 + 1.0 / prop2)


__all__ = ["HarmonicMaterialMatchmaker"]
