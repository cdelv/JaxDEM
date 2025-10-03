# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Implementation of identity bijector for free space.
"""

import jax
import jax.numpy as jnp

from typing import Tuple, Optional, Dict
from functools import partial

import distrax
from distrax._src.bijectors.bijector import Array

from . import ActionSpace


@ActionSpace.register("Free")
class FreeSpace(distrax.Bijector, ActionSpace):
    r"""
    Identity constraint (no transform).

    **Mapping**

    .. math::
        y = f(x) = x, \qquad x = f^{-1}(y) = y.

    **Jacobian**

    .. math::
        J_f(x) = I,\qquad \log\lvert\det J_f(x)\rvert = 0, \qquad \log\lvert\det J_{f^{-1}}(y)\rvert = 0.

    Parameters
    ----------
    -event_ndims_in : int
        dimensionality of a *single event* seen by the bijector (defaults to 0 for a scalar transform).

    -event_ndims_out : Optional[int]
        standard Distrax/TFP bijector flags.

    -is_constant_jacobian : bool
        standard Distrax/TFP bijector flags.

    -is_constant_log_det : bool
        standard Distrax/TFP bijector flags.

    Note
    ----------
    This bijector is **scalar** (``event_ndims_in = 0``). For vector actions,
    needs to be wrap it with ``distrax.Block(bijector, ndims=1)`. Let the model do that for you!
    """

    __slots__ = ()

    def __init__(
        self,
        event_ndims_in: int = 0,
        event_ndims_out: Optional[int] = None,
        is_constant_jacobian: bool = True,
        is_constant_log_det: Optional[bool] = True,
    ):
        super().__init__(
            event_ndims_in=event_ndims_in,
            event_ndims_out=event_ndims_out,
            is_constant_jacobian=is_constant_jacobian,
            is_constant_log_det=is_constant_log_det,
        )

    @property
    def kws(self) -> Dict:
        return dict(
            event_ndims_in=self.event_ndims_in,
            event_ndims_out=self.event_ndims_out,
            is_constant_jacobian=self.is_constant_jacobian,
            is_constant_log_det=self.is_constant_log_det,
        )

    @partial(jax.named_call, name="FreeSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> Tuple[Array, jax.Array]:
        # log|det J| = 0 for identity; shape matches x for a scalar bijector
        return x, jnp.zeros_like(x)

    @partial(jax.named_call, name="FreeSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> Tuple[Array, jax.Array]:
        # inverse is identity; log|det J_inv| = 0
        return y, jnp.zeros_like(y)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is FreeSpace  # pylint: disable=unidiomatic-typecheck


__all__ = ["FreeSpace"]
