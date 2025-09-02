# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining bijectors that constraint the action spaceof reinforcement learning policy distributions.
"""

import jax
import jax.numpy as jnp

from typing import Tuple, Optional

import distrax
from distrax._src.bijectors.bijector import Array

from ..factory import Factory


class ActionSpace(Factory):
    """
    Base class for registering bijectors (acts as namespace)
    """


@ActionSpace.register("Free")
class FreeSpace(distrax.Bijector, ActionSpace):
    """
    Identity (no constraints). Scalar bijector (event_ndims_in=0).
    Wrap with Block(ndims=1) for vector actions.
    """

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

    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        # log|det J| = 0 for identity; shape matches x for a scalar bijector
        return x, jnp.zeros_like(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        # inverse is identity; log|det J_inv| = 0
        return y, jnp.zeros_like(y)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is FreeSpace  # pylint: disable=unidiomatic-typecheck


@ActionSpace.register("Box")
class BoxSpace(distrax.Bijector, ActionSpace):
    """
    Elementwise box constraint via tanh + affine:

        y = center + half * tanh(x)

    Scalar bijector (event_ndims_in=0). Wrap with Block(ndims=1) for vectors.
    """

    # @classmethod
    # def Create(
    #     cls,
    #     x_min: Array,
    #     x_max: Array,
    #     eps: float = 1e-12,
    #     event_ndims_in: int = 0,
    #     event_ndims_out: Optional[int] = None,
    #     is_constant_jacobian: bool = False,
    #     is_constant_log_det: Optional[bool] = None,
    # ):
    #     return cls(
    #         x_min=x_min,
    #         x_max=x_max,
    #         eps=eps,
    #         event_ndims_in=event_ndims_in,
    #         event_ndims_out=event_ndims_out,
    #         is_constant_jacobian=is_constant_jacobian,
    #         is_constant_log_det=is_constant_log_det,
    #     )

    def __init__(
        self,
        x_min: Array,
        x_max: Array,
        eps: float = 1e-12,
        event_ndims_in: int = 0,
        event_ndims_out: Optional[int] = None,
        is_constant_jacobian: bool = False,
        is_constant_log_det: Optional[bool] = None,
    ):
        super().__init__(
            event_ndims_in=event_ndims_in,
            event_ndims_out=event_ndims_out,
            is_constant_jacobian=is_constant_jacobian,
            is_constant_log_det=is_constant_log_det,
        )
        x_min = jnp.asarray(x_min, dtype=float)
        x_max = jnp.asarray(x_max, dtype=float)
        if not jnp.all(x_max > x_min):
            raise ValueError("Box: require x_max > x_min elementwise.")

        self.center = (x_min + x_max) / 2.0
        self.half = (x_max - x_min) / 2.0
        self.eps = float(eps)

    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        """
        Computes log|det J(f)(x)|.
        log|dy/dx| = log|half| + log(sech^2 x)
        Stable log(sech^2 x) = 2*(log(2) - x - softplus(-2x))
        """
        return jnp.log(jnp.abs(self.half) + self.eps) + 2 * (
            jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x)
        )

    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.center + self.half * jnp.tanh(x)
        return y, self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        u = (y - self.center) / (self.half + self.eps)
        u = jnp.clip(u, -1.0 + self.eps, 1.0 - self.eps)
        x = jnp.arctanh(u)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is BoxSpace  # pylint: disable=unidiomatic-typecheck
