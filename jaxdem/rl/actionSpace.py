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

    __slots__ = ()


@ActionSpace.register("Free")
class FreeSpace(distrax.Bijector, ActionSpace):
    """
    Identity (no constraints). Scalar bijector (event_ndims_in=0).
    Wrap with Block(ndims=1) for vector actions.
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

    def forward_and_log_det(self, x: Array) -> Tuple[Array, jax.Array]:
        # log|det J| = 0 for identity; shape matches x for a scalar bijector
        return x, jnp.zeros_like(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, jax.Array]:
        # inverse is identity; log|det J_inv| = 0
        return y, jnp.zeros_like(y)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is FreeSpace  # pylint: disable=unidiomatic-typecheck


@ActionSpace.register("Box")
class BoxSpace(distrax.Bijector, ActionSpace):
    """
    Elementwise box constraint via tanh:

        y = center + half * tanh(x/w)

    Scalar bijector (event_ndims_in=0). Wrap with Block(ndims=1) for vectors.
    """

    __slots__ = ()

    def __init__(
        self,
        x_min: Array,
        x_max: Array,
        width: float = 1.0,
        eps: float = 1e-6,
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
        self.half = (1.0 - eps) * (x_max - x_min) / 2.0
        self.width = width
        self.eps = float(eps)

    @staticmethod
    def sec2_log(x):
        return 2 * (jnp.log(2) - x - jax.nn.softplus(-2.0 * x))

    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        """
        Computes log|det J(f)(x)|.
        log|dy/dx| = log|half| + log(sech^2 x)
        Stable log(sech^2 x) = 2*(log(2) - x - softplus(-2x))
        """
        return jnp.log(self.half) + self.sec2_log(x / self.width) - jnp.log(self.width)

    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.center + self.half * jnp.tanh(x / self.width)
        return y, self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        u = (y - self.center) / (self.half + self.eps)
        u = u.clip(-1.0 + self.eps, 1.0 - self.eps)
        x = self.width * jnp.arctanh(u)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is BoxSpace  # pylint: disable=unidiomatic-typecheck


@ActionSpace.register("MaxNorm")
class MaxNormSpace(distrax.Bijector, ActionSpace):
    """
    Radial max-norm constraint for vector actions:
        y = max_norm * tanh(||x||)

    """

    __slots__ = ()

    def __init__(
        self,
        max_norm: float = 1.0,
        eps: float = 1e-6,
        event_ndims_in: int = 1,
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
        self.eps = float(eps)
        self.max_norm = float(max_norm)
        self.max_norm = (1.0 - self.eps) * self.max_norm

    @staticmethod
    def sec2_log(r):
        # r is scalar radius
        return 2 * (jnp.log(2.0) - r - jax.nn.softplus(-2.0 * r))

    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        r = jnp.linalg.norm(x, axis=-1)  # shape (...,)
        x = jnp.atleast_1d(x)  # ensures x.ndim >= 1
        d = jnp.asarray(x.shape[-1], x.dtype)  # scalar, works under jit

        # Stable pieces
        log_s = jnp.log(self.max_norm + self.eps)
        log_tanh_r = jnp.log(jnp.tanh(r) + self.eps)
        log_r = jnp.log(r + self.eps)
        log_sech2_r = MaxNormSpace.sec2_log(r)

        main = d * log_s + (d - 1.0) * (log_tanh_r - log_r) + log_sech2_r
        small = d * log_s - (2.0 / 3.0) * (r * r)
        return jnp.where(r < self.eps, small, main)

    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        unit = jnp.where(r > 0.0, x / r, jnp.zeros_like(x))
        y = self.max_norm * jnp.tanh(r) * unit
        return y, self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        r = jnp.linalg.norm(y, axis=-1, keepdims=True)
        u = (r / self.max_norm).clip(-1.0 + self.eps, 1.0 - self.eps)
        unit = jnp.where(r > 0.0, y / r, jnp.zeros_like(y))
        x = jnp.arctanh(u) * unit
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is MaxNormSpace
