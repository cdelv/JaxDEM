# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Identity bijector that applies no constraint."""

from functools import partial

import distrax  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
from distrax._src.bijectors.bijector import Array  # type: ignore[import-untyped]

from . import ActionSpace


@ActionSpace.register("Free")
class FreeSpace(distrax.Bijector, ActionSpace):  # type: ignore[misc]
    r"""Identity constraint (no transform).

    **Mapping**

    .. math::
        y = f(x) = x, \qquad x = f^{-1}(y) = y.

    **Jacobian**

    .. math::
        J_f(x) = I,\qquad \log\lvert\det J_f(x)\rvert = 0, \qquad \log\lvert\det J_{f^{-1}}(y)\rvert = 0.

    Parameters
    ----------
    event_ndims_in : int
        Dimensionality of a *single event* seen by the bijector (default 0 for a scalar transform).
    event_ndims_out : Optional[int]
        Standard Distrax/TFP bijector flag.
    is_constant_jacobian : bool
        Standard Distrax/TFP bijector flag.
    is_constant_log_det : bool
        Standard Distrax/TFP bijector flag.

    Note
    ----------
    This bijector is **scalar** (``event_ndims_in = 0``). For vector actions,
    wrap it with ``distrax.Block(bijector, ndims=1)``. The model applies
    this wrapper automatically.

    """

    __slots__ = ()

    def __init__(
        self,
        event_ndims_in: int = 0,
        event_ndims_out: int | None = None,
        is_constant_jacobian: bool = True,
        is_constant_log_det: bool | None = True,
    ):
        super().__init__(
            event_ndims_in=event_ndims_in,
            event_ndims_out=event_ndims_out,
            is_constant_jacobian=is_constant_jacobian,
            is_constant_log_det=is_constant_log_det,
        )

    @partial(jax.named_call, name="FreeSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> tuple[Array, jax.Array]:
        # log|det J| = 0 for identity; shape matches x for a scalar bijector
        return x, jnp.zeros_like(x)

    @partial(jax.named_call, name="FreeSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> tuple[Array, jax.Array]:
        # inverse is identity; log|det J_inv| = 0
        return y, jnp.zeros_like(y)

    def log_det_expectation(self, mean: jax.Array, std: jax.Array) -> jax.Array:
        return jnp.zeros(mean.shape[:-1])

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is FreeSpace  # pylint: disable=unidiomatic-typecheck


__all__ = ["FreeSpace"]
