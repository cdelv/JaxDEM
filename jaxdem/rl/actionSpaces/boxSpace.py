# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Implementation of bijector for box space.
"""

import jax
import jax.numpy as jnp

from typing import Any, Dict, Optional, Tuple
from functools import partial

import distrax
from distrax._src.bijectors.bijector import Array

from . import ActionSpace


@ActionSpace.register("Box")
class BoxSpace(distrax.Bijector, ActionSpace):  # type: ignore[misc]
    r"""
    Elementwise **box** constraint implemented with a scaled `tanh`.

    **Mapping (componentwise)**

    .. math::
        y_i \;=\; c_i + h_i\,\tanh\!\left(\frac{x_i}{w}\right),
        \qquad c_i=\tfrac{1}{2}(x_{\min,i}+x_{\max,i}),
        \quad h_i=\tfrac{1-\varepsilon}{2}(x_{\max,i}-x_{\min,i}),

    with width parameter (:math:`w>0`) and small (:math:`\epsilon>0`) for numerical safety.

    **Jacobian (componentwise)**
    For each component,

    .. math::
        \frac{\partial y_i}{\partial x_i} = \frac{h_i}{w} sech^2 \left(\frac{x_i}{w}\right),
        \qquad
        \log\left| \frac{\partial y_i}{\partial x_i} \right| = \log h_i - \log w + \log\!\big(sech^2(\frac{x_i}{w})\big).

    Using the stable identity :math:`\log(sech^2 z)=2 [\log 2 - z - softplus(-2z)]`,
    which we apply for good numerical behavior.

    Parameters
    ----------
    -x_min : jax.Array
        Elementwise lower bounds of the distribution.

    -x_max : jax.Array
        Elementwise upper bounds of the distribution. Must satisfy x_max > x_min elementwise.

    -width : float
        slope control.

    -eps : float
        Small offset to avoid arctanh divergence close to bounds.

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

        self.x_min = x_min
        self.x_max = x_max
        self.center = (x_min + x_max) / 2.0
        self.half = (1.0 - eps) * (x_max - x_min) / 2.0
        self.width = width
        self.eps = float(eps)

    @property
    def kws(self) -> Dict[str, Any]:
        return dict(
            x_min=self.x_min,
            x_max=self.x_max,
            width=self.width,
            eps=self.eps,
            event_ndims_in=self.event_ndims_in,
            event_ndims_out=self.event_ndims_out,
            is_constant_jacobian=self.is_constant_jacobian,
            is_constant_log_det=self.is_constant_log_det,
        )

    @staticmethod
    @partial(jax.named_call, name="BoxSpace.sec2_log")
    def sec2_log(x: jax.Array) -> jax.Array:
        return 2 * (jnp.log(2) - x - jax.nn.softplus(-2.0 * x))

    @partial(jax.named_call, name="BoxSpace.forward_log_det_jacobian")
    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        """
        Computes log|det J(f)(x)|.
        log|dy/dx| = log|half| + log(sech^2 x)
        Stable log(sech^2 x) = 2*(log(2) - x - softplus(-2x))
        """
        return jnp.log(self.half) + self.sec2_log(x / self.width) - jnp.log(self.width)

    @partial(jax.named_call, name="BoxSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.center + self.half * jnp.tanh(x / self.width)
        return y, self.forward_log_det_jacobian(x)

    @partial(jax.named_call, name="BoxSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        u = (y - self.center) / (self.half + self.eps)
        u = u.clip(-1.0 + self.eps, 1.0 - self.eps)
        x = self.width * jnp.arctanh(u)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is BoxSpace  # pylint: disable=unidiomatic-typecheck


__all__ = ["BoxSpace"]
