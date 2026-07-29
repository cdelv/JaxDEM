# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bijector that constrains actions elementwise to a box."""

from functools import partial

import distrax  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
import numpy as np
from distrax._src.bijectors.bijector import Array  # type: ignore[import-untyped]

from ...utils.linalg import dot
from . import ActionSpace

# Gauss-Hermite quadrature nodes/weights for E_{Z~N(0,1)}[f(Z)].
_GH_N, _GH_W = np.polynomial.hermite_e.hermegauss(16)
_GH_NODES: jax.Array = jnp.asarray(_GH_N)
_GH_WEIGHTS: jax.Array = jnp.asarray(_GH_W / np.sqrt(2.0 * np.pi))


@ActionSpace.register("Box")
class BoxSpace(distrax.Bijector, ActionSpace):  # type: ignore[misc]
    r"""Elementwise **box** constraint implemented with a scaled `tanh`.

    **Mapping (componentwise)**

    .. math::
        y_i \;=\; c_i + h_i\,\tanh\!\left(\frac{x_i}{w}\right),
        \qquad c_i=\tfrac{1}{2}(x_{\min,i}+x_{\max,i}),
        \quad h_i=\tfrac{1-\varepsilon}{2}(x_{\max,i}-x_{\min,i}),

    with a width parameter :math:`w>0` and a small :math:`\epsilon>0` for numerical safety.

    **Jacobian (componentwise)**
    For each component,

    .. math::
        \frac{\partial y_i}{\partial x_i} = \frac{h_i}{w} sech^2 \left(\frac{x_i}{w}\right),
        \qquad
        \log\left| \frac{\partial y_i}{\partial x_i} \right| = \log h_i - \log w + \log\!\big(sech^2(\frac{x_i}{w})\big).

    We use the stable identity :math:`\log(sech^2 z)=2 [\log 2 - z - softplus(-2z)]`
    for good numerical behavior.

    Parameters
    ----------
    x_min : jax.Array
        Elementwise lower bounds of the box.
    x_max : jax.Array
        Elementwise upper bounds of the box. Must satisfy x_max > x_min elementwise.
    width : float
        Controls the tanh slope (default 1.0).
    eps : float
        Small offset to avoid arctanh divergence close to the bounds (default 1e-6).
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
        x_min: Array,
        x_max: Array,
        width: float = 1.0,
        eps: float = 1e-6,
        event_ndims_in: int = 0,
        event_ndims_out: int | None = None,
        is_constant_jacobian: bool = False,
        is_constant_log_det: bool | None = None,
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

    @staticmethod
    @partial(jax.named_call, name="BoxSpace.sec2_log")
    def sec2_log(x: jax.Array) -> jax.Array:
        return 2 * (jnp.log(2) - x - jax.nn.softplus(-2.0 * x))

    @partial(jax.named_call, name="BoxSpace.forward_log_det_jacobian")
    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        r"""Compute log|det J(f)(x)| = log(half) - log(width) + log(sech^2(x/width)).
        Uses the stable identity log(sech^2 z) = 2*(log(2) - z - softplus(-2z)).
        """
        return jnp.log(self.half) + self.sec2_log(x / self.width) - jnp.log(self.width)

    @partial(jax.named_call, name="BoxSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> tuple[jax.Array, jax.Array]:
        r"""Compute y = f(x) and log|det J(f)(x)|."""
        y = self.center + self.half * jnp.tanh(x / self.width)
        return y, self.forward_log_det_jacobian(x)

    @partial(jax.named_call, name="BoxSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> tuple[jax.Array, jax.Array]:
        r"""Compute x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        u = (y - self.center) / self.half
        u = u.clip(-1.0 + self.eps, 1.0 - self.eps)
        x = self.width * jnp.arctanh(u)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        """Return True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is BoxSpace  # pylint: disable=unidiomatic-typecheck

    @partial(jax.named_call, name="BoxSpace.log_det_expectation")
    def log_det_expectation(self, mean: jax.Array, std: jax.Array) -> jax.Array:
        r""":math:`\mathbb{E}_X[\sum_i \log|dJ_i/dx_i|]` via 1-D Gauss-Hermite
        quadrature (componentwise separable).
        """
        # x_i = mean_i + std_i * z_k, shape (..., d, n_pts)
        z = (mean[..., None] + std[..., None] * _GH_NODES) / self.width
        ld = jnp.log(self.half)[..., None] - jnp.log(self.width) + BoxSpace.sec2_log(z)
        return jnp.sum(dot(ld, _GH_WEIGHTS), axis=-1)


__all__ = ["BoxSpace"]
