# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bijector that constrains the norm of vector actions."""

from functools import partial
import math

import distrax  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
import numpy as np
from distrax._src.bijectors.bijector import Array  # type: ignore[import-untyped]

from ...utils.linalg import dot, norm2
from . import ActionSpace

# Gauss-Hermite nodes for tensor product quadrature over d-dimensional Normal.
_GH_N, _GH_W = np.polynomial.hermite_e.hermegauss(5)
_GH_NODES: jax.Array = jnp.asarray(_GH_N)
_GH_WEIGHTS_1D: jax.Array = jnp.asarray(_GH_W / np.sqrt(2.0 * np.pi))


@ActionSpace.register("MaxNorm")
class MaxNormSpace(distrax.Bijector, ActionSpace):  # type: ignore[misc]
    r"""**Radial max-norm** constraint for vector actions.
    Scales the radius with a `tanh` squashing and preserves the direction.

    **Mapping (vector case),** :math:`\vec{x} \in \mathbb{R}^d`:

    .. math::
        r = \lVert \vec{x} \rVert_2,\qquad
        \hat{u} = \begin{cases}
            \frac{\vec{x}}{r}, & r>0,\\[4pt]
            0, & r=0,
        \end{cases}
        \qquad
        y = s \tanh(r) \hat{u},
        \quad s = (1-\epsilon) \text{max\_norm}.

    Equivalently, :math:`y = b(r)\,x` with :math:`b(r)= s\,\tanh(r)/r` for :math:`r>0`.


    **Jacobian determinant**

    For an isotropic radial map :math:`f(x)=b(r)` with :math:`x \in \mathbb{R}^d`, the Jacobian
    eigenvalues are :math:`b` (multiplicity d-1) on the tangent subspace and :math:`b + r\,b'(r)` on the radial direction. Therefore

    .. math::
        \bigl|\det J_f(x)\bigr| = b(r)^{\,d-1}\,\bigl(b(r)+r\,b'(r)\bigr)
        = s^d \left(\frac{\tanh r}{r}\right)^{\!d-1} \text{sech}^2 r.

    Therefore

    .. math::
        \log\lvert\det J_f(x)\rvert
        = d\log s + (d-1)\bigl(\log\tanh r - \log r\bigr) + \log(\text{sech}^2 r),

    We use the stable identity :math:`\log(\text{sech}^2 z)=2 [\log 2 - z - \text{softplus}(-2z)]`
    for good numerical behavior.

    Near :math:`r\approx 0`, we use the second-order expansion

    .. math::
        \log\lvert\det J_f(x)\rvert \approx d\log s - \tfrac{d+2}{3} r^2

    to avoid division by :math:`r`.

    Parameters
    ----------
    max_norm : float
        Maximum radius after squashing (default 1.0). The bijector uses \(s=(1-\varepsilon)\,\text{max\_norm}\) to stay off the exact boundary.
    eps : float
        Numerical safety margin near \(r=0\) and \(r\to\infty\).
    event_ndims_in : int
        Dimensionality of a *single event* seen by the bijector (default 1).
    event_ndims_out : Optional[int]
        Standard Distrax/TFP bijector flag.
    is_constant_jacobian : bool
        Standard Distrax/TFP bijector flag.
    is_constant_log_det : bool
        Standard Distrax/TFP bijector flag.

    Note
    ----
    This bijector is **vector-valued** with ``event_ndims_in = 1``. It treats
    a length-\(d\) action vector as a single event. Do **not** wrap it in
    `Block` unless you want to apply it independently to multiple last-axis blocks.

    """

    __slots__ = ()

    def __init__(
        self,
        max_norm: float = 1.0,
        eps: float = 1e-6,
        event_ndims_in: int = 1,
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
        self.eps = float(eps)
        self.max_norm = float(max_norm)
        if not math.isfinite(self.max_norm) or self.max_norm <= 0.0:
            raise ValueError("max_norm must be finite and positive")
        if not math.isfinite(self.eps) or not 0.0 < self.eps < 1.0:
            raise ValueError("eps must be finite and strictly between 0 and 1")

    @staticmethod
    @partial(jax.named_call, name="MaxNormSpace.sec2_log")
    def sec2_log(r: jax.Array) -> jax.Array:
        # r is scalar radius
        return 2 * (jnp.log(2.0) - r - jax.nn.softplus(-2.0 * r))

    @staticmethod
    @partial(jax.named_call, name="MaxNormSpace._radius")
    def _radius(
        x: Array, keepdims: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the exact radius and a finite denominator at the origin."""
        r2 = norm2(x)
        if keepdims:
            r2 = r2[..., None]
        denominator = jnp.sqrt(jnp.where(r2 > 0.0, r2, 1.0))
        return jnp.where(r2 > 0.0, denominator, 0.0), denominator, r2

    @partial(jax.named_call, name="MaxNormSpace._log_det_from_r")
    def _log_det_from_r(
        self, r: jax.Array, d: jax.Array, r2: jax.Array | None = None
    ) -> jax.Array:
        """Core log|det J| for the exact radial map."""
        log_s = jnp.log((1.0 - self.eps) * self.max_norm)
        safe_r = jnp.where(r > 0.0, r, 1.0)
        exact = (
            d * log_s
            + (d - 1.0) * (jnp.log(jnp.tanh(safe_r)) - jnp.log(safe_r))
            + MaxNormSpace.sec2_log(safe_r)
        )
        if r2 is None:
            r2 = r * r
        series = d * log_s - (d + 2.0) * r2 / 3.0
        return jnp.where(r < 1e-3, series, exact)

    @partial(jax.named_call, name="MaxNormSpace.forward_log_det_jacobian")
    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        x = jnp.atleast_1d(x)
        r, _, r2 = self._radius(x)
        d = jnp.asarray(x.shape[-1], x.dtype)
        return self._log_det_from_r(r, d, r2)

    @partial(jax.named_call, name="MaxNormSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> tuple[jax.Array, jax.Array]:
        r, denominator, r2 = self._radius(x, keepdims=True)
        radial_scale = jnp.where(r > 0.0, jnp.tanh(r) / denominator, 1.0)
        y = (1.0 - self.eps) * self.max_norm * radial_scale * x
        d = jnp.asarray(jnp.atleast_1d(x).shape[-1], x.dtype)
        return y, self._log_det_from_r(r.squeeze(-1), d, r2.squeeze(-1))

    @partial(jax.named_call, name="MaxNormSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> tuple[jax.Array, jax.Array]:
        r_y, denominator, _ = self._radius(y, keepdims=True)
        one = jnp.asarray(1.0, dtype=y.dtype)
        upper = jnp.nextafter(one, jnp.asarray(0.0, dtype=y.dtype))
        s = (1.0 - self.eps) * self.max_norm
        u = jnp.minimum(r_y / s, upper)
        atanh_u = jnp.arctanh(u)
        inverse_scale = jnp.where(r_y > 0.0, atanh_u / denominator, 1.0 / s)
        x = inverse_scale * y
        r_x = atanh_u.squeeze(-1)
        d = jnp.asarray(jnp.atleast_1d(y).shape[-1], y.dtype)
        return x, -self._log_det_from_r(r_x, d)

    @partial(jax.named_call, name="MaxNormSpace.log_det_expectation")
    def log_det_expectation(self, mean: jax.Array, std: jax.Array) -> jax.Array:
        r""":math:`\mathbb{E}_X[\log|\det J_f(X)|]` via tensor-product
        Gauss-Hermite quadrature in *d* dimensions, supported for ``d <= 6``.
        """
        d = mean.shape[-1]
        if d > 6:
            raise ValueError(
                "MaxNorm entropy quadrature supports action dimensions up to 6; "
                "use BoxSpace for separable bounded actions or FreeSpace for "
                "unbounded high-dimensional actions."
            )

        # Build d-dimensional tensor-product grid: nodes (n^d, d), weights (n^d,)
        grids = jnp.meshgrid(*([_GH_NODES] * d), indexing="ij")
        nodes_nd = jnp.stack([g.ravel() for g in grids], axis=-1)
        w_grids = jnp.meshgrid(*([_GH_WEIGHTS_1D] * d), indexing="ij")
        weights_nd = jnp.prod(jnp.stack([wg.ravel() for wg in w_grids], axis=0), axis=0)

        # x = mean + std * z, shape (..., n^d, d)
        x = mean[..., None, :] + std[..., None, :] * nodes_nd
        safe_r, _, r2 = self._radius(x)  # (..., n^d)
        d_val = jnp.asarray(d, mean.dtype)
        ld = self._log_det_from_r(safe_r, d_val, r2)  # (..., n^d)
        return dot(ld, weights_nd)


__all__ = ["MaxNormSpace"]
