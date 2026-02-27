# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Implementation of bijector for max Norm space.
"""

import jax
import jax.numpy as jnp
import numpy as np

from typing import Any, Dict, Optional, Tuple
from functools import partial

import distrax
from distrax._src.bijectors.bijector import Array

from . import ActionSpace

# Gauss-Hermite nodes for tensor product quadrature over d-dimensional Normal.
_GH_N, _GH_W = np.polynomial.hermite_e.hermegauss(5)
_GH_NODES: jax.Array = jnp.asarray(_GH_N)
_GH_WEIGHTS_1D: np.ndarray = _GH_W / np.sqrt(2.0 * np.pi)


@ActionSpace.register("MaxNorm")
class MaxNormSpace(distrax.Bijector, ActionSpace):  # type: ignore[misc]
    r"""
    **Radial max-norm** constraint for vector actions:
    scales the radius with a `tanh` squashing while preserving direction.

    **Mapping (vector case,** :math:`x \in \mathbb{R}^d`

    .. math::
        r = \lVert \vec{x} \rVert_2,\qquad
        \hat{u} = \begin{cases}
            \frac{\vec{x}}{r}, & r>0,\\[4pt]
            0, & r=0,
        \end{cases}
        \qquad
        y = s \tanh(r) \hat{u},
        \quad s = (1-\epsilon) \texttt{max_norm}.

    Equivalently, :math:`y = b(r)\,x` with :math:`b(r)= s\,\tanh(r)/r` for :math:`r>0`.


    **Jacobian determinant**

    For an isotropic radial map :math:`f(x)=b(r)` with :math:`x \in \mathbb{R}^d`, the Jacobian
    eigenvalues are :math:`b` (multiplicity d-1) on the tangent subspace and :math:`b + r\,b'(r)` on the radial direction, hence

    .. math::
        \bigl|\det J_f(x)\bigr| = b(r)^{\,d-1}\,\bigl(b(r)+r\,b'(r)\bigr)
        = s^d \left(\frac{\tanh r}{r}\right)^{\!d-1} sech^2 r.

    Therefore

    .. math::
        \log\lvert\det J_f(x)\rvert
        = d\log s + (d-1)\bigl(\log\tanh r - \log r\bigr) + \log( sech^2 r),

    We use the stable identity :math:`\log(sech^2 z)=2 [\log 2 - z - softplus(-2z)]`,
    which we apply for good numerical behavior.

    Near :math:`r\approx 0`, we use the second-order expansion

    .. math::
        \log\lvert\det J_f(x)\rvert \approx d\log s - \tfrac{2}{3} r^2

    to avoid division by :math:`r`.

    Parameters
    ----------
    -max_norm : float
        target radius \(s\) after squashing (default 1.0). We actually use \(s=(1-\varepsilon)\,\texttt{max\_norm}\) to avoid the exact boundary.

    -eps : float
        numerical safety margin used near \(r=0\) and \(r\to\infty\).

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
    This bijector is **vector-valued** with ``event_ndims_in = 1`` (i.e., it operates on length-\(d\) action vectors as a
    single event). Do **not** wrap it in `Block` unless you intend to apply it independently to multiple last-axis blocks.
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

    @property
    def kws(self) -> Dict[str, Any]:
        return dict(
            max_norm=self.max_norm,
            eps=self.eps,
            event_ndims_in=self.event_ndims_in,
            event_ndims_out=self.event_ndims_out,
            is_constant_jacobian=self.is_constant_jacobian,
            is_constant_log_det=self.is_constant_log_det,
        )

    @staticmethod
    @partial(jax.named_call, name="MaxNormSpace.sec2_log")
    def sec2_log(r: jax.Array) -> jax.Array:
        # r is scalar radius
        return 2 * (jnp.log(2.0) - r - jax.nn.softplus(-2.0 * r))

    @partial(jax.named_call, name="MaxNormSpace._safe_r")
    def _safe_r(self, x: Array, keepdims: bool = False) -> jax.Array:
        """Smoothed radius sqrt(||x||² + eps²). Always > 0, gradient-safe at x = 0."""
        return jnp.sqrt(
            jnp.sum(x * x, axis=-1, keepdims=keepdims) + self.eps * self.eps
        )

    @partial(jax.named_call, name="MaxNormSpace._log_det_from_r")
    def _log_det_from_r(self, safe_r: jax.Array, d: jax.Array) -> jax.Array:
        """Core log|det J| from smoothed radius (always > 0)."""
        log_s = jnp.log((1.0 - self.eps) * self.max_norm + self.eps)
        return (
            d * log_s
            + (d - 1.0) * (jnp.log(jnp.tanh(safe_r)) - jnp.log(safe_r))
            + MaxNormSpace.sec2_log(safe_r)
        )

    @partial(jax.named_call, name="MaxNormSpace.forward_log_det_jacobian")
    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        x = jnp.atleast_1d(x)
        safe_r = self._safe_r(x)
        d = jnp.asarray(x.shape[-1], x.dtype)
        return self._log_det_from_r(safe_r, d)

    @partial(jax.named_call, name="MaxNormSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        safe_r = self._safe_r(x, keepdims=True)
        y = (1.0 - self.eps) * self.max_norm * jnp.tanh(safe_r) * (x / safe_r)
        d = jnp.asarray(jnp.atleast_1d(x).shape[-1], x.dtype)
        return y, self._log_det_from_r(safe_r.squeeze(-1), d)

    @partial(jax.named_call, name="MaxNormSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        safe_r_y = self._safe_r(y, keepdims=True)
        u = (safe_r_y / ((1.0 - self.eps) * self.max_norm)).clip(
            -1.0 + self.eps, 1.0 - self.eps
        )
        atanh_u = jnp.arctanh(u)
        x = atanh_u * (y / safe_r_y)
        safe_r_x = jnp.sqrt(atanh_u.squeeze(-1) ** 2 + self.eps * self.eps)
        d = jnp.asarray(jnp.atleast_1d(y).shape[-1], y.dtype)
        return x, -self._log_det_from_r(safe_r_x, d)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is MaxNormSpace

    @partial(jax.named_call, name="MaxNormSpace.log_det_expectation")
    def log_det_expectation(self, mean: jax.Array, std: jax.Array) -> jax.Array:
        r"""
        :math:`\mathbb{E}_X[\log|\det J_f(X)|]` via tensor-product
        Gauss-Hermite quadrature in *d* dimensions.
        """
        d = mean.shape[-1]
        n = len(_GH_N)

        # Build d-dimensional tensor-product grid: nodes (n^d, d), weights (n^d,)
        grids = jnp.meshgrid(*([_GH_NODES] * d), indexing="ij")
        nodes_nd = jnp.stack([g.ravel() for g in grids], axis=-1)
        w_grids = np.meshgrid(*([_GH_WEIGHTS_1D] * d), indexing="ij")
        weights_nd = jnp.asarray(
            np.prod(np.stack([wg.ravel() for wg in w_grids], axis=0), axis=0)
        )

        # x = mean + std * z, shape (..., n^d, d)
        x = mean[..., None, :] + std[..., None, :] * nodes_nd
        safe_r = self._safe_r(x)  # (..., n^d)
        d_val = jnp.asarray(d, mean.dtype)
        ld = self._log_det_from_r(safe_r, d_val)  # (..., n^d)
        return jnp.sum(ld * weights_nd, axis=-1)


__all__ = ["MaxNormSpace"]
