# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Implementation of bijector for max Norm space.
"""

import jax
import jax.numpy as jnp

from typing import Any, Dict, Optional, Tuple
from functools import partial

import distrax
from distrax._src.bijectors.bijector import Array

from . import ActionSpace


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

    @partial(jax.named_call, name="MaxNormSpace.forward_log_det_jacobian")
    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        r = jnp.linalg.norm(x, axis=-1)  # shape (...,)
        x = jnp.atleast_1d(x)  # ensures x.ndim >= 1
        d = jnp.asarray(x.shape[-1], x.dtype)  # scalar, works under jit

        # Stable pieces
        log_s = jnp.log((1.0 - self.eps) * self.max_norm + self.eps)
        log_tanh_r = jnp.log(jnp.tanh(r) + self.eps)
        log_r = jnp.log(r + self.eps)
        log_sech2_r = MaxNormSpace.sec2_log(r)

        main = d * log_s + (d - 1.0) * (log_tanh_r - log_r) + log_sech2_r
        small = d * log_s - (2.0 / 3.0) * (r * r)
        return jnp.where(r < self.eps, small, main)

    @partial(jax.named_call, name="MaxNormSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> Tuple[jax.Array, jax.Array]:
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        unit = jnp.where(r > 0.0, x / r, jnp.zeros_like(x))
        y = (1.0 - self.eps) * self.max_norm * jnp.tanh(r) * unit
        return y, self.forward_log_det_jacobian(x)

    @partial(jax.named_call, name="MaxNormSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> Tuple[jax.Array, jax.Array]:
        r = jnp.linalg.norm(y, axis=-1, keepdims=True)
        u = (r / ((1.0 - self.eps) * self.max_norm)).clip(
            -1.0 + self.eps, 1.0 - self.eps
        )
        unit = jnp.where(r > 0.0, y / r, jnp.zeros_like(y))
        x = jnp.arctanh(u) * unit
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is MaxNormSpace


__all__ = ["MaxNormSpace"]
