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
from numpy.typing import NDArray

from ...utils.linalg import dot, norm2
from . import ActionSpace


def _entropy_quadrature() -> tuple[NDArray[np.float64], ...]:
    """Precompute the scalar Laplace integral's nodes and spectral weights.

    The sinh/cosh product formulas give sums over integer/half-integer
    squares. Poisson summation evaluates their small-t limits without a
    slowly converging series or subtracting nearly equal large numbers.
    Eight terms suffice in either representation at the switch t=0.2.
    """
    u, weights = np.polynomial.legendre.leggauss(128)
    u = (u + 1.0) / 2.0
    t = (u / (1.0 - u)) ** 4
    weights = 2.0 * weights / (u * (1.0 - u))  # dt/t = 4 du/[u(1-u)]
    k = np.arange(1, 9)
    integers = np.exp(-(np.pi**2) * t[:, None] * k**2).sum(axis=-1)
    halves = np.exp(-(np.pi**2) * t[:, None] * (k - 0.5) ** 2).sum(axis=-1)
    reflected = np.exp(-(k**2) / t[:, None]) / np.sqrt(np.pi * t[:, None])
    difference = np.where(
        t < 0.2, -0.5 + (reflected * (1 - (-1.0) ** k)).sum(axis=-1), integers - halves
    )
    total = np.where(
        t < 0.2,
        1.0 / np.sqrt(np.pi * t) - 0.5 + (reflected * (1 + (-1.0) ** k)).sum(axis=-1),
        integers + halves,
    )
    return t, weights * difference, weights * total


_ENTROPY_NODES, _ENTROPY_DIFFERENCE, _ENTROPY_TOTAL = _entropy_quadrature()


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

    For an isotropic radial map :math:`f(x)=b(r)x` with :math:`x \in \mathbb{R}^d`, the Jacobian
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
        Maximum radius after squashing (default 1.0). The bijector uses :math:`s=(1-\varepsilon)\,\text{max\_norm}` to stay off the exact boundary.
    eps : float
        Numerical safety margin used for the output-radius margin and inverse clipping.
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
    a length-:math:`d` action vector as a single event. Do **not** wrap it in
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
        r"""Expected log-Jacobian via a scalar Laplace integral.

        For :math:`Q=\lVert X\rVert^2`, its Laplace transform is analytic:

        .. math::
            L(t)=\mathbb{E}[e^{-tQ}]
            =\prod_i\frac{
                \exp\!\left(-t\mu_i^2/(1+2t\sigma_i^2)\right)
              }{\sqrt{1+2t\sigma_i^2}}.

        The sinh/cosh infinite products (DLMF 4.36.1--2) and
        :math:`\log(1+q/a)=\int_0^\infty(1-e^{-tq})e^{-at}\,dt/t` give

        .. math::
            \mathbb{E}[\log|\det J_f(X)|]
            =d\log s+\int_0^\infty\frac{1-L(t)}{t}
              [(d-1)A(t)-(d+1)B(t)]\,dt,

        where :math:`A(t)=\sum_{k\ge1}e^{-\pi^2 k^2t}` and
        :math:`B(t)=\sum_{k\ge1}e^{-\pi^2(k-1/2)^2t}`.
        A fixed 128-point Gauss--Legendre rule after
        :math:`t=(u/(1-u))^4` approximates this one-dimensional integral.
        Cost grows linearly with action dimension; no samples or tensor
        grids are needed. Values and autodiff gradients remain numerical
        approximations, with accuracy depending on the policy parameters.
        """
        d = mean.shape[-1]
        t = jnp.asarray(_ENTROPY_NODES, dtype=mean.dtype)
        variance_term = 2.0 * std[..., :, None] ** 2 * t
        log_laplace = -0.5 * jnp.log1p(variance_term)
        log_laplace -= mean[..., :, None] ** 2 * t / (1.0 + variance_term)
        log_laplace = jnp.sum(log_laplace, axis=-2)
        weights = jnp.asarray(
            d * _ENTROPY_DIFFERENCE - _ENTROPY_TOTAL, dtype=mean.dtype
        )
        log_s = jnp.log((1.0 - self.eps) * self.max_norm)
        return d * log_s + dot(-jnp.expm1(log_laplace), weights)


__all__ = ["MaxNormSpace"]
