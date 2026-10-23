# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bijector that maps Gaussian actions into a ball with analytic entropy."""

import jax
import jax.numpy as jnp

from functools import partial
import math

import distrax  # type: ignore[import-untyped]
import numpy as np
from distrax._src.bijectors.bijector import Array  # type: ignore[import-untyped]
from scipy import special
from tensorflow_probability.substrates.jax.math import igammainv  # type: ignore[import-untyped]

from ...utils.linalg import norm2
from . import ActionSpace

# Keep direct gamma evaluations away from zero; use the scaled series below it.
_SERIES_PROBABILITY = 0.01


@ActionSpace.register("MaxNorm")
class MaxNormSpace(distrax.Bijector, ActionSpace):  # type: ignore[misc]
    r"""Constrain the norm of one action vector using a Gaussian-to-ball map.

    For an event of length d, preserve its direction and transform its radius:

    .. math::
        r=\|x\|,\quad R=(1-\epsilon)\,\mathrm{max\_norm},\qquad
        y=R\,P(d/2,r^2/2)^{1/d}\frac{x}{r},

    where P is the regularized lower incomplete gamma function. This is the
    probability integral transform of a standard Gaussian's chi-distributed
    radius, followed by the inverse radius CDF of a uniform ball. The reference
    Gaussian is fixed at zero mean and unit variance; learned Gaussian means
    and variances still control the action distribution.

    Writing c for the linear scale at the origin gives

    .. math::
        c=\frac{R}{\sqrt{2}\,\Gamma(d/2+1)^{1/d}},\qquad
        y\sim cx,\qquad
        \log|\det J(x)|=d\log c-\tfrac12\|x\|^2.

    Thus the Gaussian log-Jacobian expectation and transformed entropy are
    analytic, including for anisotropic, noncentral Gaussian policies:

    .. math::
        \mathbb E[\log|\det J(X)|]
          =d\log c-\tfrac12\sum_i(\mu_i^2+\sigma_i^2),\\
        H(Y)=\log V_d(R)
          -D_{\mathrm{KL}}(\mathcal N(\mu,\mathrm{diag}(\sigma^2))
                           \,\|\,\mathcal N(0,I)).

    The forward and inverse use a convergent gamma series near zero to avoid
    dividing by the radius or forming an underflowing d-th power. Elsewhere,
    they use the regularized gamma function and its inverse. At the boundary,
    the inverse clips to the nearest representable interior squared radius.
    Saturated actions cannot recover their original latent exactly; PPO
    retains that latent when sampling.

    Parameters
    ----------
    max_norm : float
        Positive finite upper bound on the output norm, default 1.0.
    eps : float
        Output-radius margin in (0, 1), default 1e-6.
    event_ndims_in : int
        Must be one for a single vector.
    event_ndims_out : int or None
        Must be None or one. Use ``distrax.Block`` for higher-rank events.
    is_constant_jacobian : bool
        Standard Distrax flag, default False.
    is_constant_log_det : bool or None
        Standard Distrax flag, default None.

    Notes
    -----
    This replaces the historical radial tanh map. Existing Gaussian model
    heads and constructor arguments are supported, but the same latent now
    produces different actions. One instance constrains the entire last-axis
    vector; BlockwiseSpace applies separate bounds to explicit sub-vectors.
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
        if event_ndims_in != 1 or event_ndims_out not in (None, 1):
            raise ValueError(
                "MaxNormSpace transforms one vector; use distrax.Block for higher-rank events."
            )
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

    def _log_linear_scale(self, d: int) -> float:
        """Log of the finite radial scale at the origin."""
        return (
            math.log(self.max_norm)
            + math.log1p(-self.eps)
            - 0.5 * math.log(2.0)
            - math.lgamma(d / 2 + 1) / d
        )

    @staticmethod
    def _gamma_series(
        q: jax.Array, a: float, cutoff: float
    ) -> tuple[jax.Array, jax.Array]:
        """Return S(q) and S'(q), where P(a,q)=exp(-q)*q**a*S(q)/Gamma(a+1).

        Use the positive series from DLMF 8.7.1. For q <= cutoff, consecutive
        terms have ratio at most cutoff/(a+1); bound the geometric tail by
        machine precision. Horner evaluation avoids powers and underflow.
        """
        ratio = cutoff / (a + 1)
        terms = math.ceil(
            math.log(float(np.finfo(q.dtype).eps) * (1 - ratio) / 8) / math.log(ratio)
        )

        def iteration(
            index: int, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            total, derivative = carry
            k = terms - index
            derivative = (total + q * derivative) / (a + k)
            total = 1 + q * total / (a + k)
            return total, derivative

        return jax.lax.fori_loop(
            0, terms, iteration, (jnp.ones_like(q), jnp.zeros_like(q))
        )

    @partial(jax.named_call, name="MaxNormSpace.forward_log_det_jacobian")
    def forward_log_det_jacobian(self, x: Array) -> jax.Array:
        return x.shape[-1] * self._log_linear_scale(x.shape[-1]) - 0.5 * norm2(x)

    @partial(jax.named_call, name="MaxNormSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> tuple[jax.Array, jax.Array]:
        d = x.shape[-1]
        a, q = d / 2, 0.5 * norm2(x)
        cutoff = float(special.gammaincinv(a, _SERIES_PROBABILITY))
        small = q < cutoff
        small_q = jnp.where(small, q, cutoff)
        total, _ = self._gamma_series(small_q, a, cutoff)
        lower = jnp.exp(self._log_linear_scale(d) + (-small_q + jnp.log(total)) / d)
        safe_q = jnp.where(small, cutoff, q)
        regular = (
            (1 - self.eps)
            * self.max_norm
            * jnp.exp(jnp.log(jax.scipy.special.gammainc(a, safe_q)) / d)
            / jnp.sqrt(2 * safe_q)
        )
        scale = jnp.where(small, lower, regular)
        return scale[..., None] * x, d * self._log_linear_scale(d) - q

    @partial(jax.named_call, name="MaxNormSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> tuple[jax.Array, jax.Array]:
        d, radius = y.shape[-1], (1 - self.eps) * self.max_norm
        a, u2 = d / 2, norm2(y / radius)
        one = jnp.asarray(1.0, dtype=y.dtype)
        upper = jnp.nextafter(one, jnp.asarray(0.0, dtype=y.dtype))
        clipped = jnp.minimum(u2, upper)
        cutoff = float(special.gammaincinv(a, _SERIES_PROBABILITY))
        cutoff_u2 = _SERIES_PROBABILITY ** (2 / d)
        q0 = jnp.minimum(clipped, cutoff_u2) * math.exp(math.lgamma(a + 1) / a)

        def iteration(_: int, t: jax.Array) -> jax.Array:
            # Solve for q=q0*t without taking log(q0), even when q0 is zero.
            q = q0 * t
            total, derivative = self._gamma_series(q, a, cutoff)
            residual = jnp.log(t) + (-q + jnp.log(total)) / a
            slope = 1 / t + q0 * (-1 + derivative / total) / a
            return t - residual / slope

        # The inverse flattens as d grows: allow another step per factor four.
        steps = 6 + max(0, math.ceil(math.log2(a / 8) / 2))
        t = jax.lax.fori_loop(0, steps, iteration, jnp.ones_like(u2))
        lower = math.exp(-self._log_linear_scale(d)) * jnp.sqrt(t)
        small = clipped < cutoff_u2
        safe_u2 = jnp.where(small, cutoff_u2, clipped)
        probability = jnp.minimum(jnp.exp(a * jnp.log(safe_u2)), upper)
        q = igammainv(jnp.asarray(a, dtype=y.dtype), probability)
        regular = jnp.sqrt(2 * q / jnp.where(small, cutoff_u2, u2)) / radius
        x = jnp.where(small, lower, regular)[..., None] * y
        return x, -self.forward_log_det_jacobian(x)

    @partial(jax.named_call, name="MaxNormSpace.log_det_expectation")
    def log_det_expectation(self, mean: jax.Array, std: jax.Array) -> jax.Array:
        """Exact Gaussian expectation of the quadratic log-Jacobian."""
        return mean.shape[-1] * self._log_linear_scale(mean.shape[-1]) - 0.5 * jnp.sum(
            mean * mean + std * std, axis=-1
        )


__all__ = ["MaxNormSpace"]
