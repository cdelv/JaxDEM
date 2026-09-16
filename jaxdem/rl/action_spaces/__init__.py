# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Interface for bijectors that constrain the policy probability distribution."""

from __future__ import annotations

from typing import Any

import distrax  # type: ignore[import-untyped]
import jax
from distrax._src.bijectors.bijector import Array  # type: ignore[import-untyped]

from ...factory import Factory


class ActionSpace(Factory):
    """Registry/namespace for action-space **constraints** implemented as
    ``distrax.Bijector`` objects.

    Wrap these bijectors around a base policy distribution (e.g.,
    ``MultivariateNormalDiag``) with ``distrax.Transformed``. The bijector's
    'forward_and_log_det' / 'inverse_and_log_det' methods then adjust
    sampling and log-probabilities correctly. See the Distrax/TFP bijector
    interface for details on shape semantics and 'event_ndims_in/out'.

    Example:
    --------
    To define a custom action space, inherit from :class:`distrax.Bijector` and :class:`ActionSpace` and implement its abstract methods:

    >>> @ActionSpace.register("myCustomActionSpace")
    >>> class MyCustomActionSpace(distrax.Bijector, ActionSpace):
            ...

    """

    __slots__ = ()

    @property
    def kws(self) -> dict[str, Any]:
        return self.metadata

    def log_det_expectation(self, mean: jax.Array, std: jax.Array) -> jax.Array:
        r"""Compute :math:`\mathbb{E}_{X}[\log|\det J_f(X)|]` where
        :math:`X \sim \mathcal{N}(\text{mean}, \text{diag}(\text{std}^2))`.

        Subclasses override this method to enable ``Transformed.entropy()``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement log_det_expectation"
        )


class Transformed(distrax.Transformed):  # type: ignore[misc]
    r"""``distrax.Transformed`` with entropy support for action-space bijectors.

    For :math:`Y = f(X)` where :math:`X \sim \text{base}`,

    .. math::
        H(Y) = H(X) + \mathbb{E}_X[\log|\det J_f(X)|].

    This is the differential-entropy identity for an invertible differentiable
    transform. For the diagonal-Gaussian policies used here, the bijector's
    :meth:`~ActionSpace.log_det_expectation` supplies the correction.
    FreeSpace returns it analytically; BoxSpace and MaxNormSpace approximate
    it with finite Gauss--Hermite quadrature. Those nonlinear integrands are
    not generally integrated exactly, and accuracy depends on the policy
    parameters and quadrature order.
    """

    def sample_and_log_prob_with_latent(
        self, *, seed: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sample an action and retain its latent and base log-probability.

        PPO can compute same-transform likelihood ratios in base coordinates,
        where the Jacobian cancels, without inverting a rounded/saturated action.
        Returns action, transformed log-probability, latent, base log-probability.
        """
        latent = self.distribution.sample(seed=seed, sample_shape=sample_shape)
        # Some Normal samplers compute density from the unrounded noise used
        # to generate x. Store density at the actual, representable x instead,
        # matching learner re-evaluation (important for large means/small std).
        base_log_prob = self.distribution.log_prob(latent)
        action, log_det = self.bijector.forward_and_log_det(latent)
        return action, base_log_prob - log_det, latent, base_log_prob

    def entropy(self, input_hint: Array | None = None) -> jax.Array:  # type: ignore[override, unused-ignore]
        bij = self.bijector
        inner = getattr(bij, "_bijector", bij)

        if isinstance(inner, ActionSpace):
            correction = inner.log_det_expectation(
                self.distribution.loc,
                self.distribution.scale_diag,
            )
            return self.distribution.entropy() + correction

        return super().entropy(input_hint=input_hint)


from .box_space import BoxSpace
from .free_space import FreeSpace
from .max_norm_space import MaxNormSpace

__all__ = ["ActionSpace", "BoxSpace", "FreeSpace", "MaxNormSpace", "Transformed"]
