# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Interface for defining bijectors used to constrain the policy probability distribution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

import distrax
from distrax._src.bijectors.bijector import Array

from ...factory import Factory


class ActionSpace(Factory):
    """
    Registry/namespace for action-space **constraints** implemented as
    `distrax.Bijector`s.

    These bijectors are intended to be wrapped around a base policy
    distribution (e.g., `MultivariateNormalDiag`) via
    `distrax.Transformed`, so that sampling and log-probabilities are
    correctly adjusted using the bijector's `forward_and_log_det` /
    `inverse_and_log_det` methods. See Distrax/TFP bijector interface
    for details on shape semantics and `event_ndims_in/out`.

    Example
    -------
    To define a custom action space, inherit from :class:`distrax.Bijector` and :class:`ActionSpace` and implement its abstract methods:

    >>> @ActionSpace.register("myCustomActionSpace")
    >>> class MyCustomActionSpace(distrax.Bijector, ActionSpace):
            ...
    """

    __slots__ = ()

    @property
    def kws(self) -> Dict[str, Any]:
        return dict()

    def log_det_expectation(self, mean: jax.Array, std: jax.Array) -> jax.Array:
        r"""Compute :math:`\mathbb{E}_{X}[\log|\det J_f(X)|]` where
        :math:`X \sim \mathcal{N}(\text{mean}, \text{diag}(\text{std}^2))`.

        Subclasses should override this to enable ``Transformed.entropy()``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement log_det_expectation"
        )


class Transformed(distrax.Transformed):  # type: ignore[misc]
    r"""``distrax.Transformed`` with analytical entropy support.

    For :math:`Y = f(X)` where :math:`X \sim \text{base}`,

    .. math::
        H(Y) = H(X) + \mathbb{E}_X[\log|\det J_f(X)|].

    The expectation is computed by the bijector's
    :meth:`~ActionSpace.log_det_expectation` method via
    Gauss--Hermite quadrature (exact for polynomial integrands, highly
    accurate for smooth bijectors such as scaled tanh).
    """

    def entropy(self, input_hint: Optional[Array] = None) -> jax.Array:  # type: ignore[override, unused-ignore]
        bij = self.bijector
        inner = getattr(bij, "_bijector", bij)

        if isinstance(inner, ActionSpace):
            correction = inner.log_det_expectation(
                self.distribution.loc,
                self.distribution.scale_diag,
            )
            return self.distribution.entropy() + correction

        return super().entropy(input_hint=input_hint)


from .freeSpace import FreeSpace
from .boxSpace import BoxSpace
from .maxNormSpace import MaxNormSpace

__all__ = ["ActionSpace", "Transformed", "FreeSpace", "BoxSpace", "MaxNormSpace"]
