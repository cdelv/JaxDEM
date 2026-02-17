# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Interface for defining bijectors used to constraint the policy probability distribution.
"""

from __future__ import annotations
from typing import Any, Dict

from ...factory import Factory


class ActionSpace(Factory):
    """
    Registry/namespace for action-space **constraints** implemented as
    `distrax.Bijector`s.

    These bijectors are intended to be wrapped around a base policy
    distribution (e.g., `MultivariateNormalDiag`) via
    `distrax.Transformed`, so that sampling and log-probabilities are
    correctly adjusted using the bijector’s `forward_and_log_det` /
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


from .freeSpace import FreeSpace
from .boxSpace import BoxSpace
from .maxNormSpace import MaxNormSpace

__all__ = ["ActionSpace", "FreeSpace", "BoxSpace", "MaxNormSpace"]
