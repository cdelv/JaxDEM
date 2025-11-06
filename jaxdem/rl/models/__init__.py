# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning models.
"""

from __future__ import annotations

import jax

from typing import Tuple, Dict
from abc import ABC, abstractmethod

from flax import nnx
import distrax

from ...factory import Factory


class Model(Factory, nnx.Module, ABC):
    """
    The base interface for defining reinforcement learning models. Acts as a name space.

    Models map observations to an action distribution and a value estimate.

    Example
    -------
    To define a custom model, inherit from :class:`Model` and implement its abstract methods:

    >>> @Model.register("myCustomModel")
    >>> class MyCustomModel(Model):
            ...
    """

    __slots__ = ()

    @property
    def metadata(self) -> Dict:
        return {}

    def reset(self, shape: Tuple, mask: jax.Array | None = None):
        """
        Reset the persistent LSTM carry.

        Parameters
        -----------
        lead_shape : tuple[int, ...]
            Leading dims for the carry, e.g. (num_envs, num_agents).
        mask : optional bool array
            True where to reset entries. Shape (num_envs)
        """
        return

    @abstractmethod
    def __call__(
        self, x: jax.Array, sequence: bool = False
    ) -> Tuple[distrax.Distribution, jax.Array]:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : ArrayLike: jax.Array
            Batch of observations.

        Returns
        -------
        tuple[Distribution, jax.Array]
            - A `distrax.MultivariateNormalDiag` distribution over actions.
            - A value estimate tensor of shape ``(batch, 1)``.
        """
        raise NotImplementedError


from .MLP import SharedActorCritic, ActorCritic
from .LSTM import LSTMActorCritic

__all__ = ["Model", "SharedActorCritic", "ActorCritic", "LSTMActorCritic"]
