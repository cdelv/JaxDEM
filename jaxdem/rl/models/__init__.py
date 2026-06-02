# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Interface for defining reinforcement learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import distrax  # type: ignore[import-untyped]
import jax
from flax import nnx

from ...factory import Factory


class Model(Factory, nnx.Module, ABC):
    """The base interface for defining reinforcement learning models. Acts as a namespace.

    Models map observations to an action distribution and a value estimate.

    Example:
    --------
    To define a custom model, inherit from :class:`Model` and implement its abstract methods:

    >>> @Model.register("myCustomModel")
    >>> class MyCustomModel(Model):
            ...

    """

    __slots__ = ()

    @property
    def action_space(self) -> Any:
        if hasattr(self, "bij") and self.bij is not None:
            inner = getattr(self.bij, "_bijector", self.bij)
            from ..actionSpaces import ActionSpace

            if isinstance(inner, ActionSpace):
                return inner
        return None

    def reset(self, shape: tuple[int, ...], mask: jax.Array | None = None) -> None:
        """Reset the persistent LSTM carry.

        Parameters
        ----------
        shape : tuple[int, ...]
            Leading dims for the carry, e.g. (num_envs, num_agents).
        mask : optional bool array
            True where to reset entries. Shape (num_envs)

        """
        return

    @abstractmethod
    def __call__(
        self, x: jax.Array, sequence: bool = False
    ) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass of the model.

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


from .LSTM import LSTMActorCritic
from .MLP import ActorCritic, SharedActorCritic

# from .ConvLSTM import ConvLSTMActorCritic
# from .GNNLSTM import GNNLSTMActorCritic

__all__ = [
    "ActorCritic",
    # "ConvLSTMActorCritic",
    # "GNNLSTMActorCritic",
    "LSTMActorCritic",
    "Model",
    "SharedActorCritic",
]
