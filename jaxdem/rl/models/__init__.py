# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Interface for defining reinforcement learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import distrax  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
from flax import nnx

from ...factory import Factory


class Model(Factory, nnx.Module, ABC):  # type: ignore[misc]
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

    #: Whether the model emits a categorical (discrete) action distribution.
    #: Subclasses set this in ``__init__`` before calling
    #: :meth:`_init_policy_head`.
    discrete: bool

    @property
    def action_space(self) -> Any:
        if hasattr(self, "bij") and self.bij is not None:
            inner = getattr(self.bij, "_bijector", self.bij)
            from ..action_spaces import ActionSpace

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

    def snapshot_rollout_carry(self) -> None:
        """Snapshot the current persistent carry as the rollout-initial carry.

        Trainers call this right before a rollout so that training-time
        sequence replays can start from the exact carry the rollout used.
        No-op for stateless models.
        """
        return

    def sequence_initial_carry(
        self, idx: jax.Array
    ) -> tuple[jax.Array, jax.Array] | None:
        """Return the rollout-initial carry for the given flat segment indices.

        Parameters
        ----------
        idx : jax.Array
            Integer indices into the flattened ``(num_envs * num_agents,)``
            segment axis selecting the minibatch.

        Returns
        -------
        tuple[jax.Array, jax.Array] | None
            Carry tuple (e.g. LSTM ``(c, h)``) sliced to ``idx``, or ``None``
            for stateless models.
        """
        return None

    def _init_policy_head(
        self,
        *,
        in_features: int,
        action_space_size: int,
        key: nnx.Rngs,
        actor_scale: float = 1.0,
        sigma_scale: float = 0.01,
        actor_sigma_head: bool = False,
        action_space: Any = None,
        discrete: bool = False,
    ) -> None:
        """Build the policy head shared by every actor-critic model.

        Creates ``actor_mu`` (mean head for continuous actions, logits head
        for discrete ones), ``actor_sigma`` (a learned per-action parameter or
        a dedicated dense head when ``actor_sigma_head=True``) and ``bij``
        (the action-space bijector; ``None`` for discrete actions), assigning
        them as attributes on ``self``.

        Parameters
        ----------
        in_features : int
            Feature size of the torso output feeding the heads.
        action_space_size : int
            Number of action dimensions (continuous) or discrete actions.
        key : nnx.Rngs
            RNG used for parameter initialization.
        actor_scale : float
            Orthogonal-initialization scale of the mean/logits head.
        sigma_scale : float
            Orthogonal-initialization scale of the sigma head.
        actor_sigma_head : bool
            If True, predict sigma from the features with a dense head;
            otherwise use a state-independent learned parameter.
        action_space : ActionSpace | distrax.Bijector | None
            Bijector constraining the action distribution (continuous only).
        discrete : bool
            Whether the model emits a categorical distribution.
        """
        from ..action_spaces import ActionSpace

        out_dim = int(action_space_size)
        # For discrete: logits head, for continuous: mean head
        self.actor_mu = nnx.Linear(
            in_features=in_features,
            out_features=out_dim,
            kernel_init=nnx.initializers.orthogonal(actor_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        # Sigma parameters are only needed for continuous actions; allocate
        # only the variant that is actually used.
        self.actor_sigma: Callable[[jax.Array], jax.Array]
        if not discrete:
            if actor_sigma_head:
                self._actor_sigma = nnx.Sequential(
                    nnx.Linear(
                        in_features=in_features,
                        out_features=out_dim,
                        kernel_init=nnx.initializers.orthogonal(sigma_scale),
                        bias_init=nnx.initializers.constant(-1.0),
                        rngs=key,
                    ),
                    jax.nn.softplus,
                )

                def _sigma_head(x: jax.Array) -> jax.Array:
                    return self._actor_sigma(x)

                self.actor_sigma = _sigma_head
            else:
                self._log_std = nnx.Param(jnp.zeros((1, out_dim)))

                def _sigma_param(_: jax.Array) -> jax.Array:
                    return jnp.exp(self._log_std.value)

                self.actor_sigma = _sigma_param

        # Bijector only used for continuous actions
        self.bij: distrax.Bijector | None = None
        if not discrete:
            if action_space is None:
                action_space = ActionSpace.create("Free")

            # Check if bijector is scalar
            bij = cast(distrax.Bijector, action_space)
            if getattr(bij, "event_ndims_in", 0) == 0:
                bij = distrax.Block(bij, ndims=1)
            self.bij = nnx.data(bij)

    @staticmethod
    def _critic_head(
        in_features: int, key: nnx.Rngs, scale: float = 0.01
    ) -> nnx.Linear:
        """Build the scalar value head shared by every actor-critic model."""
        return nnx.Linear(
            in_features=in_features,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

    def _policy_distribution(self, features: jax.Array) -> distrax.Distribution:
        """Map torso features to the action distribution via the policy head."""
        from ..action_spaces import Transformed

        if self.discrete:
            return distrax.Categorical(logits=self.actor_mu(features))
        pi = distrax.MultivariateNormalDiag(
            self.actor_mu(features), self.actor_sigma(features)
        )
        return Transformed(pi, self.bij)

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


from .lstm import LSTMActorCritic
from .mlp import ActorCritic, SharedActorCritic

__all__ = [
    "ActorCritic",
    "LSTMActorCritic",
    "Model",
    "SharedActorCritic",
]
