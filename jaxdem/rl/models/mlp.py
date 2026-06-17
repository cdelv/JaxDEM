# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of reinforcement learning models based on simple MLPs."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import distrax  # type: ignore[import-untyped]
import jax
from flax import nnx

from ..action_spaces import ActionSpace
from . import Model


@Model.register("SharedActorCritic")
class SharedActorCritic(Model):
    """A shared-parameter dense actor-critic model.

    This model uses a common feedforward network (the "shared torso") to
    process observations, and then branches into two separate linear heads:

    - **Actor head**: outputs the mean of a Gaussian action distribution (continuous)
      or logits for a categorical distribution (discrete).
    - **Critic head**: outputs a scalar value estimate of the state.

    Parameters
    ----------
    observation_space : int
        Shape of the observation space (excluding batch dimension).
    action_space : int
        Shape of the action space (for continuous) or number of discrete actions.
    key : nnx.Rngs
        Random number generator(s) for parameter initialization.
    architecture : Sequence[int]
        Sizes of the hidden layers in the shared network.
    in_scale : float
        Scaling factor for orthogonal initialization of the shared network layers.
    actor_scale : float
        Scaling factor for orthogonal initialization of the actor head.
    critic_scale : float
        Scaling factor for orthogonal initialization of the critic head.
    activation : Callable
        JIT-compatible activation function applied between hidden layers.
    action_space: ActionSpace
        Bijector to constrain the policy probability distribution (continuous only).
    discrete : bool
        If True, use a categorical distribution for discrete actions.
        If False (default), use a Gaussian distribution for continuous actions.

    Attributes
    ----------
    network : nnx.Sequential
        The shared feedforward network (torso).
    actor_mu : nnx.Linear
        Linear layer mapping shared features to the policy distribution means
        (continuous) or logits (discrete).
    actor_sigma : nnx.Sequential
        Linear layer mapping shared features to the policy distribution standard
        deviations if actor_sigma_head is true, else independent parameter.
        Only used for continuous actions.
    critic : nnx.Linear
        Linear layer mapping shared features to the value estimate.
    bij : distrax.Bijector
        Bijector for constraining the action space (continuous only).
    discrete : bool
        Whether this model uses discrete actions.

    """

    __slots__ = ()

    def __init__(
        self,
        *,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        architecture: Sequence[int] | None = None,
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        actor_sigma_head: bool = False,
        activation: Callable[..., Any] = nnx.gelu,
        action_space: ActionSpace | None = None,
        discrete: bool = False,
    ):
        if architecture is None:
            architecture = [42, 42, 42]
        self.observation_space_size = int(observation_space_size)
        self.action_space_size = int(action_space_size)
        self.architecture = [int(x) for x in architecture]
        self.in_scale = float(in_scale)
        self.actor_scale = float(actor_scale)
        self.critic_scale = float(critic_scale)
        self.actor_sigma_head = actor_sigma_head
        self.activation = activation
        self.discrete = discrete

        layers: list[Any] = []
        input_dim = self.observation_space_size
        out_dim = self.action_space_size

        for output_dim in self.architecture:
            layers.append(
                nnx.Linear(
                    in_features=input_dim,
                    out_features=output_dim,
                    kernel_init=nnx.initializers.orthogonal(self.in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            layers.append(self.activation)
            input_dim = output_dim

        self.network = nnx.Sequential(*layers)
        self._init_policy_head(
            in_features=input_dim,
            action_space_size=out_dim,
            key=key,
            actor_scale=self.actor_scale,
            sigma_scale=self.critic_scale,
            actor_sigma_head=self.actor_sigma_head,
            action_space=action_space,
            discrete=discrete,
        )
        self.critic = self._critic_head(input_dim, key, self.critic_scale)

    @partial(jax.named_call, name="SharedActorCritic.__call__")
    def __call__(
        self, x: jax.Array, sequence: bool = False
    ) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass of the shared actor-critic model.

        Parameters
        ----------
        x : jax.Array
            Batch of observations with shape ``(batch, obs_dim)``
            or ``(T, batch, obs_dim)`` when ``sequence=True``.
        sequence : bool
            Ignored (present for interface compatibility with recurrent
            models).

        Returns
        -------
        tuple[Distribution, jax.Array]
            - A `distrax.MultivariateNormalDiag` distribution over actions
              (continuous) or `distrax.Categorical` (discrete).
            - A value estimate tensor of shape ``(batch, 1)``.

        """
        x = self.network(x)
        return self._policy_distribution(x), self.critic(x)


@Model.register("ActorCritic")
class ActorCritic(Model):
    """An actor-critic model with separate networks for the actor and critic.

    Unlike `SharedActorCritic`, this model uses two independent feedforward
    networks:
    - **Actor torso**: processes observations into features for the policy.
    - **Critic torso**: processes observations into features for the value function.

    Parameters
    ----------
    observation_space : int
        Shape of the observation space (excluding batch dimension).
    action_space : int
        Shape of the action space (for continuous) or number of discrete actions.
    key : nnx.Rngs
        Random number generator(s) for parameter initialization.
    actor_architecture : Sequence[int]
        Sizes of the hidden layers in the actor torso.
    critic_architecture : Sequence[int]
        Sizes of the hidden layers in the critic torso.
    in_scale : float
        Scaling factor for orthogonal initialization of hidden layers.
    actor_scale : float
        Scaling factor for orthogonal initialization of the actor head.
    critic_scale : float
        Scaling factor for orthogonal initialization of the critic head.
    activation : Callable
        Activation function applied between hidden layers.
    action_space: ActionSpace
        Bijector to constrain the policy probability distribution (continuous only).
    discrete : bool
        If True, use a categorical distribution for discrete actions.
        If False (default), use a Gaussian distribution for continuous actions.

    Attributes
    ----------
    actor_torso : nnx.Sequential
        Feedforward network for the actor.
    critic : nnx.Sequential
        Feedforward network for the critic.
    actor_mu : nnx.Linear
        Linear layer mapping actor_torso's features to the policy distribution
        means (continuous) or logits (discrete).
    actor_sigma : nnx.Sequential
        Linear layer mapping actor torso features to the policy distribution
        standard deviations if actor_sigma_head is true, else independent parameter.
        Only used for continuous actions.
    bij : distrax.Bijector
        Bijector for constraining the action space (continuous only).
    discrete : bool
        Whether this model uses discrete actions.

    """

    __slots__ = ()

    def __init__(
        self,
        *,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        actor_architecture: Sequence[int] | None = None,
        critic_architecture: Sequence[int] | None = None,
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        actor_sigma_head: bool = False,
        activation: Callable[..., Any] = nnx.gelu,
        action_space: distrax.Bijector | ActionSpace | None = None,
        discrete: bool = False,
    ):
        if actor_architecture is None:
            actor_architecture = [42, 42, 42]
        if critic_architecture is None:
            critic_architecture = [42, 42, 42]
        self.observation_space_size = int(observation_space_size)
        self.action_space_size = int(action_space_size)
        self.actor_architecture = [int(x) for x in actor_architecture]
        self.critic_architecture = [int(x) for x in critic_architecture]
        self.in_scale = float(in_scale)
        self.actor_scale = float(actor_scale)
        self.critic_scale = float(critic_scale)
        self.actor_sigma_head = actor_sigma_head
        self.activation = activation
        self.discrete = discrete

        input_dim = self.observation_space_size
        out_dim = self.action_space_size

        # Build actor torso
        actor_layers: list[Any] = []
        actor_in = input_dim
        for output_dim in actor_architecture:
            actor_layers.append(
                nnx.Linear(
                    in_features=actor_in,
                    out_features=output_dim,
                    kernel_init=nnx.initializers.orthogonal(self.in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            actor_layers.append(activation)
            actor_in = output_dim
        self.actor_torso = nnx.Sequential(*actor_layers)

        # Build critic
        critic_layers: list[Any] = []
        critic_in = input_dim
        for output_dim in critic_architecture:
            critic_layers.append(
                nnx.Linear(
                    in_features=critic_in,
                    out_features=output_dim,
                    kernel_init=nnx.initializers.orthogonal(self.in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            critic_layers.append(activation)
            critic_in = output_dim

        critic_layers.append(self._critic_head(critic_in, key, self.critic_scale))
        self.critic = nnx.Sequential(*critic_layers)

        self._init_policy_head(
            in_features=actor_in,
            action_space_size=out_dim,
            key=key,
            actor_scale=self.actor_scale,
            sigma_scale=self.critic_scale,
            actor_sigma_head=self.actor_sigma_head,
            action_space=action_space,
            discrete=discrete,
        )

    @partial(jax.named_call, name="ActorCritic.__call__")
    def __call__(
        self, x: jax.Array, sequence: bool = False
    ) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass of the actor-critic model with separate torsos.

        Parameters
        ----------
        x : jax.Array
            Batch of observations with shape ``(batch, obs_dim)``
            or ``(T, batch, obs_dim)`` when ``sequence=True``.
        sequence : bool
            Ignored (present for interface compatibility with recurrent
            models).

        Returns
        -------
        tuple[Distribution, jax.Array]
            - A `distrax.MultivariateNormalDiag` distribution over actions
              (continuous) or `distrax.Categorical` (discrete).
            - A value estimate tensor of shape ``(batch, 1)``.

        """
        actor_features = self.actor_torso(x)
        return self._policy_distribution(actor_features), self.critic(x)


__all__ = ["ActorCritic", "SharedActorCritic"]
