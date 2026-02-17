# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Implementation of reinforcement learning models based on simple MLPs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import Any, Callable, Dict, Sequence, Tuple, cast
import math
from functools import partial

from flax import nnx
import distrax

from . import Model
from ..actionSpaces import ActionSpace
from ...utils import encode_callable


@Model.register("SharedActorCritic")
class SharedActorCritic(Model):
    """
    A shared-parameter dense actor-critic model.

    This model uses a common feedforward network (the "shared torso") to
    process observations, and then branches into two separate linear heads:
    - **Actor head**: outputs the mean of a Gaussian action distribution.
    - **Critic head**: outputs a scalar value estimate of the state.

    Parameters
    ----------
    observation_space : int
        Shape of the observation space (excluding batch dimension).
    action_space : int
        Shape of the action space.
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
        Bijector to constrain the policy probability distribution

    Attributes
    ----------
    network : nnx.Sequential
        The shared feedforward network (torso).
    actor_mu : nnx.Linear
        Linear layer mapping shared features to the policy distribution means.
    actor_sigma : nnx.Sequential
        Linear layer mapping LSTM features to the policy distribution standard deviations if actor_sigma_head is true, else independent parameter.
    critic : nnx.Linear
        Linear layer mapping shared features to the value estimate.
    bij: Distrax.bijector:
        Bijector for constraining the action space.
    """

    __slots__ = ()

    def __init__(
        self,
        *,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        architecture: Sequence[int] = [42, 42, 42],
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        actor_sigma_head: bool = False,
        activation: Callable[..., Any] = nnx.gelu,
        action_space: ActionSpace | None = None,
    ):
        self.observation_space_size = int(observation_space_size)
        self.action_space_size = int(action_space_size)
        self.architecture = [int(x) for x in architecture]
        self.in_scale = float(in_scale)
        self.actor_scale = float(actor_scale)
        self.critic_scale = float(critic_scale)
        self.actor_sigma_head = actor_sigma_head
        self.activation = activation

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
        self.actor_mu = nnx.Linear(
            in_features=input_dim,
            out_features=out_dim,
            kernel_init=nnx.initializers.orthogonal(self.actor_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        self._log_std = nnx.Param(jnp.zeros((1, self.action_space_size)))
        self._actor_sigma = nnx.Sequential(
            nnx.Linear(
                in_features=input_dim,
                out_features=out_dim,
                kernel_init=nnx.initializers.orthogonal(self.critic_scale),
                bias_init=nnx.initializers.constant(-1.0),
                rngs=key,
            ),
            jax.nn.softplus,
        )

        self.actor_sigma: Callable[[jax.Array], jax.Array]
        if self.actor_sigma_head:

            def _sigma_head(x: jax.Array) -> jax.Array:
                return self._actor_sigma(x)

            self.actor_sigma = _sigma_head
        else:

            def _sigma_param(_: jax.Array) -> jax.Array:
                return jnp.exp(self._log_std.value)

            self.actor_sigma = _sigma_param

        self.critic = nnx.Linear(
            in_features=input_dim,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(self.critic_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        if action_space is None:
            action_space = ActionSpace.create("Free")

        # Check if bijector is scalar
        bij = cast(distrax.Bijector, action_space)
        if getattr(bij, "event_ndims_in", 0) == 0:
            bij = distrax.Block(bij, ndims=1)
        self.bij = bij

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(
            observation_space_size=self.observation_space_size,
            action_space_size=self.action_space_size,
            architecture=self.architecture,
            in_scale=self.in_scale,
            actor_scale=self.actor_scale,
            critic_scale=self.critic_scale,
            actor_sigma_head=self.actor_sigma_head,
            activation=encode_callable(self.activation),
            action_space_type=self.bij.type_name,
            action_space_kws=self.bij.kws,
        )

    @partial(jax.named_call, name="SharedActorCritic.__call__")
    def __call__(
        self, x: jax.Array, sequence: bool = False
    ) -> Tuple[distrax.Distribution, jax.Array]:
        """
        Forward pass of the shared actor-critic model.

        Parameters
        ----------
        x : ArrayLike: jax.Array
            Batch of observations with shape ``(batch, *flatten(observation_space))``.

        Returns
        -------
        tuple[Distribution, jax.Array]
            - A `distrax.MultivariateNormalDiag` distribution over actions.
            - A value estimate tensor
        """
        x = self.network(x)
        pi = distrax.MultivariateNormalDiag(self.actor_mu(x), self.actor_sigma(x))
        pi = distrax.Transformed(pi, self.bij)
        return pi, self.critic(x)


@Model.register("ActorCritic")
class ActorCritic(Model, nnx.Module):
    """
    An actor-critic model with separate networks for the actor and critic.

    Unlike `SharedActorCritic`, this model uses two independent feedforward
    networks:
    - **Actor torso**: processes observations into features for the policy.
    - **Critic torso**: processes observations into features for the value function.

    Parameters
    ----------
    observation_space : int
        Shape of the observation space (excluding batch dimension).
    action_space : int
        Shape of the action space.
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
        Bijector to constrain the policy probability distribution

    Attributes
    ----------
    actor_torso : nnx.Sequential
        Feedforward network for the actor.
    critic : nnx.Sequential
        Feedforward network for the critic.
    actor_mu : nnx.Linear
        Linear layer mapping actor_torso's features to the policy distribution means.
    actor_sigma : nnx.Sequential
        Linear layer mapping LSTM features to the policy distribution standard deviations if actor_sigma_head is true, else independent parameter.
    bij: Distrax.bijector:
        Bijector for constraining the action space.
    """

    __slots__ = ()

    def __init__(
        self,
        *,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        actor_architecture: Sequence[int] = [42, 42, 42],
        critic_architecture: Sequence[int] = [42, 42, 42],
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        actor_sigma_head: bool = False,
        activation: Callable[..., Any] = nnx.gelu,
        action_space: distrax.Bijector | ActionSpace | None = None,
    ):
        self.observation_space_size = int(observation_space_size)
        self.action_space_size = int(action_space_size)
        self.actor_architecture = [int(x) for x in actor_architecture]
        self.critic_architecture = [int(x) for x in critic_architecture]
        self.in_scale = float(in_scale)
        self.actor_scale = float(actor_scale)
        self.critic_scale = float(critic_scale)
        self.actor_sigma_head = actor_sigma_head
        self.activation = activation

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

        critic_layers.append(
            nnx.Linear(
                in_features=critic_in,
                out_features=1,
                kernel_init=nnx.initializers.orthogonal(self.critic_scale),
                bias_init=nnx.initializers.constant(0.0),
                rngs=key,
            )
        )
        self.critic = nnx.Sequential(*critic_layers)

        # Actor heads
        self.actor_mu = nnx.Linear(
            in_features=actor_in,
            out_features=out_dim,
            kernel_init=nnx.initializers.orthogonal(self.actor_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        self._log_std = nnx.Param(jnp.zeros((1, self.action_space_size)))
        self._actor_sigma = nnx.Sequential(
            nnx.Linear(
                in_features=input_dim,
                out_features=out_dim,
                kernel_init=nnx.initializers.orthogonal(self.critic_scale),
                bias_init=nnx.initializers.constant(-1.0),
                rngs=key,
            ),
            jax.nn.softplus,
        )

        self.actor_sigma: Callable[[jax.Array], jax.Array]
        if self.actor_sigma_head:

            def _sigma_head(x: jax.Array) -> jax.Array:
                return self._actor_sigma(x)

            self.actor_sigma = _sigma_head
        else:

            def _sigma_param(_: jax.Array) -> jax.Array:
                return jnp.exp(self._log_std.value)

            self.actor_sigma = _sigma_param

        if action_space is None:
            action_space = ActionSpace.create("Free")

        # Check if bijector is scalar
        bij = cast(distrax.Bijector, action_space)
        if getattr(bij, "event_ndims_in", 0) == 0:
            bij = distrax.Block(bij, ndims=1)
        self.bij = bij

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(
            observation_space_size=self.observation_space_size,
            action_space_size=self.action_space_size,
            actor_architecture=self.actor_architecture,
            critic_architecture=self.critic_architecture,
            in_scale=self.in_scale,
            actor_scale=self.actor_scale,
            critic_scale=self.critic_scale,
            actor_sigma_head=self.actor_sigma_head,
            activation=encode_callable(self.activation),
            action_space_type=self.bij.type_name,
            action_space_kws=self.bij.kws,
        )

    @partial(jax.named_call, name="ActorCritic.__call__")
    def __call__(
        self, x: jax.Array, sequence: bool = True
    ) -> Tuple[distrax.Distribution, jax.Array]:
        """
        Forward pass of the actor-critic model with separate torsos.

        Parameters
        ----------
        x : ArrayLike
            Batch of observations with shape ``(batch, *flatten(observation_space))``.

        Returns
        -------
        tuple[Distribution, jax.Array]
            - A `distrax.MultivariateNormalDiag` distribution over actions.
            - A value estimate tensor of shape ``(batch, 1)``.
        """
        actor_features = self.actor_torso(x)
        pi = distrax.MultivariateNormalDiag(
            self.actor_mu(actor_features), self.actor_sigma(actor_features)
        )
        pi = distrax.Transformed(pi, self.bij)
        return pi, self.critic(x)


__all__ = ["SharedActorCritic", "ActorCritic"]
