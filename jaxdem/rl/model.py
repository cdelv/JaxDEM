# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning models.
"""
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Tuple, Sequence
from abc import ABC, abstractmethod
import math

from flax import nnx
from flax.nnx.nn import recurrent as rnn
import distrax

from ..factory import Factory
from .actionSpace import ActionSpace


class Model(Factory, nnx.Module, ABC):
    """
    The base interface for defining reinforcement learning models.

    Models map observations to an action distribution and a value estimate.
    """

    @abstractmethod
    def __call__(self, obs: jax.Array) -> Tuple[distrax.Distribution, jax.Array]:
        """Evaluate the model.

        Parameters
        ----------
        obs : ArrayLike
            Batch of observations with shape ``(batch, *observation_space)``.

        Returns
        -------
        tuple[Distribution, jax.Array]
            A tuple containing the action distribution and the value estimate.
        """
        raise NotImplementedError


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
    observation_space : Sequence[int]
        Shape of the observation space (excluding batch dimension).
    action_space : Sequence[int]
        Shape of the action space.
    key : nnx.Rngs
        Random number generator(s) for parameter initialization.
    architecture : Sequence[int], default=[32, 32]
        Sizes of the hidden layers in the shared network.
    in_scale : float, default=sqrt(2)
        Scaling factor for orthogonal initialization of the shared network
        layers.
    actor_scale : float, default=1.0
        Scaling factor for orthogonal initialization of the actor head.
    critic_scale : float, default=0.01
        Scaling factor for orthogonal initialization of the critic head.
    activation : Callable, default=nnx.gelu
        Activation function applied between hidden layers.

    Attributes
    ----------
    network : nnx.Sequential
        The shared feedforward network (torso).
    actor : nnx.Linear
        Linear layer mapping shared features to action means.
    critic : nnx.Linear
        Linear layer mapping shared features to a scalar value estimate.
    log_std : nnx.Param
        Learnable log standard deviation for the Gaussian action distribution.
    """

    def __init__(
        self,
        *,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        architecture: Sequence[int] = [32, 32],
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        activation=nnx.gelu,
        action_space: distrax.Bijector | None = None,
    ):
        layers = []
        input_dim = observation_space_size
        out_dim = action_space_size

        for output_dim in architecture:
            layers.append(
                nnx.Linear(
                    in_features=input_dim,
                    out_features=output_dim,
                    kernel_init=nnx.initializers.orthogonal(in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            layers.append(activation)
            input_dim = output_dim

        self.network = nnx.Sequential(*layers)
        self.actor = nnx.Linear(
            in_features=input_dim,
            out_features=out_dim,
            kernel_init=nnx.initializers.orthogonal(actor_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )
        self.critic = nnx.Linear(
            in_features=input_dim,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(critic_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )
        self.log_std = nnx.Param(jnp.zeros((1, out_dim)))

        if action_space is None:
            action_space = ActionSpace.create("Free")

        # Check if bijector is scalar
        if action_space.event_ndims_in == 0:
            self.bij = distrax.Block(action_space, ndims=1)
        else:
            self.bij = action_space

    def __call__(self, x: jax.Array):
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
            - A value estimate tensor of shape ``(batch, 1)``.
        """
        x = self.network(x)
        pi = distrax.MultivariateNormalDiag(self.actor(x), jnp.exp(self.log_std.value))
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

    The actor outputs the mean of a Gaussian action distribution, while the
    log standard deviation is a learnable, state-independent parameter.

    Parameters
    ----------
    observation_space : Sequence[int]
        Shape of the observation space (excluding batch dimension).
    action_space : Sequence[int]
        Shape of the action space.
    key : nnx.Rngs
        Random number generator(s) for parameter initialization.
    actor_architecture : Sequence[int], default=[32, 32]
        Sizes of the hidden layers in the actor torso.
    critic_architecture : Sequence[int], default=[32, 32]
        Sizes of the hidden layers in the critic torso.
    in_scale : float, default=sqrt(2)
        Scaling factor for orthogonal initialization of hidden layers.
    actor_scale : float, default=1.0
        Scaling factor for orthogonal initialization of the actor head.
    critic_scale : float, default=0.01
        Scaling factor for orthogonal initialization of the critic head.
    activation : Callable, default=nnx.gelu
        Activation function applied between hidden layers.

    Attributes
    ----------
    actor_torso : nnx.Sequential
        Feedforward network for the actor.
    critic_torso : nnx.Sequential
        Feedforward network for the critic.
    actor : nnx.Linear
        Linear layer mapping actor features to action means.
    critic : nnx.Linear
        Linear layer mapping critic features to a scalar value estimate.
    log_std : nnx.Param
        Learnable log standard deviation for the Gaussian action distribution.
    """

    def __init__(
        self,
        *,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        actor_architecture: Sequence[int] = [32, 32],
        critic_architecture: Sequence[int] = [32, 32],
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        activation=nnx.gelu,
        action_space: distrax.Bijector | None = None,
    ):
        input_dim = observation_space_size
        out_dim = action_space_size

        # Build actor torso
        actor_layers = []
        actor_in = input_dim
        for output_dim in actor_architecture:
            actor_layers.append(
                nnx.Linear(
                    in_features=actor_in,
                    out_features=output_dim,
                    kernel_init=nnx.initializers.orthogonal(in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            actor_layers.append(activation)
            actor_in = output_dim
        self.actor_torso = nnx.Sequential(*actor_layers)

        # Build critic torso
        critic_layers = []
        critic_in = input_dim
        for output_dim in critic_architecture:
            critic_layers.append(
                nnx.Linear(
                    in_features=critic_in,
                    out_features=output_dim,
                    kernel_init=nnx.initializers.orthogonal(in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            critic_layers.append(activation)
            critic_in = output_dim
        self.critic_torso = nnx.Sequential(*critic_layers)

        # Actor head
        self.actor = nnx.Linear(
            in_features=actor_in,
            out_features=out_dim,
            kernel_init=nnx.initializers.orthogonal(actor_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        # Critic head
        self.critic = nnx.Linear(
            in_features=critic_in,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(critic_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        # Global log std for Gaussian policy
        self.log_std = nnx.Param(jnp.zeros((1, out_dim)))

        if action_space is None:
            action_space = ActionSpace.create("Free")

        # Check if bijector is scalar
        if action_space.event_ndims_in == 0:
            self.bij = distrax.Block(action_space, ndims=1)
        else:
            self.bij = action_space

    def __call__(self, x: jax.Array):
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
        critic_features = self.critic_torso(x)
        pi = distrax.MultivariateNormalDiag(
            self.actor(actor_features),
            jnp.exp(self.log_std.value),
        )
        pi = distrax.Transformed(pi, self.bij)
        return pi, self.critic(critic_features)
