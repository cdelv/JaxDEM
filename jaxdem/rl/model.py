# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning models.
"""
import jax
import jax.numpy as jnp

from typing import Callable, Tuple, Sequence, cast
from abc import ABC, abstractmethod
import math

from flax import nnx
from flax.nnx.nn import recurrent as rnn
import distrax

from ..factory import Factory
from .actionSpace import ActionSpace


class Model(Factory, nnx.Module, ABC):
    """
    The base interface for defining reinforcement learning models. Acts as a name space.

    Models map observations to an action distribution and a value estimate.
    """

    __slots__ = ()

    def reset(self, shape: Tuple[int], mask: jax.Array | None = None):
        """
        Reset the persistent LSTM carry.

        Parameters
        -----------
        lead_shape : tuple[int, ...]
            Leading dims for the carry, e.g. (num_envs, num_agents).
        mask : optional bool array
            True where to reset entries. Shape (num_envs)
        """

    @abstractmethod
    def __call__(
        self, x: jax.Array, sequence: bool = True
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
    architecture : Sequence[int]
        Sizes of the hidden layers in the shared network.
    in_scale : float
        Scaling factor for orthogonal initialization of the shared network
        layers.
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
    network : nnx.Sequential
        The shared feedforward network (torso).
    actor : nnx.Linear
        Linear layer mapping shared features to action means.
    critic : nnx.Linear
        Linear layer mapping shared features to a scalar value estimate.
    log_std : nnx.Param
        Learnable log standard deviation for the Gaussian action distribution.
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
        architecture: Sequence[int] = [32, 32],
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        activation: Callable = nnx.gelu,
        action_space: distrax.Bijector | ActionSpace | None = None,
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
        bij = cast(distrax.Bijector, action_space)
        if getattr(bij, "event_ndims_in", 0) == 0:
            bij = distrax.Block(bij, ndims=1)
        self.bij = bij

    def __call__(
        self, x: jax.Array, sequence: bool = True
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

    Parameters
    ----------
    observation_space : Sequence[int]
        Shape of the observation space (excluding batch dimension).
    action_space : Sequence[int]
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
    critic_torso : nnx.Sequential
        Feedforward network for the critic.
    actor : nnx.Linear
        Linear layer mapping actor features to action means.
    critic : nnx.Linear
        Linear layer mapping critic features to a scalar value estimate.
    log_std : nnx.Param
        Learnable log standard deviation for the Gaussian action distribution.
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
        actor_architecture: Sequence[int] = [32, 32],
        critic_architecture: Sequence[int] = [32, 32],
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        activation: Callable = nnx.gelu,
        action_space: distrax.Bijector | ActionSpace | None = None,
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
        bij = cast(distrax.Bijector, action_space)
        if getattr(bij, "event_ndims_in", 0) == 0:
            bij = distrax.Block(bij, ndims=1)
        self.bij = bij

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
        critic_features = self.critic_torso(x)
        pi = distrax.MultivariateNormalDiag(
            self.actor(actor_features),
            jnp.exp(self.log_std.value),
        )
        pi = distrax.Transformed(pi, self.bij)
        return pi, self.critic(critic_features)


@Model.register("LSTMActorCritic")
class LSTMActorCritic(Model, nnx.Module):
    """
    A recurrent actor–critic with an MLP encoder and an LSTM torso.

    This model encodes observations with a small feed-forward network, passes the
    features through a single-layer LSTM, and decodes the LSTM hidden state with
    linear policy/value heads.

    **Calling modes**

    - **Sequence mode (training)**: time-major input ``x`` with shape
        ``(T, B, obs_dim)`` produces a distribution and value for **every** step:
        policy outputs ``(T, B, action_space_size)`` and values ``(T, B, 1)``.
        The LSTM carry is initialized to zeros for the sequence.

    - **Single-step mode (evaluation/rollout)**: input ``x`` with shape
        ``(..., obs_dim)`` uses and updates a persistent LSTM carry stored on the
        module (``self.h``, ``self.c``); outputs have shape
        ``(..., action_space_size)`` and ``(..., 1)``. Use :meth:`reset_carry` to
        clear state between episodes. Carry needs to be reset every new trajectory.

    Parameters
    ----------
    observation_space_size : int
        Flattened observation size (``obs_dim``).
    action_space_size : int
        Number of action dimensions.
    key : nnx.Rngs
        Random number generator(s) for parameter initialization.
    hidden_features : int
        Width of the encoder output (and LSTM input).
    lstm_features : int
        LSTM hidden/state size. Also the feature size consumed by the heads.
    activation : Callable
        Activation function applied inside the encoder.
    action_space : distrax.Bijector | ActionSpace | None, default=None
        Bijector to constrain the policy probability distribution.

    Attributes
    ----------
    obs_dim : int
        Observation dimensionality expected on the last axis of ``x``.
    lstm_features : int
        LSTM hidden/state size.
    encoder : nnx.Sequential
        MLP that maps ``obs_dim -> hidden_features``.
    cell : rnn.OptimizedLSTMCell
        LSTM cell with ``in_features = hidden_features`` and ``hidden_features = lstm_features``.
    actor : nnx.Linear
        Linear head mapping LSTM features to action means.
    critic : nnx.Linear
        Linear head mapping LSTM features to a scalar value.
    log_std : nnx.Param
        Learnable log standard deviation for the policy.
    bij : distrax.Bijector
        Action-space bijector; scalar bijectors are automatically lifted with ``Block(ndims=1)`` for vector actions.
    h, c : nnx.Variable
        Persistent LSTM carry used by single-step evaluation. Shapes are
        ``(..., lstm_features)`` and are resized lazily to match the leading batch/agent dimensions.
    """

    __slots__ = ()

    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        hidden_features: int = 64,
        lstm_features: int = 128,
        activation: Callable = nnx.gelu,
        action_space: distrax.Bijector | ActionSpace | None = None,
        cell_type=rnn.OptimizedLSTMCell,
    ):
        super().__init__()
        self.obs_dim = int(observation_space_size)
        self.lstm_features = int(lstm_features)

        self.encoder = nnx.Sequential(
            nnx.Linear(
                in_features=observation_space_size,
                out_features=hidden_features,
                rngs=key,
            ),
            activation,
            nnx.Linear(
                in_features=hidden_features, out_features=hidden_features, rngs=key
            ),
            activation,
        )

        self.cell = cell_type(
            in_features=hidden_features, hidden_features=lstm_features, rngs=key
        )

        self.actor = nnx.Linear(
            in_features=lstm_features, out_features=action_space_size, rngs=key
        )
        self.critic = nnx.Linear(in_features=lstm_features, out_features=1, rngs=key)

        self.dropout = nnx.Dropout(0.1, rngs=key)

        self.log_std = nnx.Param(jnp.zeros((1, action_space_size)))

        if action_space is None:
            action_space = ActionSpace.create("Free")

        # Check if bijector is scalar
        bij = cast(distrax.Bijector, action_space)
        if getattr(bij, "event_ndims_in", 0) == 0:
            bij = distrax.Block(bij, ndims=1)
        self.bij = bij

        # Persistent carry for SINGLE-STEP usage (lives in nnx.State)
        # shape will be lazily set to x.shape[:-1] + (lstm_features,)
        self.h = nnx.Variable(jnp.zeros((0, lstm_features)))
        self.c = nnx.Variable(jnp.zeros((0, lstm_features)))

    def reset(self, shape: Tuple[int], mask: jax.Array | None = None):
        """
        Reset the persistent LSTM carry.

        - If `self.h.value.shape != (*lead_shape, H)`, allocate fresh zeros once.
        - Otherwise, zero in-place:
            * if `mask is None`: zero everything (without materializing a zeros tensor)
            * if `mask` is provided: zero only masked entries
              (mask may be shape `lead_shape` or `(*lead_shape, 1)` / `(*lead_shape, H)`)

        Args
        ----
        lead_shape : tuple[int, ...]
            Leading dims for the carry, e.g. (num_envs, num_agents).
        mask : optional bool array
            True where you want to reset entries. If shape is `lead_shape`, it will
            be expanded across the features dim.
        """
        H = self.lstm_features
        target_shape = (*shape, H)

        # If shape changed, allocate once and return
        if self.h.value.shape != target_shape:
            zeros = jnp.zeros(target_shape, dtype=float)
            self.h.value = zeros
            self.c.value = zeros
            return

        # If shape matches and everything needs reseting
        if mask is None:
            self.h.value *= 0.0
            self.c.value *= 0.0
            return

        # If shapes matches and masked reset
        self.h.value = jnp.where(mask[..., None, None], 0.0, self.h.value)
        self.c.value = jnp.where(mask[..., None, None], 0.0, self.c.value)

    def _zeros_carry(self, lead_shape):
        z = jnp.zeros((*lead_shape, self.lstm_features), dtype=float)
        return (z, z)

    def _ensure_persistent_carry(self, shape):
        target_shape = (*shape, self.lstm_features)
        if self.h.value.shape != target_shape:
            zeros = jnp.zeros(target_shape, dtype=float)
            self.h.value = zeros
            self.c.value = zeros

    def __call__(
        self, x: jax.Array, sequence: bool = True
    ) -> Tuple[distrax.Distribution, jax.Array]:
        """
        Remember to reset the carry each time starting a new trajectory.

        Accepts:
          - sequence = False: single step: x shape (..., obs_dim) -> uses persistent carry
          - sequence = True   : x shape (T, B, obs_dim) (time-major) -> starts from zero carry
        """
        if x.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected last dim {self.obs_dim}, got {x.shape}")

        feats = self.encoder(x)  # (..., hidden)
        if sequence:
            # Time-major: (T, B, obs)
            carry, h = jax.lax.scan(
                nnx.remat(self.cell), self._zeros_carry((x.shape[1],)), feats, unroll=4
            )  # hs: (T, B, lstm)
        else:
            # Snapshot eval mode (n_envs, n_agents, obs...)
            self._ensure_persistent_carry(x.shape[:-1])
            (self.h.value, self.c.value), h = self.cell(
                (self.h.value, self.c.value), feats
            )  # h: (..., lstm)

        h = self.dropout(h)
        pi = distrax.MultivariateNormalDiag(self.actor(h), jnp.exp(self.log_std.value))
        pi = distrax.Transformed(pi, self.bij)
        return pi, self.critic(h)
