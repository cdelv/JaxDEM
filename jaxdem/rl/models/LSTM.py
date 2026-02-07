# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Implementation of reinforcement learning models based on a single layer LSTM.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import Any, Callable, Dict, Optional, Tuple, cast
from functools import partial

from flax import nnx
import flax.nnx.nn.recurrent as rnn
import distrax

from . import Model
from ..actionSpaces import ActionSpace
from ...utils import encode_callable


@Model.register("LSTMActorCritic")
class LSTMActorCritic(Model, nnx.Module):  # type: ignore[misc]
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
    actor_mu : nnx.Linear
        Linear layer mapping LSTM features to the policy distribution means.
    actor_sigma : nnx.Sequential
        Linear layer mapping LSTM features to the policy distribution standard deviations if actor_sigma_head is true, else independent parameter.
    critic : nnx.Linear
        Linear head mapping LSTM features to a scalar value.
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
        dropout_rate: float = 0.1,
        activation: Callable[..., Any] = nnx.gelu,
        action_space: distrax.Bijector | ActionSpace | None = None,
        cell_type: type[rnn.OptimizedLSTMCell] = rnn.OptimizedLSTMCell,
        remat: bool = False,
        actor_sigma_head: bool = False,
        carry_leading_shape: Tuple[int, ...] = (),
    ):
        super().__init__()
        self.obs_dim = int(observation_space_size)
        self.action_space_size = int(action_space_size)
        self.hidden_features = int(hidden_features)
        self.lstm_features = int(lstm_features)
        self.dropout_rate = float(dropout_rate)
        self.activation = activation
        self.remat = remat
        self.cell_type = cell_type
        self.actor_sigma_head = actor_sigma_head

        self.encoder = nnx.Sequential(
            nnx.Linear(
                self.obs_dim,
                self.hidden_features,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=key,
            ),
            self.activation,
        )

        self.cell = self.cell_type(
            in_features=self.hidden_features,
            hidden_features=self.lstm_features,
            rngs=key,
        )

        if self.remat:
            self.cell.__call__ = nnx.remat(self.cell.__call__)

        self.rnn = rnn.RNN(self.cell)

        self.actor_mu = nnx.Linear(
            in_features=self.lstm_features,
            out_features=self.action_space_size,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        self._log_std = nnx.Param(jnp.zeros((1, self.action_space_size)))
        self._actor_sigma = nnx.Sequential(
            nnx.Linear(
                in_features=self.lstm_features,
                out_features=self.action_space_size,
                kernel_init=nnx.initializers.orthogonal(0.01),
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
            in_features=self.lstm_features,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(0.01),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        # self.dropout = nnx.Dropout(self.dropout_rate, rngs=key)

        if action_space is None:
            action_space = ActionSpace.create("Free")

        # Check if bijector is scalar
        bij = cast(distrax.Bijector, action_space)
        if getattr(bij, "event_ndims_in", 0) == 0:
            bij = distrax.Block(bij, ndims=1)
        self.bij = bij

        # Persistent carry for SINGLE-STEP usage (lives in nnx.State)
        # shape will be lazily set to x.shape[:-1] + (lstm_features,)
        H = int(self.lstm_features)
        lead = tuple(carry_leading_shape)
        self.h = nnx.Variable(jnp.zeros(lead + (H,), dtype=float))
        self.c = nnx.Variable(jnp.zeros(lead + (H,), dtype=float))

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(
            observation_space_size=self.obs_dim,
            action_space_size=self.action_space_size,
            hidden_features=self.hidden_features,
            lstm_features=self.lstm_features,
            dropout_rate=self.dropout_rate,
            activation=encode_callable(self.activation),
            action_space_type=self.bij.type_name,
            action_space_kws=self.bij.kws,
            remat=self.remat,
            actor_sigma_head=self.actor_sigma_head,
            cell_type=encode_callable(self.cell_type),
            carry_leading_shape=self.h.value.shape[:-1],
        )

    @partial(jax.named_call, name="LSTMActorCritic.reset")
    def reset(self, shape: Tuple[int, ...], mask: jax.Array | None = None) -> None:
        """
        Reset the persistent LSTM carry.

        - If `self.h.value.shape != (*lead_shape, H)`, allocate fresh zeros once.

        - Otherwise, zero in-place:
            * if `mask is None`: zero everything
            * if `mask` is provided: zero masked entries along axis=0

        Parameters
        -----------
        shape : tuple[int, ...]
            Shape of the observation (input) tensor.
        mask : optional bool array
            Mask per environment (axis=0) to conditionally reset the carry.
        """
        H = self.lstm_features
        target_shape = (*shape[:-1], H)

        # If shape changed, allocate once and return
        if self.h.value.shape != target_shape:
            self.h.value = jnp.zeros(target_shape, dtype=float)
            self.c.value = jnp.zeros(target_shape, dtype=float)
            return

        # If shape matches and everything needs resetting
        if mask is None:
            self.h.value *= 0.0
            self.c.value *= 0.0
            return

        # If shapes matches and masked reset
        mask = mask.reshape((mask.shape[0],) + (1,) * (self.h.value.ndim - 1))
        self.h.value = jnp.where(mask, 0.0, self.h.value)
        self.c.value = jnp.where(mask, 0.0, self.c.value)

    @partial(jax.named_call, name="LSTMActorCritic.__call__")
    def __call__(
        self, x: jax.Array, sequence: bool = False
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
            y = self.rnn(feats, time_major=True)
        else:
            batch = feats.shape[:-1]
            target = (*batch, self.lstm_features)

            # Lazily allocate carry via the cell API when shape mismatches
            if self.h.value.shape != target:
                c0, h0 = self.cell.initialize_carry(
                    input_shape=feats.shape, rngs=nnx.Rngs(0)
                )
                self.c.value, self.h.value = c0, h0  # carry order = (c, h)

            (c1, h1), y = self.cell((self.c.value, self.h.value), feats)  # (..., H)
            self.c.value, self.h.value = c1, h1

        h = y
        pi = distrax.MultivariateNormalDiag(self.actor_mu(h), self.actor_sigma(h))
        pi = distrax.Transformed(pi, self.bij)
        return pi, self.critic(h)


__all__ = ["LSTMActorCritic"]
