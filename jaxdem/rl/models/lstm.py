# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of reinforcement learning models based on a single layer LSTM."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import Any
from collections.abc import Callable
from functools import partial

from flax import nnx
import flax.nnx.nn.recurrent as rnn
import distrax  # type: ignore[import-untyped]

from . import Model
from ..action_spaces import ActionSpace


@Model.register("LSTMActorCritic")
class LSTMActorCritic(Model):
    """A recurrent actor–critic with an MLP encoder and an LSTM torso.

    This model encodes observations with a small feed-forward network, passes
    the features through a single-layer LSTM, and decodes the LSTM hidden
    state with linear policy/value heads.

    **Calling modes**

    - **Sequence mode (training)**: time-major input ``x`` with shape
      ``(T, B, obs_dim)`` produces a distribution and value for **every**
      step: policy outputs ``(T, B, action_space_size)`` and values
      ``(T, B, 1)``.  The LSTM carry is initialized to zeros.

    - **Single-step mode (evaluation/rollout)**: input ``x`` with shape
      ``(..., obs_dim)`` uses and updates a persistent LSTM carry stored
      on the module (``self.h``, ``self.c``); outputs have shape
      ``(..., action_space_size)`` and ``(..., 1)``.  Use :meth:`reset`
      to clear state between episodes.

    Parameters
    ----------
    observation_space_size : int
        Flattened observation size (``obs_dim``).
    action_space_size : int
        Number of action dimensions (for continuous) or number of discrete actions.
    key : nnx.Rngs
        Random number generator(s) for parameter initialization.
    hidden_features : int
        Width of the encoder output (and LSTM input).
    lstm_features : int
        LSTM hidden/state size.  Also the feature size consumed by the
        policy and value heads.
    activation : Callable
        Activation function applied inside the encoder.
    action_space : distrax.Bijector | ActionSpace | None
        Bijector to constrain the policy probability distribution (continuous only).
    cell_type : type[rnn.OptimizedLSTMCell]
        LSTM cell class used for the recurrent layer.
    remat : bool
        If ``True``, wrap the LSTM scan body with
        ``jax.checkpoint`` to reduce memory in sequence mode.
    actor_sigma_head : bool
        If ``True``, the standard deviation is produced by a learned
        head on the LSTM output; otherwise an independent log-std
        parameter is used. Only used for continuous actions.
    carry_leading_shape : tuple[int, ...]
        Leading dimensions for the persistent carry tensors ``h`` and
        ``c``.  Typically ``()`` at construction; resized lazily at
        runtime to match the batch shape.
    discrete : bool
        If True, use a categorical distribution for discrete actions.
        If False (default), use a Gaussian distribution for continuous actions.

    Attributes
    ----------
    obs_dim : int
        Observation dimensionality expected on the last axis of ``x``.
    lstm_features : int
        LSTM hidden/state size.
    encoder : nnx.Sequential
        MLP that maps ``obs_dim → hidden_features``.
    cell : rnn.OptimizedLSTMCell
        LSTM cell with ``in_features = hidden_features`` and
        ``hidden_features = lstm_features``.
    actor_mu : nnx.Linear
        Linear layer mapping LSTM features to the policy distribution
        means (continuous) or logits (discrete).
    actor_sigma : Callable[[jax.Array], jax.Array]
        Maps LSTM features to the policy standard deviations (learned
        head when ``actor_sigma_head=True``, else independent
        parameter). Only used for continuous actions.
    critic : nnx.Linear
        Linear head mapping LSTM features to a scalar value.
    bij : distrax.Bijector
        Action-space bijector; scalar bijectors are automatically
        lifted with ``Block(ndims=1)`` for vector actions (continuous only).
    h, c : nnx.Variable
        Persistent LSTM carry used by single-step evaluation.  Shapes
        are ``(..., lstm_features)`` and are resized lazily to match
        the leading batch/agent dimensions.
    discrete : bool
        Whether this model uses discrete actions.

    """

    __slots__ = ()

    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        hidden_features: int = 64,
        lstm_features: int = 128,
        activation: Callable[..., Any] = nnx.gelu,
        action_space: distrax.Bijector | ActionSpace | None = None,
        cell_type: type[rnn.OptimizedLSTMCell] = rnn.OptimizedLSTMCell,
        remat: bool = True,
        actor_sigma_head: bool = False,
        carry_leading_shape: tuple[int, ...] = (),
        discrete: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(observation_space_size)
        self.action_space_size = int(action_space_size)
        self.hidden_features = int(hidden_features)
        self.lstm_features = int(lstm_features)
        self.activation = activation
        self.remat = remat
        self.cell_type = cell_type
        self.actor_sigma_head = actor_sigma_head
        self.discrete = discrete

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

        self._init_policy_head(
            in_features=self.lstm_features,
            action_space_size=self.action_space_size,
            key=key,
            actor_sigma_head=self.actor_sigma_head,
            action_space=action_space,
            discrete=discrete,
        )
        self.critic = self._critic_head(self.lstm_features, key)

        # Persistent carry for SINGLE-STEP usage (lives in nnx.State)
        # shape will be lazily set to x.shape[:-1] + (lstm_features,)
        H = int(self.lstm_features)
        lead = tuple(carry_leading_shape)
        self.h = nnx.Variable(jnp.zeros((*lead, H), dtype=float))
        self.c = nnx.Variable(jnp.zeros((*lead, H), dtype=float))
        # Snapshot of the carry at rollout start (see snapshot_rollout_carry):
        # training-time sequence replays start from this carry so recomputed
        # log-probs match the stored rollout log-probs at identical parameters.
        self.h0 = nnx.Variable(jnp.zeros((*lead, H), dtype=float))
        self.c0 = nnx.Variable(jnp.zeros((*lead, H), dtype=float))

    @property
    def observation_space_size(self) -> int:
        return self.obs_dim

    @property
    def carry_leading_shape(self) -> tuple[int, ...]:
        return self.h.value.shape[:-1]

    @partial(jax.named_call, name="LSTMActorCritic.reset")
    def reset(self, shape: tuple[int, ...], mask: jax.Array | None = None) -> None:
        """Reset the persistent LSTM carry.

        - If `self.h.value.shape != (*lead_shape, H)`, allocate fresh zeros once.

        - Otherwise, zero in-place:
            * if `mask is None`: zero everything
            * if `mask` is provided: zero masked entries along axis=0

        Parameters
        ----------
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

    @partial(jax.named_call, name="LSTMActorCritic.snapshot_rollout_carry")
    def snapshot_rollout_carry(self) -> None:
        """Snapshot the current persistent carry as the rollout-initial carry."""
        self.h0.value = self.h.value
        self.c0.value = self.c.value

    def sequence_initial_carry(
        self, idx: jax.Array
    ) -> tuple[jax.Array, jax.Array] | None:
        """Return the snapshotted ``(c, h)`` carry for flat segment indices."""
        H = int(self.lstm_features)
        c0 = self.c0.value.reshape((-1, H))[idx]
        h0 = self.h0.value.reshape((-1, H))[idx]
        return (jax.lax.stop_gradient(c0), jax.lax.stop_gradient(h0))

    @partial(jax.named_call, name="LSTMActorCritic.__call__")
    def __call__(
        self,
        x: jax.Array,
        sequence: bool = False,
        initial_carry: tuple[jax.Array, jax.Array] | None = None,
        dones: jax.Array | None = None,
    ) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass through encoder → LSTM → policy/value heads.

        Parameters
        ----------
        x : jax.Array
            Observations.  Shape ``(..., obs_dim)`` for single-step mode
            or ``(T, B, obs_dim)`` for sequence mode.
        sequence : bool
            If ``True``, run in sequence (training) mode. The carry starts
            from ``initial_carry`` if provided (typically the rollout-initial
            snapshot, see :meth:`sequence_initial_carry`), otherwise zeros.
            If ``False``, use and update the persistent carry stored on the
            module.  Remember to call :meth:`reset` when starting a new
            trajectory in single-step mode.
        initial_carry : tuple[jax.Array, jax.Array] | None
            Optional ``(c, h)`` carry, each shaped ``(B, lstm_features)``,
            used as the initial carry in sequence mode.
        dones : jax.Array | None
            Optional boolean episode-termination flags shaped ``(T, B)``.
            In sequence mode the carry is zeroed after each step where
            ``dones`` is ``True``, replaying the per-episode carry resets the
            rollout performed.

        Returns
        -------
        tuple[Distribution, jax.Array]
            - A ``distrax.MultivariateNormalDiag`` distribution over
              actions (continuous) or ``distrax.Categorical`` (discrete).
            - A value estimate tensor with trailing dimension 1.

        """
        if x.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected last dim {self.obs_dim}, got {x.shape}")

        feats = self.encoder(x)  # (..., hidden)
        if sequence:
            if initial_carry is not None:
                carry = initial_carry  # (c, h), each (B, H)
            else:
                carry = (
                    jnp.zeros((*feats.shape[1:-1], self.lstm_features)),
                    jnp.zeros((*feats.shape[1:-1], self.lstm_features)),
                )
            if dones is None:
                cell_fn = (
                    jax.checkpoint(lambda c, x: self.cell(c, x))
                    if self.remat
                    else self.cell
                )
                carry, y = jax.lax.scan(cell_fn, carry, feats)
            else:
                # Replay the rollout's per-episode carry resets: zero the
                # carry after every step that terminated an episode.
                keep = (1.0 - dones.astype(feats.dtype))[..., None]  # (T, B, 1)

                def cell_reset_fn(
                    carry: tuple[jax.Array, jax.Array],
                    xs: tuple[jax.Array, jax.Array],
                ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
                    x_t, keep_t = xs
                    (c_t, h_t), y_t = self.cell(carry, x_t)
                    return (c_t * keep_t, h_t * keep_t), y_t

                scan_fn = jax.checkpoint(cell_reset_fn) if self.remat else cell_reset_fn
                carry, y = jax.lax.scan(scan_fn, carry, (feats, keep))
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
        return self._policy_distribution(h), self.critic(h)


__all__ = ["LSTMActorCritic"]
