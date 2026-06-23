# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of reinforcement learning models based on MinGRU."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import distrax  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
from flax import nnx

from ..action_spaces import ActionSpace
from . import Model


@Model.register("MinGRUActorCritic")
class MinGRUActorCritic(Model):
    """A recurrent actor-critic with an MLP encoder and a MinGRU torso.

    This model uses the MinGRU architecture (https://arxiv.org/abs/2410.01201)
    and relies on a parallel associative scan for sequence-mode training, enabling
    extremely fast execution.

    Parameters
    ----------
    observation_space_size : int
        Flattened observation size (``obs_dim``).
    action_space_size : int
        Number of action dimensions (for continuous) or number of discrete actions.
    key : nnx.Rngs
        Random number generator(s) for parameter initialization.
    hidden_features : int
        Width of the encoder output.
    gru_features : int
        MinGRU hidden size. Also the feature size consumed by the policy and value heads.
    num_layers : int
        Number of MinGRU layers.
    activation : Callable
        Activation function applied inside the encoder.
    action_space : distrax.Bijector | ActionSpace | None
        Bijector to constrain the policy probability distribution (continuous only).
    remat : bool
        Accepted for compatibility but unused since associative scan is memory efficient.
    actor_sigma_head : bool
        If ``True``, the standard deviation is produced by a learned
        head on the MinGRU output; otherwise an independent log-std
        parameter is used. Only used for continuous actions.
    carry_leading_shape : tuple[int, ...]
        Leading dimensions for the persistent carry tensor ``h``.
    discrete : bool
        If True, use a categorical distribution for discrete actions.
    """

    __slots__ = ()

    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        hidden_features: int = 64,
        gru_features: int = 128,
        num_layers: int = 1,
        activation: Callable[..., Any] = nnx.gelu,
        action_space: distrax.Bijector | ActionSpace | None = None,
        remat: bool = False,
        actor_sigma_head: bool = False,
        carry_leading_shape: tuple[int, ...] = (),
        discrete: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(observation_space_size)
        self.action_space_size = int(action_space_size)
        self.hidden_features = int(hidden_features)
        self.gru_features = int(gru_features)
        self.num_layers = int(num_layers)
        self.activation = activation
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

        if self.hidden_features != self.gru_features:
            self.proj_in = nnx.Linear(
                self.hidden_features,
                self.gru_features,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=key,
            )
        else:
            self.proj_in = None

        self.mingru_layers = nnx.List(
            [
                nnx.Linear(
                    self.gru_features,
                    3 * self.gru_features,
                    use_bias=False,
                    kernel_init=nnx.initializers.orthogonal(1.0),
                    rngs=key,
                )
                for _ in range(self.num_layers)
            ]
        )

        self._init_policy_head(
            in_features=self.gru_features,
            action_space_size=self.action_space_size,
            key=key,
            actor_sigma_head=self.actor_sigma_head,
            action_space=action_space,
            discrete=discrete,
        )
        self.critic = self._critic_head(self.gru_features, key)

        def fused_init(rng: jax.Array, shape: tuple[int, ...], dtype: Any) -> jax.Array:
            k1, k2 = jax.random.split(rng)
            w1 = nnx.initializers.orthogonal(1.0)(k1, (shape[0], shape[1] - 1), dtype)
            w2 = nnx.initializers.orthogonal(0.01)(k2, (shape[0], 1), dtype)
            return jnp.concatenate([w1, w2], axis=-1)

        self.fused_head = nnx.Linear(
            in_features=self.gru_features,
            out_features=self.action_space_size + 1,
            kernel_init=fused_init,
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        H = int(self.gru_features)
        L = int(self.num_layers)
        lead = tuple(carry_leading_shape)
        self.h = nnx.Variable(jnp.zeros((*lead, L, H), dtype=float))

    @property
    def observation_space_size(self) -> int:
        return self.obs_dim

    @property
    def carry_leading_shape(self) -> tuple[int, ...]:
        return self.h.value.shape[:-2]

    @partial(jax.named_call, name="MinGRUActorCritic.reset")
    def reset(self, shape: tuple[int, ...], mask: jax.Array | None = None) -> None:
        target_shape = (*shape[:-1], self.num_layers, self.gru_features)

        if self.h.value.shape != target_shape:
            self.h.value = jnp.zeros(target_shape, dtype=float)
            return

        if mask is None:
            self.h.value *= 0.0
            return

        mask = mask.reshape((mask.shape[0],) + (1,) * (self.h.value.ndim - 1))
        self.h.value = jnp.where(mask, 0.0, self.h.value)

    @partial(jax.named_call, name="MinGRUActorCritic.__call__")
    def __call__(
        self,
        x: jax.Array,
        sequence: bool = False,
    ) -> tuple[distrax.Distribution, jax.Array]:
        if x.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected last dim {self.obs_dim}, got {x.shape}")

        feats = self.encoder(x)
        if self.proj_in is not None:
            feats = self.proj_in(feats)

        h = feats

        def _g(x_val: jax.Array) -> jax.Array:
            return jnp.where(x_val >= 0, x_val + 0.5, jax.nn.sigmoid(x_val))

        def _log_g(x_val: jax.Array) -> jax.Array:
            return jnp.where(
                x_val >= 0, jnp.log(jax.nn.relu(x_val) + 0.5), -jax.nn.softplus(-x_val)
            )

        def _highway(
            x_val: jax.Array, out_val: jax.Array, proj_val: jax.Array
        ) -> jax.Array:
            g_val = jax.nn.sigmoid(proj_val)
            return g_val * out_val + (1.0 - g_val) * x_val

        if sequence:
            carry = jnp.zeros((*h.shape[1:-1], self.num_layers, self.gru_features))

            def heinsen_operator(
                state1: tuple[jax.Array, jax.Array], state2: tuple[jax.Array, jax.Array]
            ) -> tuple[jax.Array, jax.Array]:
                log_a1, log_b1 = state1
                log_a2, log_b2 = state2
                return log_a1 + log_a2, jnp.logaddexp(log_a2 + log_b1, log_b2)

            out_h = h
            carry_out = []

            for i, layer in enumerate(self.mingru_layers):
                layer_out = layer(out_h)
                hidden, gate, proj = jnp.split(layer_out, 3, axis=-1)

                log_coeffs = -jax.nn.softplus(gate)
                log_values = -jax.nn.softplus(-gate) + _log_g(hidden)

                layer_carry = carry[..., i, :]
                log_a0 = jnp.full_like(layer_carry, -jnp.inf)
                log_b0 = jnp.where(layer_carry > 0, jnp.log(layer_carry), -jnp.inf)

                log_a_seq = jnp.concatenate([log_a0[None], log_coeffs], axis=0)
                log_b_seq = jnp.concatenate([log_b0[None], log_values], axis=0)

                _, log_h_seq = jax.lax.associative_scan(
                    heinsen_operator, (log_a_seq, log_b_seq), axis=0
                )

                log_out = log_h_seq[1:]
                out = jnp.exp(log_out)

                out_h = _highway(out_h, out, proj)
                carry_out.append(out[-1])

            h_out = out_h

        else:
            batch = h.shape[:-1]
            target = (*batch, self.num_layers, self.gru_features)

            if self.h.value.shape != target:
                self.h.value = jnp.zeros(target, dtype=float)

            out_h = h
            carry_in = self.h.value
            carry_out = []

            for i, layer in enumerate(self.mingru_layers):
                layer_out = layer(out_h)
                hidden, gate, proj = jnp.split(layer_out, 3, axis=-1)

                state = carry_in[..., i, :]
                w = jax.nn.sigmoid(gate)
                out = (1.0 - w) * state + w * _g(hidden)

                out_h = _highway(out_h, out, proj)
                carry_out.append(out)

            self.h.value = jnp.stack(carry_out, axis=-2)
            h_out = out_h

        fused = self.fused_head(h_out)

        from ..action_spaces import Transformed

        if self.discrete:
            pi = distrax.Categorical(logits=fused[..., :-1])
            return pi, fused[..., -1:]
        else:
            pi = distrax.MultivariateNormalDiag(
                fused[..., :-1], self.actor_sigma(h_out)
            )
            return Transformed(pi, self.bij), fused[..., -1:]
