# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

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


@Model.register("DeepOnetActorCritic")
class DeepOnetActorCritic(Model, nnx.Module):  # type: ignore[misc]
    """
    A DeepOnet-based Actor-Critic model with a Dynamic Weighted Combiner.

    Architecture:
    1. **Trunk (T)**: MLP encoding Goal + Velocity.
    2. **Branch (B)**: MLP encoding Lidar/Sensor data.
    3. **Weighted Combiner**: T gates B features dynamically.
    4. **Actor/Critic Heads**.

    Parameters
    ----------
    observation_space_size : int
      Total size of the observation space.
    action_space_size : int
      Size of the action space.
    key : nnx.Rngs
      Random number generator.
    trunk_architecture : Sequence[int]
      Hidden layers for the trunk (Goal/Vel).
    branch_architecture : Sequence[int]
      Hidden layers for the branch (Lidar features).
    combiner_architecture : Sequence[int]
      Hidden layers for the merging network.
    critic_architecture : Sequence[int]
      Hidden layers for the critic network (after combiner).
    basis_dim : int
      Output size of Trunk and Branch before combination.
    """

    __slots__ = ()

    def __init__(
        self,
        *,
        observation_space_size: int,
        action_space_size: int,
        key: nnx.Rngs,
        trunk_architecture: Sequence[int] = (64, 64),
        branch_architecture: Sequence[int] = (64, 64),
        combiner_architecture: Sequence[int] = (64, 64),
        critic_architecture: Sequence[int] = (64, 64),
        basis_dim: int = 64,
        activation: Callable[..., Any] = nnx.gelu,
        in_scale: float = math.sqrt(2),
        actor_scale: float = 1.0,
        critic_scale: float = 0.01,
        actor_sigma_head: bool = False,
        action_space: distrax.Bijector | ActionSpace | None = None,
    ):
        self.observation_space_size = int(observation_space_size)
        self.action_space_size = int(action_space_size)
        self.trunk_architecture = [int(x) for x in trunk_architecture]
        self.branch_architecture = [int(x) for x in branch_architecture]
        self.combiner_architecture = [int(x) for x in combiner_architecture]
        self.critic_architecture = [int(x) for x in critic_architecture]

        self.basis_dim = int(basis_dim)
        self.activation = activation
        self.in_scale = float(in_scale)
        self.actor_scale = float(actor_scale)
        self.critic_scale = float(critic_scale)
        self.actor_sigma_head = actor_sigma_head

        # --- 1. Trunk (Goal + Velocity -> 4 inputs) ---
        trunk_layers = []
        trunk_in = 4
        for width in self.trunk_architecture:
            trunk_layers.append(
                nnx.Linear(
                    in_features=trunk_in,
                    out_features=width,
                    kernel_init=nnx.initializers.orthogonal(self.in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            # pyrefly: ignore [bad-argument-type]
            trunk_layers.append(self.activation)
            trunk_in = width
        # Projection to basis_dim
        trunk_layers.append(
            nnx.Linear(in_features=trunk_in, out_features=self.basis_dim, rngs=key)
        )
        # pyrefly: ignore [bad-argument-type]
        trunk_layers.append(self.activation)
        self.trunk = nnx.Sequential(*trunk_layers)

        # --- 2. Branch (Lidar -> obs-4 inputs) ---
        branch_in_len = self.observation_space_size - 4
        if branch_in_len <= 0:
            raise ValueError("Observation space too small for split (need > 4).")

        # b. MLP layers (Input size is the raw Lidar length)
        branch_layers = []
        current_branch_in = branch_in_len

        for width in self.branch_architecture:
            branch_layers.append(
                nnx.Linear(
                    in_features=current_branch_in,
                    out_features=width,
                    kernel_init=nnx.initializers.orthogonal(self.in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            # pyrefly: ignore [bad-argument-type]
            branch_layers.append(self.activation)
            current_branch_in = width

        # c. Projection to basis_dim
        branch_layers.append(
            nnx.Linear(
                in_features=current_branch_in, out_features=self.basis_dim, rngs=key
            )
        )
        # pyrefly: ignore [bad-argument-type]
        branch_layers.append(self.activation)
        self.branch_mlp = nnx.Sequential(*branch_layers)

        # --- 3. Dynamic Gating Network (The Weighted Mixer) ---
        # T (Trunk output) is used to generate the scaling factor G
        self.gating_network = nnx.Sequential(
            nnx.Linear(
                in_features=self.basis_dim,
                out_features=self.basis_dim,
                kernel_init=nnx.initializers.orthogonal(self.in_scale),
                bias_init=nnx.initializers.constant(0.0),
                rngs=key,
            ),
            jax.nn.sigmoid,  # Sigmoid ensures the gate values are between 0 and 1
        )

        # --- 4. Combiner MLP (Takes T + Gated B) ---
        combiner_layers = []
        # Input is Trunk basis + Gated Branch basis
        combiner_in = self.basis_dim * 2

        for width in self.combiner_architecture:
            combiner_layers.append(
                nnx.Linear(
                    in_features=combiner_in,
                    out_features=width,
                    kernel_init=nnx.initializers.orthogonal(self.in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            combiner_layers.append(self.activation)
            combiner_in = width

        self.combiner = nnx.Sequential(*combiner_layers)
        feature_dim = combiner_in

        # --- 5. Actor Heads (Attached to Combiner Output) ---
        self.actor_mu = nnx.Linear(
            in_features=feature_dim,
            out_features=self.action_space_size,
            kernel_init=nnx.initializers.orthogonal(self.actor_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=key,
        )

        self._log_std = nnx.Param(jnp.zeros((1, self.action_space_size)))
        self._actor_sigma = nnx.Sequential(
            nnx.Linear(
                in_features=feature_dim,
                out_features=self.action_space_size,
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

        # --- 6. Critic MLP + Head ---
        critic_layers = []
        critic_in = feature_dim

        for width in self.critic_architecture:
            critic_layers.append(
                nnx.Linear(
                    in_features=critic_in,
                    out_features=width,
                    kernel_init=nnx.initializers.orthogonal(self.in_scale),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=key,
                )
            )
            # pyrefly: ignore [bad-argument-type]
            critic_layers.append(self.activation)
            critic_in = width

        # Final projection to scalar value
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

        # --- Action Space ---
        if action_space is None:
            action_space = ActionSpace.create("Free")
        bij = cast(distrax.Bijector, action_space)
        if getattr(bij, "event_ndims_in", 0) == 0:
            bij = distrax.Block(bij, ndims=1)
        self.bij = bij

    @property
    def metadata(self) -> Dict[str, Any]:
        """Includes all initialization parameters for model reconstruction."""
        return dict(
            observation_space_size=self.observation_space_size,
            action_space_size=self.action_space_size,
            trunk_architecture=self.trunk_architecture,
            branch_architecture=self.branch_architecture,
            combiner_architecture=self.combiner_architecture,
            critic_architecture=self.critic_architecture,
            basis_dim=self.basis_dim,
            activation=encode_callable(self.activation),
            in_scale=self.in_scale,
            actor_scale=self.actor_scale,
            critic_scale=self.critic_scale,
            actor_sigma_head=self.actor_sigma_head,
            # pyrefly: ignore [missing-attribute]
            action_space_type=self.bij.type_name,
            # pyrefly: ignore [missing-attribute]
            action_space_kws=self.bij.kws,
        )

    @partial(jax.named_call, name="DeepOnetActorCritic.__call__")
    def __call__(
        self, x: jax.Array, sequence: bool = False
    ) -> Tuple[distrax.Distribution, jax.Array]:
        # 1. Split Inputs
        trunk_input = x[..., :4]
        branch_input = x[..., 4:]  # Lidar/Sensor data

        # 2. Process Trunk
        t_out = self.trunk(trunk_input)  # Goal/Velocity features

        # 3. Process Branch (MLP)
        # Directly pass the flat branch input to the MLP
        b_out = self.branch_mlp(branch_input)

        # 4. Dynamic Weighted Mixer
        # Generate Gating weights G from the Trunk output
        gate_weights = self.gating_network(t_out)

        # Apply Gating: B_gated = B_out * G
        b_gated = b_out * gate_weights

        # Combine the original Trunk features (T_out) with the gated Branch features (B_gated)
        combined = jnp.concatenate([t_out, b_gated], axis=-1)
        features = self.combiner(combined)

        # 5. Actor
        mu = self.actor_mu(features)
        sigma = self.actor_sigma(features)
        pi = distrax.MultivariateNormalDiag(mu, sigma)
        pi = distrax.Transformed(pi, self.bij)

        # 6. Critic (MLP -> Scalar)
        val = self.critic(features)

        # pyrefly: ignore [bad-return]
        return pi, val
