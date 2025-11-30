# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Multi-agent navigation task with collision penalties."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial
from typing import Tuple

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="multi_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array, rad: float) -> jax.Array:
    i = jax.lax.iota(int, N)  # 0..N-1
    Lx, Ly = box.astype(float)

    nx = jnp.ceil(jnp.sqrt(N * Lx / Ly)).astype(int)
    ny = jnp.ceil(N / nx).astype(int)

    ix = jnp.mod(i, nx)
    iy = i // nx

    dx = Lx / nx
    dy = Ly / ny

    xs = (ix + 0.5) * dx
    ys = (iy + 0.5) * dy
    base = jnp.stack([xs, ys], axis=1)

    noise = jax.random.uniform(key, (N, 2), minval=-1.0, maxval=1.0) * jnp.asarray(
        [jnp.maximum(0.0, dx / 2 - rad), jnp.maximum(0.0, dy / 2 - rad)]
    )
    return base + noise


@Environment.register("multiNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MultiNavigator(Environment):
    """
    Multi-agent navigation environment with collision penalties.

    Agents seek fixed objectives in a 2D reflective box. Each step applies a
    force-like action, advances simple dynamics, updates LiDAR, and returns
    shaped rewards with an optional final bonus on goal.
    """

    n_lidar_rays: int = field(metadata={"static": True})
    """
    Number of lidar rays for the vision system.
    """

    @classmethod
    @partial(jax.named_call, name="MultiNavigator.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        box_padding: float = 5.0,
        max_steps: int = 5760,
        final_reward: float = 1.0,  # 1.0
        shaping_factor: float = 0.005,
        prev_shaping_factor: float = 0.0,
        global_shaping_factor: float = 0.0,
        collision_penalty: float = -0.005,
        goal_threshold: float = 2 / 3,
        lidar_range: float = 0.45,
        n_lidar_rays: int = 16,
    ) -> "MultiNavigator":
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            box_padding=jnp.asarray(box_padding, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            final_reward=jnp.asarray(final_reward, dtype=float),
            collision_penalty=jnp.asarray(collision_penalty, dtype=float),
            shaping_factor=jnp.asarray(shaping_factor, dtype=float),
            prev_shaping_factor=jnp.asarray(prev_shaping_factor, dtype=float),
            global_shaping_factor=jnp.asarray(global_shaping_factor, dtype=float),
            goal_threshold=jnp.asarray(goal_threshold, dtype=float),
            prev_rew=jnp.zeros_like(state.rad),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            goal_scale=jnp.asarray(1.0, dtype=float),
            objective_index=jnp.zeros_like(state.rad, dtype=int),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiNavigator.reset")
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Initialize the environment with randomly placed particles and velocities.

        Parameters
        ----------
        env: Environment
            Current environment instance.

        key : jax.random.PRNGKey
            JAX random number generator key.

        Returns
        -------
        Environment
            Freshly initialized environment.
        """
        root = key
        key_box = jax.random.fold_in(root, jnp.uint32(0))
        key_pos = jax.random.fold_in(root, jnp.uint32(1))
        key_objective = jax.random.fold_in(root, jnp.uint32(2))
        key_shuffle = jax.random.fold_in(root, jnp.uint32(3))
        key_vel = jax.random.fold_in(root, jnp.uint32(4))

        N = env.max_num_agents
        dim = env.state.dim
        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )

        rad = 0.05
        pos = (
            _sample_objectives(
                key_pos, int(N), box + env.env_params["box_padding"] * rad, rad
            )
            - env.env_params["box_padding"] * rad / 2
        )
        objective = _sample_objectives(key_objective, int(N), box, rad)
        env.env_params["goal_scale"] = jnp.max(box)

        base_idx = jnp.arange(N, dtype=int)
        perm = jax.random.permutation(key_shuffle, base_idx)
        env.env_params["objective"] = objective[perm]
        env.env_params["objective_index"] = perm

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )

        Rad = rad * jnp.ones(N)
        env.state = State.create(pos=pos, vel=vel, rad=Rad)

        matcher = MaterialMatchmaker.create("harmonic")
        mat_table = MaterialTable.from_materials(
            [Material.create("elastic", density=200.0, young=6e3, poisson=0.3)],
            matcher=matcher,
        )

        env.system = System.create(
            env.state.shape,
            dt=0.004,
            domain_type="reflect",
            domain_kw=dict(
                box_size=box + env.env_params["box_padding"] * rad,
                anchor=jnp.zeros_like(box) - env.env_params["box_padding"] * rad / 2,
            ),
            mat_table=mat_table,
        )

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.vecdot(delta, delta)
        env.env_params["prev_rew"] = jnp.sqrt(d) / env.env_params["goal_scale"]
        env.env_params["lidar"] = lidar(env)

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiNavigator.step")
    def step(env: "Environment", action: jax.Array) -> "Environment":
        """
        Advance one step. Actions are forces; simple drag is applied.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : jax.Array
            The vector of actions each agent in the environment should take.

        Returns
        -------
        Environment
            The updated environment state.
        """
        force = (
            action.reshape(env.max_num_agents, *env.action_space_shape)
            - jnp.sign(env.state.vel) * 0.08
        )
        env.system = env.system.force_manager.add_force(env.state, env.system, force)

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.vecdot(delta, delta)
        env.env_params["prev_rew"] = jnp.sqrt(d) / env.env_params["goal_scale"]

        env.state, env.system = env.system.step(env.state, env.system)

        env.env_params["lidar"] = lidar(env)

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.observation")
    def observation(env: "Environment") -> jax.Array:
        """
        Build per-agent observations.

        Contents per agent
        ------------------
        - Wrapped displacement to objective ``Δx`` (shape ``(2,)``).
        - Velocity ``v`` (shape ``(2,)``).
        - LiDAR proximities (shape ``(n_lidar_rays,)``).

        Returns
        -------
        jax.Array
            Array of shape ``(N, 2 * dim + n_lidar_rays)`` scaled by the
            maximum box size for normalization.
        """
        return jnp.concatenate(
            [
                env.system.domain.displacement(
                    env.env_params["objective"], env.state.pos, env.system
                )
                / env.env_params["goal_scale"],
                env.state.vel,
                env.env_params["lidar"] / env.env_params["lidar_range"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.reward")
    def reward(env: "Environment") -> jax.Array:
        r"""
        Per-agent reward with distance shaping, goal bonus, LiDAR collision penalty, and a global shaping term.

        **Equations**

        Let :math:`\delta_i=\operatorname{displacement}(\mathbf{x}_i,\mathbf{objective})`,
        :math:`d_i=\lVert\delta_i\rVert_2`, and :math:`\mathbf{1}[\cdot]` the indicator.
        With shaping factors :math:`\alpha_{\text{prev}},\alpha`, final reward :math:`R_f`,
        collision penalty math:`C`, global shaping factor math:`\beta`, and radius :math:`r_i`. Let :math:`\ell_{i,k}` be the LiDAR proximities for agent :math:`i` and ray :math:`k`,
        and :math:`h_i = \sum_k \mathbf{1}[\ell_{i,k} > (\text{LIDAR_range} - 2r_i)]` be the collision count. The rewards consists on:

        .. math::

            \mathrm{rew}^{\text{shape}}_i = \alpha_{\text{prev}}\,d^{\text{prev}}_i - \alpha\, d_i

        .. math::

            \mathrm{rew}_i = \mathrm{rew}^{\text{shape}}_i + R_f\,\mathbf{1}[\,d_i < \text{goal_threshold}\times r_i\,] + C\, h_i - \beta\, \overline{d},

        .. math::

            \overline{d} = \tfrac{1}{N}\sum_j d_j

        Parameters
        -----------
        env : Environment
            Current environment.

        Returns
        -------
        jax.Array
            Shape ``(N,)``. The normalized per-agent reward vector.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.vecdot(delta, delta)
        d = jnp.sqrt(d) / env.env_params["goal_scale"]
        rew = (
            env.env_params["prev_shaping_factor"] * env.env_params["prev_rew"]
            - env.env_params["shaping_factor"] * d
        )

        closeness_thresh = jnp.maximum(
            0.0, env.env_params["lidar_range"] - 2.0 * env.state.rad[:, None]
        )
        n_hits = (
            (env.env_params["lidar"] > closeness_thresh).sum(axis=-1).astype(rew.dtype)
        )

        on_goal = d < env.env_params["goal_threshold"] * env.state.rad
        reward = (
            rew
            + env.env_params["final_reward"] * on_goal
            + env.env_params["collision_penalty"] * n_hits
        )
        reward += jnp.mean(reward) * env.env_params["global_shaping_factor"]
        return reward.reshape(env.max_num_agents)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="MultiNavigator.done")
    def done(env: "Environment") -> jax.Array:
        """
        Returns a boolean indicating whether the environment has ended.
        The episode terminates when the maximum number of steps is reached.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Boolean array indicating whether the episode has ended.
        """
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """
        Flattened action size per agent. Actions passed to :meth:`step` have shape ``(A, action_space_size)``.
        """
        return self.state.dim

    @property
    def action_space_shape(self) -> Tuple[int]:
        """
        Original per-agent action shape (useful for reshaping inside the environment).
        """
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """
        Flattened observation size per agent. :meth:`observation` returns shape ``(A, observation_space_size)``.
        """
        return 2 * self.state.dim + self.n_lidar_rays


__all__ = ["MultiNavigator"]
