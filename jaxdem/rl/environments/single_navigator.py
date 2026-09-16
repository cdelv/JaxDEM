# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where a single agent navigates toward a target."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...state import State
from ...system import System
from ...utils.linalg import norm, unit
from . import Environment


@Environment.register("singleNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SingleNavigator(Environment):
    r"""Single-agent navigation environment toward a fixed target.

    The agent controls a force vector that acts directly on a sphere
    inside a reflective box. Each step adds viscous drag
    ``-friction * vel``. The reward uses an exponential distance potential
    measured at consecutive action checkpoints:

    .. math::

       \varphi(d) = \exp\!\left(-2 d \right)

    where :math:`d` is the distance to the objective.

    The shaping credit is :math:`F_t = \varphi(d_t) - \varphi(d_{t-1})`,
    which is positive when moving closer, negative when moving farther
    away, and zero when the distance is unchanged. Reset initializes both
    checkpoint distances to the same value.

    Per-action-checkpoint reward:

    .. math::

       \mathrm{rew}_t = F_t

    Notes
    -----
    The observation vector per agent is:

    ============================  =========
    Feature                       Size
    ============================  =========
    Unit direction to objective   ``dim``
    Clamped displacement          ``dim``
    Velocity                      ``dim``
    ============================  =========

    With ``dt = 0.002`` and ``skip_frames = 50``, each action spans 51
    physics steps (0.102 seconds). Reward shaping compares consecutive
    action checkpoints, including all of those physics steps.
    """

    @classmethod
    @partial(jax.named_call, name="SingleNavigator.Create")
    def Create(
        cls,
        dim: int = 2,
        min_box_size: float = 40.0,
        max_box_size: float = 40.0,
        max_steps: int = 20000,
        friction: float = 0.2,
    ) -> SingleNavigator:
        """Create a single-agent navigator environment.

        Parameters
        ----------
        dim : int
            Spatial dimensionality (2 or 3).
        min_box_size, max_box_size : float
            Range for the random square domain side length.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.

        Returns
        -------
        SingleNavigator
            The constructed environment. Call :meth:`reset` before use.

        """
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(
            state.shape, rotation_integrator_type=None, collider_type=None
        )

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "delta": jnp.zeros_like(state.pos),
            "curr_dist": jnp.zeros_like(state.rad),
            "prev_dist": jnp.zeros_like(state.rad),
            "curr_vel": jnp.zeros_like(state.vel),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.reset")
    def reset(env: SingleNavigator, key: ArrayLike) -> Environment:
        """Place the agent and the objective at random positions in the box.

        Parameters
        ----------
        env : SingleNavigator
            The current environment.

        key : jax.random.PRNGKey
            JAX random number generator key.

        Returns
        -------
        Environment
            The initialized environment.

        """
        key_box, key_pos, key_objective = jax.random.split(key, 3)
        N = env.max_num_agents
        dim = env.state.dim
        rad = jnp.array(1.0, dtype=float)
        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )
        min_pos = rad * jnp.ones_like(box)
        pos = jax.random.uniform(
            key_pos,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        objective = jax.random.uniform(
            key_objective,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        env.env_params["objective"] = objective
        rad = rad * jnp.ones(N)
        env.state = State.create(pos=pos, rad=rad, mass=jnp.ones(N))
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            rotation_integrator_type=None,
            domain_type="reflectsphere",
            domain_kw={"box_size": box, "anchor": jnp.zeros_like(box)},
            collider_type=None,
        )
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        dist = norm(delta)
        env.env_params["delta"] = delta
        env.env_params["curr_dist"] = dist
        env.env_params["prev_dist"] = dist
        env.env_params["curr_vel"] = env.state.vel

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.step")
    def step(env: SingleNavigator, action: jax.Array) -> Environment:
        """Advance physics with force actions and drag ``-friction * vel``.

        Measurements remain at the preceding action checkpoint. Use
        ``utils.advance_action`` to finish an action and refresh them.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : jax.Array
            The per-agent action vectors.

        Returns
        -------
        Environment
            The updated environment state.

        """
        reshaped_action = action.reshape(env.max_num_agents, *env.action_space_shape)
        force = reshaped_action - env.state.vel * env.env_params["friction"]
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.checkpoint")
    def checkpoint(env: SingleNavigator) -> Environment:
        """Refresh navigation measurements at the end of an action interval.

        Store the preceding checkpoint's distance in ``prev_dist``, then
        update ``delta``, ``curr_dist``, and ``curr_vel`` from the current
        physical state. The reward therefore measures progress over the
        complete action interval, including any repeated physics steps.

        Parameters
        ----------
        env : SingleNavigator
            The environment at the action endpoint, with measurements from
            the preceding checkpoint retained in ``env_params``.

        Returns
        -------
        Environment
            The environment with refreshed observation measurements and the
            preceding checkpoint preserved as the reward baseline.

        Notes
        -----
        Called once by :func:`jaxdem.utils.advance_action` after the repeated
        physics steps, including a final interval shortened by truncation.
        This method does not advance physics or reset the episode. Calling it
        again without advancing physics replaces the reward baseline with the
        same endpoint, making the potential-based shaping contribution zero.

        """
        env.env_params["prev_dist"] = env.env_params["curr_dist"]
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["delta"] = delta
        env.env_params["curr_dist"] = norm(delta)
        env.env_params["curr_vel"] = env.state.vel
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.observation")
    def observation(env: SingleNavigator) -> jax.Array:
        """Build per-agent observations from the latest action checkpoint.

        Contents per agent
        ------------------
        - Unit vector to objective (shape (dim,))  --> Direction
        - Clamped delta to objective (shape (dim,)) --> Local precision
        - Velocity (shape (dim,))

        Returns
        -------
        jax.Array
            Array of shape ``(N, 3 * dim)``

        """
        delta = env.env_params["delta"]
        return jnp.concatenate(
            [
                unit(delta),
                jnp.clip(delta, -3.0, 3.0),
                env.env_params["curr_vel"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.reward")
    def reward(env: SingleNavigator) -> jax.Array:
        r"""Return the change in distance potential between action checkpoints.

        .. math::

           \mathrm{rew}_t = e^{-2d_t} - e^{-2d_{t-1}}

        Here :math:`d_t` and :math:`d_{t-1}` are the current and preceding
        checkpoint distances from the agent's center to its objective.
        The reward is positive for progress toward the objective and zero
        after reset or when the distance is unchanged. This accessor reads
        the stored measurements without advancing the checkpoint.

        Parameters
        ----------
        env : Environment
            The environment with measurements from the latest checkpoint.

        Returns
        -------
        jax.Array
            Per-agent rewards of shape ``(N,)``, where ``N = 1``.

        """
        phi_curr = jnp.exp(-2 * env.env_params["curr_dist"])
        phi_prev = jnp.exp(-2 * env.env_params["prev_dist"])
        return phi_curr - phi_prev

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.truncated")
    def truncated(env: SingleNavigator) -> jax.Array:
        """Return whether the episode has ended.

        The episode ends when ``step_count`` reaches ``max_steps``.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool that is True when the episode has ended.

        """
        return jnp.asarray(env.system.step_count >= env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Flattened action size per agent. Actions passed to :meth:`step` have shape ``(A, action_space_size)``."""
        return self.state.dim

    @property
    def action_space_shape(self) -> tuple[int]:
        """Original per-agent action shape (useful for reshaping inside the environment)."""
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """Flattened observation size per agent. :meth:`observation` returns shape ``(A, observation_space_size)``."""
        return 3 * self.state.dim
