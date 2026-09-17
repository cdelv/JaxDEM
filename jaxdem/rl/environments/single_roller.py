# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where a single agent rolls toward a target on the floor."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...state import State
from ...system import System
from ...utils.linalg import cross, norm, unit
from . import Environment


@partial(jax.named_call, name="single_roller.frictional_wall_force")
def frictional_wall_force(
    pos: jax.Array, state: State, system: System
) -> tuple[jax.Array, jax.Array]:
    r"""Normal, frictional, and restitution forces for a sphere on a :math:`z = 0` plane.

    Combines a linear spring in the normal direction with Coulomb tangential
    friction and a velocity-proportional dashpot for restitution damping.

    Parameters
    ----------
    pos : jax.Array
        Particle positions, shape ``(N, 3)``.
    state : State
        Full simulation state (provides ``vel``, ``ang_vel``, ``rad``, ``mass``).
    system : System
        System configuration (provides ``dt``).

    Returns
    -------
    total_force : jax.Array
        Per-particle force, shape ``(N, 3)``.
    total_torque : jax.Array
        Per-particle torque, shape ``(N, 3)``.

    """
    k = 2e5
    mu = 0.4
    restitution = 0.6
    n = jnp.array([0.0, 0.0, 1.0])
    p = jnp.array([0.0, 0.0, 0.0])

    # Normal force
    dist = jnp.dot(pos - p, n) - state.rad
    penetration = jnp.minimum(0.0, dist)
    force_n = (-k * penetration)[..., None] * n

    # Normal velocity damping (restitution)
    v_n_scalar = jnp.sum(state.vel * n, axis=-1, keepdims=True)
    in_contact = (penetration < 0)[..., None]
    c_n = (2.0 * (1.0 - restitution) * jnp.sqrt(k * state.mass))[..., None]
    c_n = jnp.minimum(c_n, (0.5 * state.mass / system.dt)[..., None])
    force_damping = -c_n * v_n_scalar * n * in_contact

    # Velocity at contact point
    radius_vec = -state.rad[..., None] * n
    v_at_contact = state.vel + cross(state.ang_vel, radius_vec)
    v_n = jnp.sum(v_at_contact * n, axis=-1, keepdims=True) * n
    v_t = v_at_contact - v_n

    # Coulomb friction
    f_t_mag = mu * jnp.sum(force_n * n, axis=-1, keepdims=True)
    t_dir = unit(v_t)
    force_t = -f_t_mag * t_dir

    total_force = force_n + force_damping + force_t
    total_torque = cross(radius_vec, force_t)
    return total_force, total_torque


@Environment.register("SingleRoller")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SingleRoller(Environment):
    r"""Single-agent 3D navigation through torque-controlled rolling.

    The agent is a sphere resting on a :math:`z = 0` floor under gravity.
    Actions are 3-D torque vectors. Translational motion comes from
    frictional contact with the floor (see :func:`frictional_wall_force`).
    Each step applies a viscous drag ``-friction * vel`` and an angular
    damping ``-friction * ang_vel``.

    The reward uses an exponential distance potential measured from the
    action-start checkpoint to the live endpoint:

    .. math::

       \varphi(d) = \exp\!\left(-2 d\right)

    where :math:`d` is the distance to the objective.

    The shaping credit is :math:`F_t = \varphi(d_t) - \varphi(d_{t-1})`,
    which is positive when moving closer, negative when moving farther
    away, and zero when the distance is unchanged. Reset initializes the
    historical distance from the initial physical state, giving zero reward.

    Per-action reward:

    .. math::

       \mathrm{rew}_t = F_t

    Notes
    -----
    The observation vector per agent is:

    ============================  =========
    Feature                       Size
    ============================  =========
    Unit direction to objective   2
    Clamped displacement (x, y)   2
    Velocity (x, y)               2
    Angular velocity              3
    ============================  =========

    With ``dt = 0.002`` and ``skip_frames = 49``, each action spans 50
    physics steps (0.1 seconds). One checkpoint captures the distance
    before the action; the reward uses the live distance after all accepted
    physics steps. Only ``prev_dist`` is stored as reward history.
    """

    @classmethod
    @partial(jax.named_call, name="SingleRoller.Create")
    def Create(
        cls,
        min_box_size: float = 40.0,
        max_box_size: float = 40.0,
        max_steps: int = 20000,
        friction: float = 0.2,
    ) -> SingleRoller:
        """Create a single-agent roller environment.

        Parameters
        ----------
        min_box_size, max_box_size : float
            Range for the random square domain side length.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Damping coefficient applied as ``-friction * vel`` and
            ``-friction * ang_vel``.

        Returns
        -------
        SingleRoller
            The constructed environment. Call :meth:`reset` before use.
        """
        dim = 3
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape, collider_type=None)

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "prev_dist": jnp.zeros_like(state.rad),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.reset")
    def reset(env: SingleRoller, key: ArrayLike) -> Environment:
        """Place the agent and the objective at random positions on the floor.

        Parameters
        ----------
        env : Environment
            The current environment.
        key : ArrayLike
            JAX random number generator key.

        Returns
        -------
        Environment
            The initialized environment.

        """
        key_box, key_pos, key_objective = jax.random.split(key, 3)
        N = env.max_num_agents
        dim = env.state.dim
        rad_val = 1.0
        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )
        min_pos = rad_val * jnp.ones_like(box)
        pos = jax.random.uniform(
            key_pos,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        pos = pos.at[:, 2].set(rad_val)
        objective = jax.random.uniform(
            key_objective,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        objective = objective.at[:, 2].set(rad_val)
        env.env_params["objective"] = objective
        rad = rad_val * jnp.ones(N)
        env.state = State.create(pos=pos, rad=rad, mass=jnp.ones(N))
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            domain_type="reflectsphere",
            domain_kw={"box_size": box, "anchor": [0.0, 0.0, -1.0 * rad_val]},
            force_manager_kw={
                "gravity": [0.0, 0.0, -1.0],
                "force_functions": (frictional_wall_force,),
            },
            collider_type=None,
        )
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.step")
    def step(env: SingleRoller, action: jax.Array) -> Environment:
        """Apply a torque action and advance the physics by one step.

        The historical distance remains fixed throughout the action. Use
        ``utils.advance_action`` to checkpoint once before repeating physics
        steps. Observations and rewards always read the live state.

        Parameters
        ----------
        env : Environment
            Current environment.
        action : jax.Array
            3-D torque vector per agent.

        Returns
        -------
        Environment
            Updated environment after one physics step.

        """
        reshaped_action = action.reshape(env.max_num_agents, *env.action_space_shape)
        torque = reshaped_action - env.env_params["friction"] * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)
        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.checkpoint")
    def checkpoint(env: SingleRoller, action: jax.Array) -> Environment:
        """Save the starting distance before the next action interval.

        Store the live distance in ``prev_dist``. Physics steps preserve it,
        and reward compares it with the live endpoint after all repeated
        physics steps. Current quantities are computed directly from state.

        Parameters
        ----------
        env : SingleRoller
            The environment immediately before the next action.
        action : jax.Array
            The per-agent action about to be applied. This distance-only
            baseline does not depend on its value.

        Returns
        -------
        Environment
            The environment with the action-start distance saved. Read reward
            before the next checkpoint or reset replaces this baseline.
        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.observation")
    def observation(env: SingleRoller) -> jax.Array:
        """Build per-agent observations directly from the live physical state.

        Contents per agent:

        - Unit displacement to objective projected to x-y (shape ``(2,)``).
        - Clamped displacement to objective projected to x-y (shape ``(2,)``).
        - Velocity projected to x-y (shape ``(2,)``).
        - Angular velocity (shape ``(3,)``).

        Returns
        -------
        jax.Array
            Shape ``(N, 9)``.

        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        delta_2d = delta[..., :2]
        vel_2d = env.state.vel[..., :2]
        return jnp.concatenate(
            [
                unit(delta_2d),
                jnp.clip(delta_2d, -3.0, 3.0),
                vel_2d,
                env.state.ang_vel,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.reward")
    def reward(env: SingleRoller) -> jax.Array:
        r"""Return the distance-potential change since the action-start checkpoint.

        .. math::

           \mathrm{rew}_t = e^{-2d_t} - e^{-2d_{t-1}}

        Here :math:`d_t` is the live distance from the agent's center to its
        objective, and :math:`d_{t-1}` is the distance saved before the action.
        The reward is positive for progress toward the objective and zero
        after reset or when the distance is unchanged. This accessor reads
        the live state and saved baseline without advancing the checkpoint.

        Parameters
        ----------
        env : Environment
            The live environment with its action-start distance baseline.

        Returns
        -------
        jax.Array
            Shape ``(N,)``.

        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        phi_curr = jnp.exp(-2 * norm(delta))
        phi_prev = jnp.exp(-2 * env.env_params["prev_dist"])
        return phi_curr - phi_prev

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.truncated")
    def truncated(env: SingleRoller) -> jax.Array:
        """``True`` when ``step_count`` reaches ``max_steps``."""
        return jnp.asarray(env.system.step_count >= env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Per-agent flattened action dimensionality (3-D torque)."""
        return 3

    @property
    def action_space_shape(self) -> tuple[int]:
        """Per-agent action tensor shape."""
        return (3,)

    @property
    def observation_space_size(self) -> int:
        """Per-agent flattened observation dimensionality (9)."""
        return 9


__all__ = ["SingleRoller"]
