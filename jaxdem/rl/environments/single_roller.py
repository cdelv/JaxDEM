# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where a single agent rolls toward a target on the floor."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import jaxdem.utils.thermal as thermal

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

    The reward uses potential-based shaping with a proximity-gated
    kinetic-energy term:

    .. math::

       \varphi(d, K) = \exp\!\left(-2 d - \frac{K}{\text{ke\_tau}}\,e^{-\text{ke\_gate} \cdot d}\right)

    where :math:`d` is the distance to the objective and :math:`K` is the
    total (translational + rotational) kinetic energy. ``ke_tau`` is the
    KE scale that sets the overall strength of the penalty. ``ke_gate``
    controls how sharply KE sensitivity falls off with distance. A larger
    ``ke_gate`` means KE only matters very close to the objective.

    The shaping credit is :math:`F_t = \varphi(d_t, K_t) - \varphi(d_{t-1}, K_{t-1})`,
    so kinetic energy is penalized only near the objective. Far away the
    gate :math:`e^{-\text{ke\_gate} \cdot d} \to 0` and fast motion is free.

    Per-step reward:

    .. math::

       \mathrm{rew}_t = \frac{F_t + b \cdot \mathbb{1}[d_t \le r]}{b}

    where :math:`b` is the near-goal bonus and :math:`r` is the agent radius.

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

    For realistic training parameters, ``skip_frames = 50`` gives a response
    rate of 200 Hz, so ``num_steps_epoch = 100`` gives a horizon of 0.5
    seconds.
    """

    @classmethod
    @partial(jax.named_call, name="SingleRoller.Create")
    def Create(
        cls,
        min_box_size: float = 40.0,
        max_box_size: float = 40.0,
        max_steps: int = 20000,
        friction: float = 0.2,
        near_goal_bonus: float = 0.1,
        ke_tau: float = 5.0,
        ke_gate: float = 4.0,
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
        near_goal_bonus : float
            Reward bonus applied when the agent is within one radius of
            the objective.
        ke_tau : float
            Overall strength of the KE term in the potential (larger =
            less important). See class docstring.
        ke_gate : float
            Distance decay rate of KE sensitivity (larger = KE only
            matters very close to the goal). See class docstring.

        Returns
        -------
        SingleRoller
            The constructed environment. Call :meth:`reset` before use.
        """
        dim = 3
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "near_goal_bonus": jnp.asarray(near_goal_bonus, dtype=float),
            "ke_tau": jnp.asarray(ke_tau, dtype=float),
            "ke_gate": jnp.asarray(ke_gate, dtype=float),
            "delta": jnp.zeros_like(state.pos),
            "prev_dist": jnp.zeros_like(state.rad),
            "prev_ke": jnp.zeros_like(state.rad),
            "action": jnp.zeros_like(state.ang_vel),
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
            domain_type="reflect",
            domain_kw={"box_size": box, "anchor": [0.0, 0.0, -1.0 * rad_val]},
            force_manager_kw={
                "gravity": [0.0, 0.0, -1.0],
                "force_functions": (frictional_wall_force,),
            },
            dt=2e-3,
        )
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        dist = norm(delta)
        env.env_params["delta"] = delta
        env.env_params["prev_dist"] = dist

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        ke_r = thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        env.env_params["prev_ke"] = ke_t + ke_r

        env.env_params["action"] = jnp.zeros_like(env.state.ang_vel)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.step")
    def step(env: SingleRoller, action: jax.Array) -> Environment:
        """Apply a torque action and advance the physics by one step.

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
        env.env_params["action"] = reshaped_action
        torque = reshaped_action - env.env_params["friction"] * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)
        env.env_params["prev_dist"] = norm(env.env_params["delta"])
        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        ke_r = thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        env.env_params["prev_ke"] = ke_t + ke_r
        env.state, env.system = env.system.step(env.state, env.system)
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["delta"] = delta
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.observation")
    def observation(env: SingleRoller) -> jax.Array:
        """Per-agent observation vector.

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
        delta = env.env_params["delta"]
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
        r"""Return the per-agent rewards.

        Potential-based shaping with a proximity-gated KE term:

        .. math::

           \varphi(d, K) = \exp\!\left(-2 d - \frac{K}{\text{ke\_tau}}\,e^{-\text{ke\_gate} \cdot d}\right)

        The gate :math:`e^{-\text{ke\_gate} \cdot d}` suppresses the KE
        term away from the objective, so fast motion is free until the
        agent is close. ``ke_tau`` sets the overall strength of the
        penalty.

        Per-step reward:

        .. math::

           \mathrm{rew}_t = \frac{\varphi(d_t, K_t) - \varphi(d_{t-1}, K_{t-1}) + b \cdot \mathbb{1}[d_t \le r]}{b}

        where :math:`b` is the near-goal bonus and :math:`r` is the agent radius.

        Returns
        -------
        jax.Array
            Shape ``(N,)``.

        """
        curr_dist = norm(env.env_params["delta"])
        prev_dist = env.env_params["prev_dist"]

        tau = env.env_params["ke_tau"]
        alpha = env.env_params["ke_gate"]
        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        ke_r = thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        ke_curr = ke_t + ke_r

        phi_curr = jnp.exp(-2 * curr_dist - ke_curr * jnp.exp(-alpha * curr_dist) / tau)
        phi_prev = jnp.exp(
            -2 * prev_dist
            - env.env_params["prev_ke"] * jnp.exp(-alpha * prev_dist) / tau
        )
        shaping = phi_curr - phi_prev
        near = env.env_params["near_goal_bonus"] * (
            curr_dist <= env.state.rad[0]
        ).astype(float)
        return (shaping + near) / env.env_params["near_goal_bonus"]

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleRoller.done")
    def done(env: SingleRoller) -> jax.Array:
        """``True`` when ``step_count`` exceeds ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

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
