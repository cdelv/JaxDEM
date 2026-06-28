# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where a single agent navigates towards a target."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import jaxdem.utils.thermal as thermal

from ...state import State
from ...system import System
from ...utils.linalg import norm
from . import Environment


@Environment.register("singleNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SingleNavigator(Environment):
    r"""Single-agent navigation environment toward a fixed target.

    The agent controls a force vector that is applied directly to a sphere
    inside a reflective box.  Viscous drag ``-friction * vel`` is added
    each step.  The reward uses potential-based shaping with a
    proximity-gated kinetic-energy term:

    .. math::

       \varphi(d, K) = \exp\!\left(-2 d - \frac{K}{\text{ke\_tau}}\,e^{-\text{ke\_gate} \cdot d}\right)

    where :math:`d` is the distance to the objective, :math:`K` is the
    translational kinetic energy, ``ke_tau`` is the KE scale that sets the
    overall strength of the penalty, and ``ke_gate`` controls how sharply
    KE sensitivity falls off with distance — larger ``ke_gate`` means KE
    only matters very close to the objective.

    The shaping credit is :math:`F_t = \varphi(d_t, K_t) - \varphi(d_{t-1}, K_{t-1})`,
    so kinetic energy is penalised only near the objective — far away the
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
    Unit direction to objective   ``dim``
    Clamped displacement          ``dim``
    Velocity                      ``dim``
    ============================  =========

    If one wants some realistic parameters for training, ``skip_frames = 50``
    will give a response rate of 200 Hz, meaning that ``num_steps_epoch = 100``
    gives a horizon of 0.5 seconds.
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
        near_goal_bonus: float = 0.1,
        ke_tau: float = 2.0,
        ke_gate: float = 6.0,
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
        ke_tau : float
            Overall strength of the KE term in the potential (larger =
            less important). See class docstring.
        ke_gate : float
            Distance decay rate of KE sensitivity (larger = KE only
            matters very close to the goal). See class docstring.

        Returns
        -------
        SingleNavigator
            A freshly constructed environment (call :meth:`reset` before use).

        """
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape, rotation_integrator_type=None)

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
            "curr_dist": jnp.zeros_like(state.rad),
            "curr_ke": jnp.zeros_like(state.rad),
            "prev_ke": jnp.zeros_like(state.rad),
            "action": jnp.zeros_like(state.pos),
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
        """Initialize the environment with a randomly placed particle and velocity.

        Parameters
        ----------
        env: 'SingleNavigator'
            Current environment instance.

        key : jax.random.PRNGKey
            JAX random number generator key.

        Returns
        -------
        Environment
            Freshly initialized environment.

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
        )
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        dist = norm(delta)
        env.env_params["delta"] = delta
        env.env_params["prev_dist"] = dist
        env.env_params["curr_dist"] = dist

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        env.env_params["curr_ke"] = ke_t
        env.env_params["prev_ke"] = ke_t

        env.env_params["action"] = jnp.zeros_like(env.state.pos)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.step")
    def step(env: SingleNavigator, action: jax.Array) -> Environment:
        """Advance one step. Actions are forces; simple drag is applied (-friction * vel).

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
        reshaped_action = action.reshape(env.max_num_agents, *env.action_space_shape)
        env.env_params["action"] = reshaped_action
        force = reshaped_action - env.state.vel * env.env_params["friction"]
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.env_params["prev_dist"] = env.env_params["curr_dist"]
        env.env_params["prev_ke"] = env.env_params["curr_ke"]
        env.state, env.system = env.system.step(env.state, env.system)
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["delta"] = delta
        env.env_params["curr_dist"] = norm(delta)
        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        env.env_params["curr_ke"] = ke_t
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.observation")
    def observation(env: SingleNavigator) -> jax.Array:
        """Build per-agent observations.

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
        direction = (
            delta
            / jnp.where(
                env.env_params["curr_dist"] > 0, env.env_params["curr_dist"], 1.0
            )[:, None]
        )
        return jnp.concatenate(
            [
                direction,
                jnp.clip(delta, -3.0, 3.0),
                env.state.vel,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.reward")
    def reward(env: SingleNavigator) -> jax.Array:
        r"""Returns a vector of per-agent rewards.

        Potential-based shaping with a proximity-gated KE term:

        .. math::

           \varphi(d, K) = \exp\!\left(-2 d - \frac{K}{\text{ke\_tau}}\,e^{-\text{ke\_gate} \cdot d}\right)

        The gate :math:`e^{-\text{ke\_gate} \cdot d}` suppresses the KE
        term away from the objective, so fast motion is free until the
        agent is close; ``ke_tau`` sets the overall strength of the
        penalty.

        Per-step reward:

        .. math::

           \mathrm{rew}_t = \frac{\varphi(d_t, K_t) - \varphi(d_{t-1}, K_{t-1}) + b \cdot \mathbb{1}[d_t \le r]}{b}

        where :math:`b` is the near-goal bonus and :math:`r` is the agent radius.

        Parameters
        ----------
        env : Environment
            Current environment.

        Returns
        -------
        jax.Array
            Shape ``(N,)``.

        """
        tau = env.env_params["ke_tau"]
        alpha = env.env_params["ke_gate"]
        phi_curr = jnp.exp(
            -2 * env.env_params["curr_dist"]
            - env.env_params["curr_ke"]
            * jnp.exp(-alpha * env.env_params["curr_dist"])
            / tau
        )
        phi_prev = jnp.exp(
            -2 * env.env_params["prev_dist"]
            - env.env_params["prev_ke"]
            * jnp.exp(-alpha * env.env_params["prev_dist"])
            / tau
        )
        shaping = phi_curr - phi_prev
        near = env.env_params["near_goal_bonus"] * (
            env.env_params["curr_dist"] <= env.state.rad[0]
        ).astype(float)
        return (shaping + near) / env.env_params["near_goal_bonus"]

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.done")
    def done(env: SingleNavigator) -> jax.Array:
        """Returns a boolean indicating whether the environment has ended.
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
