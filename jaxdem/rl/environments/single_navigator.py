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
    ``-friction * vel``. Each physics step smooths the requested force:

    .. math::

       \mathbf{u}_t = \alpha\,\mathbf{a}_t
           + (1 - \alpha)\,\mathbf{u}_{t-1}

    Here :math:`\mathbf{a}_t` is the requested force at physics step
    :math:`t`, :math:`\mathbf{u}_t` is the applied force before drag, and
    :math:`\mathbf{u}_{t-1}` is the immediately preceding applied force.
    The parameter :math:`\alpha` is ``action_alpha``. Reset initializes
    :math:`\mathbf{u}_0 = \mathbf{0}`; checkpoints preserve this actuator
    state. With :math:`\alpha = 0.22`,
    a constant request completes 99% of its transition in 19 physics steps.

    The reward uses an exponential distance potential
    measured from the action-start checkpoint to the live endpoint:

    .. math::

       \varphi(d) = \exp\!\left(-2 d \right)

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
    Unit direction to objective   ``dim``
    Clamped displacement          ``dim``
    Velocity                      ``dim``
    ============================  =========

    With ``dt = 0.002`` and ``skip_frames = 49``, each action spans 50
    physics steps (0.1 seconds). One checkpoint captures the distance
    before the action; the reward uses the live distance after all accepted
    physics steps. Only ``prev_dist`` is stored as reward history.
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
        action_alpha: float = 0.22,
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
        action_alpha : float
            Fraction of the requested force applied by the smoothing update
            each physics step, in ``[0, 1]``. One disables smoothing.

        Returns
        -------
        SingleNavigator
            The constructed environment. Call :meth:`reset` before use.

        """
        if not 0.0 <= action_alpha <= 1.0:
            raise ValueError("action_alpha must be in [0, 1]")
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
            "action_alpha": jnp.asarray(action_alpha, dtype=float),
            "applied_action": jnp.zeros_like(state.force),
            "prev_dist": jnp.zeros_like(state.rad),
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
        env.env_params["prev_dist"] = norm(delta)
        env.env_params["applied_action"] = jnp.zeros_like(env.state.force)

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.step")
    def step(env: SingleNavigator, action: jax.Array) -> Environment:
        """Smooth the requested force, then advance physics with viscous drag.

        Smoothing uses the applied action from the immediately preceding
        physics step, including across policy actions and checkpoints.

        The historical distance remains fixed throughout the action. Use
        ``utils.advance_action`` to checkpoint once before repeating physics
        steps. Observations and rewards always read the live state.

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
        alpha = env.env_params["action_alpha"]
        applied_action = (
            alpha * reshaped_action + (1 - alpha) * env.env_params["applied_action"]
        )
        env.env_params["applied_action"] = applied_action
        force = applied_action - env.state.vel * env.env_params["friction"]
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.checkpoint")
    def checkpoint(env: SingleNavigator, action: jax.Array) -> Environment:
        """Save the starting distance before the next action interval.

        Store the live distance in ``prev_dist``. Physics steps preserve it,
        and reward compares it with the live endpoint after all repeated
        physics steps. Current quantities are computed directly from state.

        Parameters
        ----------
        env : SingleNavigator
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
    @partial(jax.named_call, name="SingleNavigator.observation")
    def observation(env: SingleNavigator) -> jax.Array:
        """Build per-agent observations directly from the live physical state.

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
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        return jnp.concatenate(
            [
                unit(delta),
                jnp.clip(delta, -3.0, 3.0),
                env.state.vel,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SingleNavigator.reward")
    def reward(env: SingleNavigator) -> jax.Array:
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
            Per-agent rewards of shape ``(N,)``, where ``N = 1``.

        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        phi_curr = jnp.exp(-2 * norm(delta))
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
