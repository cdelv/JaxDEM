# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning environments.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Dict, Any, Tuple, ClassVar, Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

from ..factory import Factory

from ..state import State
from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Environment(Factory, ABC):
    """
    Defines the interface for environments.

    - Let **A** = number of agents (A ≥ 1). **Single-agent envs must still use A=1.**
    - Observations and actions are **flattened per agent** to fixed sizes. Use
      ``observation_space_shape`` and ``action_space_shape`` to reshape if needed.

    **Required shapes**
    - Observation: ``(A, observation_space_size)``
    - Action (input to :meth:`step`): ``(A, action_space_size)``
    - Reward: ``(A,)``
    - Done: scalar boolean for the **whole environment**

    TO DO: truncated data field: per agent termination flag
    TO DO: render method
    """

    state: "State"
    """
    Simulation state
    """

    system: "System"
    """
    Simulation system's configuration
    """

    env_params: Dict[str, Any]
    """
    Environment specific parameters
    """

    max_num_agents: int = field(default=0, metadata={"static": True})
    """
    Maximun number of active agents in the environment
    """

    action_space_size: int = field(default=0, metadata={"static": True})
    """
    Flattened action size per agent: actions passed to :meth:`step` are shape ``(A, action_space_size)``.
    """

    action_space_shape: Tuple[int, ...] = field(default=(), metadata={"static": True})
    """
    Original per-agent action shape (useful for reshaping inside the env).
    """

    observation_space_size: int = field(default=0, metadata={"static": True})
    """
    Flattened observation size per agent: :meth:`observation` returns shape ``(A, observation_space_size)
    """

    observation_space_shape: Tuple[int, ...] = field(
        default=(), metadata={"static": True}
    )
    """
    Original per-agent observation shape (useful for reshaping inside the env).
    """

    _base_env_cls: ClassVar[Type["Environment"]]

    @staticmethod
    @abstractmethod
    @jax.jit
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Initialize the environment to a valid start state.

        Parameters
        ----------
        env: Environment
            Instance of the environment.

        key : jax.random.PRNGKey
            Jax random numbers key

        Returns
        -------
        Tuple[Environment, ArrayLike]
            Freshly initialized environment and new random numbers key.

        Raises
        ------
        ErrorCode
            This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def reset_if_done(
        env: "Environment", done: jax.Array, key: ArrayLike
    ) -> "Environment":
        """
        Conditionally resets the environment if the environment has reached a terminal state.

        This method checks the `done` flag and, if `True`, calls the environment's
        `reset` method to reinitialize the state. Otherwise, it returns the current
        environment unchanged.

        Parameters
        ----------
        env : Environment
            The current environment instance.

        done : jax.Array
            A boolean flag indicating whether the environment has reached a terminal state.

        key : jax.random.PRNGKey
            JAX random number generator key used for reinitialization.

        Returns
        -------
        Environment
            Either the freshly reset environment (if `done` is True) or the unchanged
            environment (if `done` is False).
        """
        base_cls = getattr(env.__class__, "_base_env_cls", env.__class__)
        return jax.lax.cond(
            done,
            lambda _: base_cls.reset(env, key),
            lambda _: env,
            operand=None,
        )

    @staticmethod
    @abstractmethod
    @jax.jit
    def step(env: "Environment", action: jax.Array) -> "Environment":
        """
        Advance the simulation by one step using **per-agent** actions.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : System
            The vector of actions each agent on the environment should take.

        Returns
        -------
        Environment
            The updated envitonment state.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        """
        Return the **per-agent** observation vector.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Vector corresponding to the environment observation.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def reward(env: "Environment") -> jax.Array:
        """
        Return the **per-agent** immediate rewards.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Vector corresponding to all the agent's rewards based on the current environment state.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def done(env: "Environment") -> jax.Array:
        """
        Return a bool indicating when the environment ended.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool indicating when the environment ended
        """
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def info(env: "Environment") -> Dict:
        """
        Return auxiliary diagnostic information.

        By default, returns an empty dict. Subclasses may override to
        provide environment specific information.

        Parameters
        ----------
        env : Environment
            The current state of the environment.

        Returns
        -------
        Dict
            A dictionary with aditional information of the environment.
        """
        return dict()


@Environment.register("singleNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class SingleNavigator(Environment):
    """
    Defines an environment with a single sphere that has to travel to
    a pre-defined point in space.
    """

    @classmethod
    def Create(
        cls,
        dim: int = 2,
        min_box_size=1.0,
        max_box_size=2.0,
        max_steps=5000,
    ):
        """
        Custom factory method for this environment.
        """
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(dim)

        env_params = dict(
            objective=jnp.zeros((N, dim)),
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_steps=max_steps,
            prev_pos=state.pos,
        )
        action_space_size = dim
        action_space_shape = (dim,)
        observation_space_size = 3 * dim
        observation_space_shape = (3, dim)

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            max_num_agents=N,
            action_space_size=action_space_size,
            action_space_shape=action_space_shape,
            observation_space_size=observation_space_size,
            observation_space_shape=observation_space_shape,
        )

    @staticmethod
    @jax.jit
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Creates a particle inside the domain at a random initial position
        with a random initial velocity.

        Parameters
        ----------
        env: Environment
            Current environment

        key : jax.random.PRNGKey
            Jax random numbers key

        Returns
        -------
        Environment
            Freshly initialized environment.
        """
        key, key_pos, key_vel, key_box, key_objective = jax.random.split(key, 5)

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

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )

        rad = rad * jnp.ones(N)
        state = State.create(pos=pos, vel=vel, rad=rad)
        system = System.create(
            env.state.dim,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=jnp.zeros_like(box)),
        )
        env = replace(env, state=state, system=system)

        return env

    @staticmethod
    @jax.jit
    def step(env: "Environment", action: jax.Array) -> "Environment":
        """
        Advances the simulation state by a time steps.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : System
            The vector of actions each agent on the environment should take.

        Returns
        -------
        Environment
            The updated envitonment state.
        """
        env.env_params["prev_pos"] = env.state.pos
        env = replace(
            env,
            state=replace(
                env.state,
                accel=action.reshape(env.max_num_agents, *env.action_space_shape),
            ),
        )
        state, system = env.system.step(env.state, env.system)
        env = replace(env, state=state, system=system)
        return env

    @staticmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        """
        Return a vector corresponding to the environment observation.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Vector corresponding to the environment observation.
        """
        return jnp.concatenate(
            [env.state.pos, env.env_params["objective"], env.state.vel], axis=-1
        )

    @staticmethod
    @jax.jit
    def reward(env: "Environment") -> jax.Array:
        """
        Return a vector corresponding to all the agent's rewards based on the current environment state.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Vector corresponding to all the agent's rewards based on the current environment state.
        """
        # pos = env.state.pos
        # prev_pos = env.env_params["prev_pos"]
        # objective = env.env_params["objective"]

        # d1 = env.system.domain.displacement(prev_pos, objective, env.system)
        # d2 = env.system.domain.displacement(pos, objective, env.system)

        # d1 = jnp.linalg.norm(d1, ord=1)
        # d2 = jnp.linalg.norm(d2, ord=1)

        # inside = d2 < 0.1 * env.state.rad[0]

        # closer = d2 < d1
        # reward = 2.0 * closer - 1.0 + 0.1 * inside

        distance = jnp.linalg.norm(
            env.system.domain.displacement(
                env.state.pos, env.env_params["objective"], env.system
            ),
            ord=1,
        )
        reward = 1.0 - distance

        return jnp.asarray(reward).reshape(env.max_num_agents)

    @staticmethod
    @jax.jit
    def done(env: "Environment") -> jax.Array:
        """
        Return a bool indicating when the environment ended.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool indicating when the environment ended
        """
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])


@Environment.register("multiNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class MultiNavigator(Environment):
    """
    Defines an environment with a multiple spheres that have to travel to
    a pre-defined point in space.
    """

    @classmethod
    def Create(
        cls,
        N: int = 2,
        dim: int = 2,
        min_box_size=1.0,
        max_box_size=2.0,
        max_steps=10000,
    ):
        """
        Custom factory method for this environment.
        """
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(dim)

        env_params = dict(
            objective=jnp.zeros((N, dim)),
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_steps=max_steps,
            prev_pos=state.pos,
        )
        action_space_size = dim
        action_space_shape = (dim,)
        observation_space_size = 2 * N * dim + dim
        observation_space_shape = (3, dim)

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            max_num_agents=N,
            action_space_size=action_space_size,
            action_space_shape=action_space_shape,
            observation_space_size=observation_space_size,
            observation_space_shape=observation_space_shape,
        )

    @staticmethod
    @jax.jit
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Creates a particle inside the domain at a random initial position
        with a random initial velocity.

        Parameters
        ----------
        env: Environment
            Current environment

        key : jax.random.PRNGKey
            Jax random numbers key

        Returns
        -------
        Environment
            Freshly initialized environment.
        """
        key, key_pos, key_vel, key_box, key_objective = jax.random.split(key, 5)

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

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )

        rad = rad * jnp.ones(N)
        state = State.create(pos=pos, vel=vel, rad=rad)
        system = System.create(
            env.state.dim,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=jnp.zeros_like(box)),
        )
        env = replace(env, state=state, system=system)

        return env

    @staticmethod
    @jax.jit
    def step(env: "Environment", action: jax.Array) -> "Environment":
        """
        Advances the simulation state by a time steps.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : System
            The vector of actions each agent on the environment should take.

        Returns
        -------
        Environment
            The updated envitonment state.
        """
        env.env_params["prev_pos"] = env.state.pos
        env = replace(
            env,
            state=replace(
                env.state,
                accel=action.reshape(env.max_num_agents, *env.action_space_shape),
            ),
        )
        state, system = env.system.step(env.state, env.system)
        env = replace(env, state=state, system=system)
        return env

    @staticmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        """
        Return a vector corresponding to the environment observation.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Vector corresponding to the environment observation.
        """
        obs = jnp.concatenate(
            [env.state.pos.reshape(-1), env.state.vel.reshape(-1)], axis=0
        )
        obs = jnp.broadcast_to(obs, (env.max_num_agents, obs.size))
        return jnp.concatenate([obs, env.env_params["objective"]], axis=1)

    @staticmethod
    @jax.jit
    def reward(env: "Environment") -> jax.Array:
        # --- base term: closer to objective is better ---
        pos = env.state.pos
        obj = env.env_params["objective"]
        sys = env.system

        distance = jnp.linalg.norm(
            sys.domain.displacement(pos, obj, sys),
            ord=1,
            axis=-1,
        )
        base = 1.0 - distance

        # --- collision penalty (same nested vmap style as compute_force) ---
        N = pos.shape[0]
        I = jax.lax.iota(jnp.int32, N)
        rad = env.state.rad

        # center-to-center separation under the domain metric
        def sep(i, j):
            d = sys.domain.displacement(pos[i], pos[j], sys)
            return jnp.linalg.norm(d, ord=2)

        # pairwise separations
        S = jax.vmap(lambda i: jax.vmap(lambda j: sep(i, j))(I))(I)  # (N, N)
        Rsum = rad[:, None] + rad[None, :]

        # collide if separation < sum of radii; ignore self-pairs
        mask_offdiag = ~jnp.eye(N, dtype=bool)
        collided = (S < 3 * Rsum) & mask_offdiag

        # penalty: 1 point per collision partner (tune if you like)
        num_collisions = collided.sum(axis=1).astype(base.dtype)
        penalty = 0.1 * num_collisions

        reward = base - penalty
        return reward.reshape(env.max_num_agents)

    @staticmethod
    @jax.jit
    def done(env: "Environment") -> jax.Array:
        """
        Return a bool indicating when the environment ended.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool indicating when the environment ended
        """
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])


# @Environment.register("pusher")
# @jax.tree_util.register_dataclass
# @dataclass(slots=True, frozen=True)
# class PusherNavigator(Environment):
#     """
#     Defines an environment where there is a single sphere that has to travel to
#     a pre defined point in space.
#     """

#     env_params: Dict[str, Any] = field(
#         default_factory=lambda: {
#             "min_box_size": 1.0,
#             "max_box_size": 1.0,
#             "max_steps": 50000,
#             "objective": None,
#             "prev_pos": None,
#         },
#     )
#     """
#     Environment specific parameters.
#     """

#     action_space_size: Tuple[int, ...] = field(default=(2,), metadata={"static": True})
#     """
#     Shape of the action space
#     """

#     observation_space_size: Tuple[int, ...] = field(
#         default=(12,), metadata={"static": True}
#     )
#     """
#     Shape of the observation space
#     """

#     @staticmethod
#     @jax.jit
#     def reset(env: "Environment", key: ArrayLike) -> "Environment":
#         """
#         Creates a particle inside the domain at a random initial position
#         with a random initial velocity.

#         Parameters
#         ----------
#         env: Environment
#             Current environment

#         key : jax.random.PRNGKey
#             Jax random numbers key

#         Returns
#         -------
#         Environment
#             Freshly initialized environment.
#         """
#         key, key_pos, key_vel, key_box, key_objective = jax.random.split(key, 5)

#         dim = 2
#         box = jax.random.uniform(
#             key_box,
#             (dim,),
#             minval=env.env_params["min_box_size"],
#             maxval=env.env_params["max_box_size"],
#             dtype=float,
#         )

#         rad = 0.065
#         min_pos = rad * jnp.ones_like(box)
#         pos = jax.random.uniform(
#             key_pos,
#             (2, dim),
#             minval=min_pos,
#             maxval=box - min_pos,
#             dtype=float,
#         )
#         rad = rad * jnp.ones(2)

#         objective = jax.random.uniform(
#             key_objective,
#             (1, dim),
#             minval=min_pos,
#             maxval=box - min_pos,
#             dtype=float,
#         )
#         env.env_params["objective"] = objective

#         vel = jnp.zeros_like(pos)

#         state = State.create(
#             pos=pos,
#             vel=vel,
#             rad=rad,
#         )
#         system = System.create(
#             env.state.dim,
#             dt=0.001,
#             domain_type="reflect",
#             domain_kw=dict(box_size=box, anchor=jnp.zeros_like(box)),
#         )
#         env = replace(env, state=state, system=system)

#         # Just checking if max_steps is in the dict
#         max_time = env.env_params["max_steps"]
#         env.env_params["prev_pos"] = state.pos

#         return env

#     @staticmethod
#     @jax.jit
#     def step(env: "Environment", action: jax.Array) -> "Environment":
#         env.env_params["prev_pos"] = env.state.pos
#         action = jnp.reshape(action, (-1,))
#         accel = env.state.accel.at[0].set(env.state.accel[0] + action)
#         accel -= 0.1 * env.state.vel
#         env = replace(env, state=replace(env.state, accel=accel))
#         state, system = env.system.step(env.state, env.system)
#         env = replace(env, state=state, system=system)
#         return env

#     @staticmethod
#     @jax.jit
#     def observation(env: "Environment") -> jax.Array:
#         """
#         Return a vector corresponding to the environment observation.

#         Parameters
#         ----------
#         env : Environment
#             The current environment.

#         Returns
#         -------
#         jax.Array
#             Vector corresponding to the environment observation.
#         """

#         return jnp.concatenate(
#             [
#                 env.state.pos.flatten(),
#                 env.env_params["objective"].flatten(),
#                 env.state.vel.flatten(),
#                 env.system.domain.box_size,
#             ],
#         )

#     @staticmethod
#     @jax.jit
#     def reward(env: "Environment") -> jax.Array:
#         """
#         Return a vector corresponding to all the agent's rewards based on the current environment state.

#         Parameters
#         ----------
#         env : Environment
#             The current environment.

#         Returns
#         -------
#         jax.Array
#             Vector corresponding to all the agent's rewards based on the current environment state.
#         """
#         prev_pos = env.env_params["prev_pos"]
#         current_pos = env.state.pos
#         objective = env.env_params["objective"]

#         r1 = env.system.domain.displacement(prev_pos[0], objective, env.system)
#         r2 = env.system.domain.displacement(current_pos[0], objective, env.system)
#         d = jnp.sum(r2 * r2) - jnp.sum(r1 * r1) <= 0

#         reward = 2.0 * d - 1.0
#         return jnp.asarray(reward)

#     @staticmethod
#     @jax.jit
#     def done(env: "Environment") -> jax.Array:
#         """
#         Return a bool indicating when the environment ended.

#         Parameters
#         ----------
#         env : Environment
#             The current environment.

#         Returns
#         -------
#         jax.Array
#             A bool indicating when the environment ended
#         """


# #         return jnp.asarray(env.system.step_count > env.env_params["max_steps"])
