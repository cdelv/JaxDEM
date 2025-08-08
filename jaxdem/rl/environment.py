# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning environments.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

from ..factory import Factory

from ..state import State
from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Environment(Factory["Environment"], ABC):
    """
    Defines the interface for environments.
    """

    state: "State" = field(default_factory=lambda: State.create(jnp.zeros((1, 2))))
    """
    Simulation state
    """

    system: "System" = field(default_factory=lambda: System.create(2))
    """
    Simulation system's configuration
    """

    env_params: Dict[str, Any] = field(default_factory=dict)
    """
    Environment specific parameters
    """

    action_space_size: Tuple[int, ...] = field(default=(), metadata={"static": True})
    """
    Shape of the action space
    """

    observation_space_size: Tuple[int, ...] = field(
        default=(), metadata={"static": True}
    )
    """
    Shape of the observation space
    """

    def __str__(self) -> str:
        """
        Method for printing more nicelly.
        """
        lines = [
            f"{self.__class__.__name__}:",
            f"\t{self.state!s}",
            f"\t{self.system!s}",
            f"\tenv_params: {self.env_params!s}",
            f"\tAction Space Size: {self.action_space_size!s}",
            f"\tObservation Space Size: {self.observation_space_size!s}",
        ]
        return "\n\n".join(lines)

    @staticmethod
    @abstractmethod
    @jax.jit
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Returns a correctly initialized state acording to the rules of the environment and a new
        random numbers key.

        Parameters
        ----------
        env: Environment
            Instance of the environment.

        key : jax.random.PRNGKey
            Jax random numbers key

        Returns
        -------
        Environment
            Freshly initialized environment and new random numbers key.

        Raises
        ------
        ErrorCode
            This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
    Defines an environment where there is a single sphere that has to travel to
    a pre defined point in space.
    """

    env_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "min_box_size": 1.0,
            "max_box_size": 1.0,
            "max_steps": 5000,
            "objective": None,
        },
    )
    """
    Environment specific parameters. 
    """

    action_space_size: Tuple[int, ...] = field(default=(2,), metadata={"static": True})
    """
    Shape of the action space
    """

    observation_space_size: Tuple[int, ...] = field(
        default=(8,), metadata={"static": True}
    )
    """
    Shape of the observation space
    """

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

        dim = 2
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
            (1, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )

        objective = jax.random.uniform(
            key_objective,
            (1, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        env.env_params["objective"] = objective

        vel = jax.random.uniform(
            key_vel, (1, dim), minval=-0.05, maxval=0.05, dtype=float
        )

        state = State.create(pos=pos, vel=vel, rad=jnp.asarray([rad]))
        system = System.create(
            env.state.dim,
            domain_type="reflect",
            domain_kw=dict(box_size=box),
        )
        env = replace(env, state=state, system=system)

        # Just checking if max_steps is in the dict
        max_time = env.env_params["max_steps"]

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
        env = replace(env, state=replace(env.state, accel=action - 0.4 * env.state.vel))
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
            [
                env.state.pos.flatten(),
                env.env_params["objective"].flatten(),
                env.state.vel.flatten(),
                env.system.domain.box_size,
            ],
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
        distance = jnp.linalg.norm(
            env.system.domain.displacement(
                env.state.pos, env.env_params["objective"], env.system
            ),
            ord=1,
        )

        # Better rewards the closer it is to the target
        reward = 1 - distance

        # Reward for beeing in the target point
        inside = distance < 0.5 * env.state.rad[0]
        reward += 0.05 * inside

        # lower_bound = env.system.domain.anchor + 1.5 * env.state.rad[:, None]
        # upper_bound = (
        #     env.system.domain.anchor
        #     + env.system.domain.box_size
        #     - 1.5 * env.state.rad[:, None]
        # )
        # outside_lower = env.state.pos < lower_bound
        # outside_upper = env.state.pos > upper_bound
        # contact = jnp.any(outside_upper | outside_lower)

        # # Punishment if hit the wall
        # reward -= 0.1 * contact

        return jnp.asarray([reward])

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
        return jnp.asarray([env.system.step_count > env.env_params["max_steps"]])
