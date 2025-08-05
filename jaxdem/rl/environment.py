# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning environments.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..factory import Factory

from ..state import State
from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Environment(Factory["Environment"], ABC):
    """
    Defines the interface for environments.
    """

    state: "State" = field(default=State.create(jnp.zeros((1, 2))))
    """
    Simulation state
    """

    system: "System" = field(default=System.create(2))
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
        # Construct lines WITHOUT trailing newlines.
        # Use !s for str() representation of nested objects for more concise output,
        # or !r if you want the full repr of nested objects too.
        # For typical printing, !s is usually what you want for nested objects.
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
    def reset(env: "Environment", key: ArrayLike) -> Tuple["Environment", ArrayLike]:
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
        Tuple["Environment", ArrayLike]
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
        EnvState
            The updated envitonment state.
        """
        raise NotImplementedError

    # @staticmethod
    # @partial(jax.jit, static_argnames=("n"))
    # def trajectory_rollout(
    #     state: "State", system: "System", n: int
    # ) -> Tuple["State", "System", Tuple["State", "System"]]:
    #     def body(carry, _):
    #         st, sys = carry
    #         st, sys = sys.step(st, sys)
    #         return (st, sys), (st, sys)

    #     (final_state, final_system), traj = jax.lax.scan(
    #         body, (state, system), xs=None, length=n
    #     )
    #     return final_state, final_system, traj

    @staticmethod
    @abstractmethod
    @jax.jit
    def observation(env: "Environment", env_params: Optional[Dict] = None) -> jax.Array:
        """
        Return a vector corresponding to the environment observation.

        Parameters
        ----------
        env : Environment
            The current environment.

        env_params : Dict
            Additional environment specific information required to perform the observation.

        Returns
        -------
        jax.Array
            Vector corresponding to the environment observation.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def reward(env: "Environment", env_params: Optional[Dict] = None) -> jax.Array:
        """
        Return a vector corresponding to all the agent's rewards based on the current environment state.

        Parameters
        ----------
        env : Environment
            The current environment.

        env_params : Dict
            Additional environment specific information required to perform the observation.

        Returns
        -------
        jax.Array
            Vector corresponding to all the agent's rewards based on the current environment state.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def done() -> bool:
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

        """
        return dict()


@Environment.register("singleNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SingleNavigator(Environment):
    """
    Defines the interface for environments.
    """

    env_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "min_box_size": 10,
            "max_box_size": 10,
            "objective": None,
        },
    )

    action_space_size: Tuple[int, ...] = field(default=(2,), metadata={"static": True})
    """
    Shape of the action space
    """

    observation_space_size: Tuple[int, ...] = field(
        default=(2,), metadata={"static": True}
    )
    """
    Shape of the observation space
    """

    @staticmethod
    @jax.jit
    def reset(env: "Environment", key: ArrayLike) -> Tuple["Environment", ArrayLike]:
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

        min_pos = jnp.ones_like(box)
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

        vel = jax.random.uniform(key_vel, (1, dim), minval=-1, maxval=1, dtype=float)

        env.state = State.create(pos=pos, vel=vel)
        env.system = System.create(
            env.state.dim,
            domain_type="reflect",
            domain_kw=dict(box_size=box),
        )

        return env, key

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
        EnvState
            The updated envitonment state.
        """
        env.state.accel = action
        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        """
        Return a vector corresponding to the environment observation.

        Parameters
        ----------
        env_state : EnvState
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
            env.system.domain.displacement(env.state.pos, env.env_params["objective"])
        )

        reward = -0.1 * distance

        return reward

    @staticmethod
    @jax.jit
    def done() -> bool:
        raise NotImplementedError
